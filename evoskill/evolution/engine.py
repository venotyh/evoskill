"""Evolution engine — selection, mutation, evaluation, and generational cycles."""

from __future__ import annotations

import random
from typing import Callable

from ..core.genome import Mutator
from ..core.skill import Skill, MutationType, create_seed_skill
from ..infra.storage import list_skills, save_skill
from ..infra.config import get_config
from .fitness import FitnessEvaluator, quick_fitness


class EvolutionEngine:
    """Orchestrates the evolutionary cycle for skills."""

    def __init__(
        self,
        population_size: int = 10,
        elite_count: int = 3,
        children_per_generation: int = 6,
        model: str | None = None,
        evaluator: FitnessEvaluator | None = None,
        on_generation: Callable | None = None,
        guided_weight: float = 0.55,
    ):
        cfg = get_config()
        self.population_size = population_size
        self.elite_count = elite_count
        self.children_per_generation = children_per_generation
        self.model = model or cfg.model
        self.provider = cfg.provider
        self.evaluator = evaluator or FitnessEvaluator(model=model)
        self.guided_weight = guided_weight
        self.on_generation = on_generation  # Callback(generation_num, population, new_children)
        self.generation = 0

    def initialize_population(self) -> list[Skill]:
        """Load existing population or seed with the primordial skill."""
        existing = list_skills()
        if existing:
            existing.sort(key=lambda s: s.fitness, reverse=True)
            self.generation = max(s.generation for s in existing)
            return existing[:self.population_size]
        seed = create_seed_skill()
        save_skill(seed)
        return [seed]

    def run_generation(self) -> dict:
        """Execute one complete evolutionary generation.

        Returns a summary dict with generation stats.
        """
        population = self.initialize_population()
        self.generation += 1

        # 1. Sort by fitness (best first)
        population.sort(key=lambda s: s.fitness, reverse=True)

        # 2. Select parents (tournament selection)
        parents = self._select_parents(population)

        # 3. Create children through mutation
        children = []
        for _ in range(self.children_per_generation):
            child = self._create_child(parents)
            children.append(child)

        # 4. Evaluate children (LLM-as-judge on 2 tasks each)
        for child in children:
            quick_fitness(child, num_tasks=2)

        # 5. Merge population + children, sort, prune
        combined = population + children
        combined.sort(key=lambda s: s.fitness, reverse=True)
        new_population = combined[:self.population_size]

        # 6. Persist all new children
        for child in children:
            save_skill(child)

        # 7. Deep evaluate top 2 elite (only if they haven't been evaluated much)
        for elite in new_population[:2]:
            if elite.task_count < 6:
                self.evaluator.evaluate_skill(elite, max_tasks=3)
                save_skill(elite)

        if self.on_generation:
            self.on_generation(self.generation, new_population, children)

        return {
            "generation": self.generation,
            "population_size": len(new_population),
            "children_created": len(children),
            "best_fitness": new_population[0].fitness if new_population else 0,
            "avg_fitness": round(
                sum(s.fitness for s in new_population) / len(new_population), 2
            ) if new_population else 0,
            "best_skill_id": new_population[0].id if new_population else None,
            "best_skill_name": new_population[0].name if new_population else "",
        }

    def _select_parents(self, population: list[Skill]) -> list[Skill]:
        """Tournament selection: pick the best from random subsets."""
        if len(population) <= 2:
            return list(population)

        parents = []
        for _ in range(max(2, self.children_per_generation // 2 + 1)):
            k = min(3, len(population))
            tournament = random.sample(population, k)
            tournament.sort(key=lambda s: s.fitness, reverse=True)
            parents.append(tournament[0])

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for p in parents:
            if p.id not in seen:
                seen.add(p.id)
                unique.append(p)
        return unique

    def _create_child(self, parents: list[Skill]) -> Skill:
        """Create a child skill from one or two parents.

        Strategy weights:
        - guided_weight: LLM-guided mutation (default 55%)
        - 20% crossover: combine two parents
        - remainder: random mutation, maintain diversity
        """
        roll = random.random()
        guided_cutoff = 0.20 + self.guided_weight

        if len(parents) >= 2 and roll < 0.20:
            # Crossover
            p1, p2 = random.sample(parents, 2)
            new_genome, desc = Mutator.crossover(p1.genome, p2.genome)
            mutation_type = MutationType.CROSSOVER
            parent_ids = [p1.id, p2.id]
            name_base = f"cross_{p1.id[:4]}_{p2.id[:4]}"

        elif roll < guided_cutoff and self.guided_weight > 0:
            # LLM-guided mutation — pick the best-scoring parent
            parents.sort(key=lambda s: s.fitness, reverse=True)
            parent = parents[0]
            new_genome, desc = Mutator.mutate_guided(parent.genome, model=self.model, provider=self.provider)
            mutation_type = MutationType.PROMPT_MUTATE
            parent_ids = [parent.id]
            name_base = f"guided_{parent.id[:4]}"

        else:
            # Random mutation — maintain diversity
            parent = random.choice(parents)
            new_genome, mutation_type_str, desc = Mutator.apply_random_mutation(parent.genome)
            mutation_type = MutationType(mutation_type_str)
            parent_ids = [parent.id]
            name_base = f"rand_{mutation_type.value}_{parent.id[:4]}"

        child = Skill(
            genome=new_genome,
            name=f"{name_base}_gen{self.generation}",
            parent_ids=parent_ids,
            generation=self.generation,
            mutation_type=mutation_type,
            mutation_desc=desc,
        )
        return child


def evolve_step(population: list[Skill], num_children: int = 4, guided_weight: float = 0.55) -> list[Skill]:
    """Standalone: run one evolution step on a population. Returns new skills only."""
    engine = EvolutionEngine(
        population_size=len(population) + num_children,
        children_per_generation=num_children,
        guided_weight=guided_weight,
    )
    engine.generation = max((s.generation for s in population), default=0) + 1

    parents = engine._select_parents(population)
    children = []
    for _ in range(num_children):
        child = engine._create_child(parents)
        children.append(child)
        save_skill(child)
    return children
