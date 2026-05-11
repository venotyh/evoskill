"""Tests for the evoskill core — no API key required."""

import os
import tempfile

import pytest

from evoskill.core.skill import Skill, create_seed_skill
from evoskill.core.genome import SkillGenome, Mutator
from evoskill.infra.storage import save_skill, load_skill, list_skills, delete_skill
from evoskill.evolution.lineage import LineageTree, sync_lineage_from_disk


@pytest.fixture
def temp_evoskill_home():
    """Use a temp directory for evoskill data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        old = os.environ.get("EVOSKILL_HOME")
        os.environ["EVOSKILL_HOME"] = tmpdir
        yield tmpdir
        if old:
            os.environ["EVOSKILL_HOME"] = old
        else:
            del os.environ["EVOSKILL_HOME"]


class TestSkill:
    def test_create_seed_skill(self):
        skill = create_seed_skill()
        assert skill.name == "primordial-seed"
        assert skill.generation == 0
        assert len(skill.genome.instructions) == 4
        assert len(skill.genome.tool_bindings) == 4
        assert skill.id  # Auto-generated

    def test_skill_to_dict_roundtrip(self):
        skill = create_seed_skill()
        skill.record_fitness(7.5)
        data = skill.to_dict()
        restored = Skill.from_dict(data)
        assert restored.name == skill.name
        assert restored.fitness == 7.5
        assert restored.genome.system_prompt == skill.genome.system_prompt

    def test_record_fitness(self):
        skill = create_seed_skill()
        skill.record_fitness(5.0)
        skill.record_fitness(9.0)
        assert skill.fitness == 7.0
        assert skill.task_count == 2
        assert skill.fitness_history == [5.0, 9.0]


class TestGenome:
    def test_clone(self):
        seed = create_seed_skill()
        clone = seed.genome.clone()
        assert clone.system_prompt == seed.genome.system_prompt
        clone.system_prompt = "changed"
        assert seed.genome.system_prompt != "changed"

    def test_mutate_prompt_adds_instruction(self):
        seed = create_seed_skill()
        g = seed.genome.clone()
        new_g, desc = Mutator.mutate_prompt(g)
        assert desc or len(new_g.instructions) > 0
        assert isinstance(new_g, SkillGenome)

    def test_mutate_tools(self):
        seed = create_seed_skill()
        g = seed.genome.clone()
        new_g, desc = Mutator.mutate_tools(g)
        assert isinstance(new_g, SkillGenome)

    def test_mutate_params(self):
        seed = create_seed_skill()
        g = seed.genome.clone()
        new_g, desc = Mutator.mutate_params(g)
        assert isinstance(new_g, SkillGenome)

    def test_crossover(self):
        s1 = create_seed_skill()
        s2 = create_seed_skill()
        s2.genome.system_prompt = "Be concise and direct."
        child_g, desc = Mutator.crossover(s1.genome, s2.genome)
        assert len(child_g.tool_bindings) > 0
        assert child_g.system_prompt in [s1.genome.system_prompt, s2.genome.system_prompt]
        assert "Crossover" in desc

    def test_apply_random_mutation(self):
        seed = create_seed_skill()
        g = seed.genome.clone()
        new_g, mut_type, desc = Mutator.apply_random_mutation(g)
        assert mut_type in ("prompt_mutate", "tool_add", "tool_drop", "param_tune")
        assert isinstance(new_g, SkillGenome)


class TestStorage:
    def test_save_and_load(self, temp_evoskill_home):
        skill = create_seed_skill()
        path = save_skill(skill)
        assert path.exists()
        loaded = load_skill(skill.id)
        assert loaded is not None
        assert loaded.name == skill.name
        assert loaded.id == skill.id

    def test_list_skills(self, temp_evoskill_home):
        s1 = create_seed_skill()
        s1.name = "alpha"
        s1.id = "skill001"
        s2 = create_seed_skill()
        s2.name = "beta"
        s2.id = "skill002"
        save_skill(s1)
        save_skill(s2)
        skills = list_skills()
        assert len(skills) == 2
        names = {s.name for s in skills}
        assert names == {"alpha", "beta"}

    def test_delete_skill(self, temp_evoskill_home):
        skill = create_seed_skill()
        save_skill(skill)
        assert delete_skill(skill.id)
        assert load_skill(skill.id) is None

    def test_lineage_persistence(self, temp_evoskill_home):
        from evoskill.infra.storage import save_lineage, load_lineage
        data = {"nodes": {"a": {"skill_id": "a", "name": "test", "parent_ids": [], "generation": 0, "fitness": 5.0, "mutation_type": "seed", "mutation_desc": "", "created_at": ""}}, "edges": []}
        save_lineage(data)
        loaded = load_lineage()
        assert "a" in loaded["nodes"]


class TestLineage:
    def test_add_and_retrieve(self, temp_evoskill_home):
        tree = LineageTree()
        skill = create_seed_skill()
        tree.add_skill(skill, persist=True)
        ancestors = tree.ancestors(skill.id)
        assert len(ancestors) == 1
        assert ancestors[0].skill_id == skill.id

    def test_parent_child_relationship(self, temp_evoskill_home):
        tree = LineageTree()
        parent = create_seed_skill()
        parent.id = "parent001"
        tree.add_skill(parent, persist=False)

        child = create_seed_skill()
        child.id = "child001"
        child.parent_ids = ["parent001"]
        child.generation = 1
        child.mutation_desc = "mutated prompt"
        tree.add_skill(child, persist=False)
        tree._rebuild_index()

        # Ancestors of child should include parent
        ancestors = tree.ancestors("child001")
        ancestor_ids = [a.skill_id for a in ancestors]
        assert "parent001" in ancestor_ids
        assert "child001" in ancestor_ids

        # Descendants of parent should include child
        descendants = tree.descendants("parent001")
        descendant_ids = [d.skill_id for d in descendants]
        assert "child001" in descendant_ids
        assert "parent001" not in descendant_ids  # Not a descendant of itself

    def test_roots(self, temp_evoskill_home):
        tree = LineageTree()
        root = create_seed_skill()
        tree.add_skill(root, persist=False)
        child = create_seed_skill()
        child.id = "child1"
        child.parent_ids = [root.id]
        tree.add_skill(child, persist=False)
        tree._rebuild_index()

        roots = tree.roots()
        assert len(roots) == 1
        assert roots[0].skill_id == root.id

    def test_ascii_tree(self, temp_evoskill_home):
        tree = LineageTree()
        root = create_seed_skill()
        tree.add_skill(root, persist=False)
        child = create_seed_skill()
        child.id = "child1"
        child.name = "mutant_child"
        child.parent_ids = [root.id]
        child.generation = 1
        child.fitness = 7.5
        tree.add_skill(child, persist=False)
        tree._rebuild_index()

        ascii_tree = tree.to_ascii_tree()
        assert "primordial-seed" in ascii_tree
        assert "mutant_child" in ascii_tree

    def test_sync_from_disk(self, temp_evoskill_home):
        parent = create_seed_skill()
        parent.id = "sync_parent"
        save_skill(parent)
        child = create_seed_skill()
        child.id = "sync_child"
        child.parent_ids = ["sync_parent"]
        child.generation = 1
        save_skill(child)

        tree = sync_lineage_from_disk()
        ancestors = tree.ancestors("sync_child")
        ancestor_ids = [a.skill_id for a in ancestors]
        assert "sync_parent" in ancestor_ids

    def test_stats(self, temp_evoskill_home):
        tree = LineageTree()
        skill = create_seed_skill()
        skill.fitness = 8.0
        tree.add_skill(skill, persist=False)
        stats = tree.stats()
        assert stats["total_skills"] == 1
        assert stats["best_fitness"] == 8.0


class TestEvolution:
    def test_initialization(self, temp_evoskill_home):
        from evoskill.evolution.engine import EvolutionEngine
        engine = EvolutionEngine(population_size=5, children_per_generation=3)
        pop = engine.initialize_population()
        assert len(pop) >= 1

    def test_evolve_step(self, temp_evoskill_home):
        from evoskill.evolution.engine import evolve_step
        seed = create_seed_skill()
        save_skill(seed)
        children = evolve_step([seed], num_children=3, guided_weight=0)
        assert len(children) == 3
        for child in children:
            assert child.generation == 1
            assert child.parent_ids
            assert child.mutation_type is not None

    def test_create_child(self, temp_evoskill_home):
        from evoskill.evolution.engine import EvolutionEngine
        engine = EvolutionEngine(population_size=5, children_per_generation=3)
        engine.generation = 1
        parents = [create_seed_skill() for _ in range(3)]
        for i, p in enumerate(parents):
            p.id = f"parent_{i}"
        child = engine._create_child(parents)
        assert child.generation == 1
        assert child.id
        assert child.parent_ids
