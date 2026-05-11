"""Skill data model — the core unit of evolution."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from .genome import SkillGenome


class MutationType(str, Enum):
    PROMPT_MUTATE = "prompt_mutate"
    TOOL_ADD = "tool_add"
    TOOL_DROP = "tool_drop"
    PARAM_TUNE = "param_tune"
    CROSSOVER = "crossover"


@dataclass
class Skill:
    """A single skill — an evolvable unit of agent capability."""

    genome: SkillGenome
    name: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    parent_ids: list[str] = field(default_factory=list)
    generation: int = 0
    fitness: float = 0.0
    fitness_history: list[float] = field(default_factory=list)
    mutation_type: Optional[MutationType] = None
    mutation_desc: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    task_count: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "genome": self.genome.to_dict(),
            "parent_ids": self.parent_ids,
            "generation": self.generation,
            "fitness": self.fitness,
            "fitness_history": self.fitness_history,
            "mutation_type": self.mutation_type.value if self.mutation_type else None,
            "mutation_desc": self.mutation_desc,
            "created_at": self.created_at,
            "task_count": self.task_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Skill:
        mt = d.get("mutation_type")
        return cls(
            id=d["id"],
            name=d["name"],
            genome=SkillGenome.from_dict(d["genome"]),
            parent_ids=d.get("parent_ids", []),
            generation=d.get("generation", 0),
            fitness=d.get("fitness", 0.0),
            fitness_history=d.get("fitness_history", []),
            mutation_type=MutationType(mt) if mt else None,
            mutation_desc=d.get("mutation_desc", ""),
            created_at=d.get("created_at", ""),
            task_count=d.get("task_count", 0),
        )

    def record_fitness(self, score: float) -> None:
        """Record a fitness score and update running average."""
        self.fitness_history.append(score)
        self.task_count += 1
        self.fitness = sum(self.fitness_history) / len(self.fitness_history)


def create_seed_skill() -> Skill:
    """Create the primordial seed skill — the root of all evolution."""
    genome = SkillGenome(
        system_prompt=(
            "You are a capable AI assistant. You solve tasks by reasoning step by step "
            "and using available tools when needed. Be thorough and precise."
        ),
        instructions=[
            "Read the task carefully before acting.",
            "Use tools when they help complete the task more effectively.",
            "Explain your reasoning before taking action.",
            "Verify results before concluding.",
        ],
        tool_bindings=["read_file", "write_file", "shell_exec", "web_search"],
        parameters={
            "temperature": 0.7,
            "max_tool_calls": 10,
            "verbose": False,
        },
    )
    return Skill(
        name="primordial-seed",
        genome=genome,
        generation=0,
        mutation_desc="The first skill — created by the gods.",
    )
