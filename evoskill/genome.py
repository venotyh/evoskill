"""Skill genome — the evolvable DNA of a skill, with mutation operators."""

from __future__ import annotations

import os
import random
from copy import deepcopy
from dataclasses import dataclass, field

ALL_AVAILABLE_TOOLS = ["read_file", "write_file", "shell_exec", "web_search", "search_files"]


@dataclass
class SkillGenome:
    """The evolvable genetic material of a skill."""

    system_prompt: str
    instructions: list[str] = field(default_factory=list)
    tool_bindings: list[str] = field(default_factory=list)
    parameters: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "system_prompt": self.system_prompt,
            "instructions": self.instructions,
            "tool_bindings": self.tool_bindings,
            "parameters": self.parameters,
        }

    @classmethod
    def from_dict(cls, d: dict) -> SkillGenome:
        return cls(
            system_prompt=d.get("system_prompt", ""),
            instructions=d.get("instructions", []),
            tool_bindings=d.get("tool_bindings", []),
            parameters=d.get("parameters", {}),
        )

    def clone(self) -> SkillGenome:
        return deepcopy(self)


class Mutator:
    """Applies evolutionary operators to skill genomes."""

    @staticmethod
    def mutate_guided(
        genome: SkillGenome,
        model: str | None = None,
        provider: str | None = None,
    ) -> tuple[SkillGenome, str]:
        """LLM-guided mutation: ask an LLM to improve the skill's prompt."""
        import sys
        model = model or os.environ.get("EVOSKILL_MODEL", "deepseek-chat")
        provider = provider or os.environ.get("EVOSKILL_PROVIDER", "deepseek")

        mutation_prompt = _build_guided_mutation_prompt(genome)
        llm_response = _call_llm_for_mutation(mutation_prompt, model, provider)
        if not llm_response:
            return Mutator.mutate_prompt(genome)

        return Mutator._apply_guided_response(genome, llm_response)

    @staticmethod
    def _apply_guided_response(genome: SkillGenome, response: str) -> tuple[SkillGenome, str]:
        """Parse LLM response and apply to genome."""
        new = genome.clone()

        # Try to parse sections from the LLM response
        sys_prompt = _extract_section(response, "SYSTEM_PROMPT")
        instructions = _extract_section(response, "INSTRUCTIONS")

        if sys_prompt:
            new.system_prompt = sys_prompt
        if instructions:
            new.instructions = [i.strip() for i in instructions.split("\n") if i.strip()][:10]

        if sys_prompt or instructions:
            desc = f"Guided mutation: {'new system_prompt' if sys_prompt else ''}{' + ' if sys_prompt and instructions else ''}{f'{len(new.instructions)} instructions' if instructions else ''}"
            return new, desc

        # If parsing failed, use the whole response as context
        new.system_prompt += "\n\n" + response[:500]
        return new, f"Guided mutation: appended {len(response[:500])} chars to system_prompt"

    @staticmethod
    def mutate_prompt(genome: SkillGenome, llm_rewrite: str | None = None) -> tuple[SkillGenome, str]:
        """Mutate the system prompt or instructions.

        If llm_rewrite is provided (from an LLM call), use it. Otherwise apply
        a simple structural mutation locally.
        """
        new = genome.clone()
        desc = ""

        if llm_rewrite:
            # LLM-provided mutation: replace or augment instructions
            lines = [l.strip() for l in llm_rewrite.split("\n") if l.strip()]
            if lines:
                choice = random.random()
                if choice < 0.4:
                    new.system_prompt = lines[0]
                    desc = f"Replaced system_prompt: {lines[0][:60]}..."
                elif choice < 0.8:
                    new.instructions = lines[:6]
                    desc = f"Replaced instructions ({len(new.instructions)} rules)"
                else:
                    new.instructions.append(lines[0])
                    desc = f"Added instruction: {lines[0][:60]}..."
        else:
            # Simple local mutation
            tactics = [
                lambda: _add_random_instruction(new),
                lambda: _remove_random_instruction(new),
                lambda: _tweak_system_prompt(new),
            ]
            desc = random.choice(tactics)()

        return new, desc

    @staticmethod
    def mutate_tools(genome: SkillGenome) -> tuple[SkillGenome, str]:
        """Add or remove a tool binding."""
        new = genome.clone()
        current = set(new.tool_bindings)
        available = set(ALL_AVAILABLE_TOOLS)

        can_add = list(available - current)
        can_drop = list(current)

        if can_add and (not can_drop or random.random() < 0.6):
            tool = random.choice(can_add)
            new.tool_bindings.append(tool)
            desc = f"Added tool: {tool}"
        elif can_drop and len(can_drop) > 1:
            tool = random.choice(can_drop)
            new.tool_bindings.remove(tool)
            desc = f"Dropped tool: {tool}"
        else:
            desc = "No tool mutation applied (bounds reached)"

        return new, desc

    @staticmethod
    def mutate_params(genome: SkillGenome) -> tuple[SkillGenome, str]:
        """Tune a parameter randomly."""
        new = genome.clone()
        params = new.parameters

        if "temperature" in params:
            old = params["temperature"]
            params["temperature"] = round(min(1.0, max(0.1, old + random.uniform(-0.2, 0.2))), 2)
            desc = f"Tuned temperature: {old} → {params['temperature']}"
        elif "max_tool_calls" in params:
            old = params["max_tool_calls"]
            params["max_tool_calls"] = max(3, old + random.choice([-2, -1, 1, 2, 5]))
            desc = f"Tuned max_tool_calls: {old} → {params['max_tool_calls']}"
        else:
            params["temperature"] = round(random.uniform(0.3, 1.0), 2)
            desc = f"Set temperature: {params['temperature']}"

        return new, desc

    @staticmethod
    def crossover(a: SkillGenome, b: SkillGenome) -> tuple[SkillGenome, str]:
        """Create a child genome by crossing over two parents."""
        child = SkillGenome(
            system_prompt=random.choice([a.system_prompt, b.system_prompt]),
            instructions=_crossover_lists(a.instructions, b.instructions),
            tool_bindings=list(set(a.tool_bindings + b.tool_bindings)),
            parameters=_crossover_params(a.parameters, b.parameters),
        )
        desc = f"Crossover: {len(child.instructions)} instructions, {len(child.tool_bindings)} tools"
        return child, desc

    @staticmethod
    def apply_random_mutation(genome: SkillGenome, llm_rewrite: str | None = None) -> tuple[SkillGenome, str, str]:
        """Apply a random mutation operator. Returns (new_genome, mutation_type, description)."""
        ops = [
            ("prompt_mutate", lambda: Mutator.mutate_prompt(genome, llm_rewrite)),
            ("tool_add" if random.random() < 0.5 else "tool_drop", lambda: Mutator.mutate_tools(genome)),
            ("param_tune", lambda: Mutator.mutate_params(genome)),
        ]
        # Weight: prompt mutate gets higher priority
        weights = [0.5, 0.3, 0.2]
        op_name, op_fn = random.choices(ops, weights=weights, k=1)[0]
        new_genome, desc = op_fn()
        return new_genome, op_name, desc


def _add_random_instruction(genome: SkillGenome) -> str:
    candidates = [
        "Break complex problems into smaller steps.",
        "Double-check your work before responding.",
        "Use concrete examples when explaining concepts.",
        "Prefer tools over manual reasoning when applicable.",
        "If a tool fails, try an alternative approach.",
        "Keep responses concise but complete.",
        "Structure output with clear headings or bullet points.",
    ]
    new_inst = random.choice(candidates)
    if new_inst not in genome.instructions:
        genome.instructions.append(new_inst)
        return f"Added instruction: {new_inst}"
    return "No change (instruction already exists)"


def _remove_random_instruction(genome: SkillGenome) -> str:
    if len(genome.instructions) > 2:
        removed = genome.instructions.pop(random.randrange(len(genome.instructions)))
        return f"Removed instruction: {removed}"
    return "No removal (minimum instructions reached)"


def _tweak_system_prompt(genome: SkillGenome) -> str:
    suffixes = [
        " Always prioritize accuracy over speed.",
        " Be creative and think outside the box.",
        " Focus on practical, actionable solutions.",
        " Maintain a helpful and friendly tone.",
        " Be rigorous and methodical in your approach.",
    ]
    suffix = random.choice(suffixes)
    if not genome.system_prompt.endswith(suffix):
        genome.system_prompt += suffix
        return f"Appended to system_prompt: {suffix[:50]}..."
    return "No change (suffix already present)"


def _crossover_lists(a: list, b: list) -> list:
    """Take some items from each parent list."""
    if not a or not b:
        return a or b
    cut = random.randint(1, max(1, max(len(a), len(b)) - 1))
    result = a[:cut]
    for item in b[cut:]:
        if item not in result:
            result.append(item)
    return result


def _crossover_params(a: dict, b: dict) -> dict:
    """Randomly pick each param from one parent."""
    merged = {}
    all_keys = set(a.keys()) | set(b.keys())
    for key in all_keys:
        merged[key] = random.choice([a.get(key), b.get(key)]) if key in a and key in b else (a.get(key) or b.get(key))
    return merged


def _build_guided_mutation_prompt(genome: SkillGenome) -> str:
    """Build the prompt for LLM-guided mutation."""
    current_prompt = genome.system_prompt
    current_instructions = "\n".join(f"{i+1}. {inst}" for i, inst in enumerate(genome.instructions))
    current_tools = ", ".join(genome.tool_bindings)

    return f"""You are improving an AI agent's skill configuration. The skill defines how the agent behaves when solving tasks.

CURRENT SKILL:
---
SYSTEM_PROMPT:
{current_prompt}

INSTRUCTIONS:
{current_instructions}

BOUND_TOOLS: {current_tools}
---

Analyze the current skill. Is the system prompt clear and directive? Are the instructions actionable? Would a better skill produce better agent outputs?

Write an IMPROVED version. Make the instructions more specific, actionable, and effective. The agent has access to these tools: {current_tools}.

Reply with this exact format (keep the section headers):

SYSTEM_PROMPT:
<improved system prompt — be more specific and directive>

INSTRUCTIONS:
<improved instructions, one per line, 2-6 rules max, each actionable>

Keep it concise. The system prompt should be 2-5 sentences. Instructions should be short, actionable rules."""


def _call_llm_for_mutation(prompt: str, model: str, provider: str) -> str | None:
    """Call an LLM for guided mutation. Returns response text or None if empty."""
    from .llm import LLMClient as _LLMClient
    client = _LLMClient(model=model, provider=provider)
    resp = client.chat(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
        temperature=0.8,
    )
    return resp.content


def _extract_section(text: str, section: str) -> str | None:
    """Extract content between a section header and the next section or EOF."""
    marker = f"{section}:"
    idx = text.find(marker)
    if idx == -1:
        # Try case-insensitive
        text_lower = text.lower()
        idx = text_lower.find(section.lower() + ":")
        if idx == -1:
            return None

    start = idx + len(marker)
    # Find next all-caps section header
    rest = text[start:]
    import re
    next_section = re.search(r"\n[A-Z_]+:", rest)
    if next_section:
        content = rest[:next_section.start()]
    else:
        content = rest
    return content.strip()
