"""Agent loop — runs a skill against an LLM backend with tool execution."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from ..infra.llm import LLMClient
from ..core.skill import Skill
from .tools import BUILTIN_TOOLS, execute_tool


class SkillAgent:
    """Wraps a Skill into an executable agent with LLM + tool loop.

    All tool execution (file ops, shell commands) is sandboxed inside
    a temporary directory that is automatically cleaned up after each run.
    """

    MAX_TOOL_ROUNDS = 8

    def __init__(self, skill: Skill, model: str | None = None):
        self.skill = skill
        self.model = model or os.environ.get("EVOSKILL_MODEL", "claude-sonnet-4-20250514")
        self.provider = os.environ.get("EVOSKILL_PROVIDER", "anthropic")

    def run(self, task: str) -> dict[str, Any]:
        """Run the skill against a task. Returns {output, tool_calls, rounds, success}.

        Executes inside a temp directory that is auto-cleaned on completion.
        """
        messages = [{"role": "user", "content": task}]
        tool_rounds = 0
        tool_calls_made = []

        bound_tool_names = set(self.skill.genome.tool_bindings)
        active_tools = [t for t in BUILTIN_TOOLS if t["name"] in bound_tool_names]

        system = self.skill.genome.system_prompt
        if self.skill.genome.instructions:
            system += "\n\nKey instructions:\n" + "\n".join(
                f"- {inst}" for inst in self.skill.genome.instructions
            )

        # ── sandbox: chdir into a temp directory ──────────────────────
        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory(prefix="evoskill_") as tmpdir:
            os.chdir(tmpdir)

            for _ in range(self.MAX_TOOL_ROUNDS):
                response = self._call_llm(system, messages, active_tools)

                if not response.tool_calls:
                    os.chdir(original_cwd)
                    return {
                        "output": response.content or "",
                        "tool_calls": tool_calls_made,
                        "rounds": tool_rounds,
                        "success": True,
                    }

                messages.append({
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": [{"name": tc.name, "arguments": tc.input} for tc in response.tool_calls],
                })

                for tc in response.tool_calls:
                    tool_calls_made.append(tc.name)
                    result = execute_tool(tc.name, tc.input)
                    messages.append({
                        "role": "tool",
                        "name": tc.name,
                        "content": f"Tool result for {tc.name}:\n{result}",
                    })

                tool_rounds += 1

        os.chdir(original_cwd)
        return {
            "output": "(max tool rounds reached)",
            "tool_calls": tool_calls_made,
            "rounds": tool_rounds,
            "success": False,
        }

    def _call_llm(self, system: str, messages: list[dict], tools: list[dict]) -> Any:
        client = LLMClient(model=self.model, provider=self.provider)
        return client.chat(
            messages=messages,
            tools=tools,
            max_tokens=2048,
            system=system,
        )
