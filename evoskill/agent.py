"""Agent loop — runs a skill against an LLM backend with tool execution."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .skill import Skill
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
                try:
                    response = self._call_llm(system, messages, active_tools)
                except Exception as e:
                    os.chdir(original_cwd)
                    return {
                        "output": f"LLM call failed: {e}",
                        "tool_calls": tool_calls_made,
                        "rounds": tool_rounds,
                        "success": False,
                        "error": str(e),
                    }

                tool_blocks = self._extract_tool_calls(response)

                if not tool_blocks:
                    os.chdir(original_cwd)
                    return {
                        "output": self._extract_text(response),
                        "tool_calls": tool_calls_made,
                        "rounds": tool_rounds,
                        "success": True,
                    }

                for tb in tool_blocks:
                    tool_name = tb.get("name", "")
                    tool_input = tb.get("input", {})
                    tool_calls_made.append(tool_name)

                    result = execute_tool(tool_name, tool_input)

                    messages.append({"role": "assistant", "content": json.dumps(tb)})
                    messages.append({"role": "user", "content": f"Tool result for {tool_name}:\n{result}"})

                tool_rounds += 1

        os.chdir(original_cwd)
        return {
            "output": "(max tool rounds reached)",
            "tool_calls": tool_calls_made,
            "rounds": tool_rounds,
            "success": False,
        }

    def _call_llm(self, system: str, messages: list[dict], tools: list[dict]) -> Any:
        if self.provider in ("openai", "deepseek"):
            return self._call_openai(system, messages, tools)
        else:
            return self._call_anthropic(system, messages, tools)

    def _call_anthropic(self, system: str, messages: list[dict], tools: list[dict]) -> Any:
        import anthropic

        client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY", "sk-placeholder")
        )

        # Convert tools to Anthropic format
        anthropic_tools = [
            {
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["parameters"],
            }
            for t in tools
        ] if tools else None

        resp = client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=system,
            messages=messages,
            tools=anthropic_tools,
        )
        return resp

    def _call_openai(self, system: str, messages: list[dict], tools: list[dict]) -> Any:
        from openai import OpenAI

        openai_api_key = os.environ.get("OPENAI_API_KEY", "sk-placeholder")
        if openai_api_key and openai_api_key != "sk-placeholder":
            client = OpenAI(openai_api_key)
        else:
            client = OpenAI(
                api_key=os.environ.get("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com"
            )

        openai_tools = [
            {"type": "function", "function": {
                "name": t["name"], "description": t["description"], "parameters": t["parameters"],
            }}
            for t in tools
        ] if tools else None

        full_messages = [{"role": "system", "content": system}] + messages

        resp = client.chat.completions.create(
            model=self.model,
            messages=full_messages,
            tools=openai_tools,
            max_tokens=2048,
        )
        return resp

    def _extract_text(self, response: Any) -> str:
        """Extract text content from an LLM response."""
        if hasattr(response, "content"):
            # Anthropic
            for block in response.content:
                if block.type == "text":
                    return block.text
            return response.content[0].text if response.content else ""
        elif hasattr(response, "choices"):
            # OpenAI
            msg = response.choices[0].message
            return msg.content or ""
        return str(response)

    def _extract_tool_calls(self, response: Any) -> list[dict]:
        """Extract tool use blocks from an LLM response."""
        if hasattr(response, "content"):
            # Anthropic
            tool_blocks = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_blocks.append({"name": block.name, "input": block.input})
            return tool_blocks
        elif hasattr(response, "choices"):
            # OpenAI
            msg = response.choices[0].message
            if msg.tool_calls:
                return [
                    {"name": tc.function.name, "input": json.loads(tc.function.arguments)}
                    for tc in msg.tool_calls
                ]
        return []
