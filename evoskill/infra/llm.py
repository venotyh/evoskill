"""Unified LLM client — single interface for Anthropic, OpenAI, and DeepSeek.

All provider-specific translation (message format, tool schema, response
parsing) lives here. Every LLM call in the project goes through this module.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    name: str
    input: dict


@dataclass
class ChatResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)


def detect_provider(model: str) -> str:
    """Auto-detect provider from model name prefix."""
    m = model.lower()
    if any(m.startswith(p) for p in ("claude-",)):
        return "anthropic"
    if any(m.startswith(p) for p in ("gpt-", "o1-", "o3-", "o4-")):
        return "openai"
    if any(m.startswith(p) for p in ("deepseek-",)):
        return "deepseek"
    return os.environ.get("EVOSKILL_PROVIDER", "anthropic")


class LLMClient:
    """Unified chat client that routes to Anthropic, OpenAI, or DeepSeek.

    Accepts messages in a normalized OpenAI-like format:
        {"role": "user", "content": "..."}
        {"role": "assistant", "content": "...", "tool_calls": [...]}  # optional
        {"role": "tool", "name": "...", "content": "..."}

    Tool schemas use the internal format:
        {"name": "...", "description": "...", "parameters": {...}}
    """

    def __init__(
        self,
        model: str | None = None,
        provider: str | None = None,
        base_url: str | None = None,
    ):
        self.model = model or os.environ.get("EVOSKILL_MODEL", "claude-sonnet-4-20250514")
        self.provider = provider or detect_provider(self.model)
        self.base_url = base_url

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        system: str | None = None,
        timeout: float = 30.0,
    ) -> ChatResponse:
        if self.provider == "anthropic":
            return self._chat_anthropic(messages, tools, max_tokens, temperature, system, timeout)
        else:
            return self._chat_openai_compat(messages, tools, max_tokens, temperature, system, timeout)

    # ── Anthropic path ──────────────────────────────────────────────────

    def _chat_anthropic(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        max_tokens: int,
        temperature: float,
        system: str | None,
        timeout: float,
    ) -> ChatResponse:
        import anthropic

        client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY", "sk-placeholder"),
            timeout=timeout,
        )
        anthropic_msgs = _to_anthropic_messages(messages)
        anthropic_tools = _to_anthropic_tools(tools) if tools else None

        # Anthropic requires at least one user message; if messages
        # start with system or assistant, insert a placeholder.
        if anthropic_msgs and anthropic_msgs[0]["role"] != "user":
            anthropic_msgs.insert(0, {"role": "user", "content": "Hello"})

        resp = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system or None,
            messages=anthropic_msgs,
            tools=anthropic_tools,
        )
        return _from_anthropic_response(resp)

    # ── OpenAI-compat path ──────────────────────────────────────────────

    def _chat_openai_compat(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        max_tokens: int,
        temperature: float,
        system: str | None,
        timeout: float,
    ) -> ChatResponse:
        from openai import OpenAI

        if self.provider == "deepseek":
            client = OpenAI(
                api_key=os.environ.get("DEEPSEEK_API_KEY"),
                base_url=self.base_url or "https://api.deepseek.com",
                timeout=timeout,
            )
        else:
            client = OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
                timeout=timeout,
            )

        openai_msgs = _to_openai_messages(messages, system)
        openai_tools = _to_openai_tools(tools) if tools else None

        resp = client.chat.completions.create(
            model=self.model,
            messages=openai_msgs,
            tools=openai_tools,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return _from_openai_response(resp)


# ── Message translation: normalized → Anthropic ──────────────────────────


def _to_anthropic_messages(messages: list[dict]) -> list[dict]:
    """Convert normalized messages to Anthropic format.

    Handles:
      - "tool" role → user message with tool_result content block
      - assistant with tool_calls → assistant with tool_use blocks
      - Merges consecutive same-role messages (Anthropic requires alternation)
    """
    result: list[dict] = []
    for msg in messages:
        role = msg.get("role", "user")
        blocks: list[dict] = []

        if role == "tool":
            # Tool result → user message with tool_result block
            tool_name = msg.get("name", "tool")
            blocks.append({
                "type": "tool_result",
                "tool_use_id": f"toolu_{uuid.uuid4().hex[:12]}",
                "content": str(msg.get("content", "")),
            })
            role = "user"

        elif role == "assistant":
            text = msg.get("content", "")
            tool_calls = msg.get("tool_calls")
            if text:
                blocks.append({"type": "text", "text": text})
            if tool_calls:
                for tc in tool_calls:
                    name = tc.get("name", "")
                    args = tc.get("arguments", tc.get("input", {}))
                    blocks.append({
                        "type": "tool_use",
                        "id": f"toolu_{uuid.uuid4().hex[:12]}",
                        "name": name,
                        "input": args if isinstance(args, dict) else json.loads(args),
                    })

        else:
            # Regular user message
            blocks.append({"type": "text", "text": str(msg.get("content", ""))})

        if not blocks:
            continue

        # Merge with previous message if same role
        if result and result[-1]["role"] == role:
            result[-1]["content"].extend(blocks)
        else:
            result.append({"role": role, "content": blocks})

    return result


def _to_anthropic_tools(tools: list[dict]) -> list[dict]:
    return [
        {
            "name": t["name"],
            "description": t.get("description", ""),
            "input_schema": t.get("parameters", t.get("input_schema", {})),
        }
        for t in tools
    ]


def _from_anthropic_response(resp: Any) -> ChatResponse:
    content_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for block in resp.content:
        if block.type == "text":
            content_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append(ToolCall(name=block.name, input=dict(block.input)))
    return ChatResponse(
        content="\n".join(content_parts) if content_parts else None,
        tool_calls=tool_calls,
    )


# ── Message translation: normalized → OpenAI ────────────────────────────


def _to_openai_messages(messages: list[dict], system: str | None) -> list[dict]:
    result: list[dict] = []
    extra_system: list[str] = []
    for msg in messages:
        role = msg.get("role")
        if role == "system":
            extra_system.append(str(msg.get("content", "")))
            continue
        entry: dict = {"role": role, "content": msg.get("content")}
        if role == "assistant" and msg.get("tool_calls"):
            entry["tool_calls"] = [
                {
                    "id": f"call_{tc.get('name', 'tool')}_{uuid.uuid4().hex[:6]}",
                    "type": "function",
                    "function": {
                        "name": tc.get("name", ""),
                        "arguments": (
                            json.dumps(tc.get("arguments", tc.get("input", {})))
                            if isinstance(tc.get("arguments", tc.get("input", {})), dict)
                            else tc.get("arguments", "{}")
                        ),
                    },
                }
                for tc in msg["tool_calls"]
            ]
        if role == "tool":
            entry["name"] = msg.get("name", "tool")
        result.append(entry)
    # Merge explicit system param with any inline system messages
    merged = [s for s in ([system] if system else []) + extra_system if s]
    if merged:
        result.insert(0, {"role": "system", "content": "\n\n".join(merged)})
    return result


def _to_openai_tools(tools: list[dict]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("parameters", t.get("input_schema", {})),
            },
        }
        for t in tools
    ]


def _from_openai_response(resp: Any) -> ChatResponse:
    msg = resp.choices[0].message
    tool_calls: list[ToolCall] = []
    if msg.tool_calls:
        for tc in msg.tool_calls:
            args = tc.function.arguments
            tool_calls.append(ToolCall(
                name=tc.function.name,
                input=json.loads(args) if isinstance(args, str) else args,
            ))
    return ChatResponse(content=msg.content, tool_calls=tool_calls)
