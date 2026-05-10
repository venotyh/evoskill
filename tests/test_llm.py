"""Integration tests for LLM providers — requires API keys."""

import os
import json
import time
import urllib.request
import urllib.error
import threading
from http.server import HTTPServer

import pytest

from evoskill.llm import LLMClient, ChatResponse, ToolCall


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def require_env(*vars: str) -> str | None:
    """Return skip reason if any required env var is missing."""
    missing = [v for v in vars if not os.environ.get(v)]
    if missing:
        return f"Missing env vars: {', '.join(missing)}"
    return None


def deepseek_model() -> str:
    return os.environ.get("EVOSKILL_DEEPSEEK_MODEL", "deepseek-chat")


# ═══════════════════════════════════════════════════════════════════════════
# LLMClient DeepSeek tests
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestDeepSeekLLMClient:
    """Test LLMClient with DeepSeek provider."""

    def test_simple_chat(self):
        """Basic chat without tools."""
        reason = require_env("DEEPSEEK_API_KEY")
        if reason:
            pytest.skip(reason)

        client = LLMClient(model=deepseek_model(), provider="deepseek")
        resp = client.chat(
            messages=[{"role": "user", "content": "Say hello in exactly one word."}],
            max_tokens=32,
        )

        assert isinstance(resp, ChatResponse)
        assert resp.content is not None
        assert len(resp.content.strip()) > 0
        assert resp.tool_calls == []

    def test_chat_with_system_prompt(self):
        """Chat with a system prompt."""
        reason = require_env("DEEPSEEK_API_KEY")
        if reason:
            pytest.skip(reason)

        client = LLMClient(model=deepseek_model(), provider="deepseek")
        resp = client.chat(
            messages=[{"role": "user", "content": "What language am I speaking?"}],
            system="Always reply in Chinese.",
            max_tokens=64,
        )

        assert isinstance(resp, ChatResponse)
        assert resp.content is not None
        # Should contain Chinese characters
        assert any('一' <= c <= '鿿' for c in resp.content)

    def test_chat_with_tools(self):
        """Chat with tool definitions — model should return a tool call for hard math."""
        reason = require_env("DEEPSEEK_API_KEY")
        if reason:
            pytest.skip(reason)

        client = LLMClient(model=deepseek_model(), provider="deepseek")
        resp = client.chat(
            messages=[{"role": "user", "content": "Calculate 3847 * 2193. You MUST use the calculator tool."}],
            tools=[{
                "name": "calculator",
                "description": "Evaluate a math expression.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Math expression to evaluate"}
                    },
                    "required": ["expression"],
                },
            }],
            max_tokens=128,
        )

        assert isinstance(resp, ChatResponse)
        assert len(resp.tool_calls) > 0, f"Expected tool call, got content: {resp.content}"
        assert resp.tool_calls[0].name == "calculator"
        assert "expression" in resp.tool_calls[0].input

    def test_auto_detect_provider(self):
        """Provider auto-detection from model name."""
        reason = require_env("DEEPSEEK_API_KEY")
        if reason:
            pytest.skip(reason)

        client = LLMClient(model="deepseek-chat")
        assert client.provider == "deepseek"

        resp = client.chat(
            messages=[{"role": "user", "content": "Reply with just 'ok'."}],
            max_tokens=16,
        )
        assert resp.content is not None


# ═══════════════════════════════════════════════════════════════════════════
# Gateway DeepSeek tests
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestDeepSeekGateway:
    """Test the gateway HTTP server routing to DeepSeek."""

    @pytest.fixture(autouse=True)
    def _gateway(self):
        """Start gateway in background, tear down after test."""
        reason = require_env("DEEPSEEK_API_KEY")
        if reason:
            pytest.skip(reason)

        from evoskill.gateway import serve_gateway, _ThreadingHTTPServer, GatewayHandler

        self._server = _ThreadingHTTPServer(("127.0.0.1", 0), GatewayHandler)
        self._port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        time.sleep(0.1)  # Let server start
        yield
        self._server.shutdown()
        self._thread.join(timeout=2)

    @property
    def _base_url(self):
        return f"http://127.0.0.1:{self._port}"

    def _post(self, path: str, body: dict) -> tuple[int, dict]:
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            f"{self._base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_health_endpoint(self):
        """GET /health returns ok."""
        with urllib.request.urlopen(f"{self._base_url}/health", timeout=5) as resp:
            data = json.loads(resp.read())
        assert data["status"] == "ok"

    def test_deepseek_simple_chat(self):
        """POST /v1/chat/completions with a DeepSeek model."""
        status, data = self._post("/v1/chat/completions", {
            "model": deepseek_model(),
            "messages": [{"role": "user", "content": "Say hello in one word."}],
            "max_tokens": 32,
        })

        assert status == 200, f"Gateway error: {data}"
        assert "choices" in data
        assert len(data["choices"]) > 0
        choice = data["choices"][0]
        assert "message" in choice
        assert choice["message"]["role"] == "assistant"
        assert choice["message"]["content"]

    def test_deepseek_with_tools(self):
        """Gateway with tool use via DeepSeek."""
        status, data = self._post("/v1/chat/completions", {
            "model": deepseek_model(),
            "messages": [{"role": "user", "content": "Calculate 3847 * 2193. You MUST use the calculator tool."}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Evaluate a math expression.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string"}
                        },
                        "required": ["expression"],
                    },
                },
            }],
            "max_tokens": 128,
        })

        assert status == 200, f"Gateway error: {data}"
        choice = data["choices"][0]
        msg = choice["message"]
        # DeepSeek should invoke the tool
        assert "tool_calls" in msg, f"Expected tool_calls, got: {msg}"
        assert len(msg["tool_calls"]) > 0
        tc = msg["tool_calls"][0]
        assert tc["function"]["name"] == "calculator"

    def test_bad_request_empty_messages(self):
        """Missing messages returns 400."""
        status, data = self._post("/v1/chat/completions", {
            "model": deepseek_model(),
        })
        assert status == 400
        assert "error" in data

    def test_not_found_path(self):
        """Unknown path returns 404."""
        status, data = self._post("/v1/nonexistent", {})
        assert status == 404
