"""Local LLM proxy gateway — exposes an OpenAI-compatible HTTP endpoint.

Routes to Anthropic, OpenAI, or DeepSeek based on model name prefix.
Uses the same LLMClient translation layer as the rest of the project.
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn


class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """HTTPServer with threading support."""
    daemon_threads = True


class GatewayHandler(BaseHTTPRequestHandler):
    """Handles /v1/chat/completions and /health."""

    def log_message(self, format, *args):
        """Suppress default access logging."""
        pass

    def _json(self, data: dict, code: int = 200):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _bad_request(self, msg: str):
        self._json({"error": {"message": msg, "type": "invalid_request_error"}}, code=400)

    def do_GET(self):
        if self.path in ("/", "/health"):
            self._json({"status": "ok", "endpoint": "/v1/chat/completions"})
        elif self.path == "/v1/models":
            self._json({
                "object": "list",
                "data": [{
                    "id": "evoskill-gateway",
                    "object": "model",
                    "created": 0,
                    "owned_by": "evoskill",
                }],
            })
        else:
            self._json({"error": {"message": "Not Found", "type": "not_found"}}, code=404)

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self._json({"error": {"message": "Not Found", "type": "not_found"}}, code=404)
            return

        # Read and parse request
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            self._bad_request("Empty request body")
            return
        body = self.rfile.read(length)
        try:
            req = json.loads(body)
        except json.JSONDecodeError as e:
            self._bad_request(f"Invalid JSON: {e}")
            return

        model = req.get("model", "claude-sonnet-4-20250514")
        messages = req.get("messages", [])
        tools = _normalize_tools(req.get("tools"))
        max_tokens = req.get("max_tokens", 2048)
        temperature = req.get("temperature", 0.7)

        if not messages:
            self._bad_request("messages is required")
            return

        # Extract system message if present as first message
        system = None
        filtered_msgs = []
        for m in messages:
            if m.get("role") == "system":
                system = m.get("content", "")
            else:
                filtered_msgs.append(m)

        try:
            from .llm import LLMClient
            client = LLMClient(model=model, provider=None)
            resp = client.chat(
                messages=filtered_msgs,
                tools=tools,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system,
            )
        except Exception as e:
            self._json(
                {"error": {"message": str(e), "type": "api_error"}},
                code=500,
            )
            return

        # Build OpenAI-compatible response
        choice = {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": resp.content,
            },
            "finish_reason": "stop",
        }
        if resp.tool_calls:
            choice["message"]["tool_calls"] = [
                {
                    "id": f"call_{tc.name}_{uuid.uuid4().hex[:6]}",
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.input),
                    },
                }
                for tc in resp.tool_calls
            ]

        self._json({
            "id": f"evoskill-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [choice],
        })


def _normalize_tools(tools: list[dict] | None) -> list[dict] | None:
    """Convert OpenAI-format tools to internal format for LLMClient."""
    if not tools:
        return None
    result = []
    for t in tools:
        if "function" in t:
            # OpenAI format: {"type": "function", "function": {"name": ..., ...}}
            inner = t["function"]
            result.append({
                "name": inner["name"],
                "description": inner.get("description", ""),
                "parameters": inner.get("parameters", {}),
            })
        else:
            # Already internal format
            result.append(t)
    return result


def serve_gateway(host: str = "127.0.0.1", port: int = 8765):
    """Start the gateway HTTP server. Blocks until Ctrl+C."""
    server = _ThreadingHTTPServer((host, port), GatewayHandler)
    print(f"EvoSkill LLM Gateway listening on http://{host}:{port}", file=sys.stderr)
    print(f"  Endpoint: http://{host}:{port}/v1/chat/completions", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nGateway stopped.", file=sys.stderr)
        server.shutdown()
