# CHANGELOG

## Unreleased

### New Modules
- **`evoskill/llm.py`** — Unified LLM client with single interface for Anthropic, OpenAI, and DeepSeek
  - Auto-detects provider from model name prefix
  - Handles all message/tool format translation internally
  - Provides normalized `ChatResponse` and `ToolCall` dataclasses
- **`evoskill/gateway.py`** — Local LLM proxy HTTP server
  - OpenAI-compatible `/v1/chat/completions` endpoint
  - Auto-routes to Anthropic/OpenAI/DeepSeek based on model name
  - Supports tool calling, system prompts, health check
  - `_normalize_tools()` converts OpenAI-format tools to internal format
- **`tests/test_llm.py`** — 9 integration tests for DeepSeek API (auto-skip when key not set)
  - 4 `LLMClient` tests: simple chat, system prompt, tool calling, auto-detect provider
  - 5 `Gateway` tests: health endpoint, chat, tool calling, validation, 404

### CLI
- `evoskill gateway` command to start the local LLM proxy

### Refactoring
- **`evoskill/agent.py`** — Replaced ~100 lines of inline Anthropic/OpenAI dispatch code with unified `LLMClient`
- **`evoskill/fitness.py`** — Judge scoring now uses `LLMClient` instead of manual provider dispatch
- **`evoskill/genome.py`** — Mutation LLM calls now use `LLMClient` instead of manual provider dispatch

### Evolution
- Added `guided_weight` parameter to `EvolutionEngine` and `evolve_step()` for controlling LLM-guided mutation rate
- Mutation selection logic now respects dynamic weight cutoff instead of hardcoded 0.75

### Config
- `.gitignore` — Added `.mcp.json` (may contain API keys)
- `pyproject.toml` — Registered `integration` pytest marker for API-key-dependent tests

### Test fixes
- `tests/test_evolution.py` — `evolve_step` call passes `guided_weight=0` to avoid LLM dependency in unit tests
