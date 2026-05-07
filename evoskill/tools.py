"""Built-in tools available to skills."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any


def tool_schema(name: str, description: str, parameters: dict) -> dict:
    """Build an OpenAI/Anthropic-compatible tool schema."""
    return {
        "name": name,
        "description": description,
        "parameters": parameters,
    }


BUILTIN_TOOLS = [
    tool_schema(
        name="read_file",
        description="Read the contents of a file at the given path.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to read."},
            },
            "required": ["path"],
        },
    ),
    tool_schema(
        name="write_file",
        description="Write content to a file, creating or overwriting it.",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to write to."},
                "content": {"type": "string", "description": "Content to write."},
            },
            "required": ["path", "content"],
        },
    ),
    tool_schema(
        name="search_files",
        description="Search for files matching a pattern in a directory.",
        parameters={
            "type": "object",
            "properties": {
                "directory": {"type": "string", "description": "Directory to search in."},
                "pattern": {"type": "string", "description": "Glob pattern (e.g. **/*.py)."},
            },
            "required": ["directory", "pattern"],
        },
    ),
    tool_schema(
        name="shell_exec",
        description="Execute a shell command and return its output.",
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute."},
            },
            "required": ["command"],
        },
    ),
    tool_schema(
        name="web_search",
        description="Search the web for information (simulated — returns mock results).",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
            },
            "required": ["query"],
        },
    ),
]


def execute_tool(name: str, params: dict[str, Any]) -> str:
    """Execute a tool and return its result as a string."""
    if name == "read_file":
        return _read_file(params.get("path", ""))
    elif name == "write_file":
        return _write_file(params.get("path", ""), params.get("content", ""))
    elif name == "search_files":
        return _search_files(params.get("directory", "."), params.get("pattern", "*"))
    elif name == "shell_exec":
        return _shell_exec(params.get("command", ""))
    elif name == "web_search":
        return _web_search(params.get("query", ""))
    else:
        return f"Unknown tool: {name}"


def _read_file(path: str) -> str:
    p = Path(path).expanduser()
    if not p.exists():
        return f"Error: file not found: {path}"
    try:
        content = p.read_text(encoding="utf-8")
        if len(content) > 5000:
            content = content[:5000] + "\n... (truncated)"
        return content
    except Exception as e:
        return f"Error reading file: {e}"


def _write_file(path: str, content: str) -> str:
    p = Path(path).expanduser()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error writing file: {e}"


def _search_files(directory: str, pattern: str) -> str:
    import glob
    try:
        results = glob.glob(pattern, root_dir=directory, recursive=True)
        if not results:
            # Fallback to Path.glob
            results = [str(p) for p in Path(directory).expanduser().glob(pattern)]
        if not results:
            return f"No files matching '{pattern}' found in {directory}"
        return "\n".join(sorted(results)[:50])
    except Exception as e:
        return f"Error searching files: {e}"


def _shell_exec(command: str) -> str:
    # Safety: block known dangerous patterns
    dangerous = ["rm -rf /", "mkfs.", "dd if=", ":(){ :|:& };:", "> /dev/sda"]
    cmd_lower = command.lower()
    for d in dangerous:
        if d in cmd_lower:
            return f"Blocked dangerous command (pattern: {d})"

    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=30, cwd=os.getcwd(),
            encoding="utf-8", errors="replace",
        )
        output = result.stdout
        if result.stderr:
            output += "\n[stderr]\n" + result.stderr
        if not output.strip():
            output = f"(exit code: {result.returncode})"
        if len(output) > 3000:
            output = output[:3000] + "\n... (truncated)"
        return output
    except subprocess.TimeoutExpired:
        return "Error: command timed out (30s)"
    except Exception as e:
        return f"Error executing command: {e}"


def _web_search(query: str) -> str:
    """Simulated web search — returns mock structured results."""
    return (
        f"Web search results for: '{query}'\n\n"
        f"1. Understanding {query} — A comprehensive guide\n"
        f"   https://example.com/guide/{query.replace(' ', '-').lower()}\n"
        f"   Summary: {query} is an important topic in computer science.\n\n"
        f"2. {query} best practices — Tips and tricks\n"
        f"   https://example.com/best-practices/{query.replace(' ', '-').lower()}\n"
        f"   Summary: Experts recommend a systematic approach to {query}.\n\n"
        f"3. {query} tutorial — Step by step\n"
        f"   https://example.com/tutorial/{query.replace(' ', '-').lower()}\n"
        f"   Summary: Learn {query} from scratch with this practical tutorial.\n"
    )
