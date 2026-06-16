"""MCP (Model Context Protocol) bridge.

Gives the crew's local Ollama agents access to MCP servers' tools, and supplies
the same server configs to Claude crew members (which speak MCP natively via the
SDK's `mcp_servers` option).

Server configs are merged from three places:
  1. Built-in defaults — a few genuinely useful Python MCP servers we ship
     enabled (fetch / git / time), run via this venv's python so they work
     without Node or uv.
  2. The panel's config.json  ("mcp_servers": { name: {command,args,env} | {url} }).
  3. ~/.claude.json mcpServers (so anything you've set up for Claude Code is reused).

Because the stdio servers are subprocesses, we connect **per call** (spawn →
initialize → call → close) inside a fresh asyncio loop. That's a little slower
than a persistent session but far simpler and crash-proof; tool discovery is
cached so the per-run overhead is just the calls the model actually makes.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_PY = sys.executable  # the venv python — runs our pip-installed MCP servers

# Useful servers we ship on by default (all pure-Python, pip-installed).
_BUILTINS: dict = {
    "fetch": {"command": _PY, "args": ["-m", "mcp_server_fetch"],
              "desc": "fetch a URL and return clean markdown"},
    "git": {"command": _PY, "args": ["-m", "mcp_server_git"],
            "desc": "git status/diff/log/commit on a repo"},
    "time": {"command": _PY, "args": ["-m", "mcp_server_time"],
             "desc": "current time and timezone conversion"},
}

_schema_cache: list | None = None      # [{server, tool, schema}]
_disabled: set = set()                 # servers that failed to start this session


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def configured_servers() -> dict:
    """name -> {command, args, env}. Built-ins + config.json + ~/.claude.json."""
    servers: dict = {}
    for name, b in _BUILTINS.items():
        servers[name] = {"command": b["command"], "args": list(b["args"]), "env": {}}

    cfg = _load_json(_ROOT / "config.json").get("mcp_servers") or {}
    claude = _load_json(Path.home() / ".claude.json").get("mcpServers") or {}
    for src in (claude, cfg):  # config.json wins over claude
        for name, spec in src.items():
            if not isinstance(spec, dict):
                continue
            if spec.get("disabled"):
                servers.pop(name, None)
                continue
            entry = {"command": spec.get("command"), "args": spec.get("args", []),
                     "env": spec.get("env", {})}
            if spec.get("url"):
                entry["url"] = spec["url"]
            servers[name] = entry
    return servers


def claude_mcp_config() -> dict:
    """Shape Claude's SDK expects for ClaudeAgentOptions.mcp_servers."""
    out = {}
    for name, s in configured_servers().items():
        if name in _disabled:
            continue
        if s.get("url"):
            out[name] = {"type": "http", "url": s["url"]}
        elif s.get("command"):
            out[name] = {"type": "stdio", "command": s["command"],
                         "args": s.get("args", []), "env": s.get("env", {})}
    return out


# --- async plumbing ---------------------------------------------------------
def _server_params(spec: dict):
    from mcp import StdioServerParameters
    env = {**os.environ, **(spec.get("env") or {})}
    return StdioServerParameters(command=spec["command"],
                                 args=spec.get("args", []), env=env)


async def _with_session(spec: dict, fn):
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client
    async with stdio_client(_server_params(spec)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await fn(session)


def _run_async(coro):
    return asyncio.run(coro)


def _text_of(result) -> str:
    parts = []
    for block in getattr(result, "content", []) or []:
        t = getattr(block, "text", None)
        if t:
            parts.append(t)
    return "\n".join(parts) if parts else str(result)


# --- discovery + execution --------------------------------------------------
def discover(force: bool = False) -> list:
    """[{server, tool, name, schema}] across all reachable servers. Cached."""
    global _schema_cache
    if _schema_cache is not None and not force:
        return _schema_cache
    out: list = []
    for name, spec in configured_servers().items():
        if not spec.get("command"):
            continue  # url servers: discovery via the bridge not supported here
        try:
            async def _list(session):
                return await session.list_tools()
            res = _run_async(_with_session(spec, _list))
            for tool in getattr(res, "tools", []) or []:
                full = f"mcp__{name}__{tool.name}"
                out.append({
                    "server": name, "tool": tool.name, "name": full,
                    "schema": {"type": "function", "function": {
                        "name": full,
                        "description": (tool.description or "")[:300],
                        "parameters": tool.inputSchema or {"type": "object", "properties": {}},
                    }},
                })
        except Exception as exc:  # noqa: BLE001 — a bad server must not kill the crew
            _disabled.add(name)
            out.append({"server": name, "tool": None, "name": None,
                        "error": f"{type(exc).__name__}: {exc}"})
    _schema_cache = [o for o in out if o.get("name")]
    return _schema_cache


def schemas() -> list:
    return [o["schema"] for o in discover()]


def tool_names() -> list:
    return [o["name"] for o in discover()]


def execute(full_name: str, args: dict | None = None) -> str:
    if not full_name.startswith("mcp__"):
        return f"Not an MCP tool: {full_name}"
    try:
        _, server, tool = full_name.split("__", 2)
    except ValueError:
        return f"Bad MCP tool name: {full_name}"
    spec = configured_servers().get(server)
    if not spec:
        return f"Unknown MCP server: {server}"

    async def _call(session):
        return await session.call_tool(tool, arguments=args or {})
    try:
        return _text_of(_run_async(_with_session(spec, _call)))[:60_000]
    except Exception as exc:  # noqa: BLE001
        return f"MCP {full_name} error: {type(exc).__name__}: {exc}"


def status() -> dict:
    """Lightweight summary for the UI / health."""
    servers = configured_servers()
    tools = discover()
    by_server: dict = {}
    for t in tools:
        by_server.setdefault(t["server"], []).append(t["tool"])
    return {
        "servers": sorted(servers.keys()),
        "disabled": sorted(_disabled),
        "tools": by_server,
        "tool_count": len(tools),
    }
