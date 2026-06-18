"""Agent tool layer.

Defines the tools an agent (local Ollama model or Claude) can call, exposes them
as OpenAI/Ollama-style function schemas, and executes them by name.

Security model: this is a *local, single-user* surface. The user explicitly opted
into a coding agent with shell access, so there's no sandbox here — instead, tools
that change state or run commands are flagged `danger=True`. The orchestrator
(crewkit/crew.py) is responsible for pausing and getting the user's approval
before executing any danger tool; `execute()` itself just runs what it's told.
Read-only tools run without ceremony.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

IS_WINDOWS = sys.platform.startswith("win")
_MAX_READ = 60_000  # chars returned from read_file / web_fetch / shell

# Tools that create/modify files — subject to per-worker file-ownership scoping
# (see crewkit/crew.py file ownership + crewkit/agents.py enforcement). run_shell
# is NOT here: it's a general escape hatch gated by user approval, not path-scoped.
WRITE_TOOLS = {"write_file"}


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict          # JSON schema for the function arguments
    handler: Callable[..., str]
    danger: bool = False      # True => orchestrator must get approval first

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


_TOOLS: dict[str, Tool] = {}


def _reg(t: Tool) -> None:
    _TOOLS[t.name] = t


def _clip(text: str, limit: int = _MAX_READ) -> str:
    text = text or ""
    if len(text) > limit:
        return text[:limit] + f"\n…[truncated, {len(text) - limit} more chars]"
    return text


# --- read-only -------------------------------------------------------------
def _list_dir(path: str) -> str:
    p = Path(os.path.expandvars(os.path.expanduser(path)))
    if not p.exists():
        return f"Path does not exist: {p}"
    if not p.is_dir():
        return f"Not a directory: {p}"
    rows = []
    try:
        for entry in sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name.lower())):
            try:
                size = entry.stat().st_size if entry.is_file() else None
            except OSError:
                size = None
            rows.append(("DIR  " if entry.is_dir() else "FILE ")
                        + entry.name + (f"  ({size} B)" if size is not None else ""))
    except OSError as exc:
        return f"Error listing {p}: {exc}"
    return f"{p}  ({len(rows)} entries)\n" + "\n".join(rows[:400])


def _read_file(path: str) -> str:
    p = Path(os.path.expandvars(os.path.expanduser(path)))
    if not p.exists() or not p.is_file():
        return f"No such file: {p}"
    try:
        return _clip(p.read_text(encoding="utf-8", errors="replace"))
    except OSError as exc:
        return f"Error reading {p}: {exc}"


def _get_time() -> str:
    now = datetime.now().astimezone()
    return now.strftime("%Y-%m-%d %H:%M:%S %Z (%A)")


def _web_fetch(url: str) -> str:
    if not url.lower().startswith(("http://", "https://")):
        url = "https://" + url
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (CoderCrew)"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read(2_000_000).decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001 — network is best-effort
        return f"Fetch failed: {exc}"
    # Crude tag strip so the model gets readable text, not markup.
    text = re.sub(r"(?is)<(script|style).*?</\1>", " ", raw)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return _clip(text.strip())


def _web_search(query: str) -> str:
    """Best-effort web search via DuckDuckGo's no-JS HTML endpoint (no API key).
    DDG serves a bot challenge for GET, so this POSTs the query as a form."""
    data = urllib.parse.urlencode({"q": query, "kl": "us-en"}).encode()
    req = urllib.request.Request(
        "https://html.duckduckgo.com/html/", data=data,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                 "Content-Type": "application/x-www-form-urlencoded"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            html = resp.read(800_000).decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        return f"Search failed: {exc}"
    results = []
    for m in re.finditer(r'result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', html):
        href, title = m.group(1), re.sub(r"<[^>]+>", "", m.group(2)).strip()
        href = urllib.parse.unquote(href)
        if "uddg=" in href:  # DDG redirect wrapper -> real url
            try:
                href = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)["uddg"][0]
            except Exception:  # noqa: BLE001
                pass
        if title:
            results.append(f"- {title}\n  {href}")
        if len(results) >= 8:
            break
    return "\n".join(results) if results else "No results."


# --- danger (state-changing / command execution) ---------------------------
def _write_file(path: str, content: str) -> str:
    p = Path(os.path.expandvars(os.path.expanduser(path)))
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    except OSError as exc:
        return f"Write failed: {exc}"
    return f"Wrote {len(content)} chars to {p}"


def shell_exec(command: str, cwd: str | None = None,
               timeout: int = 120) -> "tuple[int | None, str, bool]":
    """The single shell-execution path (NOT a sandbox). Returns
    (returncode, combined_output, timed_out). `returncode` is None when the
    process never ran (timeout or spawn failure). `cwd` only sets the working
    directory — it does NOT confine the command. Used by both the run_shell tool
    and the crew gate runner so there is ONE place commands actually execute."""
    # Put this Python's directory on PATH so bare `python`/`pytest`/`pip` resolve
    # to the same interpreter/venv running the crew — what a worker expects, and
    # what lets `pytest ... -q` acceptance checks actually run.
    env = dict(os.environ)
    env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=timeout,
            cwd=os.path.expandvars(os.path.expanduser(cwd)) if cwd else None,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return None, f"Command timed out after {timeout}s.", True
    except OSError as exc:
        return None, f"Command failed to start: {exc}", False
    parts = [f"exit_code: {result.returncode}"]
    if result.stdout:
        parts.append("stdout:\n" + result.stdout)
    if result.stderr:
        parts.append("stderr:\n" + result.stderr)
    return result.returncode, "\n".join(parts), False


def _run_shell(command: str, cwd: str | None = None) -> str:
    """Run a shell command. DANGER — gated behind user approval by the orchestrator."""
    _rc, output, _timed = shell_exec(command, cwd=cwd, timeout=120)
    return _clip(output)


# --- registry ---------------------------------------------------------------
_reg(Tool("list_dir", "List the files and folders in a directory.",
          {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
          lambda path: _list_dir(path)))
_reg(Tool("read_file", "Read a UTF-8 text file and return its contents.",
          {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
          lambda path: _read_file(path)))
_reg(Tool("get_time", "Get the current local date and time.",
          {"type": "object", "properties": {}}, lambda: _get_time()))
_reg(Tool("web_search", "Search the web; returns a list of result titles and URLs.",
          {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
          lambda query: _web_search(query)))
_reg(Tool("web_fetch", "Fetch a URL and return its text content (markup stripped).",
          {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
          lambda url: _web_fetch(url)))

_reg(Tool("write_file", "Create or overwrite a text file. Changes the filesystem.",
          {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
           "required": ["path", "content"]},
          lambda path, content: _write_file(path, content), danger=True))
_reg(Tool("run_shell", "Run a shell command on this PC and return its output.",
          {"type": "object", "properties": {"command": {"type": "string"},
                                             "cwd": {"type": "string"}}, "required": ["command"]},
          lambda command, cwd=None: _run_shell(command, cwd), danger=True))


def _launch_crew(goal: str, manager: str = "", worker: str = "") -> str:
    from . import crew   # lazy import to avoid a cycle (crew -> agents -> tools)
    m = manager or "ollama:qwen3-coder:30b"
    w = worker or m
    run = crew.MANAGER.start(goal, manager_spec=m, worker_spec=w)
    return f"Launched crew run {run.id} (manager={m}, worker={w}) for: {goal[:100]}"


_reg(Tool("launch_crew", "Launch an autonomous coder-crew run to build something (a manager "
                         "plans, workers implement + test). manager/worker are optional agent specs.",
          {"type": "object", "properties": {"goal": {"type": "string"},
                                             "manager": {"type": "string"},
                                             "worker": {"type": "string"}}, "required": ["goal"]},
          lambda goal, manager="", worker="": _launch_crew(goal, manager, worker), danger=True))


# --- public API --------------------------------------------------------------
def schemas(names: list[str] | None = None) -> list[dict]:
    """Tool schemas to hand to the model. `names` filters; None = all."""
    sel = _TOOLS.values() if names is None else [_TOOLS[n] for n in names if n in _TOOLS]
    return [t.schema() for t in sel]


def is_danger(name: str) -> bool:
    t = _TOOLS.get(name)
    return bool(t and t.danger)


def tool_names(include_danger: bool = True) -> list[str]:
    return [n for n, t in _TOOLS.items() if include_danger or not t.danger]


def _resolve(path: str, base: str) -> str:
    p = os.path.expandvars(os.path.expanduser(path or ""))
    return p if os.path.isabs(p) else os.path.join(base, p)


def execute(name: str, args: dict | None = None, base_dir: str | None = None) -> str:
    """Run a tool by name. Returns a string result (always — errors become text
    the model can read and recover from).

    `base_dir` (the crew's working folder) anchors relative paths and the default
    shell cwd, so a model that writes "add.py" / cwd="." lands in the right place
    rather than the server's directory."""
    t = _TOOLS.get(name)
    if t is None:
        return f"Unknown tool '{name}'. Available: {list(_TOOLS)}"
    args = dict(args or {})
    if base_dir:
        if name in ("write_file", "read_file", "list_dir") and args.get("path"):
            args["path"] = _resolve(args["path"], base_dir)
        elif name == "run_shell":
            cwd = args.get("cwd")
            if not cwd or cwd in (".", "./"):
                args["cwd"] = base_dir
            elif not os.path.isabs(os.path.expanduser(cwd)):
                args["cwd"] = os.path.join(base_dir, cwd)
    try:
        return str(t.handler(**args))
    except TypeError as exc:
        return f"Bad arguments for '{name}': {exc}"
    except Exception as exc:  # noqa: BLE001 — surface, don't crash the run
        return f"Tool '{name}' error: {type(exc).__name__}: {exc}"
