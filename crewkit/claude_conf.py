"""Claude / Opus connection config + status.

Chooses HOW the optional Claude path authenticates, persisted in config.json
(gitignored):

  - "off"  : don't use Claude at all — a purely local crew.
  - "code" : use your logged-in `claude` CLI (Claude Code subscription).
  - "api"  : use an Anthropic API key. PRIMARY source is the CLAUDE_API_KEY
             environment variable. A key saved via the UI is an optional FALLBACK,
             stored PLAINTEXT in config.json (gitignored — never commit or share it).

The claude-agent-sdk spawns the `claude` CLI under the hood. That CLI uses
ANTHROPIC_API_KEY if it's set, otherwise your subscription login — so "api" mode
resolves the key (env CLAUDE_API_KEY first, then config.json) and exports it as
ANTHROPIC_API_KEY; "code" mode relies on your CLI being logged in.
Availability here means "configured to use Claude + SDK present"; the actual
credentials are proven by the Test ping (server /api/claude/test).
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

_BASE = (Path(sys.executable).parent if getattr(sys, "frozen", False)
         else Path(__file__).resolve().parent.parent)
_CONFIG = _BASE / "config.json"
_DEFAULT_MODEL = "claude-opus-4-8"
_MODES = ("off", "code", "api")


def _load() -> dict:
    try:
        return json.loads(_CONFIG.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — missing/garbled config => defaults
        return {}


def get() -> dict:
    c = _load().get("claude") or {}
    mode = c.get("mode", "code")
    return {"mode": mode if mode in _MODES else "code",
            "model": c.get("model") or _DEFAULT_MODEL,
            "api_key": c.get("api_key", "")}


def save(mode: str, model: str = "", api_key: str | None = None) -> dict:
    data = _load()
    c = data.get("claude") or {}
    if mode in _MODES:
        c["mode"] = mode
    if model:
        c["model"] = model
    if api_key is not None:
        c["api_key"] = api_key.strip()
    data["claude"] = c
    _CONFIG.write_text(json.dumps(data, indent=2), encoding="utf-8")
    apply()
    return status()


def sdk_installed() -> bool:
    try:
        import claude_agent_sdk  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def cli_found() -> bool:
    return shutil.which("claude") is not None


def apply() -> None:
    """Push the saved config into the process env so the SDK/CLI pick it up.
    Call once at startup and after every save."""
    c = get()
    if c["mode"] == "off":
        os.environ["CREW_CLAUDE_OFF"] = "1"          # crew._claude_available honors this
    else:
        os.environ.pop("CREW_CLAUDE_OFF", None)
    if c["mode"] == "api":
        # PRIMARY: CLAUDE_API_KEY env. FALLBACK: a key saved in config.json (plaintext).
        key = os.environ.get("CLAUDE_API_KEY") or c["api_key"]
        if key:
            os.environ["ANTHROPIC_API_KEY"] = key


def available() -> bool:
    """Configured to use Claude AND the SDK is importable. (Whether the creds
    actually work is what the Test ping verifies.)"""
    return get()["mode"] != "off" and sdk_installed()


def status() -> dict:
    c = get()
    source = ("env:CLAUDE_API_KEY" if os.environ.get("CLAUDE_API_KEY")
              else "config.json (plaintext)" if c["api_key"]
              else "env:ANTHROPIC_API_KEY" if os.environ.get("ANTHROPIC_API_KEY")
              else "none")
    return {
        "mode": c["mode"],
        "model": c["model"],
        "sdk_installed": sdk_installed(),
        "cli_found": cli_found(),
        "api_key_set": source != "none",
        "api_key_source": source,
        "available": available(),
    }
