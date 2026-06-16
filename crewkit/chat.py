"""Local-model chat via Ollama.

A thin proxy to a locally running Ollama daemon (default 127.0.0.1:11434). The
panel never bundles or runs a model itself — it just talks to Ollama's HTTP API.
Everything here is plain stdlib so it adds no dependencies.
"""
from __future__ import annotations

import json
import os
import subprocess
import urllib.error
import urllib.request

# Where to REACH Ollama from this process. Note: we deliberately do NOT read
# OLLAMA_HOST — that is Ollama's server *bind* setting (often "0.0.0.0"), which
# is not a valid client target. Connect to loopback; honor a port from
# OLLAMA_HOST only, and allow a full override via PCP_OLLAMA_URL.
def _resolve_ollama() -> str:
    override = os.environ.get("PCP_OLLAMA_URL")
    if override:
        return override.rstrip("/")
    port = "11434"
    bind = os.environ.get("OLLAMA_HOST", "")
    if ":" in bind:
        tail = bind.rsplit(":", 1)[-1]
        if tail.isdigit():
            port = tail
    return f"http://127.0.0.1:{port}"


OLLAMA = _resolve_ollama()


def _get(path: str, timeout: float = 4.0):
    with urllib.request.urlopen(OLLAMA + path, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _post(path: str, body: dict, timeout: float = 30.0):
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(OLLAMA + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _human(n) -> str:
    n = float(n or 0)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def available() -> bool:
    """True if the Ollama daemon answers."""
    try:
        _get("/api/tags", timeout=2.0)
        return True
    except Exception:
        return False


def models() -> list[dict]:
    """Installed models, lightly normalized for the UI."""
    data = _get("/api/tags")
    out = []
    for m in data.get("models", []):
        det = m.get("details", {}) or {}
        out.append({
            "name": m.get("name", ""),
            "size": m.get("size"),
            "param_size": det.get("parameter_size"),
            "family": det.get("family"),
        })
    out.sort(key=lambda x: x["name"])
    return out


def chat(model: str, messages: list[dict], timeout: float = 600.0) -> dict:
    """Non-streaming chat completion. messages = [{role, content}, ...]."""
    body = json.dumps({"model": model, "messages": messages, "stream": False}).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA + "/api/chat", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8"))
    msg = data.get("message", {}) or {}
    return {
        "reply": msg.get("content", ""),
        "thinking": msg.get("thinking") or None,
        "model": model,
        "eval_count": data.get("eval_count"),
    }


def start_daemon() -> dict:
    """Best-effort 'ollama serve' if it isn't already running. Returns status."""
    if available():
        return {"started": False, "message": "Ollama already running."}
    exe = os.environ.get(
        "OLLAMA_EXE",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe"),
    )
    if not os.path.isfile(exe):
        return {"started": False, "message": f"ollama.exe not found at {exe}"}
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    subprocess.Popen([exe, "serve"], creationflags=flags,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return {"started": True, "message": "Starting Ollama…"}


# --- Ollama control panel ---------------------------------------------------
def running_models() -> list[dict]:
    """Resident (loaded) models — the `ollama ps` data, via GET /api/ps."""
    data = _get("/api/ps")
    out = []
    for m in data.get("models", []):
        det = m.get("details", {}) or {}
        out.append({
            "name": m.get("name") or m.get("model", ""),
            "size_vram": m.get("size_vram"),
            "size_vram_human": _human(m.get("size_vram") or 0),
            "size_human": _human(m.get("size") or 0),
            "expires_at": m.get("expires_at"),
            "param_size": det.get("parameter_size"),
        })
    return out


def unload_model(model: str) -> dict:
    """Free a resident model's VRAM. The documented mechanism is a request with
    keep_alive=0, which unloads the model immediately (same as `ollama stop`)."""
    if not model:
        return {"ok": False, "message": "No model specified."}
    try:
        _post("/api/generate", {"model": model, "prompt": "", "keep_alive": 0},
              timeout=30.0)
        return {"ok": True, "message": f"Unloaded {model} — VRAM freed."}
    except Exception as exc:  # noqa: BLE001 — surface, never crash
        return {"ok": False, "message": f"Unload failed: {exc}"}


def ollama_status() -> dict:
    """Read-only Ollama overview: reachable?, resident models (+VRAM/expiry),
    installed models, and total VRAM Ollama is holding. Degrades gracefully."""
    if not available():
        return {"available": False, "host": OLLAMA, "resident": [],
                "installed": [], "resident_count": 0, "vram_bytes": 0,
                "vram_human": _human(0)}
    resident, installed = [], []
    try:
        resident = running_models()
    except Exception:  # noqa: BLE001
        pass
    try:
        installed = models()
    except Exception:  # noqa: BLE001
        pass
    vram = sum((m.get("size_vram") or 0) for m in resident)
    return {"available": True, "host": OLLAMA, "resident": resident,
            "installed": installed, "resident_count": len(resident),
            "vram_bytes": vram, "vram_human": _human(vram)}
