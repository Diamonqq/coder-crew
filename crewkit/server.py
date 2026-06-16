"""FastAPI server for coder-crew.

Exposes the crew orchestrator over HTTP and serves the web UI. Local, single-user
surface — binds to loopback by default and has no auth (the crew can run shell
commands; don't expose this beyond localhost without a tunnel + auth in front).

Run:  python -m crewkit.server          (or: python run.py)
Then open http://127.0.0.1:8770
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Claude crew members spawn the `claude` CLI via the Agent SDK, which on Windows
# needs the Proactor event loop. Set it at import, before any loop is created.
if sys.platform == "win32":
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from . import chat, crew

_WEB = Path(__file__).resolve().parent.parent / "web"
_HOST = os.environ.get("CREW_HOST", "127.0.0.1")
_PORT = int(os.environ.get("CREW_PORT", "8770"))

_Q3_STOCK = "qwen3-coder:30b"   # recommended local model (stock build)

app = FastAPI(title="coder-crew", docs_url=None, redoc_url=None)


# --- request models ---------------------------------------------------------
class StartRequest(BaseModel):
    goal: str
    manager: str
    worker: str
    max_workers: int = 3
    cwd: str | None = None
    complexity: str = "medium"          # simple|medium|hard
    allow_escalation: bool = False      # opt-in: escalate to Opus (paid)


class AdviseRequest(BaseModel):
    idea: str


class ApproveRequest(BaseModel):
    approved: bool
    note: str = ""


# --- model catalog ----------------------------------------------------------
@app.get("/api/crew/config")
def crew_config() -> dict:
    """Models the UI offers for the manager/worker role pickers + sane defaults."""
    try:
        installed = [m["name"] for m in chat.models()]
    except Exception:  # noqa: BLE001
        installed = []
    iset = set(installed)

    def present(tag: str) -> bool:
        return any(n == tag or n.startswith(tag + ":") or n == tag.split(":")[0]
                   for n in iset)

    ollama_roles = []
    if present(_Q3_STOCK) or present(crew._Q3):
        spec = _Q3_STOCK if present(_Q3_STOCK) else crew._Q3
        ollama_roles.append({"spec": f"ollama:{spec}",
                             "label": "Qwen3-Coder 30B (recommended · agentic)",
                             "ready": True})
    elif chat.available():
        ollama_roles.append({"spec": f"ollama:{_Q3_STOCK}",
                             "label": "Qwen3-Coder 30B (recommended — not installed)",
                             "ready": False})
    seen = {r["spec"] for r in ollama_roles}
    for n in sorted(iset):
        spec = f"ollama:{n}"
        if spec in seen:
            continue
        if "coder" in n.lower() or n.startswith("gemma") or "qwen" in n.lower():
            ollama_roles.append({"spec": spec, "label": n, "ready": True})

    claude_roles = []
    if crew._claude_available():
        claude_roles = [
            {"spec": "claude:claude-opus-4-8", "label": "Claude Opus 4.8"},
            {"spec": "claude:sonnet", "label": "Claude Sonnet"},
            {"spec": "claude:haiku", "label": "Claude Haiku"},
        ]

    local_default = next((r["spec"] for r in ollama_roles if r["ready"]),
                         f"ollama:{_Q3_STOCK}")
    mgr_default = "claude:claude-opus-4-8" if claude_roles else local_default
    return {
        "ollama_available": chat.available(),
        "claude_available": crew._claude_available(),
        "ollama_roles": ollama_roles,
        "claude_roles": claude_roles,
        "defaults": {"manager": mgr_default, "worker": local_default},
    }


@app.post("/api/crew/advise")
def crew_advise(req: AdviseRequest) -> dict:
    """Autopilot: a local model turns a rough idea into a precise goal + sizing +
    recommended manager/worker combo."""
    if not req.idea.strip():
        raise HTTPException(status_code=400, detail="Describe an idea first.")
    return crew.advise(req.idea.strip())


# --- runs -------------------------------------------------------------------
@app.post("/api/crew/start")
def crew_start(req: StartRequest) -> JSONResponse:
    if not req.goal.strip():
        raise HTTPException(status_code=400, detail="Goal is required.")
    run = crew.MANAGER.start(
        req.goal.strip(), manager_spec=req.manager, worker_spec=req.worker,
        max_workers=req.max_workers, cwd=req.cwd or None,
        complexity=req.complexity, allow_escalation=req.allow_escalation)
    return JSONResponse({"run_id": run.id})


@app.get("/api/crew/runs")
def crew_runs() -> dict:
    return {"runs": crew.MANAGER.list()}


@app.get("/api/crew/runs/{run_id}")
def crew_run(run_id: str) -> JSONResponse:
    run = crew.MANAGER.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="No such run.")
    return JSONResponse(run.to_dict())


@app.post("/api/crew/runs/{run_id}/approve")
def crew_approve(run_id: str, req: ApproveRequest) -> dict:
    if not crew.MANAGER.approve(run_id, req.approved, req.note):
        raise HTTPException(status_code=409, detail="Nothing pending for this run.")
    return {"ok": True}


@app.post("/api/crew/runs/{run_id}/cancel")
def crew_cancel(run_id: str) -> dict:
    if not crew.MANAGER.cancel(run_id):
        raise HTTPException(status_code=404, detail="No such run.")
    return {"ok": True}


# --- history (SQLite) -------------------------------------------------------
@app.get("/api/crew/history")
def crew_history() -> dict:
    return {"runs": crew.crew_db.DB.recent_runs(100)}


@app.get("/api/crew/history/{run_id}")
def crew_history_detail(run_id: str) -> JSONResponse:
    detail = crew.crew_db.DB.run_detail(run_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="No such run in history.")
    return JSONResponse(detail)


# --- status -----------------------------------------------------------------
@app.get("/api/ollama")
def ollama_status() -> dict:
    return chat.ollama_status()


@app.get("/api/mcp/status")
def mcp_status() -> dict:
    try:
        from . import mcp_bridge
        return mcp_bridge.status()
    except Exception as exc:  # noqa: BLE001
        return {"servers": [], "disabled": [], "tools": {}, "tool_count": 0,
                "error": f"{type(exc).__name__}: {exc}"}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_WEB / "index.html")


def main() -> None:
    import uvicorn
    print(f"coder-crew on http://{_HOST}:{_PORT}")
    print(f"  Ollama: {'up' if chat.available() else 'NOT reachable — start `ollama serve`'}"
          f"   ·   Claude/Opus escalation: "
          f"{'available' if crew._claude_available() else 'not installed (optional)'}")
    if _HOST not in ("127.0.0.1", "localhost", "::1"):
        print("  WARNING: bound to a non-loopback host — the crew can run shell "
              "commands. Only do this behind a tunnel + auth.")
    uvicorn.run(app, host=_HOST, port=_PORT, log_level="warning")


if __name__ == "__main__":
    main()
