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

from . import chat, claude_conf, crew

claude_conf.apply()   # push saved Claude connection settings into the env at startup

# web/ is read-only and bundled INTO the PyInstaller exe (extracted to _MEIPASS);
# from source it sits next to the package.
if getattr(sys, "frozen", False):
    _WEB = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent)) / "web"
else:
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
    auto_approve: bool = False          # opt-in: run unattended, no per-tool approval


class AdviseRequest(BaseModel):
    idea: str


class ApproveRequest(BaseModel):
    approved: bool
    note: str = ""


class ClaudeConfigRequest(BaseModel):
    mode: str                       # off | code | api
    model: str = ""
    api_key: str | None = None


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
    cfg_model = claude_conf.get()["model"]
    if claude_conf.available():
        _labels = {"claude-opus-4-8": "Claude Opus 4.8", "sonnet": "Claude Sonnet",
                   "haiku": "Claude Haiku"}
        cseen = set()
        for m in [cfg_model, "claude-opus-4-8", "sonnet", "haiku"]:
            if m in cseen:
                continue
            cseen.add(m)
            claude_roles.append({"spec": f"claude:{m}", "label": _labels.get(m, "Claude " + m)})

    local_default = next((r["spec"] for r in ollama_roles if r["ready"]),
                         f"ollama:{_Q3_STOCK}")
    mgr_default = f"claude:{cfg_model}" if claude_roles else local_default
    return {
        "ollama_available": chat.available(),
        "claude_available": claude_conf.available(),
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
        complexity=req.complexity, allow_escalation=req.allow_escalation,
        auto_approve=req.auto_approve)
    return JSONResponse({"run_id": run.id})


class ResearchRequest(BaseModel):
    topic: str
    manager: str = "claude:claude-opus-4-8"
    submanager: str = "claude:sonnet"
    researcher: str = "claude:haiku"
    n_submanagers: int = 0       # 0 = flat (manager -> researchers)
    n_researchers: int = 6       # researchers (per sub-manager when tiered)
    rounds: int = 1              # >1 = looped: manager re-plans across rounds


@app.post("/api/crew/research")
def crew_research(req: ResearchRequest) -> JSONResponse:
    """Research swarm: a manager fans the topic out to researchers (and optionally
    sub-managers), then synthesizes + ranks. rounds>1 loops, re-planning on gaps."""
    if not req.topic.strip():
        raise HTTPException(status_code=400, detail="A topic is required.")
    run = crew.MANAGER.start_research(
        req.topic.strip(), manager_spec=req.manager,
        submanager_spec=req.submanager, researcher_spec=req.researcher,
        n_submanagers=req.n_submanagers, n_researchers=req.n_researchers,
        rounds=req.rounds)
    return JSONResponse({"run_id": run.id})


class AutopilotRequest(BaseModel):
    idea: str
    build: bool | None = None    # None = let the planner decide research vs build
    manager: str = "claude:claude-opus-4-8"
    researcher: str = "claude:haiku"
    builder: str = "ollama:qwen3-coder:30b"


@app.post("/api/crew/autopilot")
def crew_autopilot(req: AutopilotRequest) -> JSONResponse:
    """One prompt -> a planner sizes the whole pipeline (research vs build,
    #researchers, rounds) and launches it."""
    if not req.idea.strip():
        raise HTTPException(status_code=400, detail="Describe what you want.")
    run = crew.MANAGER.autopilot(req.idea.strip(), manager_spec=req.manager,
                                 researcher_spec=req.researcher, builder_spec=req.builder,
                                 build=req.build)
    return JSONResponse({"run_id": run.id})


class AssistantRequest(BaseModel):
    text: str
    model: str | None = None
    history: list[dict] | None = None
    incognito: bool = False


@app.post("/api/assistant/send")
def assistant_send(req: AssistantRequest) -> JSONResponse:
    """The conversational coding assistant as a tool-using agent. Returns a run_id;
    the UI polls /api/crew/runs/{id} for activity + approvals (shared gate)."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Empty message.")
    spec = req.model or "ollama:qwen3-coder:30b"
    run = crew.MANAGER.start_assistant(req.text.strip(), spec, req.history or [],
                                       incognito=req.incognito)
    return JSONResponse({"run_id": run.id})


# --- local Ollama chat (assistant model picker + private chat) --------------
@app.get("/api/chat/status")
def chat_status() -> dict:
    return {"available": chat.available(), "host": getattr(chat, "OLLAMA", "")}


@app.get("/api/chat/models")
def chat_models() -> dict:
    try:
        return {"models": chat.models()}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Ollama unreachable: {exc}")


class ChatRequest(BaseModel):
    model: str
    messages: list[dict]


@app.post("/api/chat/send")
def chat_send(req: ChatRequest) -> JSONResponse:
    if not req.model:
        raise HTTPException(status_code=400, detail="No model selected.")
    return JSONResponse(chat.chat(req.model, req.messages))


@app.post("/api/chat/start")
def chat_start() -> dict:
    try:
        return chat.start_daemon()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "message": str(exc)}


class EnhanceRequest(BaseModel):
    text: str
    model: str = ""


@app.post("/api/crew/enhance")
def crew_enhance(req: EnhanceRequest) -> JSONResponse:
    """Rewrite a rough Create idea into a sharper prompt, on the chosen model."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Nothing to improve.")
    spec = req.model or "ollama:qwen3-coder:30b"
    return JSONResponse({"prompt": crew.enhance_prompt(req.text.strip(), spec)})


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
@app.get("/api/claude/status")
def claude_status() -> dict:
    return claude_conf.status()


@app.post("/api/claude/config")
def claude_set(req: ClaudeConfigRequest) -> dict:
    return claude_conf.save(req.mode, req.model, req.api_key)


@app.post("/api/claude/test")
def claude_test() -> dict:
    """Real one-shot PONG round-trip to prove the configured Claude path works."""
    import time as _t
    from . import agents
    if not claude_conf.available():
        return {"ok": False, "error": "Claude is set to OFF or the claude-agent-sdk "
                "isn't installed — nothing to test."}
    model = claude_conf.get()["model"] or "claude-opus-4-8"
    t0 = _t.time()
    try:
        out = agents.make_agent(f"claude:{model}", cwd=None).run_task(
            "Reply with exactly one word: PONG")
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "model": model, "elapsed": round(_t.time() - t0, 1),
                "error": f"{type(exc).__name__}: {exc}"}
    ok = ("PONG" in (out or "").upper()) and not (out or "").startswith("(claude error")
    return {"ok": ok, "model": model, "elapsed": round(_t.time() - t0, 1),
            "reply": (out or "").strip()[:200]}


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
    cs = claude_conf.status()
    print(f"  Ollama: {'up' if chat.available() else 'NOT reachable — start `ollama serve`'}"
          f"   ·   Claude: mode={cs['mode']}, "
          f"{'available' if cs['available'] else 'off/SDK-missing'} "
          f"(sdk={'y' if cs['sdk_installed'] else 'n'}, cli={'y' if cs['cli_found'] else 'n'})")
    if _HOST not in ("127.0.0.1", "localhost", "::1"):
        print("  WARNING: bound to a non-loopback host — the crew can run shell "
              "commands. Only do this behind a tunnel + auth.")
    # Packaged exe (or CREW_OPEN_BROWSER=1): pop the UI open once the server is up.
    if ((getattr(sys, "frozen", False) or os.environ.get("CREW_OPEN_BROWSER"))
            and not os.environ.get("CREW_NO_BROWSER")):
        import threading
        import webbrowser
        threading.Timer(1.5, lambda: webbrowser.open(f"http://{_HOST}:{_PORT}")).start()
    uvicorn.run(app, host=_HOST, port=_PORT, log_level="warning")


if __name__ == "__main__":
    main()
