"""HTTP API surface over the swarm TaskQueue (mounted by crewkit.server).

This is ONLY an API surface — it adds no queue/worker/supervisor behaviour, it
just exposes the existing `TaskQueue` (read) and its admin transitions (write)
over `/api/swarm/*`. The server includes this router.

Safety: every WRITE route depends on `require_write_auth` — it requires a bearer
token equal to $SWARM_API_TOKEN when that env var is set, otherwise it requires
the caller to be on loopback. So writes are NEVER open to the network without a
token (safe by default), and a token unlocks remote use. Reads are open (they
can't queue code to run); see require_write_auth for the rationale.
"""
from __future__ import annotations

import hmac
import os
import time

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from .queue import STATUSES, TaskQueue

router = APIRouter(prefix="/api/swarm", tags=["swarm"])

_LOOPBACK = {"127.0.0.1", "::1", "::ffff:127.0.0.1", "localhost"}

# One queue handle per server process, opened lazily against $SWARM_DB (the same
# DB the workers/supervisor drain). TaskQueue is thread-safe (its own lock +
# WAL), so the FastAPI threadpool can share this single handle.
_QUEUE: TaskQueue | None = None


def queue() -> TaskQueue:
    global _QUEUE
    if _QUEUE is None:
        _QUEUE = TaskQueue()
    return _QUEUE


# --------------------------------------------------------------------------- #
# Safety gate (applied to writes only).
# --------------------------------------------------------------------------- #
def require_write_auth(request: Request) -> None:
    """Authorize a write. Two-layer, safe-by-default:

      * If $SWARM_API_TOKEN is set -> require `Authorization: Bearer <token>`
        (constant-time compare). Works from any host, so you can bind the server
        off-loopback and still be protected.
      * If it is NOT set -> require the client to be on loopback. Zero-config and
        still safe: a write endpoint is never reachable from another host without
        a token.

    Reads are intentionally not gated — they expose task metadata, not the ability
    to queue code for the workers to execute, which is the threat that matters."""
    token = os.environ.get("SWARM_API_TOKEN", "").strip()
    if token:
        auth = request.headers.get("authorization", "")
        presented = auth[7:].strip() if auth.lower().startswith("bearer ") else ""
        if not (presented and hmac.compare_digest(presented, token)):
            raise HTTPException(status_code=401,
                                detail="Missing or invalid bearer token "
                                       "(SWARM_API_TOKEN required for writes).")
        return
    client_host = request.client.host if request.client else ""
    if client_host not in _LOOPBACK:
        raise HTTPException(
            status_code=403,
            detail="Writes are restricted to localhost. Set SWARM_API_TOKEN to "
                   "allow authenticated writes from other hosts.")


# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #
class AddTaskRequest(BaseModel):
    description: str
    acceptance: str | None = None
    spec: str | None = None
    max_attempts: int = 2


class PurgeRequest(BaseModel):
    # Retention: delete TERMINAL rows only. None older_than = all terminal rows.
    older_than_seconds: float | None = None
    statuses: list[str] = ["done", "flagged", "cancelled"]


# --------------------------------------------------------------------------- #
# READ
# --------------------------------------------------------------------------- #
MAX_PAGE = 1000           # cap on a single /tasks page
DEFAULT_PAGE = 200        # default page size when caller doesn't specify


def _worker_view(in_progress: list[dict], now: float) -> list[dict]:
    """Derive live-worker info from the in_progress rows only: each names its holder
    + lease. A worker is 'alive' if its lease hasn't expired (it's been heart-
    beating). Available whether or not the supervisor process is running."""
    workers: dict[str, dict] = {}
    for t in in_progress:
        if not t["worker_id"]:
            continue
        lease = t["lease_expires"] or 0
        w = workers.setdefault(t["worker_id"], {
            "worker_id": t["worker_id"], "task_ids": [],
            "lease_expires": lease, "lease_fresh": False})
        w["task_ids"].append(t["id"])
        w["lease_expires"] = max(w["lease_expires"], lease)
        w["lease_fresh"] = w["lease_fresh"] or lease > now
    return sorted(workers.values(), key=lambda w: w["worker_id"])


@router.get("/status")
def swarm_status() -> dict:
    """O(status-groups + in_progress) — counts via GROUP BY, worker view from the
    in_progress rows only. Does NOT scan the whole table, so cost stays flat as
    done/flagged history accumulates (F2)."""
    q = queue()
    now = time.time()
    counts = q.stats()                         # GROUP BY count (incl. 'total')
    total = counts.pop("total")
    workers = _worker_view(q.in_progress_tasks(), now)
    return {
        "db": q.path,
        "counts": counts,
        "total": total,
        "workers": workers,
        "live_workers": sum(1 for w in workers if w["lease_fresh"]),
        "now": now,
    }


@router.get("/tasks")
def swarm_tasks(status: str | None = None, limit: int = DEFAULT_PAGE,
                offset: int = 0) -> dict:
    """Paginated (LIMIT/OFFSET, done in SQL). Response keeps `tasks` + `count`
    (this page) for backward compat and adds `total`/`limit`/`offset` for paging."""
    if status is not None and status not in STATUSES:
        raise HTTPException(status_code=400,
                            detail=f"Unknown status {status!r}; valid: {', '.join(STATUSES)}")
    limit = max(1, min(int(limit), MAX_PAGE))
    offset = max(0, int(offset))
    tasks, total = queue().list_tasks(status=status, limit=limit, offset=offset)
    return {"tasks": tasks, "count": len(tasks), "total": total,
            "limit": limit, "offset": offset}


# --------------------------------------------------------------------------- #
# WRITE  (all gated by require_write_auth)
# --------------------------------------------------------------------------- #
@router.post("/tasks", dependencies=[Depends(require_write_auth)])
def add_task(req: AddTaskRequest) -> dict:
    desc = (req.description or "").strip()
    if not desc:
        raise HTTPException(status_code=400, detail="description is required.")
    tid = queue().add(desc, acceptance=(req.acceptance or None),
                      spec=(req.spec or None), max_attempts=req.max_attempts)
    return {"ok": True, "id": tid, "task": queue().get(tid)}


@router.post("/tasks/{task_id}/requeue", dependencies=[Depends(require_write_auth)])
def requeue_task(task_id: int) -> dict:
    if not queue().requeue(task_id):
        cur = queue().get(task_id)
        if cur is None:
            raise HTTPException(status_code=404, detail="No such task.")
        raise HTTPException(status_code=409,
                            detail=f"Task {task_id} is '{cur['status']}' — only "
                                   "flagged/cancelled/in_progress can be requeued.")
    return {"ok": True, "task": queue().get(task_id)}


@router.post("/tasks/purge", dependencies=[Depends(require_write_auth)])
def purge_tasks(req: PurgeRequest) -> dict:
    """Retention: trim TERMINAL history (done/flagged/cancelled) without touching
    live work. `older_than_seconds` keeps recent terminal rows; omit to purge all
    terminal rows. (Equivalent SQL: DELETE FROM tasks WHERE status IN
    ('done','flagged','cancelled') AND ended < <cutoff>.)"""
    bad = [s for s in req.statuses if s not in ("done", "flagged", "cancelled")]
    if bad:
        raise HTTPException(status_code=400,
                            detail=f"purge only accepts terminal statuses; got {bad}")
    deleted = queue().purge_terminal(older_than=req.older_than_seconds,
                                     statuses=tuple(req.statuses))
    return {"ok": True, "deleted": deleted}


@router.post("/tasks/{task_id}/cancel", dependencies=[Depends(require_write_auth)])
def cancel_task(task_id: int) -> dict:
    if not queue().cancel(task_id):
        cur = queue().get(task_id)
        if cur is None:
            raise HTTPException(status_code=404, detail="No such task.")
        raise HTTPException(status_code=409,
                            detail=f"Task {task_id} is '{cur['status']}' — only "
                                   "pending/in_progress can be cancelled.")
    return {"ok": True, "task": queue().get(task_id)}
