"""Per-run token ledger — a minimal, dependency-free cost meter for standalone
coder-crew.

`crew.CrewRun.emit` logs one row per streamed `usage` event; `crew._ledger_totals`
reads a run's total back for the UI/CLI. In the panel build this interface is backed
by a richer SQLite ledger; standalone it is an in-memory, thread-safe aggregate that
is entirely sufficient for showing per-run token totals. Both call sites already wrap
this in try/except, so a ledger that is absent or errors degrades to "no totals" —
never a crash. Kept intentionally small: it owns numbers, nothing else.
"""
from __future__ import annotations

import threading

_lock = threading.Lock()
# run_id -> {"total": int, "by": {key: {"role", "agent", "tokens", "rate"}}}
_runs: dict[str, dict] = {}


def log(run_id: str, key: str, role, agent, tokens, rate=None) -> None:
    """Record one usage event for a run. `key` groups rows (a worker id like "w3" or a
    role name); repeated keys ACCUMULATE tokens and keep the latest rate. Best-effort:
    a bad row is ignored, never raised (the caller's guard is the backstop, this is
    belt-and-suspenders)."""
    try:
        n = int(tokens or 0)
    except (TypeError, ValueError):
        return
    with _lock:
        run = _runs.setdefault(str(run_id), {"total": 0, "by": {}})
        row = run["by"].setdefault(str(key), {"role": role, "agent": agent, "tokens": 0, "rate": None})
        row["tokens"] += n
        if role is not None:
            row["role"] = role
        if agent is not None:
            row["agent"] = agent
        if rate is not None:
            row["rate"] = rate
        run["total"] += n


def run_totals(run_id: str) -> dict:
    """A run's token total + per-key breakdown (newest-first by token count).
    {"total": int, "by": [{"key", "role", "agent", "tokens", "rate"}, ...]}."""
    with _lock:
        run = _runs.get(str(run_id))
        if not run:
            return {"total": 0, "by": []}
        by = [{"key": k, **v} for k, v in run["by"].items()]
        by.sort(key=lambda r: r.get("tokens", 0), reverse=True)
        return {"total": run["total"], "by": by}


def reset(run_id: "str | None" = None) -> None:
    """Drop one run's ledger, or ALL runs when run_id is None (tests + long-lived hosts)."""
    with _lock:
        if run_id is None:
            _runs.clear()
        else:
            _runs.pop(str(run_id), None)
