"""Persistent crew run history (SQLite) — data for the local-vs-Opus auto-router.

Mirrors foreman/db.py's pattern. On each crew run finishing we append one `runs`
row plus one `subtasks` row per subtask, capturing the FEATURES a router will
later score on: complexity, manager/worker specs, an optional expected `tag`
(bucket), and per subtask the final status, attempts, gate outcome, whether the
acceptance was weak-flagged or allowlist-rejected, whether review caught a
regression (green-in-work → red-in-review), and wall-clock. Append-on-finish
only; never changes run behavior. DB path: $CREW_DB or <project>/crew_history.db.
"""
from __future__ import annotations

import os
import sqlite3
import sys
import threading
from pathlib import Path

# Writable: beside the exe when frozen, else the repo root. ($CREW_DB overrides.)
_BASE = (Path(sys.executable).parent if getattr(sys, "frozen", False)
         else Path(__file__).resolve().parent.parent)
_DEFAULT = _BASE / "crew_history.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id           TEXT PRIMARY KEY,
    goal         TEXT,
    complexity   TEXT,            -- advisor estimate: simple|medium|hard
    tag          TEXT,            -- expected routing bucket (overnight harness); '' interactive
    manager_spec TEXT,
    worker_spec  TEXT,
    status       TEXT,            -- run-level: done|error|cancelled
    created      REAL,
    ended        REAL,
    elapsed      REAL,
    n_subtasks   INTEGER,
    n_passed     INTEGER,         -- subtasks with a REAL green gate (status 'done')
    n_failed     INTEGER,         -- subtasks with status 'failed' or 'error'
    n_unverified INTEGER DEFAULT 0  -- manual-review (None/rejected gate): NOT a pass
);

CREATE TABLE IF NOT EXISTS subtasks (
    run_id       TEXT REFERENCES runs(id),
    idx          INTEGER,
    title        TEXT,
    status       TEXT,            -- done|failed|error|running|queued
    attempts     INTEGER,
    gate_outcome TEXT,            -- passed|failed|rejected|manual|none
    weak_flagged INTEGER,         -- 0/1: acceptance heuristically weak
    rejected     INTEGER,         -- 0/1: acceptance allowlist-rejected (dropped to manual)
    regression   INTEGER,         -- 0/1: passed in work but RED at review
    elapsed      REAL,
    ran_on       TEXT DEFAULT 'local',   -- local | opus (router: where it ran)
    escalated    INTEGER DEFAULT 0,      -- 0/1: re-dispatched local -> Opus
    escalation_reason TEXT DEFAULT '',   -- failed|unverified|error|predict-concurrency
    coverage_note TEXT DEFAULT '',       -- spec-coverage review result/outcome
    PRIMARY KEY (run_id, idx)
);
"""


class CrewDB:
    def __init__(self, path: str):
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()
        # Additive migrations for DBs created before a column existed.
        for ddl in (
            "ALTER TABLE runs ADD COLUMN n_unverified INTEGER DEFAULT 0",
            "ALTER TABLE subtasks ADD COLUMN ran_on TEXT DEFAULT 'local'",
            "ALTER TABLE subtasks ADD COLUMN escalated INTEGER DEFAULT 0",
            "ALTER TABLE subtasks ADD COLUMN escalation_reason TEXT DEFAULT ''",
            "ALTER TABLE subtasks ADD COLUMN coverage_note TEXT DEFAULT ''",
        ):
            try:
                self._exec(ddl)
            except sqlite3.OperationalError:
                pass  # column already exists

    def _exec(self, sql: str, params: tuple = ()):
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur

    def _query(self, sql: str, params: tuple = ()):
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def log_run(self, run: dict, subtasks: list) -> None:
        self._exec(
            "INSERT OR REPLACE INTO runs (id, goal, complexity, tag, manager_spec, "
            "worker_spec, status, created, ended, elapsed, n_subtasks, n_passed, "
            "n_failed, n_unverified) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (run["id"], run["goal"], run["complexity"], run.get("tag", ""),
             run["manager_spec"], run["worker_spec"], run["status"], run["created"],
             run["ended"], run["elapsed"], run["n_subtasks"], run["n_passed"],
             run["n_failed"], run.get("n_unverified", 0)),
        )
        self._exec("DELETE FROM subtasks WHERE run_id = ?", (run["id"],))
        for s in subtasks:
            self._exec(
                "INSERT OR REPLACE INTO subtasks (run_id, idx, title, status, attempts, "
                "gate_outcome, weak_flagged, rejected, regression, elapsed, ran_on, "
                "escalated, escalation_reason, coverage_note) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (run["id"], s["idx"], s["title"], s["status"], s["attempts"],
                 s["gate_outcome"], int(bool(s["weak_flagged"])),
                 int(bool(s["rejected"])), int(bool(s["regression"])), s["elapsed"],
                 s.get("ran_on", "local"), int(bool(s.get("escalated"))),
                 s.get("escalation_reason", ""), s.get("coverage_note", "")),
            )

    def _honest_counts(self, run_id: str) -> dict:
        """Re-derive passed/failed/unverified from the authoritative per-subtask
        gate_outcome, so even old rows (whose stored n_passed counted manual-review
        as a pass) report honestly. REAL pass = gate_outcome 'passed' only;
        'manual'/'rejected'/'none'/'incomplete' => unverified (never verified)."""
        subs = self._query("SELECT status, gate_outcome FROM subtasks WHERE run_id = ?",
                           (run_id,))
        passed = sum(1 for s in subs if s["gate_outcome"] == "passed")
        failed = sum(1 for s in subs if s["status"] in ("failed", "error"))
        unverified = sum(1 for s in subs
                         if s["gate_outcome"] in ("manual", "rejected", "none", "incomplete")
                         and s["status"] not in ("failed", "error"))
        return {"n_passed": passed, "n_failed": failed, "n_unverified": unverified}

    def recent_runs(self, limit: int = 50) -> list:
        out = []
        for r in self._query("SELECT * FROM runs ORDER BY created DESC LIMIT ?", (limit,)):
            d = dict(r)
            d.update(self._honest_counts(d["id"]))   # honest counts (overrides stored)
            out.append(d)
        return out

    def run_detail(self, run_id: str) -> dict | None:
        rows = self._query("SELECT * FROM runs WHERE id = ?", (run_id,))
        if not rows:
            return None
        out = dict(rows[0])
        out.update(self._honest_counts(run_id))
        out["subtasks"] = [dict(r) for r in self._query(
            "SELECT * FROM subtasks WHERE run_id = ? ORDER BY idx", (run_id,))]
        return out


DB = CrewDB(os.environ.get("CREW_DB", str(_DEFAULT)))
