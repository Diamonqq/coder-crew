"""Persistent task queue for the 24/7 worker layer (SQLite, cross-process).

Reuses coder-crew's storage stack (sqlite3 + Row factory, env-var DB path like
crew_db.py). Unlike crew_db (append-on-finish history), this is a LIVE work queue
that multiple worker PROCESSES drain concurrently, so it is built for safe
cross-process contention:

  * WAL journal + a busy timeout so readers/writers don't trip over each other;
  * `claim_next` takes a BEGIN IMMEDIATE write lock and flips exactly one pending
    row to in_progress atomically — two workers can never claim the same task;
  * every terminal transition (`mark_done`/`mark_flagged`) is guarded by
    `worker_id` + current status, so a zombie worker whose task was already
    reclaimed by the reaper cannot overwrite the reassigned row.

A task is a unit of work for the crew:
  id, description, acceptance (a runnable check passed through to the crew, or
  NULL = manual review), spec (extra context), status, attempts, max_attempts,
  worker_id, last_error, result, lease_expires, and timestamps.

Status machine:
  pending  -> (claim)        -> in_progress
  in_progress -> (mark_done)    -> done
  in_progress -> (mark_flagged) -> flagged
  in_progress -> (lease expires, worker died) -> pending  (attempts remain)
                                              or flagged (attempts exhausted)
"""
from __future__ import annotations

import os
import sqlite3
import sys
import threading
import time
from pathlib import Path

# Writable: beside the exe when frozen, else the repo root. ($SWARM_DB overrides.)
_BASE = (Path(sys.executable).parent if getattr(sys, "frozen", False)
         else Path(__file__).resolve().parent.parent.parent)
_DEFAULT = _BASE / "swarm_queue.db"

STATUSES = ("pending", "in_progress", "done", "flagged", "cancelled")
DEFAULT_MAX_ATTEMPTS = 2     # one crew attempt + one retry after a crash/reclaim
DEFAULT_LEASE = 1800.0       # 30 min: a worker must heartbeat within this or be reaped

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    description   TEXT NOT NULL,
    acceptance    TEXT,                 -- runnable check for the crew gate, or NULL
    spec          TEXT,                 -- optional extra spec/context for the crew
    status        TEXT NOT NULL DEFAULT 'pending',
    attempts      INTEGER NOT NULL DEFAULT 0,
    max_attempts  INTEGER NOT NULL DEFAULT 2,
    worker_id     TEXT,                 -- holder while in_progress
    last_error    TEXT,                 -- why it last failed / was flagged
    result        TEXT,                 -- outcome summary on done / flag
    lease_expires REAL,                 -- epoch; in_progress past this = reapable
    created       REAL NOT NULL,
    updated       REAL NOT NULL,
    started       REAL,                 -- when first claimed this attempt
    ended         REAL                  -- when it reached done/flagged
);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
"""


class TaskQueue:
    def __init__(self, path: str | None = None):
        self.path = path or os.environ.get("SWARM_DB", str(_DEFAULT))
        # check_same_thread=False: the worker heartbeats from a side thread; the
        # _lock below serializes our own in-process access. Cross-PROCESS safety
        # comes from WAL + busy_timeout + the IMMEDIATE-locked claim.
        self._conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA busy_timeout=10000")
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    # -- write helpers ---------------------------------------------------------
    def _exec(self, sql: str, params: tuple = ()):
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur

    def _query(self, sql: str, params: tuple = ()):
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    # -- producer --------------------------------------------------------------
    def add(self, description: str, *, acceptance: str | None = None,
            spec: str | None = None, max_attempts: int = DEFAULT_MAX_ATTEMPTS) -> int:
        """Enqueue a task. Returns its id."""
        now = time.time()
        cur = self._exec(
            "INSERT INTO tasks (description, acceptance, spec, status, attempts, "
            "max_attempts, created, updated) VALUES (?,?,?,'pending',0,?,?,?)",
            (description, acceptance, spec, max(1, int(max_attempts)), now, now),
        )
        return int(cur.lastrowid)

    # -- consumer --------------------------------------------------------------
    def claim_next(self, worker_id: str, lease: float = DEFAULT_LEASE) -> dict | None:
        """Atomically claim the oldest pending task for `worker_id`. Returns the
        claimed row as a dict, or None if the queue has no pending work.

        Uses BEGIN IMMEDIATE so concurrent worker processes serialize on the write
        lock: each sees the other's flip and only one wins a given row.

        DISTRIBUTION (by design, not round-robin): the next FREE worker to win the
        write lock claims the next task. This is pull-based and work-conserving —
        the optimal policy for variable-length tasks, since a busy worker isn't
        contending while it runs, so claims naturally flow to available workers.
        (With instant tasks a 'hot' worker can monopolize the lock and skew the
        distribution, but that regime doesn't occur with real multi-second tasks.)
        Intentionally NOT load-balanced via quotas/round-robin: that would add bias
        to this race-free path for no real-world benefit."""
        now = time.time()
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                row = self._conn.execute(
                    "SELECT * FROM tasks WHERE status='pending' "
                    "ORDER BY id LIMIT 1").fetchone()
                if row is None:
                    self._conn.execute("ROLLBACK")
                    return None
                self._conn.execute(
                    "UPDATE tasks SET status='in_progress', worker_id=?, "
                    "attempts=attempts+1, started=?, updated=?, lease_expires=? "
                    "WHERE id=?",
                    (worker_id, now, now, now + lease, row["id"]),
                )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
            claimed = self._conn.execute(
                "SELECT * FROM tasks WHERE id=?", (row["id"],)).fetchone()
            return dict(claimed)

    def heartbeat(self, task_id: int, worker_id: str, lease: float = DEFAULT_LEASE) -> bool:
        """Extend the lease on a task this worker still holds. Returns False if the
        task is no longer ours (e.g. already reaped) — the worker should bail."""
        now = time.time()
        cur = self._exec(
            "UPDATE tasks SET lease_expires=?, updated=? "
            "WHERE id=? AND worker_id=? AND status='in_progress'",
            (now + lease, now, task_id, worker_id),
        )
        return cur.rowcount > 0

    def mark_done(self, task_id: int, worker_id: str, result: str = "") -> bool:
        """Mark a held task done. Guarded by worker_id+status so a zombie can't
        complete a task that was reclaimed out from under it."""
        now = time.time()
        cur = self._exec(
            "UPDATE tasks SET status='done', result=?, last_error=NULL, ended=?, "
            "updated=? WHERE id=? AND worker_id=? AND status='in_progress'",
            (result, now, now, task_id, worker_id),
        )
        return cur.rowcount > 0

    def mark_flagged(self, task_id: int, worker_id: str, error: str,
                     result: str = "") -> bool:
        """Flag a held task for human attention (verification failed, exhausted,
        timed out, errored). Never a silent pass. Guarded by worker_id+status."""
        now = time.time()
        cur = self._exec(
            "UPDATE tasks SET status='flagged', last_error=?, result=?, ended=?, "
            "updated=? WHERE id=? AND worker_id=? AND status='in_progress'",
            (error, result, now, now, task_id, worker_id),
        )
        return cur.rowcount > 0

    # -- recovery (used by the supervisor's reaper) ----------------------------
    def reap_expired(self, now: float | None = None) -> list[dict]:
        """Reclaim in_progress tasks whose lease has expired (the holding worker
        died/hung). A task with attempts left goes back to pending (will be
        retried); an exhausted one is flagged 'worker died, attempts exhausted'.
        Returns the list of reclaimed rows (post-transition) so the caller can log
        every one — nothing is lost silently."""
        now = now or time.time()
        reclaimed: list[dict] = []
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                rows = self._conn.execute(
                    "SELECT * FROM tasks WHERE status='in_progress' "
                    "AND lease_expires IS NOT NULL AND lease_expires < ?",
                    (now,)).fetchall()
                for r in rows:
                    if r["attempts"] >= r["max_attempts"]:
                        self._conn.execute(
                            "UPDATE tasks SET status='flagged', "
                            "last_error='worker died, attempts exhausted', "
                            "ended=?, updated=?, worker_id=NULL WHERE id=?",
                            (now, now, r["id"]))
                        new_status = "flagged"
                    else:
                        self._conn.execute(
                            "UPDATE tasks SET status='pending', worker_id=NULL, "
                            "lease_expires=NULL, "
                            "last_error='worker died mid-task, requeued', "
                            "updated=? WHERE id=?",
                            (now, r["id"]))
                        new_status = "pending"
                    d = dict(r)
                    d["status"] = new_status
                    reclaimed.append(d)
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
        return reclaimed

    # -- admin transitions (operator/API; not tied to a worker's ownership) ----
    def cancel(self, task_id: int) -> bool:
        """Operator cancel: a pending or in_progress task -> 'cancelled' (a TERMINAL
        status — history is preserved, the row is never hard-deleted). Returns False
        if the task is missing or already terminal (done/flagged/cancelled).

        For an in_progress task this only flips the row; a worker actively running it
        cannot be force-killed, but its ownership-guarded mark_done/mark_flagged will
        then no-op (status != 'in_progress') and its heartbeat will fail, so the
        in-flight result is discarded rather than recorded. worker_id/lease are
        cleared so the reaper ignores it."""
        now = time.time()
        cur = self._exec(
            "UPDATE tasks SET status='cancelled', worker_id=NULL, lease_expires=NULL, "
            "result='cancelled by operator', ended=?, updated=? "
            "WHERE id=? AND status IN ('pending','in_progress')",
            (now, now, task_id),
        )
        return cur.rowcount > 0

    def requeue(self, task_id: int) -> bool:
        """Operator requeue: send a flagged / cancelled / stuck-in_progress task back
        to 'pending' with attempts reset to 0 and worker/lease/error/timing cleared,
        so it is picked up fresh. Returns False if the task is missing, already
        pending, or already 'done' (re-running a success = add a new task instead)."""
        now = time.time()
        cur = self._exec(
            "UPDATE tasks SET status='pending', attempts=0, worker_id=NULL, "
            "lease_expires=NULL, last_error=NULL, result=NULL, started=NULL, "
            "ended=NULL, updated=? WHERE id=? AND status IN "
            "('flagged','cancelled','in_progress')",
            (now, task_id),
        )
        return cur.rowcount > 0

    # -- introspection ---------------------------------------------------------
    def get(self, task_id: int) -> dict | None:
        rows = self._query("SELECT * FROM tasks WHERE id=?", (task_id,))
        return dict(rows[0]) if rows else None

    def all_tasks(self, limit: int = 500) -> list[dict]:
        return [dict(r) for r in self._query(
            "SELECT * FROM tasks ORDER BY id LIMIT ?", (limit,))]

    def list_tasks(self, *, status: str | None = None, limit: int = 200,
                   offset: int = 0) -> "tuple[list[dict], int]":
        """Paginated task listing done IN SQL (LIMIT/OFFSET) — never materializes the
        whole table. Returns (page_rows, total_matching). O(page), not O(total)."""
        where, params = ("WHERE status=?", (status,)) if status else ("", ())
        total = self._query(f"SELECT COUNT(*) c FROM tasks {where}", params)[0]["c"]
        rows = self._query(
            f"SELECT * FROM tasks {where} ORDER BY id LIMIT ? OFFSET ?",
            (*params, max(1, limit), max(0, offset)))
        return [dict(r) for r in rows], total

    def in_progress_tasks(self) -> list[dict]:
        """Only the in_progress rows (id, worker_id, lease) — for the live-worker
        view. Index-assisted; O(in_progress), not O(total)."""
        return [dict(r) for r in self._query(
            "SELECT id, worker_id, lease_expires FROM tasks WHERE status='in_progress'")]

    def stats(self) -> dict:
        rows = self._query("SELECT status, COUNT(*) c FROM tasks GROUP BY status")
        out = {s: 0 for s in STATUSES}
        for r in rows:
            out[r["status"]] = r["c"]
        out["total"] = sum(out[s] for s in STATUSES)
        return out

    def purge_terminal(self, *, older_than: float | None = None,
                       statuses: "tuple[str, ...]" = ("done", "flagged", "cancelled")
                       ) -> int:
        """Retention: delete TERMINAL rows (done/flagged/cancelled), optionally only
        those that ended more than `older_than` seconds ago. NEVER touches live
        (pending/in_progress) work. Returns rows deleted."""
        statuses = tuple(s for s in statuses if s in ("done", "flagged", "cancelled"))
        if not statuses:
            return 0
        ph = ",".join("?" * len(statuses))
        sql = f"DELETE FROM tasks WHERE status IN ({ph})"
        params: list = list(statuses)
        if older_than is not None:
            sql += " AND (ended IS NULL OR ended < ?)"
            params.append(time.time() - older_than)
        return self._exec(sql, tuple(params)).rowcount

    def close(self) -> None:
        with self._lock:
            self._conn.close()
