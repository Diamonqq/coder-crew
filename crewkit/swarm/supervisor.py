"""Supervisor — the top loop that makes the swarm run 24/7 unattended.

Responsibilities (and nothing else — it does not touch crew internals):

  * spawn N worker PROCESSES (python -m crewkit.swarm.worker) and keep them alive;
  * RESTART any worker that exits while the supervisor is running (a crash, an OOM
    kill, an unhandled error) — with a small backoff and a restart counter;
  * run a REAPER that reclaims tasks whose lease expired because their worker died:
    requeue (attempts remain) or flag (exhausted). This is what makes a killed
    worker's in-flight task NOT get lost;
  * LOG everything to a file and never silently swallow a failure — every flag,
    every crash, every reclaim is logged, and a periodic STATUS SUMMARY (queue
    counts + the list of flagged tasks + per-worker restart counts) is written so
    a human can read later exactly what happened.

Run:
  python -m crewkit.swarm.supervisor --workers 3 --log swarm.log
Stop with Ctrl-C / SIGTERM — it shuts the workers down cleanly.
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
import time

from .queue import TaskQueue

log = logging.getLogger("swarm.supervisor")

REAP_INTERVAL = float(os.environ.get("SWARM_REAP_INTERVAL", "15"))
SUMMARY_INTERVAL = float(os.environ.get("SWARM_SUMMARY_INTERVAL", "60"))
RESTART_BACKOFF = float(os.environ.get("SWARM_RESTART_BACKOFF", "2"))


class Supervisor:
    def __init__(self, n_workers: int, *, db: str | None = None,
                 logfile: str | None = None,
                 reap_interval: float = REAP_INTERVAL,
                 summary_interval: float = SUMMARY_INTERVAL):
        self.n = max(1, n_workers)
        self.db = db
        self.logfile = logfile
        self.reap_interval = reap_interval
        self.summary_interval = summary_interval
        self.q = TaskQueue(db)
        self.procs: dict[str, subprocess.Popen] = {}
        self.restarts: dict[str, int] = {}
        self._stop = False

    # -- worker process lifecycle ---------------------------------------------
    def _spawn(self, wid: str) -> None:
        env = os.environ.copy()
        env["SWARM_WORKER_ID"] = wid
        if self.db:
            env["SWARM_DB"] = self.db
        # Children log to the SAME file (append) so the whole swarm is in one log.
        if self.logfile:
            env["SWARM_LOG"] = self.logfile
        cmd = [sys.executable, "-m", "crewkit.swarm.worker", "--id", wid]
        if self.db:
            cmd += ["--db", self.db]
        p = subprocess.Popen(cmd, env=env)
        self.procs[wid] = p
        log.info("spawned %s (pid %s)", wid, p.pid)

    def _ensure_workers(self) -> None:
        for i in range(self.n):
            wid = f"worker-{i + 1}"
            p = self.procs.get(wid)
            if p is None:
                self._spawn(wid)
            elif p.poll() is not None:   # exited
                rc = p.returncode
                self.restarts[wid] = self.restarts.get(wid, 0) + 1
                log.warning("worker %s (pid %s) exited rc=%s — RESTART #%s",
                            wid, p.pid, rc, self.restarts[wid])
                time.sleep(RESTART_BACKOFF)
                if not self._stop:
                    self._spawn(wid)

    # -- recovery + reporting --------------------------------------------------
    def _reap(self) -> None:
        try:
            reclaimed = self.q.reap_expired()
        except Exception:  # noqa: BLE001
            log.exception("reaper failed this cycle")
            return
        for t in reclaimed:
            if t["status"] == "pending":
                log.warning("REAPED task %s (worker died) -> REQUEUED for retry "
                            "(attempt %s/%s)", t["id"], t["attempts"], t["max_attempts"])
            else:
                log.error("REAPED task %s (worker died) -> FLAGGED (attempts "
                          "exhausted %s/%s): %s", t["id"], t["attempts"],
                          t["max_attempts"], t["description"][:80])

    def status_summary(self) -> str:
        stats = self.q.stats()
        flagged = [t for t in self.q.all_tasks() if t["status"] == "flagged"]
        lines = [
            "==== SWARM STATUS ====",
            f"queue: pending={stats['pending']} in_progress={stats['in_progress']} "
            f"done={stats['done']} flagged={stats['flagged']} total={stats['total']}",
            f"workers: " + ", ".join(
                f"{w}(pid {p.pid}{'' if p.poll() is None else ' DEAD'}, "
                f"restarts {self.restarts.get(w, 0)})"
                for w, p in self.procs.items()) or "workers: (none)",
        ]
        if flagged:
            lines.append(f"FLAGGED ({len(flagged)}) — need attention:")
            for t in flagged:
                lines.append(f"  #{t['id']} [{t.get('last_error','?')}] "
                             f"{t['description'][:70]}")
        else:
            lines.append("FLAGGED: none")
        lines.append("======================")
        return "\n".join(lines)

    # -- main loop -------------------------------------------------------------
    def stop(self, *_):
        log.info("supervisor stop requested — shutting workers down")
        self._stop = True

    def run(self, *, run_seconds: float | None = None,
            until_drained: bool = False) -> int:
        log.info("supervisor up: %s worker(s), db=%s, log=%s, reap=%ss",
                 self.n, self.q.path, self.logfile, self.reap_interval)
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)
        start = time.time()
        last_reap = last_summary = 0.0
        try:
            while not self._stop:
                self._ensure_workers()
                now = time.time()
                if now - last_reap >= self.reap_interval:
                    self._reap()
                    last_reap = now
                if now - last_summary >= self.summary_interval:
                    log.info("\n%s", self.status_summary())
                    last_summary = now
                if until_drained:
                    s = self.q.stats()
                    if s["pending"] == 0 and s["in_progress"] == 0 and s["total"] > 0:
                        log.info("queue drained — stopping (until_drained)")
                        break
                if run_seconds is not None and now - start >= run_seconds:
                    log.info("reached run_seconds=%s — stopping", run_seconds)
                    break
                time.sleep(1)
        finally:
            self._shutdown_workers()
            log.info("final status:\n%s", self.status_summary())
            self.q.close()
        return 0

    def _shutdown_workers(self) -> None:
        for wid, p in self.procs.items():
            if p.poll() is None:
                log.info("terminating %s (pid %s)", wid, p.pid)
                try:
                    p.terminate()
                except Exception:  # noqa: BLE001
                    pass
        deadline = time.time() + 10
        for p in self.procs.values():
            try:
                p.wait(timeout=max(0.1, deadline - time.time()))
            except Exception:  # noqa: BLE001
                try:
                    p.kill()
                except Exception:  # noqa: BLE001
                    pass


def _configure_logging(logfile: str | None) -> None:
    level = logging.DEBUG if os.environ.get("SWARM_DEBUG") else logging.INFO
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if logfile:
        handlers.append(logging.FileHandler(logfile, encoding="utf-8"))
    logging.basicConfig(
        level=level, handlers=handlers,
        format="%(asctime)s %(levelname)s %(name)s %(message)s")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="coder-crew swarm supervisor")
    ap.add_argument("--workers", type=int,
                    default=int(os.environ.get("SWARM_WORKERS", "2")))
    ap.add_argument("--db", default=None, help="queue DB path (else $SWARM_DB)")
    ap.add_argument("--log", default=os.environ.get("SWARM_LOG"),
                    help="log file (also passed to workers)")
    ap.add_argument("--reap-interval", type=float, default=REAP_INTERVAL)
    ap.add_argument("--run-seconds", type=float, default=None,
                    help="stop after N seconds (tests)")
    ap.add_argument("--until-drained", action="store_true",
                    help="stop once the queue is fully drained (tests)")
    args = ap.parse_args(argv)

    _configure_logging(args.log)
    sup = Supervisor(args.workers, db=args.db, logfile=args.log,
                     reap_interval=args.reap_interval)
    return sup.run(run_seconds=args.run_seconds, until_drained=args.until_drained)


if __name__ == "__main__":
    sys.exit(main())
