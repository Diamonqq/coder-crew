"""A single worker process that drains the queue through the real crew.

One worker = one OS process (the supervisor spawns N of them; running it directly
gives you one). The loop:

  1. claim_next() an oldest pending task (atomic across all workers);
  2. while it runs, a heartbeat thread keeps the task's lease fresh so the reaper
     won't steal an actively-running task — if THIS process dies, heartbeats stop,
     the lease expires, and the supervisor's reaper reclaims the task (requeue or
     flag). Nothing is lost;
  3. run it through crewkit.crew via runner.get_runner();
  4. mark_done ONLY on a real green gate, else mark_flagged with the captured
     error/caveat. Marks are ownership-guarded: if the task was reaped out from
     under a hung worker, the mark no-ops and we log that the result was discarded.

Idle behavior (Stage 3) when the queue is empty — configurable, NEVER invents work:
  * SWARM_IDLE=poll (default): sleep SWARM_POLL_INTERVAL, then look again;
  * SWARM_IDLE=stop:           exit 0 cleanly.

Watchdog (the outer backstop for a HANG): the heartbeat keeps a task's lease fresh
from a daemon thread regardless of whether the main thread is making progress, so a
worker wedged in an infinite loop / deadlock would otherwise pin its task forever.
A separate watchdog thread enforces a hard wall-clock cap per claimed task
(SWARM_WORKER_WATCHDOG). If a single task exceeds it, the worker SELF-TERMINATES —
exactly like a crash: it stops heartbeating, the lease expires, the supervisor's
reaper requeues/flags the task per the normal attempt rules, and the supervisor
restarts the worker (slot reclaimed). It NEVER stays pinned in_progress.
  * This is distinct from SWARM_TASK_TIMEOUT, which is enforced INSIDE the crew
    runner (runner.run_through_crew) and flags an over-long crew run cleanly. The
    watchdog is the OUTER backstop for hangs the inner timeout can't catch (a hang
    outside the crew loop, a wedged syscall, runner bug). Default is above
    SWARM_TASK_TIMEOUT so the inner timeout fires first; 0 disables the watchdog.

Run directly:
  python -m crewkit.swarm.worker --id worker-1
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
import time

from .queue import DEFAULT_LEASE, TaskQueue
from . import runner

log = logging.getLogger("swarm.worker")

POLL_INTERVAL = float(os.environ.get("SWARM_POLL_INTERVAL", "5"))
IDLE_MODE = os.environ.get("SWARM_IDLE", "poll").lower()   # poll | stop
LEASE = float(os.environ.get("SWARM_LEASE", str(DEFAULT_LEASE)))
# Outer hang backstop (seconds). Default sits ABOVE the inner crew-run timeout so
# the crew's own SWARM_TASK_TIMEOUT flags first; the watchdog only catches hangs it
# can't. 0 disables. See module docstring.
WATCHDOG = float(os.environ.get("SWARM_WORKER_WATCHDOG", "")
                 or max(2 * runner.TASK_TIMEOUT, 600.0))
_EXIT_WATCHDOG = 70   # process exit code when the watchdog trips (distinct from crashes)


class Worker:
    def __init__(self, worker_id: str, queue: TaskQueue, *,
                 idle: str = IDLE_MODE, poll_interval: float = POLL_INTERVAL,
                 lease: float = LEASE, watchdog: float = WATCHDOG):
        self.id = worker_id
        self.q = queue
        self.idle = idle
        self.poll_interval = poll_interval
        self.lease = lease
        self.watchdog = watchdog
        self._stop = threading.Event()
        self.run_fn = runner.get_runner()

    def stop(self, *_):
        log.info("[%s] stop requested — finishing current step", self.id)
        self._stop.set()

    # -- lease heartbeat while a task runs -------------------------------------
    def _heartbeat_loop(self, task_id: int, done: threading.Event):
        interval = max(2.0, self.lease / 4.0)
        while not done.wait(interval):
            alive = self.q.heartbeat(task_id, self.id, self.lease)
            if not alive:
                log.warning("[%s] task %s no longer ours (reaped) — heartbeat stop",
                            self.id, task_id)
                return

    def _watchdog_loop(self, task_id: int, done: threading.Event):
        # Wait up to the cap for the task to finish. If it doesn't, the main thread
        # is wedged — self-terminate so this looks exactly like a crash: lease
        # expires, reaper applies the normal attempt rules, supervisor restarts us.
        if not done.wait(self.watchdog):
            log.error("[%s] task %s EXCEEDED watchdog cap (%.0fs) — self-terminating "
                      "(rc=%d) so the lease is reaped and the slot is reclaimed",
                      self.id, task_id, self.watchdog, _EXIT_WATCHDOG)
            os._exit(_EXIT_WATCHDOG)

    def _run_task(self, task: dict) -> None:
        tid = task["id"]
        log.info("[%s] claimed task %s (attempt %s): %s",
                 self.id, tid, task["attempts"], task["description"][:80])
        done = threading.Event()
        hb = threading.Thread(target=self._heartbeat_loop, args=(tid, done), daemon=True)
        hb.start()
        if self.watchdog > 0:
            threading.Thread(target=self._watchdog_loop, args=(tid, done),
                             daemon=True).start()
        try:
            try:
                result = self.run_fn(task)
            except Exception as exc:  # noqa: BLE001 — a crashing run must FLAG, not pass
                log.exception("[%s] task %s raised during execution", self.id, tid)
                ok = self.q.mark_flagged(
                    tid, self.id, f"worker exception: {type(exc).__name__}: {exc}")
                self._log_mark(tid, "flagged", ok, "execution raised")
                return
        finally:
            done.set()
            hb.join(timeout=2)

        if result.is_done:
            ok = self.q.mark_done(tid, self.id, result.summary)
            self._log_mark(tid, "done", ok, result.summary)
        else:
            err = f"[{result.kind}] {result.summary}"
            ok = self.q.mark_flagged(tid, self.id, err, result.detail[:4000])
            self._log_mark(tid, "flagged", ok, err)

    def _log_mark(self, tid: int, intended: str, ok: bool, msg: str) -> None:
        if ok:
            level = logging.INFO if intended == "done" else logging.WARNING
            log.log(level, "[%s] task %s -> %s: %s", self.id, tid, intended.upper(), msg)
        else:
            # ownership-guarded write failed => the task was reclaimed mid-run.
            log.warning("[%s] task %s result DISCARDED (task was reclaimed; "
                        "intended %s: %s)", self.id, tid, intended, msg)

    # -- main loop -------------------------------------------------------------
    def run(self, *, max_tasks: int | None = None, once: bool = False) -> int:
        log.info("[%s] worker up (idle=%s, poll=%ss, lease=%ss, watchdog=%ss, runner=%s)",
                 self.id, self.idle, self.poll_interval, self.lease,
                 (f"{self.watchdog:.0f}" if self.watchdog > 0 else "off"),
                 self.run_fn.__name__)
        completed = 0
        while not self._stop.is_set():
            try:
                task = self.q.claim_next(self.id, self.lease)
            except Exception:  # noqa: BLE001 — a transient DB lock shouldn't kill a worker
                log.exception("[%s] claim failed; backing off", self.id)
                self._sleep(self.poll_interval)
                continue

            if task is None:
                if self.idle == "stop":
                    log.info("[%s] queue empty and idle=stop — exiting cleanly", self.id)
                    return 0
                log.debug("[%s] queue empty — idling %ss", self.id, self.poll_interval)
                if once:
                    return 0
                self._sleep(self.poll_interval)
                continue

            self._run_task(task)
            completed += 1
            if once or (max_tasks is not None and completed >= max_tasks):
                log.info("[%s] reached task limit (%s) — exiting", self.id, completed)
                return 0
        log.info("[%s] stopped after %s task(s)", self.id, completed)
        return 0

    def _sleep(self, secs: float) -> None:
        # interruptible sleep so stop() / SIGTERM is responsive
        self._stop.wait(secs)


def _identify(worker_id: str, db_path: str) -> None:
    """Make a worker identifiable by the DB it serves even when launched with only
    $SWARM_DB (F4). Best-effort process title: setproctitle if installed (shows in
    the cross-platform process list), else the Windows console title. Always logged.
    NOTE: the most robust identifier is `--db` on the command line — the supervisor
    always passes it, and direct launches should too (it shows in the process list)."""
    title = f"swarm-worker {worker_id} db={db_path}"
    try:
        import setproctitle  # optional; not a hard dependency
        setproctitle.setproctitle(title)
    except Exception:  # noqa: BLE001
        if sys.platform == "win32":
            try:
                import ctypes
                ctypes.windll.kernel32.SetConsoleTitleW(title)
            except Exception:  # noqa: BLE001
                pass


def _configure_logging(worker_id: str) -> None:
    if logging.getLogger().handlers:   # supervisor already configured root
        return
    level = logging.DEBUG if os.environ.get("SWARM_DEBUG") else logging.INFO
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    logfile = os.environ.get("SWARM_LOG")
    if logfile:
        handlers.append(logging.FileHandler(logfile, encoding="utf-8"))
    logging.basicConfig(
        level=level, handlers=handlers,
        format="%(asctime)s %(levelname)s %(name)s %(message)s")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="coder-crew swarm worker")
    ap.add_argument("--id", default=os.environ.get("SWARM_WORKER_ID")
                    or f"worker-{os.getpid()}")
    ap.add_argument("--db", default=None, help="queue DB path (else $SWARM_DB)")
    ap.add_argument("--once", action="store_true", help="one cycle then exit (tests)")
    ap.add_argument("--max-tasks", type=int, default=None)
    args = ap.parse_args(argv)

    _configure_logging(args.id)
    q = TaskQueue(args.db)
    _identify(args.id, q.path)            # F4: surface the served DB (title + log)
    log.info("[%s] serving queue db=%s", args.id, q.path)
    w = Worker(args.id, q)
    signal.signal(signal.SIGINT, w.stop)
    signal.signal(signal.SIGTERM, w.stop)
    try:
        return w.run(max_tasks=args.max_tasks, once=args.once)
    finally:
        q.close()


if __name__ == "__main__":
    sys.exit(main())
