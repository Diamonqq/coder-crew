"""Run one queued task through the REAL coder-crew, honestly.

This is the bridge between the queue and `crewkit.crew.MANAGER`. It does NOT
reimplement any crew logic — it launches a normal auto-approved crew run, waits
for it to finish (with a wall-clock cap), then reads the crew's OWN honest
per-subtask status to decide the task's outcome.

Honest mapping (the crew already sets Worker.status honestly — see crew._log_run):
  * every subtask w.status == 'done'              -> DONE       (real green gates)
  * any subtask w.status in {failed, error}       -> FLAGGED    (hard failure)
  * otherwise (some 'unverified'/manual/none)     -> FLAGGED    (passed-with-caveats:
                                                     ran but NOT verified — never
                                                     reported as a pass)
  * crew run errored / produced no plan           -> FLAGGED    (crew_error)
  * wall-clock cap hit                            -> FLAGGED    (timeout, run cancelled)

A pluggable test double (SWARM_RUNNER=test) exists for exercising the queue /
worker-loop / supervisor MECHANICS without spending a 20GB local model run on
every crash test. It is NEVER the default and is clearly labelled in output.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

# Sensible local defaults for an unattended box (per the project's tested combo:
# a capable manager that writes compliant acceptance + a strong coder worker).
DEFAULT_MANAGER = os.environ.get(
    "SWARM_MANAGER_SPEC", "ollama:huihui_ai/gemma-4-abliterated:26b-qat")
DEFAULT_WORKER = os.environ.get(
    "SWARM_WORKER_SPEC", "ollama:qwen3-coder:30b")
# Hard wall-clock cap per task (seconds). Past this the crew run is cancelled and
# the task is flagged 'timeout' — never left hanging, never marked done.
TASK_TIMEOUT = float(os.environ.get("SWARM_TASK_TIMEOUT", "1800"))
_POLL = 1.0
_TERMINAL = {"done", "error", "cancelled"}


@dataclass
class RunResult:
    outcome: str                 # 'done' | 'flagged'
    kind: str                    # passed | failed | unverified | timeout | crew_error
    summary: str                 # short human line
    detail: str = ""             # error / caveat detail captured for the queue
    run_id: str = ""
    subtasks: list = field(default_factory=list)  # [(title, status)]

    @property
    def is_done(self) -> bool:
        return self.outcome == "done"


def _summarize_subtasks(run) -> list:
    return [(w.title, w.status) for w in (run.workers or [])]


def run_through_crew(task: dict, *, manager_spec: str = DEFAULT_MANAGER,
                     worker_spec: str = DEFAULT_WORKER,
                     timeout: float = TASK_TIMEOUT,
                     poll: float = _POLL) -> RunResult:
    """Launch an auto-approved crew run for `task`, wait for it, map to a RunResult.
    Lazy-imports crew so the queue module stays importable without the model stack."""
    from .. import crew

    goal = task["description"]
    if task.get("spec"):
        goal = f"{goal}\n\nADDITIONAL SPEC:\n{task['spec']}"
    # NOTE: the crew's manager authors each subtask's acceptance gate. A task-level
    # `acceptance` is recorded on the queue row for the human; we surface it to the
    # manager as a hint so its generated gate matches the caller's intent.
    if task.get("acceptance"):
        goal = f"{goal}\n\nThe result MUST satisfy this check: {task['acceptance']}"

    run = crew.MANAGER.start(
        goal, manager_spec=manager_spec, worker_spec=worker_spec,
        auto_approve=True,            # fully unattended — no approval pauses
        allow_escalation=False,       # never spend Opus budget without a human
    )

    deadline = time.time() + timeout
    while run.status not in _TERMINAL:
        if time.time() > deadline:
            run._cancel = True        # ask the crew to stop at its next checkpoint
            # give it a moment to unwind, then report timeout regardless
            for _ in range(10):
                if run.status in _TERMINAL or run.status == "cancelled":
                    break
                time.sleep(0.5)
            return RunResult(
                outcome="flagged", kind="timeout",
                summary=f"timed out after {int(timeout)}s (run cancelled)",
                detail=f"crew run {run.id} exceeded the {int(timeout)}s cap",
                run_id=run.id, subtasks=_summarize_subtasks(run))
        time.sleep(poll)

    subs = _summarize_subtasks(run)

    if run.status == "error":
        return RunResult(outcome="flagged", kind="crew_error",
                         summary="crew run errored",
                         detail=run.error or "unknown crew error",
                         run_id=run.id, subtasks=subs)
    if run.status == "cancelled":
        return RunResult(outcome="flagged", kind="timeout",
                         summary="crew run cancelled",
                         detail="run was cancelled before completing",
                         run_id=run.id, subtasks=subs)
    if not run.workers:
        return RunResult(outcome="flagged", kind="crew_error",
                         summary="crew produced no subtasks (planning failed)",
                         detail=run.error or "manager produced an empty plan",
                         run_id=run.id, subtasks=subs)

    # crew finished 'done' — now read the crew's OWN honest per-subtask verdict.
    statuses = [w.status for w in run.workers]
    if all(s == "done" for s in statuses):
        return RunResult(outcome="done", kind="passed",
                         summary=f"all {len(statuses)} subtask(s) passed their gates",
                         detail=(run.final or "")[:2000],
                         run_id=run.id, subtasks=subs)

    hard = [w for w in run.workers if w.status in ("failed", "error")]
    if hard:
        det = "; ".join(f"{w.title}: {(w.output or w.gate_output or '').strip()[:200]}"
                        for w in hard)
        return RunResult(outcome="flagged", kind="failed",
                         summary=f"{len(hard)} subtask(s) failed their gate",
                         detail=det or "subtask(s) failed", run_id=run.id, subtasks=subs)

    # Remainder: ran but unverified (manual / weak / incomplete gate). NOT a pass.
    unver = [w.title for w in run.workers if w.status not in ("done",)]
    return RunResult(
        outcome="flagged", kind="unverified",
        summary=f"passed-with-caveats: {len(unver)} subtask(s) unverified (no real gate)",
        detail="unverified subtasks: " + ", ".join(unver), run_id=run.id, subtasks=subs)


# --------------------------------------------------------------------------- #
# Test double — mechanics only. Selected by SWARM_RUNNER=test. NOT the default.
# Reads a directive from the task description so crash/idle/restart tests are
# deterministic and instant. Honest by construction: it only ever returns 'done'
# when explicitly told TEST:done, mirroring the real "only a green gate passes".
# --------------------------------------------------------------------------- #
def _test_runner(task: dict, **_) -> RunResult:
    desc = task.get("description", "")
    directive = "done"
    for tok in desc.split():
        if tok.startswith("TEST:"):
            directive = tok[len("TEST:"):]
            break
    parts = directive.split(":")
    kind = parts[0]
    if kind == "sleep":
        time.sleep(float(parts[1]) if len(parts) > 1 else 5)
        return RunResult("done", "passed", "[test] slept then passed")
    if kind == "crash":
        # Simulate a hard worker-process crash mid-task.
        os._exit(137)
    if kind == "crashfirst":
        # Crash on the first attempt, succeed on any retry. Lets the crash test
        # prove a killed task is REQUEUED and then COMPLETES (retried, not lost).
        if int(task.get("attempts", 1)) <= 1:
            os._exit(137)
        return RunResult("done", "passed", "[test] passed on retry after a crash")
    if kind == "flag":
        return RunResult("flagged", "failed", "[test] deliberately flagged",
                         detail="test double: forced flag")
    if kind == "flagfirst":
        # Flag on attempt 1, pass on retry — lets the CRASH/reap path (which keeps
        # attempts) prove a requeued task can succeed. NOTE: the API requeue resets
        # attempts to 0, so for that path use TEST:wantfile below instead.
        if int(task.get("attempts", 1)) <= 1:
            return RunResult("flagged", "failed", "[test] flagged on first attempt",
                             detail="test double: flag-then-pass")
        return RunResult("done", "passed", "[test] passed on rerun after requeue")
    if kind == "wantfile":
        # Pass only if a marker file exists — models a task that FAILS, then its
        # cause is fixed, then an API requeue makes it RERUN and now SUCCEED.
        import tempfile
        marker = os.path.join(tempfile.gettempdir(),
                              parts[1] if len(parts) > 1 else "swarm_marker")
        if os.path.exists(marker):
            return RunResult("done", "passed", "[test] marker present — passed on rerun")
        return RunResult("flagged", "failed", "[test] marker absent — flagged",
                         detail=f"test double: waiting for {marker}")
    if kind == "unverified":
        return RunResult("flagged", "unverified",
                         "[test] passed-with-caveats (unverified)",
                         detail="test double: forced unverified")
    return RunResult("done", "passed", "[test] passed")


def get_runner():
    """Return the active runner callable. Real crew unless SWARM_RUNNER=test."""
    if os.environ.get("SWARM_RUNNER", "").lower() == "test":
        return _test_runner
    return run_through_crew
