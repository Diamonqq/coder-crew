# crewkit.swarm — 24/7 unattended worker layer

A persistent layer **on top of** coder-crew so it runs unattended: workers drain a
standing task queue, self-gate every result through the existing crew gates, and a
supervisor keeps everything alive and **surfaces** failures instead of plowing
through them. It **wraps** `crewkit.crew.MANAGER` + the acceptance gate — it does
not modify or redesign any crew internals.

## Honest-accounting rule
A task is marked **`done` only on a REAL green acceptance gate** (the crew's own
per-subtask verdict). Repair-exhaustion, timeout, an unverified/manual result, a
crew error, or a worker crash all end as **`flagged`** with the error captured —
never a silent pass. "passed-with-caveats" (ran but no real gate) is flagged, not
done.

## Pieces
| module | role |
|---|---|
| `queue.py` | persistent SQLite queue (WAL, cross-process); atomic `claim_next`, ownership-guarded `mark_done`/`mark_flagged`, lease-based `reap_expired` |
| `runner.py` | launches one auto-approved crew run per task, waits (wall-clock capped), maps the crew's honest per-subtask status to done/flagged |
| `worker.py` | one worker **process**: claim → heartbeat lease → run → mark; idles safely when empty |
| `supervisor.py` | top loop: keeps N worker processes alive, restarts crashes, reaps dead workers' tasks, logs everything + a status summary |

## Seed the queue
```python
from crewkit.swarm.queue import TaskQueue
q = TaskQueue()                       # uses $SWARM_DB or ./swarm_queue.db
q.add("Create calc.py with add(a,b) and a pytest asserting add(2,3)==5.",
      acceptance="pytest test_calc.py -q",   # optional hint to the crew's manager
      max_attempts=2)
q.close()
```

## Launch the swarm
```bash
# point both at the same DB
set SWARM_DB=.\swarm_queue.db                     # PowerShell: $env:SWARM_DB=...
python -m crewkit.swarm.supervisor --workers 3 --log swarm.log
```
Stop with Ctrl-C — workers shut down cleanly. Read `swarm.log` (and the periodic
`==== SWARM STATUS ====` block) to see done/flagged counts and every flagged task.

Run a single worker without the supervisor: `python -m crewkit.swarm.worker --id w1`.

## Key env knobs
| var | default | meaning |
|---|---|---|
| `SWARM_DB` | `./swarm_queue.db` | queue DB path |
| `SWARM_WORKERS` | 2 | worker count (or `--workers`) |
| `SWARM_IDLE` | `poll` | `poll` (sleep+retry) or `stop` (exit when empty) |
| `SWARM_POLL_INTERVAL` | 5 | idle poll seconds |
| `SWARM_LEASE` | 1800 | task lease seconds; a worker must heartbeat within this or be reaped |
| `SWARM_TASK_TIMEOUT` | 1800 | inner cap per crew run (runner flags `timeout`) |
| `SWARM_WORKER_WATCHDOG` | `max(2×TASK_TIMEOUT, 600)` | **outer** hang backstop: hard wall-clock cap per claimed task. If exceeded the worker self-terminates (like a crash) so the lease is reaped and the slot reclaimed — a wedged worker can't pin a task forever. `0` disables. Default sits above `SWARM_TASK_TIMEOUT` so the inner timeout flags cleanly first; the watchdog only catches hangs *outside* the crew loop. |
| `SWARM_MANAGER_SPEC` | `ollama:huihui_ai/gemma-4-abliterated:26b-qat` | crew manager model |
| `SWARM_WORKER_SPEC` | `ollama:qwen3-coder:30b` | crew worker model |
| `SWARM_LOG` | — | log file (supervisor passes it to workers too) |
| `SWARM_RUNNER` | (real crew) | set to `test` for the deterministic test double (mechanics tests only — never produces a fake pass) |

## HTTP API + GUI (mounted on the existing coder-crew server)
`crewkit/swarm/api.py` adds an `/api/swarm/*` router (included by `crewkit.server`)
and a **🐝 Swarm** tab in the web GUI for monitoring + control. Start the normal
server (`python run.py`, port 8770) and open the Swarm tab.

| method | route | gated | purpose |
|---|---|---|---|
| GET  | `/api/swarm/status` | no | counts per status (GROUP BY) + live workers (from in_progress leases). O(status-groups + in_progress) — **does not scan the whole table**, so cost stays flat as history grows |
| GET  | `/api/swarm/tasks?status=&limit=&offset=` | no | **paginated** task list (SQL LIMIT/OFFSET; default 200, max 1000). Response: `{tasks, count, total, limit, offset}` |
| POST | `/api/swarm/tasks` | **yes** | add `{description, acceptance?, spec?, max_attempts?}` |
| POST | `/api/swarm/tasks/{id}/requeue` | **yes** | flagged/cancelled/stuck → pending, attempts reset |
| POST | `/api/swarm/tasks/{id}/cancel` | **yes** | pending/in_progress → cancelled (terminal; never hard-deleted) |
| POST | `/api/swarm/tasks/purge` | **yes** | retention: delete TERMINAL rows only `{older_than_seconds?, statuses?}`. Live work never touched. SQL equiv: `DELETE FROM tasks WHERE status IN ('done','flagged','cancelled') AND ended < <cutoff>` |

**Write safety (`require_write_auth`)** — safe by default, two layers:
- `$SWARM_API_TOKEN` set → writes require `Authorization: Bearer <token>` (works from any host);
- unset → writes require a **loopback** client (off-localhost writes are rejected 403).

Reads are open (they expose metadata, not the ability to queue code). The GUI sends
the token from the panel's "API token" field (stored in `localStorage`), so on
localhost no token is needed.

Notes:
- `live_workers`/`status.workers` are derived from in_progress task leases, so an
  **idle** worker (holding no task) shows as 0 live — it reflects active holders,
  not idle pollers.
- Cancelling an **in_progress** task flips the row to `cancelled`; the worker's
  ownership-guarded `mark_*` then no-ops (result discarded) — but an in-flight crew
  run can't be force-killed and finishes in the background.
- **Work distribution** is pull-based: the next *free* worker to win the claim lock
  takes the next task (work-conserving, optimal for variable-length tasks). It is
  intentionally NOT round-robin — that would bias the race-free claim path for no
  real-world gain. With real (non-instant) tasks the distribution is naturally even.
- **Worker identity**: every worker logs `serving queue db=<path>` at startup and
  sets a best-effort process title. The supervisor passes `--db` to each worker
  (shows in the process list); pass `--db` to direct launches for the same.

## Verify
- Stage 1 queue gate: `python -m crewkit.swarm._verify_stage1`
- API read gate: `python -m crewkit.swarm._verify_api_read`
- Write-safety gate: `python -m crewkit.swarm._verify_auth`
- Mechanics (idle / crash / restart): set `SWARM_RUNNER=test` and use `TEST:done`,
  `TEST:flag`, `TEST:sleep:N`, `TEST:crash`, `TEST:crashfirst` directives in a task
  description.
