"""crewkit.swarm — a 24/7 unattended worker layer ON TOP of coder-crew.

This package WRAPS the existing crew (crewkit.crew.MANAGER + gates); it does not
modify or redesign crew internals. It adds three things:

  * queue.py      — a persistent SQLite task queue (pending/in_progress/done/flagged)
  * worker.py     — a worker process that drains the queue through the real crew,
                    self-gates each result, and idles safely when empty
  * supervisor.py — a top loop that keeps N worker processes alive, restarts
                    crashed ones, reclaims their in-flight tasks, and logs/surfaces
                    every flag and crash instead of swallowing it

Honest-accounting rule (inherited from the crew): a task is only ever marked
`done` on a REAL green acceptance gate. Repair-exhaustion, timeout, an unverified
result, or a worker crash all end as `flagged` — never a silent pass.
"""
__version__ = "0.1.0"
