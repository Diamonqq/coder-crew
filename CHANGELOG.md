# Changelog

All notable changes to **coder-crew** are documented here.
This project adheres to [Semantic Versioning](https://semver.org).

## [1.0.0] — 2026-06-30

First stable release. coder-crew matured from a verified single-run coder into a
local-first system you can also run **unattended for hours** — with the same
verification discipline held constant the whole way: a subtask counts as *passed*
only on a real green gate, and unverified work is surfaced, never dressed up as a
pass.

### Packaging & hardening (publish prep, 2026-07-07)
- **`pyproject.toml`** — `pip install`-able (PEP 621); `coder-crew` console script →
  `crewkit.server:main`; optional extras `[escalation]` (claude-agent-sdk) and `[mcp]`.
- **`crewkit/ledger.py`** — a minimal, thread-safe, in-memory per-run token ledger
  backing `crew`'s already-guarded `ledger.log`/`run_totals` calls, so standalone runs
  show real per-run token totals instead of silently no-opping (the panel build backs the
  same interface with a richer store).
- **`tests/test_smoke.py`** — a standalone suite (no host app) exercising REAL paths: the
  acceptance allowlist rejecting chained/substituted commands, the gate actually running
  pytest (pass and fail), and the **completeness-honesty invariant** — a subtask with a
  missing required deliverable is `incomplete`, never `passed`, even with a green own-gate.
- Dropped a dead host-only token-meter feed from the local-agent streamer (the standalone
  package has no such surface; per-run totals go through `crewkit.ledger`).

### Added
- **24/7 unattended swarm layer** (`crewkit/swarm/`) — a persistent SQLite task
  queue with atomic, race-free claiming and lease-based recovery; **workers** that
  mark a task `done` **only on a real green gate** (repair-exhaustion, timeout, an
  unverified result, or a crash all end `flagged`, never a silent pass), each with a
  per-task watchdog backstop; a **supervisor** that keeps N workers alive, restarts
  crashes, and reaps dead workers' tasks; and a `/api/swarm/*` surface plus a
  **🐝 Swarm** tab to add / requeue / cancel / purge and watch the queue.
  Stress-tested for throughput, 16-worker claim race-freedom, crash storms,
  lease/timeout edges, WAL contention + crash-consistency, API-under-load, and
  resource leaks — it does not lose, duplicate, or corrupt work.
- **Swarm write gate** — `/api/swarm/*` write routes require
  `Authorization: Bearer $SWARM_API_TOKEN` when that variable is set; otherwise they
  are restricted to loopback callers. Reads stay open.

### Changed
- Public-release hygiene: removed an extraction-leftover launch tile from the UI,
  genericized a machine-specific path in the swarm docs, and excluded dev
  verification scratch (`_verify_*.py`) and the swarm queue DB from the shipped tree.

## [0.2.1] — 2026-06-18

### Changed
- Local research that **actually researches** — live web snippets + read-the-page
  retrieval + fuller synthesis. (Still an unverified aid: it has no correctness
  gate — verify the claims and sources yourself.)

## [0.2.0] — 2026-06-17

### Added
- **Research swarm** — a manager fans a topic out to parallel, **read-only**
  researcher agents (web search + read, no file writes / no shell), then synthesizes
  and ranks the findings into a report you can read as a document or slides and
  export (Markdown / HTML / PDF / CSV). Flat, or tiered with sub-managers.
- **Coding assistant** — a conversational, tool-using agent for quick help, with
  every dangerous tool going through the **same approval gate** as the crew.
- **Connection panel + unattended mode** — UI control of the optional Claude path,
  plus an auto-approve mode for hands-off runs.
- **Completeness + spec-coverage verification** — a deliverable shipped with no test
  that actually ran is flagged `unverified`; and with a Claude/Opus manager, a
  **spec-coverage review** critiques the tests against the spec and can turn an
  omitted-case bug into a caught `failed` instead of a shipped one.
- Claude-style UI and `.exe` packaging.

## [0.1.0] — 2026-06-16

- Initial public release: a **local-first agentic coding crew**. A *manager* model
  plans a goal into subtasks; *worker* models implement each with file + shell tools
  behind an approval gate; **every subtask is gated on a runnable acceptance check**;
  the manager reviews and integrates the result. An optional local-vs-Opus
  auto-router is **off by default** (no surprise spend). The differentiator from day
  one: *acceptance-gated, not vibes* — unverified work is never counted as passed.

[1.0.0]: https://github.com/Diamonqq/coder-crew/releases/tag/v1.0.0
[0.2.1]: https://github.com/Diamonqq/coder-crew/releases/tag/v0.2.1
[0.2.0]: https://github.com/Diamonqq/coder-crew/releases/tag/v0.2.0
[0.1.0]: https://github.com/Diamonqq/coder-crew/releases/tag/v0.1.0
