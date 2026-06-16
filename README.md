# coder-crew

A **local-first agentic coding crew**. You give it a goal; a *manager* model plans
it into subtasks, *worker* models implement each one with real tools (file + shell)
gated behind your approval, every subtask is checked by a **runnable acceptance
gate**, and the manager reviews and integrates the result into a final answer.

It runs entirely on local models via [Ollama](https://ollama.com). An **optional,
opt-in auto-router** can escalate a subtask that local can't finish (or can't
verify) to Claude/Opus — off by default, so there's never surprise spend.

```
manager (plan) ──► worker · worker · worker ──► manager (review + synthesize)
                      │
                      └─ each subtask: implement → run its acceptance gate → repair (≤3) 
```

## Why it's different

- **Acceptance-gated, not vibes.** Every subtask carries a runnable check
  (`pytest …`). A subtask only counts as **passed** when a *real* gate goes green.
  Checks that just prove code "imports/runs" are rejected — they verify nothing.
- **Honest accounting.** Manual-review / unverifiable subtasks are tracked as
  `unverified`, never silently counted as passed.
- **Disjoint file ownership.** The planner assigns each subtask the exact files it
  may write; the tools refuse writes outside that set, so parallel-decomposed work
  doesn't clobber itself.
- **Local-vs-Opus auto-router** *(optional)*. Default: everything runs local.
  - *Reactive*: a subtask that fails its gate, can't be self-verified, or crashes
    is re-dispatched to Opus — which, for the unverifiable case, is also asked to
    **author a real test gate** (validated by the same allowlist) so the result is
    actually trustworthy.
  - *Predictive*: true threading/shared-state concurrency subtasks (which local
    models reliably stall on) route to Opus upfront.
  - Everything is surfaced in the UI/history as `local` / `local→Opus` / `Opus`.
- **You approve the dangerous bits.** File writes and shell commands pause for your
  approval (or auto-approve for unattended use). Opus, when used, runs through the
  *same* approval gate via the SDK's permission callback — read-only tools auto-
  allow, everything else (including unknown/new tools) is gated fail-safe.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running locally with a coding model. Recommended:
  ```
  ollama pull qwen3-coder:30b
  ```
  (A 30B-A3B MoE — fast despite its size, strong at agentic tool use. Smaller
  coders work too; pick whatever fits your VRAM.)
- *(optional)* `claude-agent-sdk` for the Opus escalation path.

## Install & run

```bash
git clone https://github.com/Diamonqq/coder-crew
cd coder-crew
python -m venv .venv && .venv/Scripts/activate      # (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt

python run.py            # or: python -m crewkit.server
```

Open **http://127.0.0.1:8770**, describe a goal, pick a manager + worker model, and
**Launch**. Use **Autopilot** to have a local model sharpen a rough idea into a
precise goal and suggest a model combo.

### Config (env)

| var | default | meaning |
|-----|---------|---------|
| `CREW_HOST` | `127.0.0.1` | bind host |
| `CREW_PORT` | `8770` | bind port |
| `CREW_DB` | `./crew_history.db` | run-history SQLite path |
| `PCP_OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama base URL |

## Model roles

Roles are pluggable agent specs:

- `ollama:<model>` — e.g. `ollama:qwen3-coder:30b` (we drive the tool-calling loop)
- `claude:<model>` — e.g. `claude:claude-opus-4-8` (a real Claude Code session via
  the Agent SDK; needs `claude-agent-sdk` + a logged-in `claude` CLI)

A common setup is a **Claude/Opus manager** (the strongest planner/reviewer) with a
**local worker** doing the building — or all-local for zero cost. On a single GPU,
workers run sequentially; the win is decomposition + verification quality.

## Security

This is a **local, single-user** tool. Workers can run shell commands, so the
server binds to loopback and has no auth. Don't expose it on a non-loopback host
without a tunnel + authentication in front of it. Acceptance-check commands are
restricted to a single test-runner invocation (no shell chaining) by an allowlist —
that allowlist, not a sandbox, is what keeps a model-authored check from being an
arbitrary command.

## License

MIT — see [LICENSE](LICENSE).
