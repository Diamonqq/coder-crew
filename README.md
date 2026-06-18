# coder-crew

> **A local-first agentic coding crew that codes what it can and is honest about
> what it can't** — every subtask is gated on a runnable test, and unverified work
> is never counted as passed.

You give it a goal; a *manager* model plans it into subtasks, *worker* models
implement each one with real tools (file + shell) gated behind your approval, every
subtask is checked by a **runnable acceptance gate**, and the manager reviews and
integrates the result into a final answer.

It runs entirely on local models via [Ollama](https://ollama.com). An **optional,
opt-in auto-router** can escalate a subtask that local can't finish (or can't
verify) to Claude/Opus — off by default, so there's never surprise spend.

```
manager (plan) ──► worker · worker · worker ──► manager (review + synthesize)
                      │
                      └─ each subtask: implement → run its acceptance gate → repair (≤3) 
```

## What it does and doesn't guarantee

Honesty about the boundary is the whole point:

- **Catches:** failed gates; **missing tests** (the completeness check flags a
  module shipped without a test that actually ran — counted `unverified`, not
  passed); and — with a **Claude/Opus manager** — many **incomplete-test gaps**
  (the coverage review critiques tests against the spec and can turn an
  omitted-case bug into a caught `failed`).
- **Does NOT guarantee that a passing test suite is COMPLETE.** A green gate proves
  the tests *pass*, not that they *cover the spec* — test completeness is undecidable
  in general. The coverage review **reduces** this gap (with an Opus manager) but
  cannot eliminate it; pure-local runs rely on the gate plus **your own review**.
  This is the honest boundary, and it's the differentiator — unverified work is
  surfaced as `unverified`/`failed`, never dressed up as a pass.

## Also included: research swarm + assistant

Two more local-first surfaces alongside the coder crew:

- **Research swarm** — a manager model fans a topic out to parallel researcher
  agents (web search + read, **strictly read-only** — no file writes, no shell),
  then synthesizes and ranks the findings into a report you can read as a document
  or slides and export (Markdown / HTML / PDF / CSV). Flat, or tiered with
  sub-managers for broad topics.
- **Assistant** — a conversational, tool-using agent for quick help: it can read/
  write files, run shell, search the web, and `launch_crew` — every dangerous tool
  going through the **same approval gate** as the crew.

> **⚠ The research feature is NOT verified.** Unlike the coder crew, research has
> **no verification gate** — its synthesis is never checked for correctness, and it
> can confidently state wrong things or cite weak/irrelevant sources with nothing to
> catch it. The coder crew's gate proves *tests ran*; it says nothing about whether
> a research claim is *true*. **Treat research output as an aid and verify the
> claims and sources yourself.**

## Why it's different

- **Acceptance-gated, not vibes.** Every subtask carries a runnable check
  (`pytest …`). A subtask only counts as **passed** when a *real* gate goes green.
  Checks that just prove code "imports/runs" are rejected — they verify nothing.
- **Honest accounting.** Manual-review / unverifiable subtasks are tracked as
  `unverified`, never silently counted as passed.
- **Completeness check.** A subtask that shipped a deliverable (e.g. a module)
  with no test that actually ran is flagged `unverified` — and a local retry first
  tries to write the missing test before giving up honestly.
- **Spec-coverage review** *(Claude/Opus manager; default ON only there)*. After a
  subtask is green, the manager critiques its tests against the spec; missing spec
  cases trigger a local retry that adds them — and if an added test then fails, the
  subtask becomes `failed` and **names the real bug** (an incomplete-test bug caught
  instead of shipped). Best-effort, conservative, degrades to a no-op with a weak/
  local manager; never fabricates a failure or fakes a pass.
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
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows
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

Pick the config honestly for what you need:

- **All-local** (e.g. `qwen3-coder:30b` for both) — free, private, no API. You get
  gating + the completeness check, but the **weakest verification**: no spec-coverage
  review (it's a no-op for a local manager), so test *completeness* is on you.
- **Opus manager + local worker** *(recommended)* — Opus plans/reviews and runs the
  **spec-coverage review** (catches many incomplete-test gaps); the local model does
  the building (cheap). Best verification per dollar.
- **Opus for both** — strongest, priciest.

`coverage_review` defaults **ON only when the manager is a Claude spec** (a local
manager's critique is too weak to be worth the latency). On a single GPU, workers
run sequentially; the win is decomposition + verification quality, not throughput.

### Connecting Claude (optional)

The **Connection** panel in the UI controls the optional Claude path:

- **Off** — local-only; Claude isn't used or offered.
- **Claude Code** — uses your logged-in `claude` CLI (Claude Code subscription).
- **API key** — set the **`CLAUDE_API_KEY`** environment variable (preferred), or
  paste a key in the UI. A pasted key is saved to `config.json` **in plaintext**
  (git-ignored — **never commit or share it**). Either source is exported as
  `ANTHROPIC_API_KEY` for the SDK/CLI; the env var wins if both are set.

Hit **🛰 Test connection** to do a real one-word round-trip and confirm it actually
works before you rely on it. Either way you need `pip install claude-agent-sdk`.

## Security

This is a **local, single-user** tool. Workers can run shell commands, so the
server binds to loopback and has no auth. Don't expose it on a non-loopback host
without a tunnel + authentication in front of it. Acceptance-check commands are
restricted to a single test-runner invocation (no shell chaining) by an allowlist —
that allowlist, not a sandbox, is what keeps a model-authored check from being an
arbitrary command.

The optional unattended auto-approve mode runs file writes and shell commands
**without review** — using it trades the approval gate for autonomy, so only point
it at work (and a folder) you trust.

**API keys:** prefer the `CLAUDE_API_KEY` environment variable. A key entered in the
UI is stored **in plaintext** in `config.json` (git-ignored) — never commit or share
that file. "Claude Code" mode and "Off" store no key.

## License

MIT — see [LICENSE](LICENSE).
