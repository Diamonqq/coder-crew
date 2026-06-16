"""Local 'Coder Crew' orchestration.

A manager model decomposes a goal into subtasks, worker models implement each
one (with tools + user-approved shell), and the manager reviews/integrates the
results into a final answer.

  manager (plan) ──► worker · worker · worker ──► manager (review + synthesize)

Roles are pluggable agent specs (see crewkit.agents.make_agent):
  ollama:<model>            e.g. ollama:qwen3-coder:30b
  claude:<model>            e.g. claude:claude-opus-4-8  (real Claude Code session)

Design notes
------------
* Single GPU => workers run **sequentially**. Parallel workers on one card don't
  speed up (shared compute + VRAM swap thrash); the win is decomposition+review
  quality, not throughput. (For a Claude worker, "sequential" is moot.)
* A run executes in a background thread. The HTTP layer polls `to_dict()` and
  posts approvals. When a worker calls a danger tool, the run goes `blocked` and
  the worker thread waits on an Event until the user approves/denies.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from . import agents, gate
from . import crew_db
from . import tools as toolmod

log = logging.getLogger("crew")

_MAX_EVENTS = 400
_APPROVAL_TIMEOUT = 900  # seconds a worker waits for the user before auto-deny
# Total attempts per subtask, INCLUDING the first (so MAX_REPAIR=3 means 1 initial
# try + up to 2 repairs). Each attempt is a fresh worker invocation with its own
# max_steps budget, so a hard subtask can multiply wall-clock by up to MAX_REPAIR.
MAX_REPAIR = 3
# Per-attempt worker step budget, scaled by the run's estimated complexity. Trims
# the flailing tail on easy goals; hard goals keep the full 14. (MAX_REPAIR is NOT
# scaled — the repair loop's quality is kept; only the within-attempt budget moves.)
_STEP_BUDGET = {"simple": 6, "medium": 10, "hard": 14}

# Where auto-created run folders live (when the user doesn't specify one).
_WORKSPACE = Path(__file__).resolve().parent.parent / "crew-workspace"


def _auto_workspace(goal: str) -> str:
    """Make a fresh, named folder for a run so workers always have somewhere to
    write — no need for the user to pick a path."""
    slug = re.sub(r"[^a-z0-9]+", "-", goal.lower()).strip("-")[:32] or "run"
    stamp = datetime.now().strftime("%m%d-%H%M%S")
    path = _WORKSPACE / f"{slug}-{stamp}"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


@dataclass
class Worker:
    id: int
    title: str
    detail: str
    acceptance: object = None   # runnable check: shell str | {"type":"pytest","code"} | None
    check_rationale: str = ""   # manager's one-line justification for the check
    owns: list = field(default_factory=list)  # files this subtask may write (empty => unscoped)
    status: str = "queued"      # queued | running | done | unverified | error | failed
    output: str = ""
    attempts: int = 0           # how many times the worker has been invoked
    gate_passed: object = None  # work-phase gate: True | False | None (not yet / manual)
    gate_output: str = ""       # work-phase acceptance-check output
    review_gate_passed: object = None  # review-time re-run: True | False | None (manual)
    review_gate_output: str = ""       # review-time acceptance-check output
    started_at: float | None = None    # wall-clock for the subtask (history/routing data)
    ended_at: float | None = None
    ran_on: str = "local"              # local | opus — where the (final) attempt ran
    escalated: bool = False            # True if re-dispatched to Opus after local
    escalation_reason: str = ""        # failed | unverified | error | predict-concurrency


@dataclass
class CrewRun:
    id: str
    goal: str
    manager_spec: str
    worker_spec: str
    max_workers: int
    cwd: str | None
    status: str = "planning"    # planning | working | reviewing | done | blocked | error | cancelled
    complexity: str = "medium"  # simple|medium|hard — scales the worker step budget
    tag: str = ""               # expected routing bucket (overnight harness); '' interactive
    # Tool restrictions (None => normal defaults; set by the unattended harness for
    # a structural no-network posture). Do NOT set these for interactive runs.
    worker_tools: object = None     # None => all tools except launch_crew
    manager_tools: object = None    # None => _READONLY
    worker_use_mcp: bool = True     # harness sets False (no MCP network sinks)
    # Auto-router: escalate a locally-failed/unverified subtask to Opus. OFF by
    # default (no surprise Opus spend); opt-in per run / via the UI.
    allow_escalation: bool = False
    escalation_spec: str = "claude:claude-opus-4-8"
    plan: list = field(default_factory=list)
    contract: object = None     # shared interface contract from the plan (or None)
    workers: list = field(default_factory=list)
    events: list = field(default_factory=list)
    final: str = ""
    error: str = ""
    created: float = field(default_factory=time.time)
    ended: float | None = None

    # --- approval / control plumbing (not serialized) ---
    pending: dict | None = None          # {tool, args, worker_id}
    _gate: threading.Event = field(default_factory=threading.Event, repr=False)
    _decision: tuple | None = None       # (approved: bool, note: str)
    _cancel: bool = False

    def emit(self, ev: dict) -> None:
        ev["t"] = time.time()
        self.events.append(ev)
        if len(self.events) > _MAX_EVENTS:
            del self.events[: len(self.events) - _MAX_EVENTS]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "goal": self.goal,
            "manager": self.manager_spec,
            "worker_model": self.worker_spec,
            "cwd": self.cwd,
            "status": self.status,
            "plan": self.plan,
            "contract": self.contract,
            "workers": [vars(w) for w in self.workers],
            "events": self.events[-120:],
            "final": self.final,
            "error": self.error,
            "pending": self.pending,
            "created": self.created,
            "ended": self.ended,
            "elapsed": round((self.ended or time.time()) - self.created, 1),
        }


# --- prompts ----------------------------------------------------------------
_PLAN_SYS = (
    "You are the MANAGER of a small team of coder agents. Plan the user's goal in "
    "TWO parts. You may use read-only tools (list_dir, read_file, web_search) first.\n\n"
    "PART 1 — a shared CONTRACT every worker must conform to, so they don't make "
    "incompatible assumptions or clobber each other:\n"
    "  • files: each filename that will exist and its purpose;\n"
    "  • signatures: the EXACT public function signatures / data schemas that cross "
    "between subtasks (name, params, return type);\n"
    "  • notes: shared constants or conventions.\n\n"
    "PART 2 — a short ordered list of SUBTASKS (1-{maxw}), each independent. Each MUST have:\n"
    "  • title, detail (conforming to the contract);\n"
    "  • owns: the list of file paths THIS subtask may create/edit. Ownership MUST "
    "be DISJOINT — no two subtasks may own the same file (a worker can ONLY write "
    "files it owns; the tools refuse otherwise). A subtask's own test file goes in "
    "its owns list.\n"
    "  • acceptance: a RUNNABLE check (never prose). It MUST be a SINGLE test-runner "
    "command — \"pytest <file> -q\", \"python -m pytest ...\", or \"python -m unittest "
    "...\" — with NO shell chaining/redirection (no ; | & > < ` $( ) ). The test FILE "
    "it runs MUST contain at least one real ASSERTION on behavior (assert / == on an "
    "expected value). PREFER creating a test file + a one-line \"pytest <file> -q\"; a "
    "{{\"type\":\"pytest\",\"code\":\"...\"}} object is allowed only if a separate file "
    "makes no sense.\n"
    "  • check_rationale: ONE sentence on why the check is sufficient.\n\n"
    "FORBIDDEN — a `python -c \"...; print('Success')\"` smoke test, an import-only "
    "check, or anything that just proves the code IMPORTS/RUNS without crashing. That "
    "verifies NOTHING, is NOT acceptable, and will be rejected (the subtask then counts "
    "as UNVERIFIED, not passed). The check must assert that behavior is CORRECT.\n"
    "Make checks DEMANDING — exercise EDGE CASES (empty, boundary, malformed, unicode, "
    "zero/negative), assert on ACTUAL OUTPUT VALUES, and cover error paths.\n"
    "For STATEFUL / object / decorator / concurrency code this is where checks usually "
    "go wrong: do NOT just construct the object. Construct it, perform operations, and "
    "ASSERT the resulting state or return. Example (state machine): a pytest file with "
    "`m = FSM(states, transitions, 'green'); m.fire('go'); assert m.state == 'yellow'` "
    "and a case asserting an invalid transition raises — run via "
    "\"pytest tests/test_fsm.py -q\".\n\n"
    "Reply with ONLY a JSON object, no prose:\n"
    '{{"contract": {{"files": {{"util.py": "slugify()", "tests/test_slugify.py": '
    '"slugify tests"}}, "signatures": ["slugify(text: str) -> str"], "notes": "no '
    'leading/trailing hyphen"}}, "subtasks": [{{"title": "Implement slugify with tests", '
    '"detail": "Add slugify(text) to util.py (lowercase, strip, collapse non-alphanumeric '
    'runs to single hyphens, no leading/trailing hyphen); create tests/test_slugify.py.", '
    '"owns": ["util.py", "tests/test_slugify.py"], '
    '"acceptance": "pytest tests/test_slugify.py -q", '
    '"check_rationale": "Covers normal text, consecutive separators, unicode, empty '
    'string, and all-punctuation collapsing to empty."}}]}}'
)
_WORKER_SYS = (
    "You are a senior coder WORKER. You complete the subtask by ACTUALLY CALLING "
    "TOOLS — never just describe or paste code in prose. If a file should exist, "
    "you MUST create it with the write_file tool. To run or test something, you "
    "MUST use the run_shell tool. Work in small steps: write a file, run it, read "
    "the output, fix if needed. Do NOT claim something works unless you ran it and "
    "saw the output. ONLY after the work is actually done and verified, reply with "
    "a short plain-text report of what you did and the file paths involved."
)
_REVIEW_SYS = (
    "You are the MANAGER doing a VERIFYING review. You are given each subtask's "
    "VERIFIED status — its acceptance gate was just RE-RUN against the real files, "
    "so trust that over any worker's prose claims. Use your read-only tools "
    "(read_file, list_dir) to inspect the ACTUAL files in the working folder and "
    "confirm what was built. Then write ONE final answer for the user: what was "
    "built, where it lives, and how to use it.\n"
    "You MUST explicitly report EVERY problem in the PROBLEMS list — red gates, "
    "failed subtasks, and crashed/errored subtasks — even if a worker claimed "
    "success. Never smooth over or omit a failure. If everything passed, say so "
    "plainly."
)


_REPAIR_DIRECTIVE = (
    "Your previous attempt did NOT pass the acceptance check. The check output is "
    "above — it tells you exactly why it failed. Edit the files in your working "
    "folder so the check passes, then verify. Do not explain in prose — make the "
    "change and re-run."
)


def _repair_task(w: "Worker", prev_output: str) -> str:
    """Augmented task for a repair attempt: original detail + what the worker last
    reported + the failing gate output (the critical signal) + a fix directive."""
    return (
        f"Subtask: {w.title}\n\n{w.detail}\n\n"
        f"--- YOUR PREVIOUS ATTEMPT REPORTED ---\n{(prev_output or '(nothing)')[:2000]}\n\n"
        f"--- ACCEPTANCE CHECK FAILED — its output ---\n{(w.gate_output or '')[:2500]}\n\n"
        f"{_REPAIR_DIRECTIVE}"
    )


# Acceptance shell commands come straight from the manager MODEL and are executed
# UNATTENDED by the gate (no approval) through tools.shell_exec, which is NOT
# sandboxed. So we ALLOWLIST them to a single test-runner invocation: any shell
# metacharacter that could chain/redirect/subshell is rejected, and the command
# must start with a recognized test runner. A rejected command becomes None
# (manual review) — we never "fix" it. The {"type":"pytest","code":...} form is
# safe (we write it to a temp file and run pytest ourselves — no model shell string).
_ACCEPT_META = set(";|&`><\n\r")
_ACCEPT_RUNNER_RE = re.compile(
    r'^\s*("?[^"\s]*python[0-9.]*(?:\.exe)?"?\s+-m\s+(?:pytest|unittest)\b|pytest\b)',
    re.IGNORECASE)


def _is_allowed_acceptance_cmd(cmd: str) -> bool:
    """True iff `cmd` is a SINGLE test-runner invocation with no shell chaining."""
    s = (cmd or "").strip()
    if not s or any(c in s for c in _ACCEPT_META) or "$(" in s:
        return False
    return bool(_ACCEPT_RUNNER_RE.match(s))


# When we escalate an UNVERIFIED subtask, Opus is asked to author a real gate and
# state its run command on a line "ACCEPTANCE_CMD: <cmd>". Parse the LAST such line.
_ACCEPT_CMD_RE = re.compile(r'ACCEPTANCE_CMD:\s*(.+?)\s*$', re.IGNORECASE | re.MULTILINE)


def _extract_acceptance_cmd(text: str) -> str | None:
    matches = _ACCEPT_CMD_RE.findall(text or "")
    if not matches:
        return None
    cmd = matches[-1].strip().strip("`").strip().strip('"').strip()
    return cmd or None


def _rejected_cmd(raw) -> str | None:
    """If `raw` was a non-empty shell command that FAILS the allowlist (so it gets
    dropped to manual review), return it for observability; else None."""
    if isinstance(raw, str):
        c = raw.strip()
    elif isinstance(raw, dict) and raw.get("type") != "pytest":
        c = str(raw.get("cmd") or raw.get("command") or raw.get("shell") or "").strip()
    else:
        c = ""
    return c if (c and not _is_allowed_acceptance_cmd(c)) else None


def _clean_acceptance(acc):
    """Normalize a subtask's acceptance check. Returns an ALLOWLISTED test-runner
    shell command string, a {"type":"pytest","code":...} dict, or None (manual
    review). Anything missing/malformed/disallowed becomes None — we never invent
    or repair a check."""
    if isinstance(acc, str):
        cmd = acc.strip()
        return cmd if (cmd and _is_allowed_acceptance_cmd(cmd)) else None
    if isinstance(acc, dict):
        if acc.get("type") == "pytest" and str(acc.get("code", "")).strip():
            return {"type": "pytest", "code": str(acc["code"])}
        cmd = acc.get("command") or acc.get("shell")  # tolerate {"type":"shell",...}
        if isinstance(cmd, str) and cmd.strip() and _is_allowed_acceptance_cmd(cmd.strip()):
            return cmd.strip()
    return None


def _loads_lenient(s: str):
    """json.loads, but tolerant of the #1 local-model mistake: raw newlines/tabs
    inside string values (e.g. a multi-line pytest `code` field), which is invalid
    JSON. On failure, escape control chars that sit *inside* quoted strings and
    retry. Returns the parsed value or None."""
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    out, instr, esc = [], False, False
    for ch in s:
        if esc:
            out.append(ch); esc = False; continue
        if ch == "\\":
            out.append(ch); esc = True; continue
        if ch == '"':
            instr = not instr; out.append(ch); continue
        if instr and ch in "\n\r\t":
            out.append({"\n": "\\n", "\r": "\\r", "\t": "\\t"}[ch]); continue
        out.append(ch)
    try:
        result = json.loads("".join(out))
        log.warning("WARN plan recovered via lenient parse "
                    "(model emitted invalid JSON — likely raw newlines in a string)")
        return result
    except json.JSONDecodeError:
        return None


def _clean_owns(owns) -> list:
    """Normalize a subtask's owned-file list to a list of non-empty path strings."""
    if isinstance(owns, str):
        owns = [owns]
    if not isinstance(owns, list):
        return []
    return [str(p).strip() for p in owns if str(p).strip()]


def _parse_plan(text: str, goal: str, maxw: int) -> "tuple[list, object]":
    """Parse the manager's plan into (subtasks, contract). Accepts BOTH the new
    object form {"contract": {...}, "subtasks": [...]} and the legacy bare array
    [...] (contract=None). Each subtask carries acceptance, check_rationale, and
    `owns`. Falls back to a single un-gated, unscoped subtask if parsing fails."""
    contract = None
    arr = None
    obj = re.search(r"\{.*\}", text, re.S)
    bracket = re.search(r"\[.*\]", text, re.S)
    # Prefer an object that actually has a subtasks array; else a bare array.
    if obj:
        parsed = _loads_lenient(obj.group(0))
        if isinstance(parsed, dict) and isinstance(parsed.get("subtasks"), list):
            contract = parsed.get("contract")
            arr = parsed["subtasks"]
    if arr is None and bracket:
        parsed = _loads_lenient(bracket.group(0))
        if isinstance(parsed, list):
            arr = parsed
    if isinstance(arr, list):
        out = []
        for item in arr[:maxw]:
            if isinstance(item, dict) and item.get("title"):
                out.append({"title": str(item["title"]),
                            "detail": str(item.get("detail", "")),
                            "acceptance": _clean_acceptance(item.get("acceptance")),
                            "acceptance_rejected": _rejected_cmd(item.get("acceptance")),
                            "check_rationale": str(item.get("check_rationale", "")),
                            "owns": _clean_owns(item.get("owns"))})
            elif isinstance(item, str):
                out.append({"title": item, "detail": "", "acceptance": None,
                            "acceptance_rejected": None,
                            "check_rationale": "", "owns": []})
        if out:
            return out, contract
    return ([{"title": "Complete the goal", "detail": goal, "acceptance": None,
              "acceptance_rejected": None, "check_rationale": "", "owns": []}], None)


def _weak_reason(acc) -> str | None:
    """Heuristic flag for a trivially weak acceptance check (observability only —
    never blocks). Returns a reason string or None. We can't inspect a `pytest
    <file>` command's asserts (the file isn't written at plan time), so those
    aren't flagged; we only catch import-only shell checks and thin pytest snippets."""
    if acc is None:
        return None
    if isinstance(acc, dict) and acc.get("type") == "pytest":
        n = str(acc.get("code", "")).count("assert")
        if n < 2:
            return f"embedded pytest has {n} assert(s) — too few to cover edge cases"
        return None
    if isinstance(acc, str):
        s = acc.lower()
        # A `python -c` one-liner that neither asserts nor sets a failing exit code
        # (e.g. it only imports, or only prints results) passes even when the output
        # is wrong. Don't flag exit-code-based checks (sys.exit / exit()).
        if ("python" in s and " -c" in s and "assert" not in s and "==" not in s
                and "sys.exit" not in s and "exit(" not in s):
            return ("shell check runs code but makes no assertion (no assert / == / "
                    "exit-code) — it passes even if the output is wrong")
    return None


_READONLY = ["list_dir", "read_file", "web_search", "web_fetch", "get_time"]


# --- Autopilot / advisor ----------------------------------------------------
_Q3 = "huihui_ai/qwen3-coder-abliterated:30b"   # primary: agentic MoE, 3.3B active
_CODER = "huihui_ai/qwen2.5-coder-abliterate"   # dense fallbacks
# Rough perf/quality facts for an RX 7900 XTX (24GB). tps = generated tok/s.
_MODEL_INFO = {
    _Q3: {"label": "Qwen3-Coder 30B", "tps": 60, "quality": 5, "vram": 19.0},
    f"{_CODER}:7b":  {"label": "2.5 7B",  "tps": 75, "quality": 3, "vram": 4.7},
    f"{_CODER}:14b": {"label": "2.5 14B", "tps": 45, "quality": 4, "vram": 9.0},
    f"{_CODER}:32b": {"label": "2.5 32B", "tps": 20, "quality": 5, "vram": 19.5},
    "gemma4vision:latest": {"label": "gemma 25B", "tps": 22, "quality": 4, "vram": 16},
}
_QUALITY_WORD = {3: "good", 4: "very good", 5: "excellent"}
_ADVISE_SYS = (
    "You are a planning assistant for a local coder-agent crew. Given the user's "
    "rough idea, write a clear, detailed, actionable GOAL the crew can build, and "
    "judge its size. Reply with ONLY a JSON object:\n"
    '{"goal": "<a precise, self-contained spec — what to build, key requirements, '
    'how to verify>", "complexity": "simple|medium|hard", "suggested_tasks": <1-6>, '
    '"summary": "<one short line>"}'
)


def _installed_models() -> set:
    try:
        from . import chat
        return {m["name"] for m in chat.models()}
    except Exception:  # noqa: BLE001
        return set()


def _ready(model: str, installed: set) -> bool:
    """Is a model spec available among the installed Ollama models? Rule:
      • exact name match (the common case — specs are fully tagged), OR
      • if the spec is given WITHOUT a tag, any installed model sharing that base
        name (before the ':') counts.
    We do NOT treat different tags of the same base as interchangeable — a ':7b'
    must never satisfy a ':30b' request (different model)."""
    if model in installed:
        return True
    if ":" not in model:  # untagged spec => any tag of this base is acceptable
        return any(n.split(":")[0] == model for n in installed)
    return False


def _estimate(mgr: str, wkr: str, complexity: str, tasks: int) -> dict:
    mi, wi = _MODEL_INFO[mgr], _MODEL_INFO[wkr]
    # crude token budgets per task by complexity
    per_task = {"simple": 1200, "medium": 3500, "hard": 8000}.get(complexity, 3500)
    worker_tokens = per_task * tasks
    mgr_tokens = 800 + 500 * tasks            # plan + review
    swap = 0
    # Same model for both roles => only one copy is ever loaded (no swap). Two
    # different models that can't co-reside in 24GB => swap on each handoff.
    if mgr != wkr and mi["vram"] + wi["vram"] > 22:
        swap = 5 * (tasks + 2)
    secs = worker_tokens / wi["tps"] + mgr_tokens / mi["tps"] + swap + 4
    # quality: manager weights planning/review, worker weights the actual code
    q = round(0.45 * mi["quality"] + 0.55 * wi["quality"], 1)
    return {
        "seconds": int(secs),
        "time": f"~{int(secs)}s" if secs < 90 else f"~{round(secs/60,1)}min",
        "quality": q,
        "quality_word": _QUALITY_WORD.get(round(q), "good"),
        "resident": swap == 0,
    }


def _claude_available() -> bool:
    try:
        import claude_agent_sdk  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def build_combos(installed: set, complexity: str, tasks: int) -> list:
    """Candidate manager×worker combos with estimated time + quality for THIS
    task size. Local combos are filtered to installed models; Claude combos are
    offered (when the SDK is present) for premium manager/worker mixes."""
    # (manager, worker, name, note) — Qwen3-Coder first as the default pick.
    candidates = [
        (_Q3, _Q3, "Qwen3-Coder (recommended)", "one fast agentic MoE for both roles — no swap"),
        (_Q3, f"{_CODER}:7b", "Qwen3 mgr + 2.5 7B wkr", "big agentic brain, tiny fast workers"),
        (f"{_CODER}:14b", f"{_CODER}:7b", "2.5 Fast", "dense 14B + 7B, both resident"),
        (f"{_CODER}:32b", f"{_CODER}:7b", "2.5 Max", "32B brain, swaps workers"),
        (f"{_CODER}:7b", f"{_CODER}:7b", "Lightest", "smallest, quickest, lowest quality"),
    ]
    combos = []
    for mgr, wkr, name, note in candidates:
        if mgr not in _MODEL_INFO or wkr not in _MODEL_INFO:
            continue
        if not (_ready(mgr, installed) and _ready(wkr, installed)):
            continue
        est = _estimate(mgr, wkr, complexity, tasks)
        combos.append({"name": name, "note": note, "manager": f"ollama:{mgr}",
                       "worker": f"ollama:{wkr}",
                       "manager_label": _MODEL_INFO[mgr]["label"],
                       "worker_label": _MODEL_INFO[wkr]["label"], **est})
    # Premium: Claude manager steering a local worker (your Opus+Qwen idea).
    if _claude_available():
        for mdl, lbl in (("claude-opus-4-8", "Opus 4.8"), ("sonnet", "Sonnet")):
            wkr = _Q3 if _ready(_Q3, installed) else f"{_CODER}:7b"
            if wkr not in _MODEL_INFO or not _ready(wkr, installed):
                continue
            combos.append({
                "name": f"{lbl} + Qwen3", "note": "Claude plans & reviews, local model builds — uses credits",
                "manager": f"claude:{mdl}", "worker": f"ollama:{wkr}",
                "manager_label": lbl, "worker_label": _MODEL_INFO[wkr]["label"],
                "seconds": 0, "time": "varies", "quality": 5.0,
                "quality_word": "excellent", "resident": True, "premium": True})
    return combos


def advise(idea: str, advisor_spec: str | None = None) -> dict:
    installed = _installed_models()
    # Compose the goal with the best available local model (gemma > 14B > 7B).
    if not advisor_spec:
        for m in ("gemma4vision:latest", f"{_CODER}:14b", f"{_CODER}:7b"):
            if _ready(m, installed):
                advisor_spec = m
                break
    parsed = {"goal": idea, "complexity": "medium", "suggested_tasks": 3,
              "summary": idea[:80]}
    if advisor_spec:
        try:
            from . import chat
            model = advisor_spec.split("ollama:")[-1]
            res = chat.chat(model, [{"role": "system", "content": _ADVISE_SYS},
                                    {"role": "user", "content": idea}], timeout=180)
            m = re.search(r"\{.*\}", res.get("reply", ""), re.S)
            if m:
                got = json.loads(m.group(0))
                parsed.update({k: got[k] for k in
                               ("goal", "complexity", "suggested_tasks", "summary")
                               if k in got})
        except Exception:  # noqa: BLE001 — fall back to the raw idea as the goal
            pass
    complexity = parsed.get("complexity", "medium")
    tasks = max(1, min(int(parsed.get("suggested_tasks", 3) or 3), 6))
    combos = build_combos(installed, complexity, tasks)
    local = [c for c in combos if not c.get("premium")]
    recommended = None
    # Prefer Claude Opus as manager + local worker when the SDK is available — Opus
    # is the strongest planner/reviewer ("the ideas part"); the local model builds.
    opus = next((c for c in combos if c["manager"] == "claude:claude-opus-4-8"), None)
    # Otherwise the Qwen3-Coder all-roles combo — best free local pick (fast MoE).
    q3 = next((c for c in local if c["manager"] == f"ollama:{_Q3}"
               and c["worker"] == f"ollama:{_Q3}"), None)
    if opus:
        recommended = opus
    elif q3:
        recommended = q3
    elif local:
        if complexity == "hard":
            recommended = max(local, key=lambda c: c["quality"])
        elif complexity == "simple":
            recommended = min(local, key=lambda c: c["seconds"])
        else:
            resident = [c for c in local if c["resident"]] or local
            recommended = max(resident, key=lambda c: c["quality"])
    return {
        "goal": parsed.get("goal", idea),
        "summary": parsed.get("summary", ""),
        "complexity": complexity,
        "suggested_tasks": tasks,
        "advisor": advisor_spec,
        "combos": combos,
        "recommended": recommended,
    }


# Predictive router signal: TRUE threading/shared-state concurrency, which the
# overnight data showed local both times-out on AND can't write a valid gate for.
# Deliberately CONSERVATIVE — it does NOT match plain asyncio ("async",
# "concurrency", "semaphore" alone), which local handled fine (~4/5). Err toward
# letting local try; reactive escalation catches the misses.
_CONCURRENCY_RE = re.compile(
    r"\b(thread|threads|threading|thread-?safe|threadsafe|mutex|deadlock|"
    r"race condition|producer.?consumer)\b", re.IGNORECASE)


def _is_concurrency(text: str) -> bool:
    t = text or ""
    if _CONCURRENCY_RE.search(t):
        return True
    # blocking-queue semantics (e.g. "blocking put/get") => threading, not asyncio
    return bool(re.search(r"\bblocking\b", t, re.I) and re.search(r"\b(queue|put|get)\b", t, re.I))


def _subtask_gate_outcome(w: "Worker", rejected: bool) -> str:
    if rejected:
        return "rejected"
    if w.acceptance is None:
        return "manual"
    if w.gate_passed is True:
        return "passed"
    if w.gate_passed is False:
        return "failed"
    return "none"


def _log_run(run: CrewRun) -> None:
    """Persist a finished run's routing features to crew_db. Reads only data
    already on the run/Worker objects — no behavior change."""
    workers = run.workers or []
    subs = []
    for i, w in enumerate(workers):
        rejected = bool(run.plan[i].get("acceptance_rejected")) if i < len(run.plan) else False
        elapsed = (round(w.ended_at - w.started_at, 1)
                   if (w.started_at and w.ended_at) else None)
        subs.append({
            "idx": i,
            "title": w.title,
            "status": w.status,
            "attempts": w.attempts,
            "gate_outcome": _subtask_gate_outcome(w, rejected),
            "weak_flagged": _weak_reason(w.acceptance) is not None,
            "rejected": rejected,
            "regression": bool(w.gate_passed is True and w.review_gate_passed is False),
            "elapsed": elapsed,
            "ran_on": w.ran_on,
            "escalated": w.escalated,
            "escalation_reason": w.escalation_reason,
        })
    run_row = {
        "id": run.id, "goal": run.goal, "complexity": run.complexity,
        "tag": run.tag, "manager_spec": run.manager_spec,
        "worker_spec": run.worker_spec, "status": run.status,
        "created": run.created, "ended": run.ended,
        "elapsed": round((run.ended or time.time()) - run.created, 1),
        "n_subtasks": len(workers),
        # Honest accounting: only a REAL green gate counts as passed; manual-review
        # (None/rejected gate) is its own "unverified" bucket, in NEITHER passed nor
        # failed. passed + failed(incl error) + unverified == n_subtasks.
        "n_passed": sum(1 for w in workers if w.status == "done"),
        "n_failed": sum(1 for w in workers if w.status in ("failed", "error")),
        "n_unverified": sum(1 for w in workers if w.status == "unverified"),
    }
    crew_db.DB.log_run(run_row, subs)


class _CrewManager:
    def __init__(self):
        self.runs: dict[str, CrewRun] = {}
        self._lock = threading.Lock()

    def start(self, goal: str, *, manager_spec: str, worker_spec: str,
              max_workers: int = 3, cwd: str | None = None,
              complexity: str = "medium", tag: str = "",
              worker_tools=None, manager_tools=None, worker_use_mcp: bool = True,
              allow_escalation: bool = False,
              escalation_spec: str = "claude:claude-opus-4-8") -> CrewRun:
        # No folder given => make a fresh one so workers always have a home.
        if not cwd:
            cwd = _auto_workspace(goal)
        run = CrewRun(id=uuid.uuid4().hex[:12], goal=goal,
                      manager_spec=manager_spec, worker_spec=worker_spec,
                      max_workers=max(1, min(max_workers, 6)), cwd=cwd,
                      complexity=complexity if complexity in _STEP_BUDGET else "medium",
                      tag=tag, worker_tools=worker_tools, manager_tools=manager_tools,
                      worker_use_mcp=worker_use_mcp, allow_escalation=allow_escalation,
                      escalation_spec=escalation_spec)
        with self._lock:
            self.runs[run.id] = run
        threading.Thread(target=self._drive, args=(run,), daemon=True).start()
        return run

    def get(self, run_id: str) -> CrewRun | None:
        return self.runs.get(run_id)

    def list(self) -> list:
        return [r.to_dict() for r in sorted(self.runs.values(),
                                            key=lambda r: r.created, reverse=True)]

    def approve(self, run_id: str, approved: bool, note: str = "") -> bool:
        run = self.runs.get(run_id)
        if not run or not run.pending:
            return False
        run._decision = (approved, note)
        run._gate.set()
        return True

    def cancel(self, run_id: str) -> bool:
        run = self.runs.get(run_id)
        if not run:
            return False
        run._cancel = True
        run._gate.set()  # release any pending wait
        return True

    # -- the run, on a worker thread --
    def _drive(self, run: CrewRun) -> None:
        try:
            self._plan(run)
            if run._cancel:
                return self._finish(run, "cancelled")
            self._work(run)
            if run._cancel:
                return self._finish(run, "cancelled")
            self._review(run)
            self._finish(run, "done")
        except Exception as exc:  # noqa: BLE001
            run.error = f"{type(exc).__name__}: {exc}"
            run.emit({"type": "error", "text": run.error})
            self._finish(run, "error")

    def _approver(self, run: CrewRun, worker_id: int):
        def approve(name: str, args: dict):
            run.pending = {"tool": name, "args": args, "worker_id": worker_id}
            run.status = "blocked"
            run._gate.clear()
            run._decision = None
            run.emit({"type": "approval_needed", "tool": name, "args": args,
                      "worker_id": worker_id})
            ok = run._gate.wait(timeout=_APPROVAL_TIMEOUT)
            run.pending = None
            run.status = "working"
            if run._cancel:
                return False, "run cancelled"
            if not ok or run._decision is None:
                return False, "approval timed out"
            return run._decision
        return approve

    def _plan(self, run: CrewRun) -> None:
        run.status = "planning"
        run.emit({"type": "phase", "text": "manager planning"})
        mgr = agents.make_agent(run.manager_spec,
                                tool_names=run.manager_tools or _READONLY,
                                max_steps=8, cwd=run.cwd)
        sys_prompt = _PLAN_SYS.format(maxw=run.max_workers)
        reply = mgr.run_task(run.goal, system=sys_prompt,
                             on_event=lambda e: run.emit({**e, "role": "manager"}))
        run.plan, run.contract = _parse_plan(reply, run.goal, run.max_workers)
        run.workers = [Worker(id=i, title=p["title"], detail=p["detail"],
                              acceptance=p.get("acceptance"),
                              check_rationale=p.get("check_rationale", ""),
                              owns=p.get("owns", []))
                       for i, p in enumerate(run.plan)]
        run.emit({"type": "plan", "plan": run.plan, "contract": run.contract})
        # Observability only (no blocking): surface (a) acceptance commands the
        # allowlist DROPPED to manual review (a non-test-runner command the model
        # tried to gate with), and (b) subtasks with thin gates.
        for w, p in zip(run.workers, run.plan):
            rejected = p.get("acceptance_rejected")
            if rejected:
                log.warning("WARN rejected acceptance on subtask %d (%s): not an "
                            "allowlisted test-runner command, dropped to manual "
                            "review: %s", w.id, w.title, rejected[:200])
                run.emit({"type": "rejected_acceptance", "worker_id": w.id,
                          "title": w.title, "command": rejected[:200]})
            weak = _weak_reason(w.acceptance)
            if weak:
                log.warning("WARN weak acceptance check on subtask %d (%s): %s",
                            w.id, w.title, weak)
                run.emit({"type": "weak_check", "worker_id": w.id,
                          "title": w.title, "reason": weak})

    def _work(self, run: CrewRun) -> None:
        run.status = "working"
        context = f"Overall goal:\n{run.goal}"
        # Inject the shared contract so each worker conforms to the agreed
        # filenames/signatures (the cheap structured alternative to forwarding
        # prose between workers). Skipped entirely when there's no contract.
        if run.contract:
            context += ("\n\nSHARED CONTRACT — conform to these agreed interfaces "
                        "(filenames, signatures, conventions):\n"
                        + json.dumps(run.contract, indent=1)[:2000])
        for w in run.workers:
            if run._cancel:
                return
            self._work_subtask(run, w, context)

    def _work_subtask(self, run: CrewRun, w: "Worker", context: str) -> None:
        """Run one subtask, gate it, and re-dispatch on failure up to MAX_REPAIR
        total attempts. Un-gated subtasks (acceptance=None) run exactly once — the
        gate returns pass with "manual review" and there's nothing to converge on."""
        prev_output = ""
        w.started_at = time.time()
        # PREDICTIVE OVERRIDE (thin): true threading/shared-state concurrency
        # structurally fails local (timeouts + can't self-gate), so route it to
        # Opus UPFRONT — but only when escalation is enabled (no surprise spend).
        # On Opus-unavailable or Opus-error we fall through to local. Predict-route
        # ran on Opus from the start, so escalated stays False (not a local->Opus).
        if run.allow_escalation and _is_concurrency(f"{w.title}\n{w.detail}"):
            if _claude_available():
                log.warning("predict-routed subtask %d to Opus: concurrency signal", w.id)
                run.emit({"type": "escalation", "worker_id": w.id,
                          "reason": "predict-concurrency", "status": "predict-routed",
                          "to": run.escalation_spec})
                if self._dispatch_opus(run, w, context, "predict-concurrency"):
                    run.emit({"type": "escalation", "worker_id": w.id,
                              "reason": "predict-concurrency", "status": "done",
                              "outcome": w.status})
                    return
                log.warning("predict-route of subtask %d failed; running local", w.id)
            else:
                log.warning("would predict-route subtask %d to Opus (concurrency) but "
                            "SDK unavailable; running local", w.id)
            w.escalation_reason = ""   # fell back to local; drop the predict tag
            w.ran_on = "local"
        for attempt in range(1, MAX_REPAIR + 1):
            if run._cancel:
                return
            w.attempts = attempt
            w.status = "running"
            run.emit({"type": "phase", "worker_id": w.id,
                      "text": f"worker {w.id} · {w.title} · attempt {attempt}/{MAX_REPAIR}"})
            # Workers get all tools EXCEPT launch_crew (no recursive crew spawning).
            # `owns` scopes which files this worker may WRITE (enforced in the agent
            # loop); empty => unscoped (back-compat, no behavior change).
            # Default: all tools except launch_crew. The unattended harness passes
            # run.worker_tools (a no-network set) + worker_use_mcp=False instead.
            wkr_tools = (run.worker_tools if run.worker_tools is not None
                         else [n for n in toolmod.tool_names() if n != "launch_crew"])
            steps = _STEP_BUDGET.get(run.complexity, 10)  # scale by goal complexity
            agent = agents.make_agent(run.worker_spec, max_steps=steps, cwd=run.cwd,
                                      use_mcp=run.worker_use_mcp, tool_names=wkr_tools,
                                      owns=w.owns)
            owns_note = ("\n\nFILE OWNERSHIP: you may create/edit ONLY these files: "
                         + ", ".join(w.owns) + ". Writes to any other path will be "
                         "REFUSED by the tools — do not modify files owned by other "
                         "subtasks.") if w.owns else ""
            task = ((f"Subtask: {w.title}\n\n{w.detail}".strip() + owns_note) if attempt == 1
                    else _repair_task(w, prev_output) + owns_note)
            try:
                prev_output = agent.run_task(
                    task, system=_WORKER_SYS, context=context,
                    on_event=lambda e, wid=w.id: run.emit({**e, "role": "worker",
                                                           "worker_id": wid}),
                    approver=self._approver(run, w.id))  # repairs are gated too
                w.output = prev_output
            except Exception as exc:  # noqa: BLE001 — a hard crash is terminal, no retry
                w.status = "error"
                w.output = f"(worker error: {exc})"
                w.ended_at = time.time()
                run.emit({"type": "error", "worker_id": w.id, "text": w.output})
                break

            if run._cancel:
                return
            # Gate runs UNATTENDED (no approver): it's the manager's own acceptance
            # check in cwd, not a worker-requested command, so it routes around no
            # approval that would otherwise fire. Worker tool calls above still gate.
            passed, gate_out = gate.run_gate(w.acceptance, run.cwd)
            w.gate_passed = passed
            w.gate_output = gate_out
            run.emit({"type": "gate", "worker_id": w.id, "attempt": attempt,
                      "passed": passed, "manual": w.acceptance is None,
                      "output": gate_out[:600]})
            if passed:
                # run_gate returns True both for a REAL green gate and for a
                # None/rejected acceptance (manual review). The latter was never
                # actually verified, so label it "unverified" — NOT done/passed.
                # Accounting only: the run proceeds exactly as before either way.
                w.status = "done" if w.acceptance is not None else "unverified"
                w.ended_at = time.time()
                break

        # Fell through all attempts without a terminal status -> failed the gate.
        if w.status not in ("done", "unverified", "error"):
            w.status = "failed"
            w.ended_at = time.time()
            run.emit({"type": "phase", "worker_id": w.id,
                      "text": f"worker {w.id} FAILED gate after {w.attempts} attempts"})

        # REACTIVE ESCALATION (opt-in): a local subtask that failed / couldn't be
        # self-verified / crashed is re-dispatched to Opus. OFF by default.
        if run.allow_escalation and w.status in ("failed", "unverified", "error"):
            self._escalate(run, w, context)

    def _dispatch_opus(self, run: CrewRun, w: "Worker", context: str, reason: str) -> bool:
        """Run a subtask on Opus (Claude path), re-gate its work, set the subtask
        state. Returns True if Opus actually ran; False if the SDK/Opus is
        unavailable or errored (caller decides the fallback). Never raises."""
        w.escalation_reason = reason
        if not _claude_available():
            return False
        # UNVERIFIED case (no allowlisted gate exists — the FSM / can't-self-verify
        # task): re-gating with the same None acceptance would just resolve back to
        # "unverified". So have Opus ALSO author a real gate, then verify against it.
        needs_gate = w.acceptance is None
        agent = agents.make_agent(run.escalation_spec, cwd=run.cwd)   # ClaudeAgent
        if needs_gate:
            task = (f"Subtask: {w.title}\n\n{w.detail}\n\n"
                    f"(Routed to you for escalation (reason: {reason}). There was NO "
                    f"automated way to verify this subtask locally. Do TWO things:\n"
                    f"1. Implement the subtask correctly.\n"
                    f"2. Write a REAL pytest test file (named test_*.py) with concrete "
                    f"`assert` statements that actually exercise the behavior — NOT a "
                    f"print/smoke test.\n"
                    f"Then make the LAST line of your reply EXACTLY:\n"
                    f"ACCEPTANCE_CMD: pytest <your_test_file>.py -q\n"
                    f"It must be a single pytest/unittest command — no shell pipes, "
                    f"redirects, chaining, or `python -c` smoke tests.)")
        else:
            task = (f"Subtask: {w.title}\n\n{w.detail}\n\n(Routed to you because a local "
                    f"model {reason} on this subtask. Implement it correctly and verify it.)")
        try:
            out = agent.run_task(
                task, system=_WORKER_SYS, context=context,
                on_event=lambda e, wid=w.id: run.emit({**e, "role": "opus", "worker_id": wid}),
                approver=self._approver(run, w.id))
        except Exception as exc:  # noqa: BLE001
            out = f"(escalation error: {exc})"
        # ClaudeAgent returns "(claude error: …)" if the SDK can't run (e.g. not authed).
        if out.startswith("(claude error") or out.startswith("(escalation error"):
            log.warning("Opus dispatch for subtask %d errored: %s", w.id, out[:120])
            return False
        w.output = out
        w.ran_on = "opus"
        w.ended_at = time.time()

        if needs_gate:
            # Adopt Opus's stated gate ONLY if it passes the SAME allowlist local
            # models are held to (Opus is NOT exempt — a smoke test is rejected).
            derived = _extract_acceptance_cmd(out)
            if derived and _is_allowed_acceptance_cmd(derived):
                w.acceptance = derived                       # record the real gate
                if w.id < len(run.plan):
                    # The old recorded acceptance (if any) was rejected/None; Opus
                    # replaced it with an allowlisted one, so clear the stale flag
                    # (data update — the honest-accounting LOGIC is untouched).
                    run.plan[w.id]["acceptance_rejected"] = None
                passed, gate_out = gate.run_gate(derived, run.cwd)
                w.gate_passed = passed
                w.gate_output = gate_out
                w.status = "done" if passed else "failed"
                log.warning("subtask %d: Opus authored gate %r -> %s",
                            w.id, derived, "passed" if passed else "failed")
                run.emit({"type": "gate", "worker_id": w.id, "phase": "opus",
                          "passed": passed, "manual": False, "output": gate_out[:600]})
            else:
                # Opus didn't produce a verifiable, allowlisted gate either — keep it
                # UNVERIFIED but say so LOUDLY (visible, not silent).
                why = ("no ACCEPTANCE_CMD line" if not derived
                       else f"gate rejected by allowlist: {derived!r}")
                w.status = "unverified"
                log.warning("subtask %d: Opus could NOT self-verify (%s) — leaving it "
                            "UNVERIFIED", w.id, why)
                run.emit({"type": "gate", "worker_id": w.id, "phase": "opus",
                          "passed": None, "manual": True,
                          "output": f"opus could not self-verify: {why}"})
            return True

        # failed/error case: an allowlisted gate already exists — re-gate as before.
        passed, gate_out = gate.run_gate(w.acceptance, run.cwd)
        w.gate_passed = passed
        w.gate_output = gate_out
        w.status = "done" if passed else "failed"
        run.emit({"type": "gate", "worker_id": w.id, "phase": "opus",
                  "passed": passed, "manual": False, "output": gate_out[:600]})
        return True

    def _escalate(self, run: CrewRun, w: "Worker", context: str) -> None:
        """REACTIVE: re-dispatch a locally failed/unverified/crashed subtask to
        Opus. Degrades gracefully — on unavailable/error it leaves the local
        terminal state untouched."""
        reason = w.status   # failed | unverified | error
        if not _claude_available():
            log.warning("would escalate subtask %d (%s) to Opus, but the SDK is "
                        "unavailable — leaving it %s", w.id, reason, w.status)
            run.emit({"type": "escalation", "worker_id": w.id, "reason": reason,
                      "status": "opus_unavailable"})
            return
        log.warning("escalating subtask %d to %s (reason: %s)",
                    w.id, run.escalation_spec, reason)
        run.emit({"type": "escalation", "worker_id": w.id, "reason": reason,
                  "status": "escalating", "to": run.escalation_spec})
        if self._dispatch_opus(run, w, context, reason):
            w.escalated = True   # ran local first, then Opus
            run.emit({"type": "escalation", "worker_id": w.id, "reason": reason,
                      "status": "done", "outcome": w.status})
        else:
            run.emit({"type": "escalation", "worker_id": w.id, "reason": reason,
                      "status": "opus_unavailable"})   # keep local terminal state

    def _review(self, run: CrewRun) -> None:
        run.status = "reviewing"
        run.emit({"type": "phase", "text": "manager reviewing"})

        # 1) Re-run every gated subtask's acceptance FRESH against the current files.
        #    This is what catches integration breakage — e.g. a later worker
        #    clobbering a shared file that an earlier worker's gate depended on, a
        #    regression per-subtask gating during _work cannot see. Unattended,
        #    same path as Step 3 (the manager's own check, no approver).
        sections, problems = [], []
        for w in run.workers:
            if run._cancel:
                break
            line = (f"### Subtask {w.id}: {w.title}\n"
                    f"work-phase status: {w.status} (attempts {w.attempts})")
            if w.acceptance is not None:
                passed, out = gate.run_gate(w.acceptance, run.cwd)
                w.review_gate_passed = passed
                w.review_gate_output = out
                run.emit({"type": "gate", "worker_id": w.id, "phase": "review",
                          "passed": passed, "manual": False, "output": out[:600]})
                line += f"\nreview gate: {'PASS' if passed else 'FAIL'}"
                if not passed:
                    line += f"\nreview gate output:\n{out[:1500]}"
                    if w.gate_passed:  # green during work, red now = a regression
                        problems.append(f"Subtask {w.id} '{w.title}' PASSED during "
                                        "work but its gate is RED now — a later step "
                                        "likely broke a shared file.")
                    else:
                        problems.append(f"Subtask {w.id} '{w.title}' gate is RED.")
            else:
                line += "\nreview gate: (no automated check — manual review)"
            # Distinct terminal states must BOTH be reported, not just `failed`.
            if w.status == "failed":
                problems.append(f"Subtask {w.id} '{w.title}' FAILED — gate never "
                                f"passed after {w.attempts} attempts.")
            elif w.status == "error":
                problems.append(f"Subtask {w.id} '{w.title}' ERRORED (worker crashed): "
                                f"{(w.output or '')[:200]}")
            line += f"\nworker report:\n{(w.output or '')[:1200]}"
            sections.append(line)

        report = "\n\n".join(sections) or "(no workers ran)"
        problems_block = ("\n".join(f"- {p}" for p in problems)
                          if problems else "- none detected")

        # 2) Manager reads the real files and writes the final answer from VERIFIED
        #    status + the PROBLEMS list (which it is instructed to surface in full).
        mgr = agents.make_agent(run.manager_spec,
                                tool_names=run.manager_tools or _READONLY,
                                max_steps=6, cwd=run.cwd)
        task = (f"Original goal:\n{run.goal}\n\n"
                f"VERIFIED SUBTASK STATUS (gates re-run just now against the real "
                f"files):\n{report}\n\n"
                f"PROBLEMS DETECTED (you MUST surface every one of these):\n"
                f"{problems_block}\n\n"
                "Inspect the actual files in the working folder, then write the final "
                "integrated answer for the user.")
        run.final = mgr.run_task(task, system=_REVIEW_SYS,
                                 on_event=lambda e: run.emit({**e, "role": "manager"}))

    def _finish(self, run: CrewRun, status: str) -> None:
        run.status = status
        run.ended = time.time()
        run.emit({"type": "phase", "text": f"run {status}"})
        try:
            _log_run(run)
        except Exception as exc:  # noqa: BLE001 — logging must never affect the run
            log.warning("crew history logging failed: %s", exc)


MANAGER = _CrewManager()
