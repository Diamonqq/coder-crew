"""Local 'Coder Crew' orchestration.

A manager model decomposes a goal into subtasks, worker models implement each
one (with tools + user-approved shell), and the manager reviews/integrates the
results into a final answer.

  manager (plan) ──► worker · worker · worker ──► manager (review + synthesize)

Roles are pluggable agent specs (see agent.agents.make_agent):
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
_REPORTS = Path(__file__).resolve().parent.parent / "data" / "research"


def _save_report(run) -> str | None:
    """Persist a finished research run's synthesis as a markdown file on disk so the
    user has a real artifact. Best-effort: a write failure must not break the run."""
    if not (run.final or "").strip():
        return None
    try:
        slug = re.sub(r"[^a-z0-9]+", "-", (run.goal or "report").lower()).strip("-")[:48] or "report"
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        _REPORTS.mkdir(parents=True, exist_ok=True)
        path = _REPORTS / f"{slug}-{stamp}.md"
        header = f"# Research: {run.goal}\n\n_Generated {datetime.now():%Y-%m-%d %H:%M} · {run.manager_spec}_\n\n---\n\n"
        path.write_text(header + run.final, encoding="utf-8")
        return str(path)
    except OSError:
        return None


class _Cancelled(Exception):
    """Raised at a research checkpoint when the user hits Stop (run._cancel)."""


def _ledger_totals(run_id: str) -> dict:
    try:
        from . import ledger
        return ledger.run_totals(run_id)
    except Exception:  # noqa: BLE001
        return {"total": 0, "by": []}


def list_reports() -> list:
    """Saved research reports on disk, newest first — so reports survive restarts
    (in-memory runs don't)."""
    try:
        files = sorted(_REPORTS.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        return []
    out = []
    for p in files[:100]:
        try:
            st = p.stat()
        except OSError:
            continue
        out.append({"name": p.name, "size": st.st_size, "mtime": st.st_mtime})
    return out


def read_report(name: str) -> str | None:
    """Read one saved report by bare filename (path-traversal-safe)."""
    if not name or "/" in name or "\\" in name or ".." in name:
        return None
    p = _REPORTS / name
    if not p.is_file():
        return None
    try:
        return p.read_text(encoding="utf-8")
    except OSError:
        return None


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
    incomplete_reason: str = ""        # completeness pass: a required deliverable
    #                                    (e.g. a module's test) is missing/never-ran
    #                                    => forces gate_outcome off "passed" (unverified).
    coverage_missing: list = field(default_factory=list)  # spec cases the tests omitted
    coverage_note: str = ""            # spec-coverage review summary + outcome


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
    # Auto-approve mode: danger tools (file writes / shell) run WITHOUT pausing for
    # approval — fully unattended. OFF by default; opt-in per run / via the UI.
    auto_approve: bool = False
    # Incognito assistant run — kept out of the runs list and never persisted, so
    # the conversation leaves no trace.
    incognito: bool = False
    # Research-swarm config (None for normal coder runs): see start_research.
    research_cfg: object = None
    # Spec-coverage review: after a green subtask, the manager critiques whether the
    # tests cover the spec's named cases; missing cases trigger a local retry. Best-
    # effort. Default ON only when the manager is a Claude spec (local critique is weak).
    coverage_review: bool = False
    plan: list = field(default_factory=list)
    contract: object = None     # shared interface contract from the plan (or None)
    workers: list = field(default_factory=list)
    events: list = field(default_factory=list)
    final: str = ""
    error: str = ""
    created: float = field(default_factory=time.time)
    ended: float | None = None
    report_file: str | None = None     # saved markdown path (research runs)
    code_rounds: int = 1               # >1 = looped autonomous coder (manager re-plans each round)
    support_researchers: int = 0       # >0 = research-augmented coder: N researchers/round feed the build
    support_spec: str = ""             # model the support researchers use
    research_notes: str = ""           # latest round's gathered research, injected into plan + build

    # --- approval / control plumbing (not serialized) ---
    pending: dict | None = None          # {tool, args, worker_id}
    _gate: threading.Event = field(default_factory=threading.Event, repr=False)
    _decision: tuple | None = None       # (approved: bool, note: str)
    _cancel: bool = False

    def emit(self, ev: dict) -> None:
        ev["t"] = time.time()
        self.events.append(ev)
        # Persist token usage to the granular ledger (survives the event cap below).
        if ev.get("type") == "usage":
            try:
                from . import ledger
                key = ("w" + str(ev["worker_id"])) if ev.get("worker_id") is not None \
                    else (ev.get("role") or ev.get("agent") or "?")
                ledger.log(self.id, key, ev.get("role"), ev.get("agent"),
                           ev.get("tokens"), ev.get("rate"))
            except Exception:  # noqa: BLE001
                pass
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
            "events": self.events[-250:],
            "final": self.final,
            "error": self.error,
            "pending": self.pending,
            "created": self.created,
            "ended": self.ended,
            "elapsed": round((self.ended or time.time()) - self.created, 1),
            "report_file": self.report_file,
            "code_rounds": self.code_rounds,     # >1 = looped autonomous coder
            "ledger": _ledger_totals(self.id),   # persisted per-role/model token totals
            "research": ({"n_submanagers": self.research_cfg.get("n_submanagers"),
                          "n_researchers": self.research_cfg.get("n_researchers"),
                          "submanager": self.research_cfg.get("submanager_spec"),
                          "researcher": self.research_cfg.get("researcher_spec")}
                         if self.research_cfg else None),
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
_ASSISTANT_SYS = (
    "You are coder-crew's built-in coding assistant. You can not only chat — you "
    "can TAKE ACTIONS via tools: read/write files, run shell commands, list "
    "directories, search/fetch the web, and launch_crew (spin up an autonomous "
    "coder crew to build something larger). When the user asks you to DO something, "
    "actually call the tool — don't just describe it. Anything risky (shell, file "
    "writes, launching a crew) pauses for the user's approval before it runs, so act "
    "decisively and let them approve. Be concise."
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
_COVERAGE_SYS = (
    "You are a strict TEST-COVERAGE reviewer. You are given a goal, a subtask, a "
    "module's source, and its test file. Your ONLY job: list spec-required behaviors "
    "/ edge cases that the goal or subtask EXPLICITLY named or CLEARLY implied, and "
    "that the test file does NOT actually assert. This is about COVERAGE (did the "
    "tests check everything the spec asked for) — NOT about whether the code is "
    "correct. Be CONSERVATIVE: only list a case if the spec clearly requires it; if "
    "you're unsure, leave it out (a false 'missing' triggers needless work). Reply "
    'with ONLY a JSON object: {"missing": ["<short specific case>", ...], "covered": '
    '["..."]}. An empty "missing" list means coverage looks complete.'
)


_REPAIR_DIRECTIVE = (
    "Your previous attempt did NOT pass the acceptance check. The check output is "
    "above — it tells you exactly why it failed. Edit the files in your working "
    "folder so the check passes, then verify. Do not explain in prose — make the "
    "change and re-run."
)


def _distill_gate(out: str) -> str:
    """Structured error feedback (#3): pull just the lines that matter (errors,
    assertions, the pytest summary) instead of dumping raw output the model drowns in.
    Cuts repair attempts because the signal isn't buried in noise."""
    out = out or ""
    lines = out.splitlines()
    pat = re.compile(r"(?i)(error|assert|fail|exception|traceback|^E\s|importerror|"
                     r"syntaxerror|nameerror|typeerror|attributeerror|\bnot found\b)")
    key = [ln for ln in lines if pat.search(ln)]
    picked, seen = [], set()
    for ln in key[:18] + lines[-6:]:            # key error lines + the summary tail
        s = ln.strip()
        if s and s not in seen:
            seen.add(s)
            picked.append(ln.rstrip())
    return ("\n".join(picked)[:1400]) or out[:1200]


def _repair_task(w: "Worker", prev_output: str) -> str:
    """Repair-attempt prompt: original spec + the DISTILLED failure (the lines that
    matter) + a fix directive. Focused signal → far fewer wasted retries."""
    return (
        f"Subtask: {w.title}\n\n{w.detail}\n\n"
        f"--- THE ACCEPTANCE TEST FAILED. The lines that matter ---\n{_distill_gate(w.gate_output)}\n\n"
        f"Fix ONLY what these errors point to (don't rewrite passing parts). {_REPAIR_DIRECTIVE}"
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


_READONLY = ["list_dir", "read_file", "system_stats", "web_search",
             "web_fetch", "get_time", "read_notes"]

# Tools a read-only research worker (Claude) must NOT have — keeps an autonomous,
# auto-approved swarm from writing files or running commands.
_RESEARCH_BLOCK = ["Bash", "Write", "Edit", "MultiEdit", "NotebookEdit"]
# Local-model research workers get only these.
_RESEARCH_TOOLS = ["web_search", "web_fetch", "read_file", "get_time"]


# --- Autopilot / advisor ----------------------------------------------------
_Q3 = "qwen3-coder:30b"        # primary: agentic MoE, 3.3B active (stock build)
_CODER = "qwen2.5-coder"       # dense fallbacks (stock)
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


_ENHANCE_SYS = (
    "You are a prompt engineer for an autonomous build/research agent. Rewrite the "
    "user's rough idea into ONE clear, specific, self-contained prompt the agent can "
    "act on directly. PRESERVE their intent and scope — do not invent a different "
    "project or pad it out. Make it concrete: state exactly what to build or research, "
    "the key requirements, and how success is verified. If it's a software build, "
    "require tests and say it must actually run them. Keep it tight — a short paragraph "
    "or a few bullet lines. Output ONLY the improved prompt: no preamble, no "
    "explanation, no surrounding quotes or markdown headers."
)


def enhance_prompt(text: str, spec: str) -> str:
    """Rewrite a rough Create-tab idea into a sharper prompt, using the chosen model.
    ollama: specs go through the chat endpoint (one fast call); claude: specs use a
    one-turn tool-less agent. Any failure falls back to the original text unchanged."""
    text = (text or "").strip()
    if not text:
        return ""
    try:
        if spec.startswith("claude:"):
            out = agents.make_agent(spec, max_turns=1).run_task(text, system=_ENHANCE_SYS)
        else:
            from . import chat
            model = spec.split("ollama:")[-1]
            res = chat.chat(model, [{"role": "system", "content": _ENHANCE_SYS},
                                    {"role": "user", "content": text}], timeout=120)
            out = res.get("reply", "")
        return (out or "").strip() or text
    except Exception:  # noqa: BLE001 — never fail the box; return what they typed
        return text


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
    import os
    if os.environ.get("CREW_CLAUDE_OFF"):   # connection set to "off" (see claude_conf)
        return False
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


def _is_test_name(name: str) -> bool:
    return name.startswith("test_") or name.endswith("_test.py")


def _all_rel_files(cwd: str) -> list:
    """Relative paths of every file under cwd, skipping caches."""
    root = Path(cwd)
    out = []
    for p in root.rglob("*"):
        if p.is_file() and not any(part in ("__pycache__", ".pytest_cache", ".git")
                                   for part in p.parts):
            out.append(p.relative_to(root).as_posix())
    return out


def _tests_that_ran(run: "CrewRun") -> "tuple[set, bool]":
    """Which test files were executed by a PASSING gate. Returns (set of test-file
    basenames named in passing gate commands, whole_suite_ran). A bare `pytest` /
    `python -m pytest [dir]` that passed ran the WHOLE suite."""
    ran: set = set()
    whole = False
    for w in run.workers:
        if not (w.gate_passed is True or w.review_gate_passed is True):
            continue
        acc = w.acceptance
        if isinstance(acc, str):
            pyfiles = re.findall(r"[\w./\\-]+\.py", acc)
            tests = {Path(f).name for f in pyfiles if _is_test_name(Path(f).name)}
            if tests:
                ran |= tests
            elif not pyfiles and re.match(r"\s*(\"?[^\"\s]*python|pytest)", acc, re.I):
                whole = True
    return ran, whole


def _subtask_gate_outcome(w: "Worker", rejected: bool) -> str:
    if rejected:
        return "rejected"
    # A required deliverable (e.g. a module's test) is missing or never ran — cannot
    # be "passed" however green its own gate was. Counts as unverified.
    if getattr(w, "incomplete_reason", ""):
        return "incomplete"
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
            "coverage_note": w.coverage_note,
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
              escalation_spec: str = "claude:claude-opus-4-8",
              auto_approve: bool = False,
              coverage_review: "bool | None" = None, rounds: int = 1,
              support_researchers: int = 0, support_spec: str = "") -> CrewRun:
        # No folder given => make a fresh one so workers always have a home.
        if not cwd:
            cwd = _auto_workspace(goal)
        # Default coverage review ON only for a capable (Claude) manager.
        cov = (coverage_review if coverage_review is not None
               else manager_spec.startswith("claude:"))
        rounds = max(1, min(int(rounds), 6))
        run = CrewRun(id=uuid.uuid4().hex[:12], goal=goal,
                      manager_spec=manager_spec, worker_spec=worker_spec,
                      max_workers=max(1, min(max_workers, 6)), cwd=cwd,
                      complexity=complexity if complexity in _STEP_BUDGET else "medium",
                      tag=tag, worker_tools=worker_tools, manager_tools=manager_tools,
                      worker_use_mcp=worker_use_mcp, allow_escalation=allow_escalation,
                      escalation_spec=escalation_spec, auto_approve=auto_approve,
                      coverage_review=cov, code_rounds=rounds,
                      support_researchers=max(0, min(int(support_researchers), 8)),
                      support_spec=support_spec or worker_spec)
        # rounds>1 → autonomous looped coder (manager re-plans on results each round).
        # Force auto_approve so it can run unattended.
        if rounds > 1:
            run.auto_approve = True
        with self._lock:
            self.runs[run.id] = run
        driver = self._drive_loop if rounds > 1 else self._drive
        threading.Thread(target=driver, args=(run,), daemon=True).start()
        return run

    def start_assistant(self, text: str, model_spec: str,
                        history: list | None = None,
                        incognito: bool = False) -> CrewRun:
        """Run the chat-bubble assistant as a tool-using agent. It reuses the
        SAME CrewRun + _approver machinery as workers, so any danger/MCP tool the
        assistant calls pauses for approval exactly like a worker's would."""
        run = CrewRun(id=uuid.uuid4().hex[:12], goal=text, manager_spec=model_spec,
                      worker_spec=model_spec, max_workers=1, cwd=None,
                      incognito=incognito)
        with self._lock:
            self.runs[run.id] = run
        threading.Thread(target=self._drive_assistant,
                         args=(run, history or []), daemon=True).start()
        return run

    def _drive_assistant(self, run: CrewRun, history: list) -> None:
        run.status = "working"
        # Full toolset (incl. launch_crew) + MCP. The approver is the IDENTICAL
        # function workers use — danger/MCP calls block on run.pending until the
        # user approves via the same /api/crew/runs/{id}/approve endpoint.
        agent = agents.make_agent(run.manager_spec, max_steps=10, use_mcp=True)
        ctx = "\n".join(f"{m.get('role')}: {m.get('content','')}"
                        for m in history[-8:] if m.get("content"))
        try:
            run.final = agent.run_task(
                run.goal, system=_ASSISTANT_SYS, context=ctx,
                on_event=lambda e: run.emit({**e, "role": "assistant"}),
                approver=self._approver(run, 0))  # <-- same gating path as workers
            run.status = "done"
        except Exception as exc:  # noqa: BLE001
            run.error = f"{type(exc).__name__}: {exc}"
            run.status = "error"
        run.ended = time.time()

    # -- research swarm --------------------------------------------------------
    def start_research(self, topic: str, *, manager_spec: str,
                       submanager_spec: str, researcher_spec: str,
                       n_submanagers: int, n_researchers: int,
                       rounds: int = 1) -> CrewRun:
        """Fan a topic out to a swarm of researchers, then synthesize + rank.
          • flat (n_submanagers=0): manager -> n_researchers -> ranked report
          • tiered: manager -> n_submanagers -> n_researchers each -> report
          • LOOPED (rounds>1, flat): the manager runs multiple bounded ROUNDS —
            after each it reviews the findings and either declares DONE or names the
            specific remaining GAPS as the next round's angles, then synthesizes all
            rounds. This is the looped-Opus orchestrator (re-plan on info from leaves).
        Researchers are read-only; auto-approved. Trades tokens for breadth/depth."""
        cfg = {"submanager_spec": submanager_spec, "researcher_spec": researcher_spec,
               "n_submanagers": max(0, min(int(n_submanagers), 8)),
               "n_researchers": max(1, min(int(n_researchers), 10)),
               "rounds": max(1, min(int(rounds), 5))}
        run = CrewRun(id=uuid.uuid4().hex[:12], goal=topic, manager_spec=manager_spec,
                      worker_spec=researcher_spec, max_workers=cfg["n_researchers"],
                      cwd=None, auto_approve=True)
        run.research_cfg = cfg
        with self._lock:
            self.runs[run.id] = run
        driver = self._drive_looped if cfg["rounds"] > 1 else self._drive_research
        threading.Thread(target=driver, args=(run,), daemon=True).start()
        return run

    def autopilot(self, idea: str, *, manager_spec: str, researcher_spec: str,
                  builder_spec: str, build=None) -> CrewRun:
        """One-prompt orchestrator: a planner model reads the idea, decides the whole
        pipeline (research vs build, #researchers, rounds, refined goal), then launches
        it. `build` (the UI toggle) overrides the build decision when not None."""
        planner = agents.make_agent(manager_spec, tool_names=[], max_steps=2) \
            if manager_spec.startswith("ollama:") else agents.make_agent(manager_spec, max_turns=2)
        prompt = (
            "You are sizing a task for an autonomous agent swarm. The user's idea:\n\n"
            f"{idea}\n\n"
            "Reply with ONLY a JSON object, no prose:\n"
            '{"build": <true if they want software CREATED/built, false if it is a question/'
            'topic to research>, "n_researchers": <2-8, how many parallel researchers fit the '
            'breadth>, "rounds": <1-4, how many iterative rounds the depth warrants>, '
            '"goal": "<one clear sentence restating the goal>"}')
        cfgj = {}
        try:
            reply = planner.run_task(prompt)
            m = re.search(r"\{.*\}", reply or "", re.S)
            if m:
                cfgj = json.loads(m.group(0))
        except Exception:  # noqa: BLE001
            cfgj = {}
        do_build = build if build is not None else bool(cfgj.get("build"))
        goal = (cfgj.get("goal") or idea).strip()
        n = max(2, min(int(cfgj.get("n_researchers", 4) or 4), 8))
        rounds = max(1, min(int(cfgj.get("rounds", 1) or 1), 4))
        if do_build:
            return self.start(goal, manager_spec=manager_spec, worker_spec=builder_spec,
                              max_workers=3, rounds=max(2, rounds),
                              support_researchers=n, support_spec=researcher_spec)
        return self.start_research(goal, manager_spec=manager_spec,
                                   submanager_spec=researcher_spec, researcher_spec=researcher_spec,
                                   n_submanagers=0, n_researchers=n, rounds=rounds)

    def _research_agent(self, spec: str, *, max_steps: int = 6, role: str = "researcher"):
        """A read-only agent (web + read tools only) for research. Per-role tuning
        (Tier-1 #1/#5): researchers get a tiny context + short keep_alive so they
        can't evict the resident manager, and a tight timeout (the watchdog — a
        stuck leaf fails fast instead of hanging the swarm). Ignored for Claude."""
        budgets = {"manager":    {"num_ctx": 16384, "keep_alive": "20m", "timeout": 600.0},
                   "submanager": {"num_ctx": 8192,  "keep_alive": "5m",  "timeout": 300.0},
                   "researcher": {"num_ctx": 4096,  "keep_alive": "60s", "timeout": 180.0}}
        b = budgets.get(role, budgets["researcher"])
        if spec.startswith("claude:"):
            return agents.make_agent(spec, max_turns=max_steps,
                                     disallowed_tools=_RESEARCH_BLOCK)
        return agents.make_agent(spec, tool_names=_RESEARCH_TOOLS,
                                 max_steps=max_steps, use_mcp=False,
                                 num_ctx=b["num_ctx"], keep_alive=b["keep_alive"],
                                 timeout=b["timeout"])

    def _split(self, agent, topic: str, n: int, kind: str, *, emit=None) -> list:
        prompt = (f"Break this topic into EXACTLY {n} distinct {kind}, each a short "
                  f"specific phrase worth researching. Topic: {topic}\n\n"
                  f"Reply with ONLY a numbered list (one item per line), no preamble.")
        try:
            reply = agent.run_task(prompt, on_event=emit) if emit else agent.run_task(prompt)
        except Exception:  # noqa: BLE001
            reply = ""
        items = []
        for ln in (reply or "").splitlines():
            s = re.sub(r"^[\s\d.)\-*•]+", "", ln).strip()
            if s:
                items.append(s)
        items = items[:n]
        while len(items) < n:
            items.append(f"{topic} — additional angle {len(items) + 1}")
        return items

    def _research_one(self, run: CrewRun, spec: str, topic: str, angle: str,
                      idx: int, branch: str | None = None, branch_id=None, rnd=None):
        # branch_id (the sub-area index, or None when flat) lets the tree nest each
        # researcher under its sub-manager; rnd (looped runs) tags which round it ran in.
        tag = {"role": "worker", "worker_id": idx, "branch": branch_id, "round": rnd}
        run.emit({**tag, "type": "tool_call", "angle": angle[:120],
                  "tool": "research", "args": {"angle": angle[:90]}})
        agent = self._research_agent(spec, max_steps=6)
        ctx = f"Overall topic: {topic}" + (f"\nSub-area: {branch}" if branch else "")
        # Forceful + structured so small local models actually SEARCH and return
        # specifics instead of generic prose (the main cause of weak local research).
        task = ("Research the angle below and return SUBSTANTIVE findings — not generic advice.\n"
                "STEP 1: Call web_search at least once with a focused query to get current, real info "
                "(search again if the first results are thin).\n"
                "STEP 2: Write 4-6 bullet points. Each MUST be a specific fact, number, tool name, "
                "technique, or concrete trade-off you actually found — with a short source. "
                "No vague filler like 'consider best practices'.\n"
                "STEP 3: Finish with the 2-3 strongest, most actionable takeaways for the overall topic.\n"
                "Do NOT write files or run shell commands.\n\n"
                "Angle: " + angle)
        emit = lambda e: run.emit({**e, **tag})
        try:
            out = agent.run_task(task, context=ctx, on_event=emit)
        except Exception as exc:  # noqa: BLE001
            out = f"(researcher error: {exc})"
        run.emit({**tag, "type": "tool_result", "tool": "research", "result": f"✓ {angle[:70]}"})
        return (angle, out)

    def _synthesize(self, spec: str, topic: str, findings: list, *, master: bool = False,
                    emit=None) -> str:
        # Synthesis is PURE GENERATION — build a TOOL-LESS agent so a local model can't
        # loop calling web_search and hit "max tool steps" (which produced empty reports).
        what = "sub-area reports" if master else "researcher findings"

        def _build(cap: int) -> tuple[str, str]:
            joined = "\n\n".join(f"### {a}\n{(t or '').strip()[:cap]}" for a, t in findings)
            prompt = (f"You are synthesizing {what} on: {topic}\n\n{joined}\n\n"
                      f"Write a structured report in Markdown:\n"
                      f"- A 1-2 sentence summary.\n"
                      f"- '## Key findings' — the concrete, specific facts/numbers/tools from the "
                      f"research above, DEDUPLICATED (merge repeats).\n"
                      f"- '## Ranked recommendations' — best first, each a specific actionable item "
                      f"with a one-line why.\n"
                      f"Use the actual details from the findings; do NOT pad with generic advice "
                      f"('follow best practices', 'it depends'). If the findings are thin, say so "
                      f"briefly rather than inventing filler. Do not call any tools — just write it.")
            return prompt, joined

        is_local = not spec.startswith("claude:")
        # A bare "(model error/(synthesis error" sentinel from run_task means the call
        # failed (e.g. an Ollama 500 when 16k ctx on a 30B spills VRAM on 24GB). Local
        # synthesis runs at a SAFE 8k ctx; on failure we retry smaller, then fall back
        # to the raw findings so the user always gets a usable report — never an error.
        def _bad(s: str) -> bool:
            return (not s) or s.lstrip().startswith(("(model error", "(synthesis error"))

        attempts = [(8192, 4000), (4096, 1500)] if is_local else [(None, 12000)]
        last_joined = ""
        for ctx, cap in attempts:
            prompt, last_joined = _build(cap)
            try:
                if spec.startswith("claude:"):
                    agent = agents.make_agent(spec, max_turns=2, disallowed_tools=_RESEARCH_BLOCK)
                else:
                    agent = agents.make_agent(spec, tool_names=[], max_steps=2, num_ctx=ctx)
                out = agent.run_task(prompt, on_event=emit) if emit else agent.run_task(prompt)
                if not _bad(out):
                    return out
            except Exception:  # noqa: BLE001
                pass  # fall through to the next (smaller) attempt
        # Every attempt failed — return the gathered findings directly so the run is
        # still useful instead of a one-line error.
        return (f"## Findings on: {topic}\n\n_(Automatic synthesis was unavailable — "
                f"showing the raw researcher findings below.)_\n\n{last_joined}")

    def _drive_research(self, run: CrewRun) -> None:
        import concurrent.futures as cf
        cfg = run.research_cfg
        topic = run.goal
        M, K = cfg["n_submanagers"], cfg["n_researchers"]
        pool = 8  # bounded concurrent researchers (each spawns a model process)
        # Role-tagged emitters so manager / sub-manager token usage surfaces in the
        # tree just like the researchers' (worker_id keeps each node distinct;
        # sub-managers use a high base to never collide with researcher indices).
        mgr_emit = lambda e: run.emit({**e, "role": "manager"})
        def ck():
            if run._cancel:
                raise _Cancelled()
        try:
            run.status = "planning"
            manager = self._research_agent(run.manager_spec, max_steps=4, role="manager")
            ck()

            if M <= 0:  # flat
                angles = self._split(manager, topic, K, "research angles", emit=mgr_emit)
                ck()
                run.emit({"type": "text", "role": "manager",
                          "text": f"Dispatching {len(angles)} researchers: " + "; ".join(angles)})
                run.status = "working"
                findings = [None] * len(angles)
                with cf.ThreadPoolExecutor(max_workers=min(pool, len(angles))) as ex:
                    futs = {ex.submit(self._research_one, run, cfg["researcher_spec"],
                                      topic, ang, i): i for i, ang in enumerate(angles)}
                    for fut in cf.as_completed(futs):
                        findings[futs[fut]] = fut.result()
                        ck()
                run.status = "reviewing"
                run.emit({"type": "text", "role": "manager", "text": "Synthesizing & ranking…"})
                run.final = self._synthesize(run.manager_spec, topic, findings, emit=mgr_emit)
            else:  # tiered: manager -> sub-managers -> researchers
                branches = self._split(manager, topic, M, "major sub-areas", emit=mgr_emit)
                ck()
                run.emit({"type": "text", "role": "manager",
                          "text": f"{M} sub-managers × {K} researchers = {M * K} researchers. "
                                  f"Sub-areas: " + "; ".join(branches)})
                run.status = "working"
                submgrs, tasks = [], []
                for bi, br in enumerate(branches):
                    sm = self._research_agent(cfg["submanager_spec"], max_steps=4, role="submanager")
                    submgrs.append(sm)
                    sm_emit = lambda e, b=bi: run.emit({**e, "role": "submanager",
                                                        "worker_id": 1000 + b, "branch": b})
                    for ang in self._split(sm, f"{topic} — sub-area: {br}", K,
                                           "research angles", emit=sm_emit):
                        tasks.append((bi, br, ang))
                # surface each sub-area title so the tree can label sub-manager nodes
                for bi, br in enumerate(branches):
                    run.emit({"type": "branch", "role": "submanager",
                              "worker_id": 1000 + bi, "branch": bi, "title": br})
                by_branch: dict = {bi: [] for bi in range(len(branches))}
                with cf.ThreadPoolExecutor(max_workers=pool) as ex:
                    futs = {ex.submit(self._research_one, run, cfg["researcher_spec"],
                                      topic, ang, i, br, bi): bi
                            for i, (bi, br, ang) in enumerate(tasks)}
                    for fut in cf.as_completed(futs):
                        by_branch[futs[fut]].append(fut.result())
                        ck()
                run.status = "reviewing"
                branch_reports = []
                for bi, br in enumerate(branches):
                    run.emit({"type": "text", "role": "submanager",
                              "text": f"Sub-manager synthesizing: {br}"})
                    sm_emit = lambda e, b=bi: run.emit({**e, "role": "submanager",
                                                        "worker_id": 1000 + b, "branch": b})
                    branch_reports.append(
                        (br, self._synthesize(cfg["submanager_spec"], f"{topic} :: {br}",
                                              by_branch[bi], emit=sm_emit)))
                run.emit({"type": "text", "role": "manager", "text": "Master synthesis…"})
                run.final = self._synthesize(run.manager_spec, topic, branch_reports,
                                             master=True, emit=mgr_emit)
            run.report_file = _save_report(run)
            run.status = "done"
        except _Cancelled:
            run.emit({"type": "text", "role": "manager", "text": "Stopped by user."})
            run.status = "cancelled"
        except Exception as exc:  # noqa: BLE001
            run.error = f"{type(exc).__name__}: {exc}"
            run.emit({"type": "error", "text": run.error})
            run.status = "error"
        run.ended = time.time()

    def _replan(self, agent, topic, findings, k, rnd, emit=None):
        """Manager reviews findings and returns up to k NEW angles targeting the gaps,
        or [] if it judges coverage sufficient (replies just DONE)."""
        joined = "\n\n".join(f"- {a}: {str(t)[:300]}" for a, t in findings[-12:])
        prompt = (f"You are leading a multi-round research effort on: {topic}\n\n"
                  f"Findings so far (after round {rnd + 1}):\n{joined}\n\n"
                  f"Decide the next step. If the topic is now well-covered, reply with EXACTLY the "
                  f"single word DONE. Otherwise identify the most important REMAINING GAPS and reply "
                  f"with EXACTLY {k} new, SPECIFIC research angles targeting those gaps — a numbered "
                  f"list, one per line, no preamble. Do NOT repeat angles already covered.")
        try:
            reply = agent.run_task(prompt, on_event=emit) if emit else agent.run_task(prompt)
        except Exception:  # noqa: BLE001
            return []
        if reply and re.search(r"\bDONE\b", reply) and len(reply.strip()) < 14:
            return []
        items = []
        for ln in (reply or "").splitlines():
            s = re.sub(r"^[\s\d.)\-*•]+", "", ln).strip()
            if not s or len(s) <= 4:
                continue
            if s.endswith(":") or (s.isupper() and len(s) < 42):   # skip headers like "REMAINING GAPS:"
                continue
            items.append(s)
        return items[:k]

    def _drive_looped(self, run: CrewRun) -> None:
        """Looped-Opus orchestrator (flat): the manager runs bounded ROUNDS, re-planning
        on the researchers' findings each round, then synthesizes everything. Stop:
        hard round cap (cfg.rounds) OR the manager declaring DONE early."""
        import concurrent.futures as cf
        cfg = run.research_cfg
        topic = run.goal
        K = cfg["n_researchers"]
        rounds = cfg["rounds"]
        pool = 8
        mgr_emit = lambda e: run.emit({**e, "role": "manager"})
        def ck():
            if run._cancel:
                raise _Cancelled()
        try:
            run.status = "planning"
            manager = self._research_agent(run.manager_spec, max_steps=4, role="manager")
            ck()
            angles = self._split(manager, topic, K, "research angles", emit=mgr_emit)
            all_findings, idx = [], 0
            for rnd in range(rounds):
                ck()
                run.emit({"type": "text", "role": "manager", "round": rnd,
                          "text": f"Round {rnd + 1}/{rounds}: {len(angles)} researchers — "
                                  + "; ".join(a[:55] for a in angles)})
                run.status = "working"
                results = [None] * len(angles)
                with cf.ThreadPoolExecutor(max_workers=min(pool, len(angles))) as ex:
                    futs = {ex.submit(self._research_one, run, cfg["researcher_spec"],
                                      topic, ang, idx + i, rnd=rnd): i for i, ang in enumerate(angles)}
                    for fut in cf.as_completed(futs):
                        results[futs[fut]] = fut.result(); ck()
                idx += len(angles)
                all_findings += [r for r in results if r]
                if rnd >= rounds - 1:
                    break
                run.status = "reviewing"
                run.emit({"type": "text", "role": "manager", "round": rnd,
                          "text": "Reviewing findings & planning the next round…"})
                angles = self._replan(manager, topic, all_findings, K, rnd, emit=mgr_emit)
                if not angles:
                    run.emit({"type": "text", "role": "manager",
                              "text": "Manager judged coverage sufficient — stopping early."})
                    break
            ck()
            run.status = "reviewing"
            run.emit({"type": "text", "role": "manager", "text": "Final synthesis across all rounds…"})
            run.final = self._synthesize(run.manager_spec, topic, all_findings, emit=mgr_emit)
            run.report_file = _save_report(run)
            run.status = "done"
        except _Cancelled:
            run.emit({"type": "text", "role": "manager", "text": "Stopped by user."})
            run.status = "cancelled"
        except Exception as exc:  # noqa: BLE001
            run.error = f"{type(exc).__name__}: {exc}"
            run.emit({"type": "error", "text": run.error})
            run.status = "error"
        run.ended = time.time()

    def get(self, run_id: str) -> CrewRun | None:
        return self.runs.get(run_id)

    def list(self) -> list:
        # Incognito assistant runs are hidden from the runs list (leave no trace).
        return [r.to_dict() for r in sorted(self.runs.values(),
                                            key=lambda r: r.created, reverse=True)
                if not getattr(r, "incognito", False)]

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
            self._verify_completeness(run)
            if run._cancel:
                return self._finish(run, "cancelled")
            self._coverage_review(run)
            if run._cancel:
                return self._finish(run, "cancelled")
            self._review(run)
            self._finish(run, "done")
        except Exception as exc:  # noqa: BLE001
            run.error = f"{type(exc).__name__}: {exc}"
            run.emit({"type": "error", "text": run.error})
            self._finish(run, "error")

    def _drive_loop(self, run: CrewRun) -> None:
        """Looped AUTONOMOUS coder: plan → build (test-gated) → the manager reviews the
        results + workspace and re-plans the next round (fix failures, fill gaps) in the
        SAME folder, until all subtasks pass or the round cap. Auto-approved end-to-end.
        This is the 'builds good things on its own' prototype."""
        try:
            rounds = max(1, run.code_rounds)
            prev_fail = None   # stasis tracking: set of still-failing subtasks last round
            for rnd in range(rounds):
                run.emit({"type": "phase", "role": "manager", "round": rnd,
                          "text": f"round {rnd + 1}/{rounds}"})
                if run.support_researchers:
                    self._gather_support(run, rnd)   # sub-agents research → run.research_notes
                if run._cancel:
                    return self._finish(run, "cancelled")
                if rnd == 0:
                    self._plan(run)
                elif not self._replan_code(run, rnd):
                    run.emit({"type": "phase", "role": "manager",
                              "text": "manager: goal complete — finishing"})
                    break
                if run._cancel:
                    return self._finish(run, "cancelled")
                self._work(run)
                if run._cancel:
                    return self._finish(run, "cancelled")
                self._verify_completeness(run)
                if run._cancel:
                    return self._finish(run, "cancelled")
                if run.workers and all(w.status == "done" for w in run.workers):
                    run.emit({"type": "phase", "role": "manager",
                              "text": "all subtasks passed — finishing"})
                    break
                # STASIS: if a round ends with the EXACT same set of failing subtasks as
                # the previous round, the loop isn't making progress — stop early instead
                # of burning identical rounds (a worse model can spin here forever).
                fail_sig = frozenset((w.title, w.status) for w in run.workers if w.status != "done")
                if fail_sig and fail_sig == prev_fail and rnd < rounds - 1:
                    run.emit({"type": "phase", "role": "manager",
                              "text": "no progress since last round (identical failures) — stopping early"})
                    break
                prev_fail = fail_sig
            self._coverage_review(run)
            self._review(run)
            self._finish(run, "done")
        except Exception as exc:  # noqa: BLE001
            run.error = f"{type(exc).__name__}: {exc}"
            run.emit({"type": "error", "text": run.error})
            self._finish(run, "error")

    def _replan_code(self, run: CrewRun, rnd: int) -> bool:
        """Manager reviews the prior round's outcomes + the workspace and produces the
        NEXT round's plan, or returns False if the goal is complete (replies DONE)."""
        lines = []
        for w in run.workers:
            gate = "PASS" if w.gate_passed is True else ("FAIL" if w.gate_passed is False else "manual/none")
            # Structured failure context so the manager re-plans against the REAL error,
            # not a 160-char truncation: name the check that ran + give fuller output.
            acc = (w.acceptance if isinstance(w.acceptance, str)
                   else "embedded tests" if w.acceptance is None else str(w.acceptance))
            entry = f"- [{w.status}] {w.title} — gate {gate}"
            if w.status != "done":
                entry += f"\n    check: {acc}\n    output: {(w.output or '').strip()[:400] or '(none)'}"
            else:
                entry += f"; {(w.output or '')[:120]}"
            lines.append(entry)
        prev = "\n".join(lines) or "(nothing built yet)"
        mgr = agents.make_agent(run.manager_spec,
                                tool_names=run.manager_tools or _READONLY,
                                max_steps=8, cwd=run.cwd)
        prompt = (f"GOAL:\n{run.goal}\n\nThis is round {rnd + 1} of an iterative build. "
                  f"Workspace: {run.cwd}\nPrior subtask results:\n{prev}\n\n"
                  f"Inspect the existing files (list_dir / read_file) as needed. If the GOAL is now "
                  f"fully met and tests pass, reply with EXACTLY the single word DONE. Otherwise produce "
                  f"the NEXT plan — fix the failures and fill the gaps — in the SAME format (contract + "
                  f"subtasks, each with owns + a runnable acceptance). Build on the existing files; do "
                  f"NOT redo work that already passed."
                  + (f"\n\nCURRENT RESEARCH (fresh, use it):\n{run.research_notes}" if run.research_notes else ""))
        reply = mgr.run_task(prompt, system=_PLAN_SYS.format(maxw=run.max_workers),
                             on_event=lambda e: run.emit({**e, "role": "manager"}))
        if reply and re.search(r"\bDONE\b", reply) and len(reply.strip()) < 14:
            return False
        plan, contract = _parse_plan(reply, run.goal, run.max_workers)
        if not plan:
            return False
        run.plan, run.contract = plan, contract
        run.workers = [Worker(id=i, title=p["title"], detail=p["detail"],
                              acceptance=p.get("acceptance"),
                              check_rationale=p.get("check_rationale", ""),
                              owns=p.get("owns", []))
                       for i, p in enumerate(plan)]
        run.emit({"type": "plan", "plan": run.plan, "contract": run.contract, "round": rnd})
        return True

    def _gather_support(self, run: CrewRun, rnd: int) -> None:
        """Research-augmented build: sub-agent researchers gather current info for this
        round (round 0 = the goal; later rounds = the open gaps/failures), and the joined
        findings are stashed on run.research_notes → injected into the manager's plan AND
        the builders' context so they work from fresh info, not just memory."""
        import concurrent.futures as cf
        n = run.support_researchers
        spec = run.support_spec or run.worker_spec
        mgr_emit = lambda e: run.emit({**e, "role": "manager"})
        run.emit({"type": "phase", "role": "manager", "round": rnd,
                  "text": f"researching ({n} agents) to inform the build…"})
        focus = run.goal
        if rnd > 0:
            gaps = "; ".join(f"{w.title} [{w.status}]" for w in run.workers)
            focus = f"{run.goal}\n\nOpen gaps/failures to research: {gaps}"
        mgr = self._research_agent(run.manager_spec, max_steps=4, role="manager") \
            if run.manager_spec.startswith("claude:") else agents.make_agent(
                run.manager_spec, tool_names=_READONLY, max_steps=4)
        try:
            angles = self._split(mgr, focus, n, "concrete technical questions worth researching to build this well", emit=mgr_emit)
        except Exception:  # noqa: BLE001
            angles = [run.goal]
        findings = []
        try:
            with cf.ThreadPoolExecutor(max_workers=min(8, len(angles))) as ex:
                futs = [ex.submit(self._research_one, run, spec, run.goal, ang, 5000 + rnd * 100 + i, rnd=rnd)
                        for i, ang in enumerate(angles)]
                for f in cf.as_completed(futs):
                    if run._cancel:
                        break
                    findings.append(f.result())
        except Exception as exc:  # noqa: BLE001
            run.emit({"type": "error", "text": f"support research: {exc}"})
        notes = "\n\n".join(f"### {a}\n{str(t)[:600]}" for a, t in findings if t)
        run.research_notes = notes[:6000]
        run.emit({"type": "phase", "role": "manager",
                  "text": f"research done — {len(findings)} findings feeding the build"})

    def _approver(self, run: CrewRun, worker_id: int):
        def approve(name: str, args: dict):
            # Auto-approve mode: run unattended — grant without pausing. (The
            # tool_call is still emitted by the agent loop, so it stays visible.)
            if run.auto_approve:
                run.emit({"type": "auto_approved", "tool": name,
                          "worker_id": worker_id})
                return True, "auto-approved"
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
        task = run.goal + (f"\n\nCURRENT RESEARCH (fresh, gathered for this build — use it):\n{run.research_notes}"
                           if run.research_notes else "")
        reply = mgr.run_task(task, system=sys_prompt,
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
        if run.research_notes:
            context += ("\n\nRESEARCH (current info the team gathered for this build — "
                        "prefer it over your own assumptions):\n" + run.research_notes[:3000])
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
        prev_gate = None   # for stasis detection (#6): stop retrying on identical failures
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
            # STASIS (#6): the SAME failing gate output twice means the worker is stuck
            # in a loop — retrying again just burns tokens. Stop and let it fail/escalate.
            go = (gate_out or "").strip()
            if attempt > 1 and go and go == prev_gate:
                run.emit({"type": "phase", "worker_id": w.id,
                          "text": f"worker {w.id} stalled — identical failure twice; stopping retries"})
                break
            prev_gate = go

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

    def _verify_completeness(self, run: CrewRun) -> None:
        """COMPLETENESS PASS (local, honest): a subtask cannot be 'passed' if a
        deliverable it DECLARED (via `owns`) is missing, or an owned/expected test
        was never RUN by a gate. Such a subtask is flagged UNVERIFIED (not passed),
        and a local retry first TRIES to write the missing test. Conservative — it
        only checks what subtasks declared they'd produce; never invents requirements."""
        run.emit({"type": "phase", "text": "completeness check"})
        if not run.cwd:
            return
        for w in run.workers:
            if run._cancel:
                return
            if w.status in ("failed", "error"):
                continue   # already worse than unverified — leave it
            issues = self._check_one(run, w)
            if issues and not run._cancel:
                self._retry_for_completeness(run, w, issues)   # local fix attempt
                issues = self._check_one(run, w)               # re-check new files
            if issues:
                w.incomplete_reason = "; ".join(issues)
                if w.status == "done":
                    w.status = "unverified"
                log.warning("completeness: subtask %d '%s' UNVERIFIED — %s",
                            w.id, w.title, w.incomplete_reason)
                run.emit({"type": "completeness", "worker_id": w.id,
                          "status": "unverified", "reason": w.incomplete_reason})
            elif w.incomplete_reason:
                w.incomplete_reason = ""   # a retry resolved it
                run.emit({"type": "completeness", "worker_id": w.id, "status": "resolved"})

    def _check_one(self, run: CrewRun, w: "Worker") -> list:
        """Fresh completeness check for one subtask (re-reads disk + gate state)."""
        try:
            all_files = _all_rel_files(run.cwd)
        except OSError:
            return []
        ran_tests, whole_suite = _tests_that_ran(run)

        def ran(basenames) -> bool:
            return whole_suite or any(b in ran_tests for b in basenames)

        return self._completeness_issues(run, w, all_files, ran)

    def _retry_for_completeness(self, run: CrewRun, w: "Worker", issues: list) -> None:
        """LOCAL retry (worker fixes its own gap — never Opus): re-dispatch the
        worker to WRITE the missing test, then gate it. Bounded by MAX_REPAIR; on
        give-up the caller leaves the subtask UNVERIFIED. Reuses the same agent /
        approver / gate / allowlist machinery as the work loop."""
        steps = _STEP_BUDGET.get(run.complexity, 10)
        owns = list(w.owns or [])
        extra = []
        for rel in list(owns):
            base = Path(rel).name
            if base.endswith(".py") and not _is_test_name(base) and base != "__init__.py":
                mod, d = base[:-3], Path(rel).parent.as_posix()
                for cand in ({f"test_{mod}.py", f"tests/test_{mod}.py"}
                             | ({f"{d}/test_{mod}.py"} if d not in ("", ".") else set())):
                    if cand not in owns and cand not in extra:
                        extra.append(cand)
        retry_owns = owns + extra
        base_tools = (run.worker_tools if run.worker_tools is not None
                      else [n for n in toolmod.tool_names() if n != "launch_crew"])
        for attempt in range(1, MAX_REPAIR + 1):
            if run._cancel:
                return
            run.emit({"type": "phase", "worker_id": w.id,
                      "text": f"worker {w.id} · {w.title} · completeness retry "
                              f"{attempt}/{MAX_REPAIR} (write the missing test)"})
            agent = agents.make_agent(run.worker_spec, max_steps=steps, cwd=run.cwd,
                                      use_mcp=run.worker_use_mcp, tool_names=base_tools,
                                      owns=retry_owns)
            task = (f"Subtask: {w.title}\n\n{w.detail}\n\n"
                    f"COMPLETENESS PROBLEM: {'; '.join(issues)}.\n"
                    f"The module(s) you delivered have NO real test that verifies them. "
                    f"Write a pytest test FILE (named test_<module>.py) with concrete "
                    f"`assert` statements exercising the behavior — INCLUDING the edge / "
                    f"invalid cases the goal requires — and make sure it passes. You may "
                    f"create these files: {', '.join(retry_owns)}.\n"
                    f"Make the LAST line of your reply EXACTLY:\n"
                    f"ACCEPTANCE_CMD: pytest <your_test_file> -q\n"
                    f"It must be a single pytest/unittest command — no shell chaining, "
                    f"redirects, or `python -c` smoke tests.")
            try:
                out = agent.run_task(
                    task, system=_WORKER_SYS, context=f"Overall goal:\n{run.goal}",
                    on_event=lambda e, wid=w.id: run.emit({**e, "role": "worker", "worker_id": wid}),
                    approver=self._approver(run, w.id))
            except Exception as exc:  # noqa: BLE001
                run.emit({"type": "error", "worker_id": w.id, "text": f"(retry error: {exc})"})
                continue
            w.output = out
            derived = _extract_acceptance_cmd(out)
            if derived and _is_allowed_acceptance_cmd(derived):
                gate_cmd = derived
            elif isinstance(w.acceptance, str):
                gate_cmd = w.acceptance
            else:
                continue
            passed, gout = gate.run_gate(gate_cmd, run.cwd)
            run.emit({"type": "gate", "worker_id": w.id, "phase": "completeness",
                      "passed": passed, "output": gout[:600]})
            if passed:
                w.acceptance = gate_cmd
                w.gate_passed = True
                w.gate_output = gout
                w.attempts += 1
                return

    def _completeness_issues(self, run: CrewRun, w: "Worker", all_files, ran) -> list:
        """The deliverable gaps for one subtask (empty list => complete)."""
        issues, owns = [], (w.owns or [])
        # (A) every DECLARED deliverable must exist; declared TESTS must also have run.
        for rel in owns:
            base = Path(rel).name
            if not (Path(run.cwd) / rel).exists():
                kind = "test" if _is_test_name(base) else "file"
                issues.append(f"declared {kind} '{rel}' was never created")
            elif _is_test_name(base) and not ran({base}):
                issues.append(f"test '{rel}' exists but no gate ran it")
        # (B) an owned MODULE with no declared test of its own — and not self-verified
        #     by an embedded pytest — must have SOME corresponding test that ran.
        declared_test = any(_is_test_name(Path(r).name) for r in owns)
        embedded_ok = (isinstance(w.acceptance, dict)
                       and w.acceptance.get("type") == "pytest"
                       and (w.gate_passed is True or w.review_gate_passed is True))
        if not declared_test and not embedded_ok:
            for rel in owns:
                base = Path(rel).name
                if (not base.endswith(".py") or _is_test_name(base)
                        or base == "__init__.py"):
                    continue
                mod = base[:-3]
                cands = {f"test_{mod}.py", f"{mod}_test.py"}
                present = [f for f in all_files if Path(f).name in cands]
                if not present:
                    issues.append(f"module '{rel}' shipped with NO test")
                elif not ran({Path(f).name for f in present}):
                    issues.append(f"module '{rel}' has a test but no gate ran it")
        return issues

    def _coverage_review(self, run: CrewRun) -> None:
        """SPEC-COVERAGE REVIEW (best-effort, manager-driven): for each genuinely
        green subtask, have the manager critique whether the tests cover the spec's
        named cases. Missing cases trigger a local retry (add them). Degrades to a
        no-op when the manager can't critique (empty result) — NEVER fabricates a
        failure and NEVER turns into a fake pass."""
        if not run.coverage_review or not run.cwd:
            return
        run.emit({"type": "phase", "text": "spec-coverage review"})
        for w in run.workers:
            if run._cancel:
                return
            if w.status != "done":            # only critique REAL green subtasks
                continue
            mods = [r for r in (w.owns or [])
                    if r.endswith(".py") and not _is_test_name(Path(r).name)
                    and Path(r).name != "__init__.py" and (Path(run.cwd) / r).exists()]
            tests = [r for r in (w.owns or [])
                     if _is_test_name(Path(r).name) and (Path(run.cwd) / r).exists()]
            if not mods or not tests:
                continue
            missing = self._critique_coverage(run, w, mods, tests)
            if missing:
                w.coverage_missing = missing
                w.coverage_note = "missing spec cases: " + "; ".join(missing)
                log.warning("coverage: subtask %d '%s' missing spec cases: %s",
                            w.id, w.title, missing)
                run.emit({"type": "coverage", "worker_id": w.id, "status": "missing",
                          "missing": missing})
                self._retry_for_coverage(run, w, missing)
            else:
                w.coverage_note = "coverage looks complete"
                run.emit({"type": "coverage", "worker_id": w.id, "status": "complete",
                          "missing": []})

    def _critique_coverage(self, run: CrewRun, w: "Worker", mods: list, tests: list) -> list:
        """Ask the manager which spec-required cases the tests omit. Conservative;
        returns [] on parse failure / weak critique (=> no-op)."""
        src = ""
        for f in mods + tests:
            try:
                src += f"\n----- {f} -----\n{(Path(run.cwd) / f).read_text(encoding='utf-8', errors='replace')[:4000]}\n"
            except OSError:
                continue
        mgr = agents.make_agent(run.manager_spec,
                                tool_names=run.manager_tools or _READONLY,
                                max_steps=4, cwd=run.cwd)
        prompt = (f"GOAL:\n{run.goal}\n\nSUBTASK: {w.title}\n{w.detail}\n\n"
                  f"FILES (module source + its tests):\n{src}\n\n"
                  f"List spec-required cases the tests do NOT assert (conservative).")
        try:
            reply = mgr.run_task(prompt, system=_COVERAGE_SYS,
                                 on_event=lambda e: run.emit({**e, "role": "manager"}))
        except Exception as exc:  # noqa: BLE001 — critique failure must not break the run
            log.warning("coverage critique for subtask %d errored: %s", w.id, exc)
            return []
        m = re.search(r"\{.*\}", reply or "", re.S)
        if not m:
            return []
        data = _loads_lenient(m.group(0))
        if not isinstance(data, dict):
            return []
        missing = data.get("missing") or []
        if not isinstance(missing, list):
            return []
        return [str(x).strip() for x in missing if str(x).strip()][:8]

    def _retry_for_coverage(self, run: CrewRun, w: "Worker", missing: list) -> None:
        """LOCAL retry: add the missing spec cases to the TEST file(s) only — the
        worker may NOT edit the module, so a genuine bug surfaces as a gate FAILURE
        (subtask becomes `failed` and names the real bug) instead of being silently
        patched. If the added assertions pass, coverage improved, stays done."""
        test_owns = [r for r in (w.owns or []) if _is_test_name(Path(r).name)
                     and (Path(run.cwd) / r).exists()]
        if not test_owns:
            return
        steps = _STEP_BUDGET.get(run.complexity, 10)
        cases = "; ".join(missing)
        for attempt in range(1, MAX_REPAIR + 1):
            if run._cancel:
                return
            run.emit({"type": "phase", "worker_id": w.id,
                      "text": f"worker {w.id} · {w.title} · coverage retry "
                              f"{attempt}/{MAX_REPAIR} (add missing spec cases)"})
            agent = agents.make_agent(run.worker_spec, max_steps=steps, cwd=run.cwd,
                                      use_mcp=run.worker_use_mcp,
                                      tool_names=[n for n in toolmod.tool_names() if n != "launch_crew"],
                                      owns=test_owns)   # TEST files only — module off-limits
            task = (f"Subtask: {w.title}\n\n{w.detail}\n\n"
                    f"Your tests PASS but do NOT cover these spec-required cases: {cases}.\n"
                    f"Add a pytest assertion for EACH missing case to the test file(s): "
                    f"{', '.join(test_owns)}. Do NOT modify the module under test — only the "
                    f"test file(s) (writes elsewhere are refused). Then make the LAST line of "
                    f"your reply EXACTLY:\nACCEPTANCE_CMD: pytest <your_test_file> -q\n"
                    f"a single pytest/unittest command — no shell chaining or `python -c` smoke tests.")
            try:
                out = agent.run_task(
                    task, system=_WORKER_SYS, context=f"Overall goal:\n{run.goal}",
                    on_event=lambda e, wid=w.id: run.emit({**e, "role": "worker", "worker_id": wid}),
                    approver=self._approver(run, w.id))
            except Exception as exc:  # noqa: BLE001
                run.emit({"type": "error", "worker_id": w.id, "text": f"(coverage retry error: {exc})"})
                continue
            w.output = out
            derived = _extract_acceptance_cmd(out)
            if derived and _is_allowed_acceptance_cmd(derived):
                gate_cmd = derived
            elif isinstance(w.acceptance, str):
                gate_cmd = w.acceptance
            else:
                continue
            passed, gout = gate.run_gate(gate_cmd, run.cwd)
            run.emit({"type": "gate", "worker_id": w.id, "phase": "coverage",
                      "passed": passed, "output": gout[:600]})
            w.acceptance = gate_cmd
            w.gate_output = gout
            if passed:
                w.gate_passed = True
                w.coverage_note = "coverage gap fixed (added + passing): " + cases
                run.emit({"type": "coverage", "worker_id": w.id, "status": "fixed",
                          "missing": missing})
            else:
                w.gate_passed = False
                w.status = "failed"
                w.coverage_note = "REAL BUG surfaced by added coverage: " + cases
                log.warning("coverage: subtask %d FAILED — real bug on spec case(s): %s",
                            w.id, cases)
                run.emit({"type": "coverage", "worker_id": w.id, "status": "bug-surfaced",
                          "missing": missing})
            return

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
            if w.incomplete_reason:   # completeness pass: a required deliverable is missing
                problems.append(f"Subtask {w.id} '{w.title}' UNVERIFIED — not everything it "
                                f"was responsible for is verified: {w.incomplete_reason}.")
                line += f"\ncompleteness: UNVERIFIED — {w.incomplete_reason}"
            if w.coverage_note:       # spec-coverage review outcome
                line += f"\ncoverage review: {w.coverage_note}"
                if "REAL BUG" in w.coverage_note:
                    problems.append(f"Subtask {w.id} '{w.title}' — spec-coverage review caught a "
                                    f"REAL BUG: {w.coverage_note.split(': ',1)[-1]} "
                                    f"(a spec case the original tests omitted).")
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
