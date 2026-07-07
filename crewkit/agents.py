"""Agent backends.

An *agent* takes a task (plus optional shared context) and returns a final text
result, autonomously calling tools along the way. Two backends share one
interface (`run_task`) so the crew orchestrator can mix them freely:

  * **OllamaAgent** — a local model (e.g. qwen3-coder). We drive the
    tool-calling loop ourselves against Ollama's /api/chat, executing tools from
    `agent.tools`. Danger tools (shell, file writes, panel actions) are routed
    through an `approver` callback so the UI can require the user's OK.
  * **ClaudeAgent** — a real Claude Code session via claude-agent-sdk (the same
    engine the Fleet section uses). It brings its *own* tools and permission
    system, so it runs the task as a one-shot Claude Code agent. Imported lazily
    so the panel still works when the SDK isn't installed.

Events (tool calls, results, text, approvals) are emitted via an `on_event`
callback so a run can be streamed to the UI.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.request
from typing import Callable

from . import tools as toolmod
from .chat import OLLAMA

log = logging.getLogger("crew.agent")

# Workers/managers run Ollama at this context window. We PIN it (rather than rely
# on Ollama's default) so the compaction threshold below can't silently desync —
# a drifting default is exactly how the "worker forgets its task" overflow happens.
# 16384: the KV cache for a 30B at 32K is several GB of VRAM on top of the ~17GB
# weights — that overhead (not the weights) is why qwen3-coder sits at ~20-21GB,
# not 17. Halving the context roughly halves the KV cache, freeing VRAM headroom
# (e.g. to keep a small 3-4B researcher model resident alongside the 30B). The
# deep-research run found agent prompts should stay <5K and quality decays well
# before 32K, so 16K is ample here. Reversible — bump back up for big-file coding.
# Bigger lever still: set OLLAMA_FLASH_ATTENTION=1 + OLLAMA_KV_CACHE_TYPE=q8_0 on
# the Ollama service to quantize the KV cache (~half the VRAM, near-lossless).
#
# TIERED CONTEXT — smaller ctx = much faster (smaller KV cache + far less prompt
# to process; a 4K window benches ~3x the tok/s of 16K). Size the window to the
# role. The research swarm already tiers these (see crew._research_agent budgets:
# researcher 4K / sub-manager 8K / manager 16K). The CODER path uses
# WORKER_NUM_CTX (the make_agent default) for BOTH its manager and workers, kept
# EQUAL so the single 30B never reloads between plan<->work phases (a ctx change
# forces a ~3-6s model reload). 8K is ample for these app builds; compaction
# (below) handles overflow. 16K stays reserved for the research manager only.
CTX_SMALL = 4096      # researchers, quick judges, chat — short & focused
CTX_MEDIUM = 8192     # coder manager + workers, sub-managers — code + a few files
CTX_LARGE = 16384     # research manager / heavy reasoning over many findings
WORKER_NUM_CTX = CTX_MEDIUM   # coder default (was 16384 → spilled to CPU on 24GB)
# Compact older turns once the estimated prompt size crosses ~70% of num_ctx,
# leaving headroom for the next model turn before Ollama would truncate the FRONT
# (which is where the system prompt + task live).
COMPACT_AT_TOKENS = int(0.70 * WORKER_NUM_CTX)  # ~5734
# Per-step OUTPUT cap. Ollama defaults to unbounded (-1), so a model stuck in a
# repetition loop can generate until num_ctx fills — pure wasted wall-clock. We
# cap it. NOTE: a worker writing a whole file emits the file's contents as the
# tool-call arguments, which COUNT as generated tokens — so this must be well
# above any legitimate single write. 8192 tok ≈ ~600 lines of code; a true
# runaway generates far more. (2000, the first suggestion, would truncate a
# moderate module mid-write into a broken tool call.)
WORKER_NUM_PREDICT = 8192
# Keep the model resident between steps/phases/runs so Ollama doesn't unload a
# ~19GB model and eat a multi-second reload on the next call.
WORKER_KEEP_ALIVE = "10m"


def _est_tokens(messages: list[dict]) -> int:
    """Cheap token estimate (chars/4) over content + serialized tool_calls. No
    tokenizer dependency."""
    total = 0
    for m in messages:
        total += len(m.get("content") or "")
        tc = m.get("tool_calls")
        if tc:
            total += len(json.dumps(tc))
    return total // 4


def _progress_note(middle: list[dict]) -> str:
    """Deterministically compress the older middle turns: keep file-WRITE records
    and shell/gate OUTCOMES (the load-bearing facts), trim verbose read/list dumps
    (re-fetchable on demand)."""
    lines = ["[Context compacted to save room — older steps are summarized below. "
             "Your system instructions and the original task above STILL APPLY. "
             "Continue from here.]", "PROGRESS SO FAR (older steps):"]
    for m in middle:
        role = m.get("role")
        c = (m.get("content") or "").strip()
        if role == "assistant":
            for tc in (m.get("tool_calls") or []):
                fn = tc.get("function") or {}
                lines.append(f"  · called {fn.get('name')}"
                             f"({json.dumps(fn.get('arguments'))[:80]})")
            if c:
                lines.append(f"  · note: {c[:160]}")
        elif role == "tool":
            if c.startswith("Wrote ") or c.startswith("exit_code:") or c.startswith("Launched "):
                lines.append(f"  -> {c[:220]}")          # action / outcome: keep
            elif c:
                first = c.splitlines()[0][:80]
                lines.append(f"  -> (output trimmed, {len(c)} chars; began: {first})")
    return "\n".join(lines[:100])


def _compact_messages(messages: list[dict]) -> "tuple[list[dict], bool]":
    """Replace the older middle of the message list with one compact progress
    note. ALWAYS keeps: [0] system prompt, [1] original task, and the last ~2
    assistant turns (current working state). Returns (messages, did_compact)."""
    if len(messages) < 6:
        return messages, False
    head, body = messages[:2], messages[2:]   # system + task are sacrosanct
    a_idx = [i for i, m in enumerate(body) if m.get("role") == "assistant"]
    if len(a_idx) < 3:
        return messages, False                 # not enough turns to bother
    keep_from = a_idx[-2]                       # keep last 2 assistant turns intact
    middle, tail = body[:keep_from], body[keep_from:]
    if not middle:
        return messages, False
    return head + [{"role": "user", "content": _progress_note(middle)}] + tail, True

# Some local models (notably the older abliterated qwen2.5-coder builds) don't
# emit Ollama's native `tool_calls` — they print the call into the text content in
# one of a few shapes. We parse those as a fallback so tool use works anyway.
# This protocol line nudges them toward the easiest shape to parse.
_TOOL_PROTOCOL = (
    "\n\nWhen you want to use a tool, output ONLY a single line of the form "
    '<tool_call>{"name": "<tool>", "arguments": {<json args>}}</tool_call> '
    "and nothing else. When you are completely finished, reply with your final "
    "answer as plain text (no tool_call)."
)


def _norm_call(name, args) -> "tuple[str, dict] | None":
    if not name:
        return None
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    return str(name), (args or {})


def _json_objects(s: str):
    """Yield every balanced, string-aware top-level {...} substring in s.
    Handles nested objects (e.g. an `arguments` object) and multiple calls."""
    i, n = 0, len(s)
    while i < n:
        if s[i] != "{":
            i += 1
            continue
        depth = 0
        instr = esc = False
        start = i
        while i < n:
            c = s[i]
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                instr = not instr
            elif not instr:
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        yield s[start:i + 1]
                        break
            i += 1
        i += 1


def extract_tool_calls(content: str, known: set[str]) -> list:
    """Recover tool calls a model printed into its text instead of returning
    them natively. Accepts {"name"/"tool", "arguments"/"args"} JSON (fenced,
    bare, or wrapped in <tool_call> tags), and a self-closing <tool attr=.../>
    form. Only names in `known` are accepted, to avoid treating example JSON in
    prose as a real call."""
    calls: list = []
    seen: set = set()
    for objstr in _json_objects(content):
        try:
            o = json.loads(objstr)
        except json.JSONDecodeError:
            continue
        if not isinstance(o, dict):
            continue
        nm = o.get("name") or o.get("tool")
        if nm not in known:
            continue
        args = o.get("arguments") if "arguments" in o else o.get("args")
        nc = _norm_call(nm, args)
        if not nc:
            continue
        key = (nc[0], json.dumps(nc[1], sort_keys=True, default=str))
        if key not in seen:
            seen.add(key)
            calls.append(nc)
    if calls:
        return calls
    # self-closing <toolname attr="v" .../> for a known tool
    for m in re.finditer(r"<([a-z_]+)((?:\s+\w+=\"[^\"]*\")+)\s*/?>", content):
        if m.group(1) in known:
            nc = _norm_call(m.group(1), dict(re.findall(r'(\w+)="([^"]*)"', m.group(2))))
            if nc:
                calls.append(nc)
    return calls

# Callbacks
EventFn = Callable[[dict], None]
# approver(tool_name, args) -> (approved: bool, note: str)
ApproveFn = Callable[[str, dict], "tuple[bool, str]"]


def _owns_ok(write_path: str, owns: list, cwd: str | None) -> "tuple[bool, str]":
    """Is `write_path` within this worker's owned files? Resolves both the target
    and each owned entry against `cwd` (matching tools' base_dir behavior). Allows
    an exact file match or a path under an owned directory."""
    if not write_path:
        return False, "no path given"
    base = cwd or os.getcwd()

    def norm(p: str) -> str:
        p = os.path.expanduser(p or "")
        if not os.path.isabs(p):
            p = os.path.join(base, p)
        return os.path.normcase(os.path.normpath(p))

    target = norm(write_path)
    for o in owns:
        owned = norm(o)
        if target == owned or target.startswith(owned + os.sep):
            return True, ""
    return False, (f"path '{write_path}' is NOT owned by this subtask. You may only "
                   f"write these files: {owns}. Do not modify files owned by other "
                   f"subtasks — put your work in your own files.")


def _noop_event(_e: dict) -> None:
    pass


def _auto_approve(_n: str, _a: dict) -> "tuple[bool, str]":
    return True, ""


def _ollama_chat(model: str, messages: list[dict], tools: list[dict] | None,
                 timeout: float = 600.0, num_ctx: int | None = None,
                 keep_alive: str | None = None, on_delta=None) -> dict:
    # Pin num_ctx so workers run at a KNOWN context window that matches the
    # compaction threshold (see WORKER_NUM_CTX). Per-role overrides allowed.
    #
    # on_delta(tokens_so_far, rate) — when given, we STREAM the response (NDJSON)
    # and call it every few tokens so the UI shows a LIVE tok/s during generation
    # instead of one frozen number per call. We reassemble the streamed chunks
    # into the same dict shape the non-streaming call returns (message + the
    # final eval_count/prompt_eval_count/eval_duration totals), so the caller's
    # accounting + tool-call handling is unchanged.
    stream = on_delta is not None
    body = {"model": model, "messages": messages, "stream": stream,
            "keep_alive": keep_alive or WORKER_KEEP_ALIVE,
            "options": {"num_ctx": num_ctx or WORKER_NUM_CTX,
                        "num_predict": WORKER_NUM_PREDICT}}
    if tools:
        body["tools"] = tools
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(OLLAMA + "/api/chat", data=data,
                                 headers={"Content-Type": "application/json"})
    if not stream:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    # Streaming: read newline-delimited JSON objects, accumulate content +
    # tool_calls, and emit a live rate as tokens arrive. Each content chunk is
    # ~one token, so the chunk count is a good live proxy until the final object
    # carries the exact eval_count.
    parts: list[str] = []
    tool_calls: list = []
    final: dict = {}
    n = 0
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:  # iterates the response body line by line
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw.decode("utf-8"))
            except Exception:  # noqa: BLE001 — skip a partial/garbled line
                continue
            m = obj.get("message") or {}
            c = m.get("content") or ""
            if c:
                parts.append(c)
                n += 1
                if n % 6 == 0:  # throttle UI events (~every 6 tokens)
                    el = time.monotonic() - t0
                    on_delta(n, round(n / el, 1) if el > 0 else None)
            if m.get("tool_calls"):
                tool_calls.extend(m["tool_calls"])
            if obj.get("done"):
                final = obj
    # Reassemble into the non-streaming shape.
    msg = dict(final.get("message") or {})
    msg["role"] = msg.get("role") or "assistant"
    msg["content"] = "".join(parts)
    if tool_calls and not msg.get("tool_calls"):
        msg["tool_calls"] = tool_calls
    final["message"] = msg
    return final


class OllamaAgent:
    """Local tool-calling model. We own the loop."""

    backend = "ollama"

    def __init__(self, model: str, tool_names: list[str] | None = None,
                 max_steps: int = 12, cwd: str | None = None, use_mcp: bool = False,
                 owns: list | None = None, num_ctx: int | None = None,
                 keep_alive: str | None = None, timeout: float | None = None):
        self.model = model
        self.tool_names = tool_names  # None = all tools
        self.max_steps = max_steps
        self.cwd = cwd
        self.use_mcp = use_mcp
        # Per-worker file ownership. Truthy => writes are scoped to these paths;
        # None/empty => enforcement OFF (assistant, review, back-compat plans).
        self.owns = owns
        # Per-ROLE tuning (deep-research Tier-1 #1/#5): a small researcher gets a
        # tiny context + short keep_alive (so it can't evict the resident manager)
        # + a tight timeout (the watchdog — a stuck leaf fails fast, never hangs the
        # swarm). Defaults preserve the old global behavior.
        self.num_ctx = num_ctx or WORKER_NUM_CTX
        self.keep_alive = keep_alive or WORKER_KEEP_ALIVE
        self.timeout = timeout or 600.0

    def run_task(self, task: str, *, system: str = "", context: str = "",
                 on_event: EventFn = _noop_event,
                 approver: ApproveFn = _auto_approve) -> str:
        schemas = toolmod.schemas(self.tool_names)
        known = set(self.tool_names) if self.tool_names else set(toolmod.tool_names())
        mcp_known: set = set()
        if self.use_mcp:
            try:
                from . import mcp_bridge
                schemas = schemas + mcp_bridge.schemas()
                mcp_known = set(mcp_bridge.tool_names())
                known |= mcp_known
            except Exception:  # noqa: BLE001 — MCP optional; never break a run
                pass
        messages: list[dict] = []
        sys_content = (system or "") + _TOOL_PROTOCOL
        if self.cwd:
            sys_content += (f"\n\nYour working folder is {self.cwd} . Put new files "
                            f"there (use absolute paths under it) and leave run_shell's "
                            f"cwd unset to run there.")
        messages.append({"role": "system", "content": sys_content})
        user = task if not context else f"{context}\n\n---\n\nTask:\n{task}"
        messages.append({"role": "user", "content": user})

        final_text = ""
        for step in range(self.max_steps):
            # Compact older turns before they push the prompt past num_ctx (where
            # Ollama would truncate the FRONT — the system prompt + task).
            est = _est_tokens(messages)
            if est > int(0.70 * self.num_ctx):   # threshold tracks THIS agent's num_ctx
                messages, did = _compact_messages(messages)
                if did:
                    after = _est_tokens(messages)
                    log.warning("WARN context compaction fired: ~%d -> ~%d tok "
                                "(kept system+task+last turns)", est, after)
                    on_event({"type": "compaction", "before": est, "after": after})
            # Live tok/s: stream the response and emit partial usage events as
            # tokens arrive. These carry tokens=0 (so the ledger — which skips
            # 0-token rows — never double-counts) and a climbing `total` estimate
            # (the frontend takes max(total)); the exact count is emitted once the
            # call completes (below).
            base_total = getattr(self, "_toks_total", 0)

            def _delta(nchunks: int, rate) -> None:
                on_event({"type": "usage", "agent": self.model, "tokens": 0,
                          "total": base_total + nchunks, "rate": rate, "partial": True})

            try:
                data = _ollama_chat(self.model, messages, schemas,
                                    timeout=self.timeout, num_ctx=self.num_ctx,
                                    keep_alive=self.keep_alive, on_delta=_delta)
            except Exception as exc:  # noqa: BLE001
                on_event({"type": "error", "text": f"{self.model}: {exc}"})
                return final_text or f"(model error: {exc})"
            msg = data.get("message", {}) or {}
            content = (msg.get("content") or "").strip()

            # Live token accounting. Ollama returns exact counts on the final
            # (non-streamed) object: prompt_eval_count (input) + eval_count
            # (generated) and eval_duration in NANOSECONDS, which is the honest
            # generation tok/s. Emit per-call so the tree can show a live rate.
            ec = int(data.get("eval_count") or 0)
            pe = int(data.get("prompt_eval_count") or 0)
            ed = int(data.get("eval_duration") or 0)  # ns, generation only
            if ec or pe:
                self._toks_total = getattr(self, "_toks_total", 0) + ec + pe
                rate = round(ec / (ed / 1e9), 1) if ed > 0 else None
                on_event({"type": "usage", "agent": self.model,
                          "tokens": ec + pe, "total": self._toks_total,
                          "rate": rate})

            # Prefer native tool_calls; fall back to parsing them from content.
            # Each entry is (name, args, call_id); call_id is the native tool_call
            # id (carried onto the tool result for OpenAI-compatible endpoints) or
            # None for the text-extracted fallback (no id to fabricate).
            calls: list = []
            for c in (msg.get("tool_calls") or []):
                fn = c.get("function") or {}
                nc = _norm_call(fn.get("name"), fn.get("arguments"))
                if nc:
                    calls.append((nc[0], nc[1], c.get("id")))
            if not calls and content:
                calls = [(n, a, None) for (n, a) in extract_tool_calls(content, known)]

            messages.append({"role": "assistant", "content": msg.get("content") or "",
                             "tool_calls": msg.get("tool_calls") or []})

            if not calls:
                if content:
                    final_text = content
                    on_event({"type": "text", "agent": self.model, "text": content})
                return final_text  # model is done

            for name, args, call_id in calls:
                # Tool result message, tagged with the native call id when we have
                # one (stricter OpenAI-compatible endpoints associate by id).
                def tool_msg(content: str) -> dict:
                    m = {"role": "tool", "content": content}
                    if call_id:
                        m["tool_call_id"] = call_id
                    return m
                on_event({"type": "tool_call", "agent": self.model,
                          "tool": name, "args": args})
                # Per-worker file-ownership enforcement (OFF when self.owns is
                # falsy). Scoped here in the worker loop — NOT global — so the
                # assistant/review aren't affected. Reads are never scoped; only
                # WRITE tools. Refuse before the approver so no prompt even fires.
                if self.owns and name in toolmod.WRITE_TOOLS:
                    ok, why = _owns_ok(args.get("path"), self.owns, self.cwd)
                    if not ok:
                        on_event({"type": "ownership_refused", "tool": name,
                                  "path": args.get("path"), "owns": self.owns})
                        messages.append(tool_msg("REFUSED: " + why))
                        continue
                is_mcp = name in mcp_known or name.startswith("mcp__")
                if is_mcp or toolmod.is_danger(name):  # MCP tools are always gated
                    approved, note = approver(name, args)
                    if not approved:
                        result = f"DENIED by user. {note}".strip()
                        on_event({"type": "tool_denied", "tool": name, "note": note})
                        messages.append(tool_msg(result))
                        continue
                if is_mcp:
                    from . import mcp_bridge
                    result = mcp_bridge.execute(name, args)
                else:
                    result = toolmod.execute(name, args, base_dir=self.cwd)
                on_event({"type": "tool_result", "tool": name,
                          "result": result[:600]})
                messages.append(tool_msg(result))

        on_event({"type": "text", "agent": self.model,
                  "text": "(stopped: hit max tool steps)"})
        return final_text or "(no final answer — hit max tool steps)"


# can_use_tool fail-safe: auto-allow ONLY clearly read-only SDK tools; gate
# EVERYTHING else — writes, Bash, network, MCP, and any unknown/new tool. So an
# unrecognized tool defaults to GATED (routes to the approver), never allowed.
_READ_ONLY_TOOLS = {"Read", "Glob", "Grep", "TodoWrite"}


def _opus_tool_gated(name: str) -> bool:
    """True => route through the crew approver; False => auto-allow (read-only)."""
    return name not in _READ_ONLY_TOOLS


# Panel-unique tools worth surfacing to a Claude agent. Claude already has native
# Read/Write/Bash/Glob/Grep/WebSearch/WebFetch, so we skip those duplicates and
# expose only what this panel provides and Claude otherwise can't reach.
_PANEL_TOOLS_FOR_CLAUDE = ["system_stats", "run_action", "launch_crew",
                           "save_note", "read_notes", "get_time"]
_PANEL_MCP_PREFIX = "mcp__panel__"


def _panel_mcp_server(cwd: str | None, tool_names: list[str]):
    """In-process SDK MCP server that re-exposes the panel's own tools to a
    Claude-backed agent. Without this, a Claude assistant only sees Claude Code's
    native tools and none of the panel's (system_stats, run_action, launch_crew,
    …) — which is exactly why it reported the tools "aren't wired up". Each tool
    just forwards to toolmod.execute on a worker thread (it's blocking)."""
    from claude_agent_sdk import create_sdk_mcp_server, tool
    sdk_tools = []
    for name in tool_names:
        t = toolmod.get_tool(name)
        if not t:
            continue

        def _make(tname: str):
            async def _handler(args):
                import asyncio
                out = await asyncio.to_thread(toolmod.execute, tname, args or {}, cwd)
                return {"content": [{"type": "text", "text": out}]}
            return _handler

        sdk_tools.append(tool(t.name, t.description, t.parameters)(_make(t.name)))
    return create_sdk_mcp_server("panel", tools=sdk_tools)


class ClaudeAgent:
    """A real Claude Code session via claude-agent-sdk. Brings its own tools."""

    backend = "claude"

    def __init__(self, model: str = "sonnet", max_turns: int = 20,
                 cwd: str | None = None, disallowed_tools: list | None = None):
        self.model = model        # "claude-opus-4-8" / "sonnet" / "haiku" / full id
        self.max_turns = max_turns
        self.cwd = cwd
        # e.g. block Bash/Write/Edit for read-only research workers.
        self.disallowed_tools = disallowed_tools

    def run_task(self, task: str, *, system: str = "", context: str = "",
                 on_event: EventFn = _noop_event,
                 approver: ApproveFn = _auto_approve) -> str:
        # Imported here so the panel runs without the SDK; mirrors agent/main.py.
        import asyncio

        async def _go() -> str:
            from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

            from claude_agent_sdk import (PermissionResultAllow,
                                          PermissionResultDeny)
            opts = ClaudeAgentOptions()
            if self.model:
                opts.model = self.model
            if self.disallowed_tools:
                opts.disallowed_tools = list(self.disallowed_tools)
            if self.cwd:
                opts.cwd = self.cwd
            # Cap autonomous turns so an escalated/assistant Opus session can't run
            # (and bill) unbounded if it loops. Without this the SDK default is
            # uncapped — exactly the runaway we don't want on a paid agent.
            if self.max_turns:
                opts.max_turns = self.max_turns
            # Headless: there's no interactive prompt, so the SDK's default mode
            # silently blocks writes/Bash (Opus narrates "needs permission" but
            # nothing happens). Instead of a blanket bypass, route Opus's tool use
            # through the SAME crew approver local workers use. FAIL-SAFE direction:
            # only clearly read-only tools auto-allow (_READ_ONLY_TOOLS); everything
            # else — writes, Bash, network, MCP, and any unknown/new tool — is gated.
            async def _can_use_tool(name, tool_input, ctx):
                # Panel tools (mcp__panel__*): gate only the danger ones; read-only
                # vitals/notes/clock auto-allow so the assistant can read freely.
                if name.startswith(_PANEL_MCP_PREFIX):
                    bare = name[len(_PANEL_MCP_PREFIX):]
                    if not toolmod.is_danger(bare):
                        return PermissionResultAllow()
                elif not _opus_tool_gated(name):
                    return PermissionResultAllow()
                approved, note = await asyncio.to_thread(approver, name, tool_input)
                if approved:
                    return PermissionResultAllow()
                return PermissionResultDeny(message=note or "denied by approver")

            opts.can_use_tool = _can_use_tool
            # Give Claude the panel's own tools (in-process SDK MCP server) PLUS the
            # same external MCP servers the local agents use. Without the panel
            # server, a Claude assistant has no system_stats/run_action/launch_crew.
            servers: dict = {}
            try:
                from . import mcp_bridge
                servers.update(mcp_bridge.claude_mcp_config() or {})
            except Exception:  # noqa: BLE001
                pass
            try:
                servers["panel"] = _panel_mcp_server(self.cwd, _PANEL_TOOLS_FOR_CLAUDE)
            except Exception as exc:  # noqa: BLE001 — never block the run on this
                on_event({"type": "error", "text": f"panel tools unavailable: {exc!r}"})
            if servers:
                opts.mcp_servers = servers
            prompt = task if not context else f"{context}\n\n---\n\nTask:\n{task}"
            if system:
                prompt = f"{system}\n\n{prompt}"
            chunks: list[str] = []
            total_toks = 0
            out_toks = 0
            t0 = time.monotonic()
            last_t = t0          # wall time of the previous usage event
            last_out = 0         # out_toks at the previous usage event
            seen_ids: set = set()
            tag = f"claude:{self.model}"
            async with ClaudeSDKClient(options=opts) as client:
                await client.query(prompt)
                async for message in client.receive_response():
                    # surface tool use (so a watcher can see "what it's doing")
                    for block in (getattr(message, "content", None) or []):
                        tname = getattr(block, "name", None)
                        if tname and getattr(block, "input", None) is not None:
                            on_event({"type": "tool", "agent": tag, "tool": tname})
                    text = _extract_text(message)
                    if text:
                        chunks.append(text)
                        on_event({"type": "text", "agent": tag, "text": text})
                    # token accounting (deduped by message id) — this is what was
                    # missing, so crew/research/assistant Claude usage never counted
                    u = getattr(message, "usage", None)
                    mid = getattr(message, "message_id", None)
                    if u and (mid is None or mid not in seen_ids):
                        if mid:
                            seen_ids.add(mid)
                        # Count only "new" tokens: fresh input + generated output +
                        # cache WRITES. We deliberately EXCLUDE cache_read_input_tokens
                        # — a Claude Code session re-reads its entire cached context
                        # (system prompt + tools + whole transcript) EVERY turn, so
                        # counting reads makes a multi-turn agent loop report millions
                        # of "tokens" for a few thousand tokens of actual work (and on
                        # a subscription those reads are ~free). Reads aren't work.
                        d = ((u.get("input_tokens") or 0) + (u.get("output_tokens") or 0)
                             + (u.get("cache_creation_input_tokens") or 0))
                        if d:
                            total_toks += d
                            out_toks += (u.get("output_tokens") or 0)
                            # INSTANTANEOUS rate: this turn's output / the wall time
                            # since the previous usage event — not a cumulative
                            # average (which idle/tool time between turns drags down
                            # and makes the number drift/freeze). Cache/input reads
                            # aren't "work", so rate is from OUTPUT only.
                            now = time.monotonic()
                            dt = now - last_t
                            dout = out_toks - last_out
                            rate = round(dout / dt, 1) if dt > 0 and dout else None
                            last_t, last_out = now, out_toks
                            on_event({"type": "usage", "agent": tag,
                                      "tokens": d, "total": total_toks,
                                      "rate": rate})
            # (Per-run token totals are recorded by crew.CrewRun.emit via crewkit.ledger;
            # the panel build additionally fed an all-time host meter here — dropped in the
            # standalone package, which has no such host surface.)
            return "\n".join(chunks).strip()

        try:
            return asyncio.run(_go())
        except Exception as exc:  # noqa: BLE001
            on_event({"type": "error", "text": f"claude:{self.model}: {exc}"})
            return f"(claude error: {exc})"


def _extract_text(message) -> str:
    """Pull plain text out of an SDK message (best-effort across versions)."""
    content = getattr(message, "content", None)
    if content is None:
        return ""
    out = []
    if isinstance(content, str):
        return content
    for block in content:
        t = getattr(block, "text", None)
        if t:
            out.append(t)
    return "".join(out)


def make_agent(spec: str, **kwargs):
    """Build an agent from a 'backend:model' spec, e.g.
       'ollama:qwen3-coder:30b' or 'claude:claude-opus-4-8'.
    """
    if spec.startswith("claude:"):
        return ClaudeAgent(spec.split(":", 1)[1], **{k: v for k, v in kwargs.items()
                                                     if k in ("max_turns", "cwd", "disallowed_tools")})
    model = spec.split(":", 1)[1] if spec.startswith("ollama:") else spec
    return OllamaAgent(model, **{k: v for k, v in kwargs.items()
                                 if k in ("tool_names", "max_steps", "cwd", "use_mcp",
                                          "owns", "num_ctx", "keep_alive", "timeout")})
