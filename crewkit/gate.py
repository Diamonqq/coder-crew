"""Acceptance-gate runner for the crew.

A subtask's `acceptance` is a RUNNABLE check (see crew._clean_acceptance):

  * None                                  -> manual review (always passes the gate)
  * "pytest tests/test_x.py -q"           -> shell command; pass = exit 0   (PRIMARY)
  * {"type": "shell", "cmd": "..."}       -> shell command; pass = exit 0
  * {"type": "pytest", "code": "..."}     -> write to a temp file, run pytest (FALLBACK)

The common, encouraged pattern is the worker writing a real test file into the run
folder and the acceptance being a one-line command that runs it — clean JSON,
inspectable tests. Embedded pytest source is a rarely-used fallback.

Everything runs through `tools.shell_exec`, the SAME subprocess path `run_shell`
uses. NOTE: `shell_exec` is NOT sandboxed — `cwd` only sets the working directory,
it does not confine the command. Acceptance shell commands are therefore
ALLOWLISTED upstream (crew._clean_acceptance) to a single test-runner invocation
(pytest/unittest, no shell chaining); that allowlist — not any sandbox — is what
keeps a model-authored acceptance from being an arbitrary unapproved command.
`run_gate` never raises: any internal failure becomes `(False, "<reason>")` so the
caller's repair loop reacts instead of the run thread crashing.
"""
from __future__ import annotations

import logging
import sys
import uuid
from pathlib import Path

from . import tools

GATE_TIMEOUT = 120  # seconds
log = logging.getLogger("crew.gate")


def run_gate(acceptance, cwd: str | None) -> "tuple[bool, str]":
    """Run a subtask's acceptance check. Returns (passed, output). Never raises."""
    try:
        if acceptance is None:
            log.debug("gate: none (manual review)")
            return True, "no automated check — manual review"

        # Decide the branch. Embedded pytest = explicit dict, or a bare multi-line
        # string that actually looks like Python test source. (Upstream the
        # acceptance allowlist already rejects multi-line shell strings, so this
        # bare-string case is rare; we only treat it as pytest when it clearly is
        # Python — contains "def test" or "import" — never a random newline string.)
        pytest_code = None
        if isinstance(acceptance, dict) and acceptance.get("type") == "pytest":
            pytest_code = str(acceptance.get("code", ""))
        elif (isinstance(acceptance, str) and "\n" in acceptance.strip()
              and ("def test" in acceptance or "import" in acceptance)):
            pytest_code = acceptance

        if pytest_code is not None:
            return _gate_pytest(pytest_code, cwd)

        # Shell command (primary path).
        if isinstance(acceptance, str):
            cmd = acceptance.strip()
        elif isinstance(acceptance, dict):
            cmd = (acceptance.get("cmd") or acceptance.get("command") or "").strip()
        else:
            cmd = ""
        if not cmd:
            return False, f"gate: unrunnable acceptance {acceptance!r}"
        return _gate_shell(cmd, cwd)
    except Exception as exc:  # noqa: BLE001 — a gate must never crash the run thread
        log.debug("gate: internal error: %s", exc)
        return False, f"gate error: {type(exc).__name__}: {exc}"


def _gate_shell(cmd: str, cwd: str | None) -> "tuple[bool, str]":
    log.debug("gate: shell -> %s", cmd)
    rc, output, timed_out = tools.shell_exec(cmd, cwd=cwd, timeout=GATE_TIMEOUT)
    if timed_out:
        return False, f"gate timed out after {GATE_TIMEOUT}s"
    if rc is None:
        return False, tools._clip(output)  # spawn failure, message in output
    return rc == 0, tools._clip(output)


def _gate_pytest(code: str, cwd: str | None) -> "tuple[bool, str]":
    log.debug("gate: pytest-fallback (%d chars)", len(code or ""))
    if not cwd:
        return False, "gate: pytest fallback needs a working folder"
    # Name must match pytest's collection pattern AND be a valid module name —
    # a leading-dot name like "._gate_x.py" is hidden/uncollectible and errors on
    # import, so use "test_gate_<hex>.py".
    tmp = Path(cwd) / f"test_gate_{uuid.uuid4().hex[:8]}.py"
    try:
        try:
            tmp.write_text(code, encoding="utf-8")
        except OSError as exc:
            return False, f"gate: could not write temp test: {exc}"
        # Run via this interpreter's pytest so the venv's pytest is used; only the
        # temp file, with cwd set to the run folder (note: cwd is not a sandbox).
        cmd = f'"{sys.executable}" -m pytest -q "{tmp.name}"'
        rc, output, timed_out = tools.shell_exec(cmd, cwd=cwd, timeout=GATE_TIMEOUT)
        if timed_out:
            return False, f"gate timed out after {GATE_TIMEOUT}s"
        if rc is None:
            return False, tools._clip(output)
        return rc == 0, tools._clip(output)
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass
