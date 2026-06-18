"""coder-crew — a local-first agentic coding crew.

A manager model plans a goal into subtasks, worker models implement each one with
real tools (file + shell) gated behind your approval, every subtask is checked by
a runnable acceptance gate, and the manager reviews + integrates the result.

Local-first (Ollama). Optional opt-in escalation of a failed/unverifiable subtask
to Claude/Opus via claude-agent-sdk (OFF by default).
"""
__version__ = "0.2.1"
