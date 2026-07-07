"""Launch coder-crew. Equivalent to `python -m crewkit.server`.

Env:
  CREW_HOST   bind host (default 127.0.0.1)
  CREW_PORT   bind port (default 8770)
  CREW_DB     SQLite history path (default ./crew_history.db)
  PCP_OLLAMA_URL  override Ollama base URL (default http://127.0.0.1:11434)
"""
from crewkit.server import main

if __name__ == "__main__":
    main()
