@echo off
REM Launch coder-crew. Uses .venv if present, else the system python.
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m crewkit.server
) else (
  python -m crewkit.server
)
