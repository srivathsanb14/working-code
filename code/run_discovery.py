"""Launcher for the discovery loop (code/discovery/loop.py); see its docstring for usage.
Run from working-code/:  .venv/bin/python code/run_discovery.py --help
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from discovery.loop import main  # noqa: E402

if __name__ == "__main__":
    main()
