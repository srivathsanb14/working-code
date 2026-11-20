"""Launcher for model discovery.  Run from working-code/:

  DEFAULT  closed-form discovery, no fitting anywhere, islands + mutation + crossover (code/discovery/evolve.py):
      .venv/bin/python code/run_discovery.py --run cf --proposer dry --generations 1      # write every prompt, call nothing
      .venv/bin/python code/run_discovery.py --run cf --proposer mock --generations 3     # no LLM: mechanical operators
      .venv/bin/python code/run_discovery.py --run cf --proposer claude --generations 20  # the real search
      .venv/bin/python code/run_discovery.py --help                                       # all options

  SELF-TEST  planted truth on synthetic data (shipped floor plans, made-up inputs): the scorer must give the
             truth ~0 K and rank it first, and the whole loop runs offline -> outputs/discovery/selftest/REPORT.md
      .venv/bin/python code/run_discovery.py --selftest

  FITTED   the earlier modes that fit constants (comparison only; code/discovery/loop.py):
      .venv/bin/python code/run_discovery.py --deployment offline|online|both ...
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--selftest" in argv:
        from discovery.selftest import main
        main()
    elif "--deployment" in argv and argv[argv.index("--deployment") + 1] in ("offline", "online", "both"):
        from discovery.loop import main
        main()
    else:
        argv = [x for i, x in enumerate(argv) if not (x == "--deployment" or (i and argv[i - 1] == "--deployment"))]
        from discovery.evolve import main_args
        main_args(argv)
