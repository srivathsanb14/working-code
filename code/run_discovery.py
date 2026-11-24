"""Launcher for model discovery.  Run from working-code/.

  INPUT    a folder of finished episodes made by the data repository (the one that runs the starter kit):
           index.json + one .npz per episode.  Nothing is simulated or generated here.
  DEFAULT  closed-form discovery, no fitting anywhere, every candidate written by the Claude API
           (code/discovery/evolve.py; needs `pip install -r requirements-llm.txt` and ANTHROPIC_API_KEY):
      .venv/bin/python code/run_discovery.py --dataset /path/to/dataset --run main --generations 20 --test --max-usd 50
      .venv/bin/python code/run_discovery.py --help                                       # all options

  SELF-TEST  the scorer on a planted truth (synthetic inputs on the shipped floor plans): the truth must score ~0 K
             and beat every hand-written model.  No API call.
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
