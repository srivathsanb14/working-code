"""Shared paths and helpers for the working-code tools.

The starter kit is located by the STARTER_KIT environment variable, else as the sibling
folder ../smart-buildings-starter-kit.  It is only ever read: imported from its folder,
or its command-line tools run with outputs pointed into working-code/data.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

WC = Path(__file__).resolve().parents[1]
KIT = Path(os.environ.get("STARTER_KIT", WC.parent / "smart-buildings-starter-kit")).resolve()
DATA = Path(os.environ.get("SB_DATA", WC / "data")).resolve()
NUMBA_CACHE = WC / ".numba_cache"

os.environ.setdefault("NUMBA_CACHE_DIR", str(NUMBA_CACHE))


def kit_env():
    """Environment for running starter-kit tools without writing into the kit."""
    env = dict(os.environ)
    env.update(NUMBA_CACHE_DIR=str(NUMBA_CACHE), PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg",
               PYTHONPATH=str(KIT) + os.pathsep + env.get("PYTHONPATH", ""))
    return env


def run_kit(args, log_path: Path | None = None, timeout=None):
    """Run `python -m <module> ...` from the kit root with outputs elsewhere. Returns (code, tail)."""
    cmd = [sys.executable, "-m", *args]
    with (open(log_path, "a") if log_path else open(os.devnull, "w")) as log:
        log.write("\n$ " + " ".join(map(str, cmd)) + "\n"); log.flush()
        p = subprocess.run(cmd, cwd=KIT, env=kit_env(), stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    tail = Path(log_path).read_text()[-2000:] if log_path else ""
    return p.returncode, tail


def kit_git_status():
    """Untracked or modified files in the kit ('' = untouched), or None if not a git checkout."""
    try:
        out = subprocess.run(["git", "status", "--porcelain"], cwd=KIT, capture_output=True, text=True, timeout=30)
    except Exception:  # noqa: BLE001
        return None
    if out.returncode != 0:
        return None
    lines = [l for l in out.stdout.splitlines() if not l.endswith(".DS_Store")]
    return "\n".join(lines)


def import_kit():
    """Import the kit's evaluate module from its folder (gin paths are relative to the kit root)."""
    if str(KIT) not in sys.path:
        sys.path.insert(0, str(KIT))
    cwd = os.getcwd(); os.chdir(KIT)
    try:
        from smart_buildings import evaluate as ev
    finally:
        os.chdir(cwd)
    return ev
