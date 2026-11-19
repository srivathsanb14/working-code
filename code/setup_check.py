"""Step 0: check that this computer can run the pipeline.  Writes nothing.

    python code/setup_check.py

Checks Python, the packages, that the starter kit is found and unmodified, and that one
shipped scenario loads through the kit (metadata only, no simulation step).
"""
import importlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import KIT, WC, DATA, kit_git_status, import_kit  # noqa: E402

ok = True


def check(name, cond, detail=""):
    global ok
    print(f"[{'ok' if cond else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
    ok &= bool(cond)


check("python >= 3.12", sys.version_info >= (3, 12), sys.version.split()[0])
for pkg in ("numpy", "scipy", "pandas", "gin", "numba", "cv2", "gymnasium", "yaml", "matplotlib"):
    try:
        m = importlib.import_module(pkg); check(f"import {pkg}", True, getattr(m, "__version__", ""))
    except Exception as e:  # noqa: BLE001
        check(f"import {pkg}", False, str(e))
try:
    import anthropic  # noqa: F401
    print("[ok] import anthropic (optional, for --proposer claude)")
except Exception:  # noqa: BLE001
    print("[--] anthropic not installed (only needed for --proposer claude: pip install -r requirements-llm.txt)")
check("starter kit found", (KIT / "smart_buildings" / "evaluate.py").exists(), str(KIT))
st = kit_git_status()
check("starter kit unmodified", st in ("", None), "not a git checkout (cannot verify)" if st is None else (st[:300] or "clean"))
scen = KIT / "smart_buildings" / "training_scenarios" / "building_train_8" / "atlanta_1"
try:
    ev = import_kit()
    cwd = os.getcwd(); os.chdir(KIT)
    try:
        env = ev.load_hybrid_environment(scen / "config.gin")
        meta = ev.create_building_meta_data(env, scen)
    finally:
        os.chdir(cwd)
    check("load a shipped scenario through the kit", len(meta["observation_field_names"]) == 283,
          f"{len(meta['observation_field_names'])} observation fields, {meta['n_continuous_action']} continuous actions")
except Exception as e:  # noqa: BLE001
    check("load a shipped scenario through the kit", False, f"{type(e).__name__}: {e}")
print(f"working code: {WC}\ndata root:    {DATA}  (override with SB_DATA)")
st2 = kit_git_status()
check("starter kit still unmodified after the checks", st2 in ("", None), st2[:300] if st2 else "clean")
sys.exit(0 if ok else 1)
