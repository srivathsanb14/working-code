"""Build the folder to upload as the Kaggle Dataset attached to the submission notebook.

    python code/package_submission.py                                   # policy code only (pid_water + safety)
    python code/package_submission.py --model outputs/discovery/main/best.py   # also the discovered formulas

Writes dist/kaggle_dataset/ (code/ with only what the notebook imports, optional best.py, MANIFEST.txt) and checks
that it imports in a clean interpreter without the starter kit.  Upload the folder as a Kaggle Dataset and attach it
to kaggle/submission.ipynb.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WC = HERE.parent
FILES = ["room_model.py", "room_coeff_targets.py",
         "discovery/__init__.py", "discovery/closedform.py", "discovery/coefform.py", "discovery/contract.py", "discovery/dlib.py",
         "discovery/episodes.py", "discovery/universal_constants.json",
         "runtime/__init__.py", "runtime/inputs.py", "runtime/closedform_tracker.py",
         "submission/__init__.py", "submission/layout.py", "submission/safety.py", "submission/rule_policy.py",
         "submission/transfer_policy.py", "submission/vendor/__init__.py", "submission/vendor/pid.py", "submission/vendor/pid_plus.py",
         "control/__init__.py", "control/costs.py", "control/occupancy.py", "control/schedule.py", "control/planner.py",
         "control/policies.py"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, help="closed-form best.py to include")
    ap.add_argument("--out", type=Path, default=WC / "dist" / "kaggle_dataset")
    a = ap.parse_args()
    if a.out.exists():
        shutil.rmtree(a.out)
    for f in FILES:
        dst = a.out / "code" / f; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(HERE / f, dst)
    if a.model:
        shutil.copy2(a.model, a.out / "best.py")
    (a.out / "MANIFEST.txt").write_text("\n".join(["code/" + f for f in FILES] + (["best.py"] if a.model else [])) + "\n")
    check = ("import sys; sys.path.insert(0, sys.argv[1]); "
             "from submission.transfer_policy import TransferPolicyCore; from runtime.closedform_tracker import ClosedFormTracker; "
             "from control.policies import MPCPolicy, FeedForwardPolicy; "
             "print('imports ok without the starter kit')")
    r = subprocess.run([sys.executable, "-I", "-c", check, str(a.out / "code")], capture_output=True, text=True, cwd=a.out)
    print(r.stdout.strip() or r.stderr.strip())
    if r.returncode:
        raise SystemExit("packaged code does not import on its own")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
