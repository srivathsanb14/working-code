"""Dump every entry of the observation and action vectors for each building.

Read-only with respect to the starter kit: imports it, loads one scenario per
building, resets once, and writes CSVs into ../outputs. Run:
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<starter-kit> python dump_vectors.py
"""
import csv, json, sys
from pathlib import Path
import numpy as np

KIT = Path("/Users/srivathsan/Desktop/smart-buildings-starter-kit")
OUT = Path(__file__).resolve().parent.parent / "outputs"
sys.path.insert(0, str(KIT))
from smart_buildings import evaluate as ev  # noqa: E402

buildings = ["building_train_%d" % i for i in (1, 2, 4, 5, 7, 8)]
for b in buildings:
    d = sorted((KIT / "smart_buildings/training_scenarios" / b).iterdir())[0]
    env = ev.load_hybrid_environment(d / "config.gin")
    ts = env.reset()
    obs = np.asarray(ts[0] if isinstance(ts, tuple) else ts.observation)
    names = list(env.field_names)
    with open(OUT / f"{b}_observation.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["index", "field_id", "initial_value"])
        for i, (n, v) in enumerate(zip(names, obs)):
            w.writerow([i, n, float(v)])
    acts = [(i, dev, nm) for i, (dev, nm) in enumerate(env._device_action_tuples)]
    with open(OUT / f"{b}_action.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["tuple_index", "device_id", "setpoint_name", "kind"])
        for i, dev, nm in acts:
            w.writerow([i, dev, nm, "discrete" if "run_command" in nm else "continuous"])
    meta = {
        "scenario": str(d.name),
        "n_obs": len(names),
        "n_action_tuples": len(acts),
        "action_spec_keys": list(env.action_spec().keys()),
        "action_indices": ev.get_action_indices(env),
        "ahu_zone_assignments": ev.get_ahu_zone_assignments(env),
    }
    (OUT / f"{b}_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print(b, d.name, len(names), len(acts))
