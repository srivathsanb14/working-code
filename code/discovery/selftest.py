"""Planted-truth self-test of the closed-form discovery machinery (no simulator data, no LLM).

Why: before trusting what the loop says about real data, check that it can recognise a right answer.
A known closed-form map (TRUTH_SRC) generates synthetic room temperatures on the REAL floor plans the
starter kit ships (geometry only), under made-up weather and HVAC inputs.  Then:
  1. the truth itself must score ~0 at every horizon (catches any misalignment of history, inputs
     and targets in the scorer: the classic off-by-one);
  2. the seeds and persistence must score worse than the truth, and the scorer must rank the truth first;
  3. the full evolutionary loop (islands, mutation, crossover, migration, logs, report, plots) runs with
     the offline MockProposer, so every file the real run writes can be inspected.
Everything is written to outputs/discovery/selftest/ and labelled SYNTHETIC.

    .venv/bin/python -m discovery.selftest            (from working-code/code)
    .venv/bin/python code/run_discovery.py --selftest
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

from . import closedform as cf
from .episodes import Episode, _save_cached, geometry

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from common import KIT  # noqa: E402

TRUTH_SRC = '''"""Synthetic truth: one air node per room, a 4 h slow store from history, facade U = 3.5 W/m2K."""
import numpy as np
import dlib
HISTORY = 48
def prepare(geo, U):
    face = U["CELL"] * U["HEIGHT"]
    G_nb = (geo["wall_cells"] * face * 3.0 + geo["door_cells"] * face * 40.0) * (1 - np.eye(geo["V"]))
    return dict(C=U["RHO_C_AIR"] * U["HEIGHT"] * geo["area"], G_ext=geo["facade_faces"] * face * 3.5,
                G_nb=G_nb, G_nb_sum=G_nb.sum(1), G_slow=6.0 * geo["area"])
def step(coef, geo, hist, u, U):
    T = hist["T"][:, -1]; C = coef["C"]; dt = U["DT"]
    w = np.exp(-np.arange(HISTORY)[::-1] / 48.0); T_slow = np.tensordot(hist["T"], w / w.sum(), axes=([1], [0]))
    T_nb = (T @ coef["G_nb"].T) / np.maximum(coef["G_nb_sum"], 1e-9)
    Q = u["Q_reheat"] + u["Q_gain"] + U["OCC_W"] * u["occupants"][:, None] / geo["V"]
    return dlib.assemble(T, [(u["g"] * dt / C, u["T_supply"]), (coef["G_nb_sum"] * dt / C, T_nb),
                             (coef["G_ext"] * dt / C, u["T_out"][:, None]), (coef["G_slow"] * dt / C, T_slow)], Q * dt / C)
'''


def shipped_geometries(limit=6):
    """Geometry of the floor plans the starter kit ships (metadata arrays only)."""
    root = KIT / "smart_buildings" / "training_scenarios"
    out = []
    for b in sorted(p for p in root.iterdir() if p.is_dir())[:limit]:
        w = sorted(p for p in b.iterdir() if (p / "floor_plan.npy").exists())[0]
        fp, zm = np.load(w / "floor_plan.npy"), np.load(w / "zone_map.npy")
        import room_model as rm
        V = int(rm.grid_geometry(fp, zm)["nroom"])
        rooms = [f"room_{i + 1}" for i in range(V)]
        out.append((b.name, geometry(fp, zm, rooms, {"ahu_1": rooms})))
    return out


def synthetic_inputs(geo, n, rng):
    V = geo["V"]; t = np.arange(n)
    hour = (t * 5 / 60) % 24; day = (t * 5 / 60 // 24) % 7
    comfort = ((hour >= 7) & (hour < 19) & (day < 5)).astype(float)
    T_out = 276 + 7 * np.sin(2 * np.pi * (hour - 9) / 24) + rng.normal(0, 0.3, n).cumsum() * 0.05
    on = comfort[:, None] * (rng.random((1, V)) > 0.15)
    g = on * (300 + 900 * rng.random((n // 12 + 1, V)).repeat(12, 0)[:n])
    Q_re = on * (T_out[:, None] < 280) * 1500 * rng.random((n // 6 + 1, V)).repeat(6, 0)[:n]
    return dict(T_out=T_out, g=g, T_supply=np.full((n, V), 288.0) - 2 * (T_out[:, None] > 283), Q_reheat=Q_re,
                Q_gain=10.0 * geo["area"][None] * np.where(comfort[:, None] > 0, 1.0, 0.2),
                occupants=comfort * V * 0.6, comfort=comfort,
                hour_sin=np.sin(2 * np.pi * hour / 24), hour_cos=np.cos(2 * np.pi * hour / 24),
                dow_sin=np.sin(2 * np.pi * day / 7), dow_cos=np.cos(2 * np.pi * day / 7))


def generate(U, n=864, seed=0):
    """SYNTHETIC episodes: TRUTH_SRC rolled out from 294 K on each shipped floor plan, two weathers each."""
    from . import contract
    mod, _ = contract.load_closedform(TRUTH_SRC)
    rng = np.random.default_rng(seed)
    eps = []
    for b, geo in shipped_geometries():
        for wk in range(2):
            inp = synthetic_inputs(geo, n, rng)
            e = Episode(name=f"{b}_synth{wk}_mixed", building=b, weather=f"synth{wk}", policy="mixed",
                        scenario=f"{b}/synth{wk}", geo=geo, inputs=inp, T_meas=np.zeros((n, geo["V"])))
            prep = cf.Prepared(e, mod.HISTORY)
            T = cf.rollout(mod, mod.prepare(geo, U), prep, U, [0], n)[0]   # free run from T0: the "measurements"
            e.T_meas = T
            eps.append(e)
    return eps


def main(out=ROOT / "outputs" / "discovery" / "selftest", generations=2, islands=2):
    U, _ = cf.universal_constants()
    eps = generate(U)
    ds = out.parent / "selftest_dataset"
    if ds.exists():
        shutil.rmtree(ds)
    ds.mkdir(parents=True)
    idx = {}
    for e in eps:
        _save_cached(e, ds / f"{e.name}.npz")
        idx[e.name] = dict(file=f"{e.name}.npz", status="ok", policy=e.policy, building=e.building)
    blds = sorted({e.building for e in eps})
    split = {"_note": "SYNTHETIC self-test split", "train": blds[:3], "val": blds[3:5], "test": blds[5:]}
    (ds / "index.json").write_text(json.dumps(idx, indent=1)); (ds / "split.json").write_text(json.dumps(split, indent=1))
    tr = [e for e in eps if e.building in split["train"]]; va = [e for e in eps if e.building in split["val"]]

    print("1. the planted truth must score ~0 (scorer alignment)")
    rt = cf.evaluate(TRUTH_SRC, tr, va, U)
    print(f"   truth: status {rt['status']}, one-step {rt.get('val_one_step_mean', float('nan')):.2e} K, "
          f"3 h {rt.get('val_3h_mean', float('nan')):.2e} K")
    ok1 = rt["status"] == "ok" and rt["val_3h_mean"] < 1e-6 and rt["val_one_step_mean"] < 1e-6
    print("2. seeds and persistence must be worse")
    seeds = sorted((Path(__file__).parent / "seeds_cf").glob("*.py"))
    res = {}
    for s in seeds:
        r = cf.evaluate(s.read_text(), tr, va, U); res[s.stem] = r
        print(f"   {s.stem}: {r['status']}, 3 h {r.get('val_3h_mean', float('nan')):.3f} K "
              f"(persistence {r.get('val_persistence_3h_mean', float('nan')):.3f})" + (f" - {r.get('reason')}" if r['status'] != 'ok' else ""))
    ok2 = all(r["status"] != "ok" or r["score"] > rt["score"] for r in res.values())
    print("3. the evolutionary loop with the offline mock proposer")
    from . import evolve
    if out.exists():
        shutil.rmtree(out)
    evolve.main_args(["--run", out.name, "--dataset", str(ds), "--proposer", "mock", "--generations", str(generations),
                      "--islands", str(islands), "--migrate-every", "1", "--every", "12", "--ceiling"])
    ok3 = (out / "REPORT.md").exists()
    verdict = dict(truth_scores_zero=ok1, truth_beats_seeds=ok2, loop_and_report=ok3)
    (out / "selftest.json").write_text(json.dumps(verdict, indent=1))
    print(json.dumps(verdict, indent=1))
    if not all(verdict.values()):
        raise SystemExit("self-test FAILED")
    print(f"self-test passed; open {out / 'REPORT.md'}")


if __name__ == "__main__":
    main()
