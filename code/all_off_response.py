"""With every air handler and the boiler switched off, how fast does each room's temperature
follow the outside and its neighbours?  Per-room regression
    T[t+1]-T[t] = a (T_out-T) + n (mean_nbr(T)-T) + g*occupants + c
and a comparison of `a` across room types.  Run from the starter-kit root; writes ../outputs.
"""
import csv, json, re, sys
from pathlib import Path
import numpy as np
KIT = Path(__import__("os").environ.get("STARTER_KIT", Path(__file__).resolve().parent.parent.parent / "smart-buildings-starter-kit")); OUT = Path(__file__).resolve().parent.parent / "outputs"
sys.path.insert(0, str(KIT)); from smart_buildings import evaluate as ev
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 864
bl = sys.argv[2:] or ["building_train_8", "building_train_2"]
summary = {}
for b in bl:
    d = sorted((KIT / "smart_buildings/training_scenarios" / b).iterdir())[0]
    env = ev.load_hybrid_environment(d / "config.gin"); names = list(env.field_names); ix = {n: i for i, n in enumerate(names)}
    rooms = sorted(int(m.group(1)) for n in names if (m := re.fullmatch(r"VAV_(\d+)_zone_air_temperature_sensor", n)))
    zi = [ix[f"VAV_{r}_zone_air_temperature_sensor"] for r in rooms]; oi = ix["ahs_ahu_1_outside_air_temperature_sensor"]; ni = ix["num_occupants"]
    spec = env.action_spec(); z = {"continuous_action": np.zeros(spec["continuous_action"].shape, np.float32), "discrete_action": np.zeros(spec["discrete_action"].shape, np.int32)}
    ts = env.reset(); o = np.asarray(ts[0] if isinstance(ts, tuple) else ts.observation); O = [o]
    for _ in range(STEPS): O.append(np.asarray(env.step(z)[0]))
    O = np.array(O); T = O[:, zi] * 5 + 295; Tout = O[:, oi]; occ = O[:, ni]
    info = {int(r["room"].split("_")[1]): r for r in csv.DictReader(open(OUT / f"{b}_rooms.csv"))}
    pos = {r: k for k, r in enumerate(rooms)}; rows = []
    for r in rooms:
        nb = [int(x.split("_")[1]) for x in (info[r]["wall_neighbours"].split() + info[r]["door_neighbours"].split())]; nb = sorted(set(nb))
        Tn = T[:, [pos[x] for x in nb]].mean(1) if nb else T[:, pos[r]]
        t = T[:, pos[r]]; y = t[1:] - t[:-1]
        X = np.c_[Tout[:-1] - t[:-1], Tn[:-1] - t[:-1], occ[:-1], np.ones(len(y))]
        coef, *_ = np.linalg.lstsq(X, y, rcond=None); r2 = 1 - ((y - X @ coef) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        ec = int(info[r]["ext_N"]) + int(info[r]["ext_S"]) + int(info[r]["ext_W"]) + int(info[r]["ext_E"])
        rows.append(dict(room=r, type=info[r]["type"], area_m2=float(info[r]["area_m2"]), ext_cells=ec, ext_per_m2=ec / float(info[r]["area_m2"]),
                         a_env=coef[0], n_nbr=coef[1], g_occ=coef[2], c=coef[3], r2=r2, T_range=float(t.max() - t.min())))
    with open(OUT / f"{b}_alloff_response.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    S = {"scenario": d.name, "steps": STEPS, "Tout_range_K": [float(Tout.min()), float(Tout.max())], "zone_T_range_K": [float(T.min()), float(T.max())]}
    for ty in ("corner", "edge", "interior"):
        a = np.array([x["a_env"] for x in rows if x["type"] == ty])
        if len(a): S[ty] = dict(n=len(a), a_env_mean=float(a.mean()), a_env_std=float(a.std()), a_env_min=float(a.min()), a_env_max=float(a.max()))
    A = np.array([x["a_env"] for x in rows]); E = np.array([x["ext_per_m2"] for x in rows]); S["corr_a_env_vs_ext_cells_per_m2"] = float(np.corrcoef(A, E)[0, 1])
    S["median_r2"] = float(np.median([x["r2"] for x in rows]))
    summary[b] = S; print(b, json.dumps(S))
(OUT / "alloff_response_summary.json").write_text(json.dumps(summary, indent=1))
