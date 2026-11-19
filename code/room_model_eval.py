"""Open-loop test of lumped room models against the simulator, with oracle inputs.

For each logged rollout (room_model_data.py) the lumped network is instantiated
from the metadata arrays and the TRUE material scalars, driven by the exact
per-room heat inputs the solver received, and rolled out open loop from the
reset state for the whole run (no correction from measurements).  Several
reductions of the same network are compared.

Run:  <venv>/python <this file>      (from anywhere; reads ../outputs/roomdata_*.npz)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "outputs"
sys.path.insert(0, str(HERE))
import room_model as rm  # noqa: E402


def theta_from(D):
    m = D["mat"]
    return dict(k_iw=m[0], rc_iw=m[1], k_ext=m[2], rc_ext=m[3], rc_mass=m[5], h=float(D["h"][0]))


SPREAD = None


def variants(M):
    C, L, b, kinds, owner = rm.full_network(M)
    V = M["V"]
    air = list(range(V))
    # slab nodes that belong to room air nodes, in room order
    n0 = len(M["nodes"])
    has = np.nonzero(M["Cm"] > 0)[0]
    slab_room = [n0 + q for q, i in enumerate(has) if i < V]
    facade = [i for i, k in enumerate(kinds) if k == "facade" and len(owner[i]) == 1]
    out = {}
    out["full_lumped"] = (np.diag(C), L, b, np.zeros(len(C)))
    out["air+slab+facade (3/room)"] = rm.guyan(C, L, b, air + slab_room + facade)
    out["air+slab (2/room)"] = rm.guyan(C, L, b, air + slab_room)
    out["air only (1/room)"] = rm.guyan(C, L, b, air)
    return out, len(C)


def main():
    global SPREAD
    SPREAD = dict(zip(("c", "a", "af"), json.loads((OUT / "room_spread_fit.json").read_text())["c_a_af"]))
    files = sorted(OUT.glob("roomdata_*.npz"))
    res = {}
    for f in files:
        D = np.load(f)
        G = rm.grid_geometry(D["floor_plan"], D["zone_map"])
        M = rm.cluster_model(G, theta_from(D))
        V = M["V"]
        Ms = rm.cluster_model(G, theta_from(D), spread=SPREAD)
        Tt = D["Ttrue_after"]
        r = {"V": V}
        vs, n_full = variants(M)
        vs2, _ = variants(Ms)
        vs = {"lumped " + k: v for k, v in vs.items() if k in ("full_lumped", "air+slab+facade (3/room)")}
        vs.update({"spread " + k: v for k, v in vs2.items()})
        r["n_nodes_full"] = n_full
        for name, (Cr, Lr, br, cr) in vs.items():
            t0 = time.time()
            P = rm.simulate_reduced(Cr, Lr, br, cr, V, D["Tamb"], D["g"], D["Tsup"], D["Qroom"], 294.0)
            e = P - Tt
            r[name] = dict(states=int(Cr.shape[0]), rmse_K=float(np.sqrt((e ** 2).mean())), max_abs_K=float(np.abs(e).max()),
                           rmse_day1=float(np.sqrt((e[:288] ** 2).mean())), rmse_day3=float(np.sqrt((e[576:] ** 2).mean())),
                           sec=round(time.time() - t0, 2))
        r["true_T_range_K"] = [float(Tt.min()), float(Tt.max())]
        r["true_T_std_K"] = float(Tt.std())
        res[f.stem] = r
        print(f.stem, json.dumps({k: (v if not isinstance(v, dict) else {kk: round(vv, 3) if isinstance(vv, float) else vv for kk, vv in v.items()}) for k, v in r.items()}))
    (OUT / "room_model_eval.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
