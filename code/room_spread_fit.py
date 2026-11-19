"""Fit the two global spreading constants (c, a) of room_model.spreading_resistance
so that the lumped network, Kron-reduced onto room air nodes, reproduces the
exact steady-state room conductance matrix of the grid.  Fit on three buildings
(true and random materials), test on the other three.
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
HERE = Path(__file__).resolve().parent; OUT = HERE.parent / "outputs"
KIT = Path(__import__("os").environ.get("STARTER_KIT", HERE.parent.parent / "smart-buildings-starter-kit"))
sys.path.insert(0, str(HERE))
import room_model as rm
from room_coeff_targets import exact_Y, RANGES

val = json.loads((OUT / "room_model_validate.json").read_text())
rng = np.random.default_rng(1)
cases = []
for b, rec in val.items():
    d = sorted((KIT / "smart_buildings/training_scenarios" / b).iterdir())[0]
    G = rm.grid_geometry(np.load(d / "floor_plan.npy"), np.load(d / "zone_map.npy"))
    cid, nodes = rm.cluster_ids(G)
    ths = [rec["theta"]] + [{k: float(np.exp(rng.uniform(np.log(lo), np.log(hi)))) for k, (lo, hi) in RANGES.items()} for _ in range(2)]
    for th in ths:
        cases.append(dict(b=b, G=G, cid=cid, nodes=nodes, th=th, Y=exact_Y(G, th)))
train = {"building_train_8", "building_train_2", "building_train_5"}

def Ycl(case, par):
    M = rm.cluster_model(case["G"], case["th"], case["cid"], case["nodes"], spread=None if par is None else dict(c=par[0], a=par[1], af=par[2]))
    C, L, b, kinds, owner = rm.full_network(M); V = M["V"]; E = np.arange(V, len(C))
    return L[:V, :V] - L[:V][:, E] @ np.linalg.solve(L[np.ix_(E, E)], L[E][:, :V])

def resid(par, cs):
    r = []
    for c in cs:
        Y, Yc = c["Y"], Ycl(c, par); V = len(Y); off = ~np.eye(V, dtype=bool)
        k, kc = -Y[off], -Yc[off]; m = k > 2
        r.append((kc[m] - k[m]) / np.sqrt(k[m]))
        e, ec = Y.sum(1), Yc.sum(1); m = np.abs(e) > 2
        r.append((ec[m] - e[m]) / np.sqrt(np.abs(e[m])))
    return np.concatenate(r)

def report(par, cs):
    out = {}
    K, Kc, E, Ec = [], [], [], []
    for c in cs:
        Y, Yc = c["Y"], (Ycl(c, par)); V = len(Y); off = ~np.eye(V, dtype=bool)
        K.append(-Y[off]); Kc.append(-Yc[off]); E.append(Y.sum(1)); Ec.append(Yc.sum(1))
    K, Kc, E, Ec = map(np.concatenate, (K, Kc, E, Ec))
    for nm, y, p in (("kappa", K, Kc), ("eps", E, Ec)):
        m = np.abs(y) > 5
        out[nm] = dict(r2=float(1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()), median_rel_err=float(np.median(np.abs(p[m] - y[m]) / np.abs(y[m]))),
                       total_ratio=float(p.sum() / y.sum()))
    return out

tr = [c for c in cases if c["b"] in train]; te = [c for c in cases if c["b"] not in train]
fit = least_squares(resid, x0=[0.01, 0.7, 0.7], args=(tr,), bounds=([0, 0, 0], [20, 20, 20]))
res = {"c_a_af": fit.x.tolist(), "train": report(fit.x, tr), "test": report(fit.x, te), "test_without_spreading": report(None, te)}
print(json.dumps(res, indent=1)); (OUT / "room_spread_fit.json").write_text(json.dumps(res, indent=1))
