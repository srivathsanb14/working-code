"""Exact steady-state room coefficients from the rebuilt grid, and the geometry
features a policy can compute from the metadata arrays.

For a building and material vector theta, the grid (room_model.grid_geometry +
cell_properties) is solved at steady state for a unit of heat spread uniformly
over each room, reading back every room's mean temperature (exactly how the
simulator injects heat and senses temperature).  Inverting that V x V response
gives the exact steady-state room conductance matrix Y: off-diagonals are the
inter-room couplings kappa_ij (W/K) and row sums the loss to ambient eps_i.

Features per room / pair, from floor_plan and zone_map only:
  A_i    floor area (m2)
  F_i    faces between room air and exterior-material cells
  P_ij   partition cells that touch both rooms (1-cell walls)
  D_ij   door cells that touch both rooms
  Lx,Ly  room bounding-box size; depth = A / F (m) for facade rooms

Writes ../outputs/room_coeff_targets.npz (one row per pair and per room).
Usage: <venv>/python <this file> [n_random_theta_per_building]
"""
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy import ndimage

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "outputs"
KIT = Path(__import__("os").environ.get("STARTER_KIT", HERE.parent.parent / "smart-buildings-starter-kit"))
sys.path.insert(0, str(HERE))
import room_model as rm  # noqa: E402

RANGES = dict(k_ext=(0.15, 0.55), rc_ext=(350 * 840, 2400 * 1000), rc_mass=(10 * 1000, 40 * 1000),
              k_iw=(0.45, 0.60), rc_iw=(110 * 900, 700 * 900), h=(8.8, 26.7))


def grid_operator(G, theta):
    k, C, Cm, Gm, amb = rm.cell_properties(G, theta)
    gc, gr = rm.face_conductances(G, k)
    H, W = G["shape"]
    idx = np.arange(H * W).reshape(H, W)
    s = G["solved"].ravel()
    rows = np.r_[idx[:, :-1].ravel(), idx[:-1, :].ravel()]
    cols = np.r_[idx[:, 1:].ravel(), idx[1:, :].ravel()]
    g = np.r_[gc.ravel(), gr.ravel()]
    K = sp.csr_matrix((np.r_[g, g], (np.r_[rows, cols], np.r_[cols, rows])), shape=(H * W, H * W))
    tot = np.asarray(K.sum(1)).ravel()
    Ks = K[s][:, s]
    A = (sp.diags(tot[s] + amb.ravel()[s]) - Ks).tocsc()
    lab = G["lab"].ravel()[s]
    V = G["nroom"]
    P = sp.csr_matrix((np.ones((lab > 0).sum()), (np.nonzero(lab > 0)[0], lab[lab > 0] - 1)), shape=(s.sum(), V))
    cnt = np.asarray(P.sum(0)).ravel()
    P = P.multiply(1 / cnt).tocsc()
    return A, P, dict(C=C.ravel()[s], Cm=Cm.ravel()[s], Gm=Gm.ravel()[s])


def exact_Y(G, theta):
    A, P, _ = grid_operator(G, theta)
    lu = sla.splu(A)
    Z = P.T @ lu.solve(P.toarray())
    return np.linalg.inv(Z)


def features(G):
    lab, cls, door = G["lab"], G["cls"], G["door"]
    V = G["nroom"]
    area = np.bincount(lab[lab > 0] - 1, minlength=V) * rm.CV ** 2
    ext = cls == rm.SHELL_OR_EXT
    F = np.zeros(V)
    for sh in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nb = np.roll(ext, sh, axis=(0, 1))
        m = (lab > 0) & nb
        F += np.bincount(lab[m] - 1, minlength=V)
    # rooms touched by each partition / door cell (4-neighbourhood)
    P = np.zeros((V, V)); Dm = np.zeros((V, V))
    for mask, out in (((cls == rm.IWALL), P), (door, Dm)):
        rr, cc = np.nonzero(mask)
        for i, j in zip(rr, cc):
            rs = {int(lab[a, b]) for a, b in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)) if lab[a, b] > 0}
            rs = sorted(rs)
            for x in range(len(rs)):
                for y in range(x + 1, len(rs)):
                    out[rs[x] - 1, rs[y] - 1] += 1; out[rs[y] - 1, rs[x] - 1] += 1
    # door count per pair (connected door components)
    dl, nd = ndimage.label(door)
    ndoor = np.zeros((V, V))
    for d in range(1, nd + 1):
        m = dl == d
        ring = ndimage.binary_dilation(m) & ~m
        rs = sorted(set(lab[ring][lab[ring] > 0].tolist()))
        for x in range(len(rs)):
            for y in range(x + 1, len(rs)):
                ndoor[rs[x] - 1, rs[y] - 1] += 1; ndoor[rs[y] - 1, rs[x] - 1] += 1
    sl = ndimage.find_objects(np.where(lab > 0, lab, 0))
    Lx = np.array([(s_[1].stop - s_[1].start) * rm.CV for s_ in sl]); Ly = np.array([(s_[0].stop - s_[0].start) * rm.CV for s_ in sl])
    return dict(area=area, F=F, P=P, D=Dm, ndoor=ndoor, Lx=Lx, Ly=Ly)


def main():
    nrand = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    val = json.loads((OUT / "room_model_validate.json").read_text())
    rng = np.random.default_rng(0)
    pair_rows, room_rows = [], []
    for b, rec in val.items():
        d = sorted((KIT / "smart_buildings/training_scenarios" / b).iterdir())[0]
        G = rm.grid_geometry(np.load(d / "floor_plan.npy"), np.load(d / "zone_map.npy"))
        f = features(G)
        thetas = [rec["theta"]] + [{k: float(np.exp(rng.uniform(np.log(lo), np.log(hi)))) for k, (lo, hi) in RANGES.items()} for _ in range(nrand)]
        for ti, th in enumerate(thetas):
            Y = exact_Y(G, th)
            V = len(Y)
            for i in range(V):
                room_rows.append([hash(b) % 1000, ti, i, f["area"][i], f["F"][i], f["Lx"][i], f["Ly"][i], Y[i].sum(), Y[i, i],
                                  th["k_ext"], th["h"], th["k_iw"], th["rc_ext"], th["rc_mass"], th["rc_iw"]])
                for j in range(i + 1, V):
                    if f["P"][i, j] + f["D"][i, j] > 0 or -Y[i, j] > 0.5:
                        pair_rows.append([hash(b) % 1000, ti, i, j, f["P"][i, j], f["D"][i, j], f["ndoor"][i, j], f["area"][i], f["area"][j],
                                          -Y[i, j], th["k_iw"], th["k_ext"]])
            print(b, "theta", ti, "V", V, "sum kappa", round(float(-(Y - np.diag(np.diag(Y))).sum() / 2), 1), "sum eps", round(float(Y.sum()), 1))
    np.savez(OUT / "room_coeff_targets.npz", pairs=np.array(pair_rows), rooms=np.array(room_rows),
             pair_cols=np.array(["bld", "theta", "i", "j", "P", "D", "ndoor", "Ai", "Aj", "kappa", "k_iw", "k_ext"]),
             room_cols=np.array(["bld", "theta", "i", "A", "F", "Lx", "Ly", "eps", "Yii", "k_ext", "h", "k_iw", "rc_ext", "rc_mass", "rc_iw"]))


if __name__ == "__main__":
    main()
