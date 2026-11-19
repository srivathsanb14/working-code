"""Fit closed-form coefficient templates to the exact steady-state room
coefficients (room_coeff_targets.npz) and test them on held-out buildings.

Templates (k_air = 50 W/mK and z = 3 m are fixed simulator constants; every
other symbol is a geometry count from the metadata arrays or a material scalar):

  wall path   g_w(k_iw)        = z*CV / (2*CV/(2*k_air) + CV/k_iw)         per partition cell
  door path   g_d              = z*CV / (2*CV/(2*k_air) + CV/k_air)        per door cell (75 W/K)
  facade      g_f(k_ext, h)    = z*CV / (CV/(2*k_air) + CV/(2*k_ext) + 1/h) per facade face

  kappa_ij  (lumped)   = g_w P_ij + g_d D_ij
  kappa_ij  (spread)   = g_w P_ij + 1 / ( 1/(g_d D_ij) + a/(pi k_air z) * [ln(sqrt(A_i)/b) + ln(sqrt(A_j)/b)] )
  eps_i     (lumped)   = g_f F_i
  eps_i     (spread)   = 1 / ( 1/(g_f F_i) + c * d_i / (3 k_air z L_i) ),  L_i = F_i*CV, d_i = A_i / L_i

a, b, c are three global constants fitted on three buildings and tested on the other three.
"""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "outputs"
CV, Z, KA = 0.2, 3.0, 50.0

T = np.load(OUT / "room_coeff_targets.npz")
pc = {c: i for i, c in enumerate(T["pair_cols"])}; rc = {c: i for i, c in enumerate(T["room_cols"])}
Pr, Rr = T["pairs"], T["rooms"]
blds = np.unique(Rr[:, rc["bld"]])
train_b, test_b = blds[::2], blds[1::2]


def g_w(k):
    return Z * CV / (CV / KA + CV / k)


G_D = Z * CV / (CV / KA + CV / KA)


def g_f(k, h):
    return Z * CV / (CV / (2 * KA) + CV / (2 * k) + 1 / h)


def kappa(par, X, spread=True):
    P, D, Ai, Aj, kiw = X[:, pc["P"]], X[:, pc["D"]], X[:, pc["Ai"]], X[:, pc["Aj"]], X[:, pc["k_iw"]]
    wall = g_w(kiw) * P
    if not spread:
        return wall + G_D * D
    a, b = par
    Rs = a / (np.pi * KA * Z) * (np.log(np.sqrt(Ai) / b) + np.log(np.sqrt(Aj) / b))
    door = np.where(D > 0, 1 / (1 / (G_D * np.maximum(D, 1e-9)) + np.maximum(Rs, 0)), 0.0)
    return wall + door


def eps(par, X, spread=True):
    A, F, k, h = X[:, rc["A"]], X[:, rc["F"]], X[:, rc["k_ext"]], X[:, rc["h"]]
    base = g_f(k, h) * F
    if not spread:
        return base
    (c,) = par
    L = np.maximum(F * CV, 1e-9); d = A / L
    return np.where(F > 0, 1 / (1 / np.maximum(base, 1e-12) + c * d / (3 * KA * Z * L)), 0.0)


def stats(pred, y, mask_sig):
    e = pred - y
    m = mask_sig
    return dict(r2=float(1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()),
                median_abs_rel_err=float(np.median(np.abs(e[m]) / np.abs(y[m]))),
                p90_abs_rel_err=float(np.percentile(np.abs(e[m]) / np.abs(y[m]), 90)),
                total_ratio=float(pred.sum() / y.sum()))


res = {}
ptr = np.isin(Pr[:, pc["bld"]], train_b); pte = ~ptr
rtr = np.isin(Rr[:, rc["bld"]], train_b); rte = ~rtr
yk = Pr[:, pc["kappa"]]; ye = Rr[:, rc["eps"]]
sigk = yk > 5; sige = ye > 5
fk = least_squares(lambda p: (kappa(p, Pr[ptr]) - yk[ptr]) / np.sqrt(1 + np.abs(yk[ptr])), x0=[1.0, 0.3], bounds=([0.01, 0.01], [20, 20]))
fe = least_squares(lambda p: (eps(p, Rr[rtr]) - ye[rtr]) / np.sqrt(1 + np.abs(ye[rtr])), x0=[1.0], bounds=([0.0], [50]))
res["kappa_params_a_b"] = fk.x.tolist(); res["eps_param_c"] = fe.x.tolist()
for name, sel in (("train_buildings", ptr), ("test_buildings", pte)):
    res[f"kappa_lumped_{name}"] = stats(kappa(None, Pr[sel], False), yk[sel], sigk[sel])
    res[f"kappa_spread_{name}"] = stats(kappa(fk.x, Pr[sel]), yk[sel], sigk[sel])
for name, sel in (("train_buildings", rtr), ("test_buildings", rte)):
    res[f"eps_lumped_{name}"] = stats(eps(None, Rr[sel], False), ye[sel], sige[sel])
    res[f"eps_spread_{name}"] = stats(eps(fe.x, Rr[sel]), ye[sel], sige[sel])
# pairs that the grid couples but no wall/door connects (diagonal contact, corners)
nc = (Pr[:, pc["P"]] + Pr[:, pc["D"]]) == 0
res["pairs_without_shared_wall_or_door"] = dict(n=int(nc.sum()), max_kappa=float(yk[nc].max()) if nc.any() else 0.0,
                                                 share_of_total_kappa=float(yk[nc].sum() / yk.sum()))
res["n_pairs"] = int(len(yk)); res["n_rooms"] = int(len(ye))
print(json.dumps(res, indent=1))
(OUT / "room_coeff_fit.json").write_text(json.dumps(res, indent=1))
