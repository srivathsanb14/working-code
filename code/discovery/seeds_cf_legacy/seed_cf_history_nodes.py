"""Starting model: room air plus a slow store, with the outside-wall U-value computed from the measured history.

Nodes: every room's air and one slow store per room (slab and furniture).  Rooms exchange heat with neighbours
through shared walls and doors (contact area x a nominal conductance per m2), with their slow store (floor area x
4 W/m2K, capacity set so the store's time constant is 6 h), and, for rooms with an outside wall, with the outside
air through the wall area x U.  U is not fixed: every REFRESH steps it is computed from the last 24 h of
measurements as a shrunk least-squares ratio (how much the exterior rooms' measured changes follow T_out - T beyond
what the other terms explain), pulled toward a nominal value while the history is short.  Nothing is fitted.
"""
import numpy as np
import dlib

HISTORY = 288                                            # one day of 5-minute steps
REFRESH = 36                                             # U is recomputed every 3 hours
U_NOMINAL, U_LO, U_HI, SHRINK = 2.0, 0.3, 8.0, 5.0        # W/m2K; SHRINK in units of sum(x^2)
H_SLOW, TAU_SLOW = 4.0, 6 * 3600.0                       # W/m2K store coupling; store time constant, s


def _geo(geo, U):
    face = U["CELL"] * U["HEIGHT"]
    G_nb = (geo["wall_cells"] * face * 2.5 + geo["door_cells"] * face * 25.0) * (1 - np.eye(geo["V"]))
    return face, G_nb


def nodes(geo, U):
    V = geo["V"]
    _, G_nb = _geo(geo, U)
    ni, nj, _ = dlib.pairs(G_nb)
    ext = np.where(geo["facade_faces"] > 0)[0]
    rooms = np.arange(V)
    return {"N": 2 * V, "links": (np.concatenate([rooms, ni]), np.concatenate([V + rooms, nj])), "outside": ext}


def facade_u(geo, hist, U):
    """Closed-form estimate of the outside-wall U-value from the history buffer, pooled over exterior rooms."""
    face, G_nb = _geo(geo, U)
    dt = U["DT"]; Cair = U["RHO_C_AIR"] * U["HEIGHT"] * geo["area"]
    Th = hist["T"][0]                                                 # (H, V); input j drives T[j-1] -> T[j]
    prev, nxt = Th[:-1], Th[1:]
    inp = {k: hist[k][0, 1:] for k in ("g", "T_supply", "Q_reheat", "Q_gain", "occupants", "T_out")}
    w = np.exp(-np.arange(len(Th))[::-1] * dt / TAU_SLOW)
    T_slow = (w / w.sum()) @ Th                                       # the store, summarised by an average of history
    G_sum = G_nb.sum(1)
    T_nb = (prev @ G_nb.T) / np.maximum(G_sum, 1e-9)
    Q = inp["Q_reheat"] + inp["Q_gain"] + U["OCC_W"] * inp["occupants"][:, None] / geo["V"]
    terms = [(inp["g"] * dt / Cair, inp["T_supply"]), (G_sum * dt / Cair, T_nb), (H_SLOW * geo["area"] * dt / Cair, T_slow)]
    base = dlib.assemble(prev, terms, Q * dt / Cair)
    den = 1 + sum(np.maximum(wk, 0) for wk, _ in terms)
    A_fac = geo["facade_faces"] * face
    x = A_fac * dt / Cair * (inp["T_out"][:, None] - prev) / den      # effect of U = 1 W/m2K
    r = nxt - base
    ext = A_fac > 0
    sxy, sxx = float((x * r)[:, ext].sum()), float((x * x)[:, ext].sum())
    return float(np.clip((sxy + SHRINK * U_NOMINAL) / (sxx + SHRINK), U_LO, U_HI))


def coefficients(geo, hist, U, net):
    V = geo["V"]
    face, G_nb = _geo(geo, U)
    _, _, g_nb = dlib.pairs(G_nb)
    A = geo["area"]; G_slow = H_SLOW * A
    C = np.concatenate([U["RHO_C_AIR"] * U["HEIGHT"] * A, G_slow * TAU_SLOW])
    ext = np.where(geo["facade_faces"] > 0)[0]
    G_out = geo["facade_faces"][ext] * face * facade_u(geo, hist, U)
    return {"C": C, "G": np.concatenate([G_slow, g_nb]), "G_out": G_out}
