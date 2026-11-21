# SYSTEM

see ../../../../system_prompt.md

# USER

Generation 1, island 1.  Operator: CROSSOVER.

### Parent A: seed_cf_history_slab  [g000_i1_0001, island 1, score 2.5778]
Hypothesis: Room air plus a slow store summarised from history, and the facade U-value inferred from history.
```python
"""Room air plus a slow store summarised from history, and the facade U-value inferred from history.

Templates (each answers one physical question):
  passive   neighbour exchange (walls, doors) as in the one-node seed;
            slow store: a room drops fast, then slowly, because heavy materials hold heat.  Instead of a
            hidden slab state, the store's temperature is an exponentially weighted average of the room's
            measured history (averaging a linear system gives a linear system), coupled through floor area.
            facade loss: facade area x U, where U is NOT fixed: a closed-form shrunk least-squares estimate
            over the history buffer (how much the exterior rooms' measured changes follow T_out - T beyond
            what the other templates explain), pulled toward a nominal value while the buffer is short.
  HVAC      supply air treated at the end of the step; reheat as direct heat.
  disturb.  lighting/equipment schedule; occupants spread evenly over rooms.
Nothing is fitted before deployment: the only estimation is the closed-form history expression.
"""
import numpy as np
import dlib

HISTORY = 288                                            # one day of 5-minute steps
U_NOMINAL, U_LO, U_HI, SHRINK = 2.0, 0.3, 8.0, 5.0        # W/m2K; SHRINK in units of sum(x^2)
H_SLOW, TAU_SLOW = 4.0, 6 * 12                           # W/m2K store coupling; averaging time constant, steps


def prepare(geo, U):
    face = U["CELL"] * U["HEIGHT"]
    G_nb = (geo["wall_cells"] * face * 2.5 + geo["door_cells"] * face * 25.0) * (1 - np.eye(geo["V"]))
    return dict(C=U["RHO_C_AIR"] * U["HEIGHT"] * geo["area"], A_fac=geo["facade_faces"] * face,
                G_nb=G_nb, G_nb_sum=G_nb.sum(1), G_slow=H_SLOW * geo["area"])


def _known(coef, geo, T, inp, U, T_slow):
    """Heat-balance terms other than the facade, as (terms, b) for dlib.assemble."""
    C, dt = coef["C"], U["DT"]
    T_nb = (T @ coef["G_nb"].T) / np.maximum(coef["G_nb_sum"], 1e-9)
    Q = inp["Q_reheat"] + inp["Q_gain"] + U["OCC_W"] * inp["occupants"][..., None] / geo["V"]
    return [(inp["g"] * dt / C, inp["T_supply"]), (coef["G_nb_sum"] * dt / C, T_nb),
            (coef["G_slow"] * dt / C, T_slow)], Q * dt / C


def _slow(hist_T):
    w = np.exp(-np.arange(hist_T.shape[1])[::-1] / TAU_SLOW)
    return np.tensordot(hist_T, w / w.sum(), axes=([1], [0]))         # (B, V)


def facade_u(coef, geo, hist, U):
    """Closed-form estimate of the facade U-value from the buffer (pooled over exterior rooms, per batch row)."""
    Th = hist["T"]                                                    # (B, H, V); input j drives T[j-1] -> T[j]
    prev, nxt = Th[:, :-1], Th[:, 1:]
    inp = {k: hist[k][:, 1:] for k in ("g", "T_supply", "Q_reheat", "Q_gain", "occupants", "T_out")}
    T_slow = _slow(Th)[:, None, :]
    terms, b = _known(coef, geo, prev, inp, U, T_slow)
    base = dlib.assemble(prev, terms, b)
    den = 1 + sum(np.maximum(w, 0) for w, _ in terms)
    x = coef["A_fac"] * U["DT"] / coef["C"] * (inp["T_out"][..., None] - prev) / den   # effect of U = 1 W/m2K
    r = nxt - base
    ext = coef["A_fac"] > 0
    sxy = (x * r)[..., ext].sum((1, 2)); sxx = (x * x)[..., ext].sum((1, 2))
    return np.clip((sxy + SHRINK * U_NOMINAL) / (sxx + SHRINK), U_LO, U_HI)    # (B,)


def step(coef, geo, hist, u, U):
    T = hist["T"][:, -1]
    terms, b = _known(coef, geo, T, u, U, _slow(hist["T"]))
    Uf = facade_u(coef, geo, hist, U)
    terms.append((coef["A_fac"] * Uf[:, None] * U["DT"] / coef["C"], u["T_out"][:, None]))
    return dlib.assemble(T, terms, b)
```


### Parent B: seed_cf_network_observer  [g000_i1_0003, island 1, score 2.4331]
Hypothesis: Stateful seed: the audited room network (air, slab, facade, partitions, doors) with a measurement-corrected state.
```python
"""Stateful seed: the audited room network (air, slab, facade, partitions, doors) with a measurement-corrected state.

Nothing is fitted.  Every material number is a universal constant: the middle (geometric mean for log-drawn
properties) of the starter-kit generator's range (smart_control/utils/materials.py), and the outdoor film is the
middle of the stations' range.  The state is corrected every step by the measured room temperatures (room-air
nodes take the measurement; slab, facade and partition nodes follow), so the slow stores are estimated from
history instead of being fitted.  Stateful contract: init_state / predict / correct (see prompts.CF_CONTRACT).
"""
import numpy as np
import dlib

K_EXT = (0.15 * 0.55) ** 0.5               # W/mK   facade conductivity, log-middle of 0.15-0.55
RC_EXT = (350 * 2400) ** 0.5 * 920.0       # J/m3K  facade density (log-middle 350-2400) x heat capacity (middle 840-1000)
K_IW = 0.525                               # W/mK   partition conductivity, middle of 0.45-0.60
RC_IW = (110 * 700) ** 0.5 * 900.0         # J/m3K  partition density (log-middle) x fixed heat capacity 900
RC_MASS = (10 * 40) ** 0.5 * 1000.0        # J/m3K  slab/furniture density (log-middle 10-40) x 1000
H_FILM = 17.75                             # W/m2K  outdoor film, middle of the stations' 8.8-26.7
H_SLAB, A_SPREAD, K_AIR = 5.0, 0.6, 50.0   # slab-air coupling per m2, spreading-resistance strength, mixed-air conductivity


def _spread(area, length):
    return A_SPREAD * np.log1p(np.sqrt(area) / np.maximum(length, 1e-9)) / (np.pi * K_AIR * dlib.HEIGHT)


def prepare(geo, U):
    V = geo["V"]; A = geo["area"]; H = U["HEIGHT"]; c = U["CELL"]; face = H * c
    wi, wj, wn = dlib.pairs(geo["wall_cells"]); di, dj, dn = dlib.pairs(geo["door_cells"])
    N = 3 * V + len(wi)
    C = np.empty(N); G = np.zeros((N, N)); G_amb = np.zeros(N)
    air, slab, fac = np.arange(V), V + np.arange(V), 2 * V + np.arange(V)
    C[air] = U["RHO_C_AIR"] * H * A
    C[slab] = RC_MASS * H * A
    G[air, slab] = G[slab, air] = H_SLAB * A
    nf = geo["facade_faces"]; has_f = nf > 0
    C[fac] = np.where(has_f, RC_EXT * nf * face * c / 2, 1.0)
    g_in = nf * face / (c / (2 * K_AIR) + c / (4 * K_EXT))
    G[air, fac] = G[fac, air] = np.where(has_f, 1 / (1 / np.maximum(g_in, 1e-9) + _spread(A, nf * c)), 0.0)
    G_amb[fac] = np.where(has_f, nf * face / (c / (4 * K_EXT) + 1 / H_FILM), 0.0)
    for k in range(len(wi)):
        i, j, n = wi[k], wj[k], wn[k]; w = 3 * V + k
        C[w] = RC_IW * n * face * c
        g_half = n * face / (c / (2 * K_AIR) + c / (2 * K_IW))
        G[i, w] = G[w, i] = 1 / (1 / g_half + _spread(A[i], n * c))
        G[j, w] = G[w, j] = 1 / (1 / g_half + _spread(A[j], n * c))
    for i, j, n in zip(di, dj, dn):
        g_door = n * face / (2 * c / K_AIR)
        G[i, j] = G[j, i] = G[i, j] + 1 / (1 / g_door + _spread(A[i], n * c) + _spread(A[j], n * c))
    return dict(net=dlib.LinearNetwork(C, G, G_amb, V), V=V)


def init_state(coef, geo, U):
    return np.full(coef["net"].N, U["T0"])          # every cell starts at 294 K


def predict(coef, geo, state, u, U):
    Q = u["Q_reheat"] + u["Q_gain"] + U["OCC_W"] * np.asarray(u["occupants"])[:, None] / geo["V"]
    return dlib.network_predict(coef["net"], state, u["T_out"], u["g"], u["T_supply"], Q)


def correct(coef, geo, state, T_measured, U):
    return dlib.correct_rooms(state, T_measured, coef["V"])
```


Pre-crossover analysis:
Where each parent is better (A vs B, main-horizon rollout RMSE, K):
  all rooms                              2.500 vs 2.284  -> B
  exterior rooms                         2.739 vs 2.480  -> B
  interior rooms                         1.645 vs 1.603  -> B
  most exposed third of exterior rooms   3.026 vs 2.773  -> B
  HVAC air on                            1.520 vs 0.810  -> B
  HVAC air off                           3.391 vs 3.361  -> B
  comfort hours                          1.520 vs 0.810  -> B
  night/weekend                          3.391 vs 3.361  -> B
  room being reheated                    1.814 vs 0.746  -> B
  room not reheated                      2.640 vs 2.520  -> B
  colder half of outside temps           3.489 vs 3.324  -> B
  warmer half                            1.288 vs 0.833  -> B
B is clearly better on: all rooms, exterior rooms, interior rooms, most exposed third of exterior rooms, HVAC air on, comfort hours, room being reheated, room not reheated, colder half of outside temps, warmer half
templates (top-level functions): same in both: none; different: prepare; only in A: _known, _slow, facade_u, step; only in B: _spread, init_state, predict, correct

Write 2 children that COMBINE the parents: take each template from the parent that is better where that template matters (e.g. the exterior-wall template from the parent better on exterior rooms), make the pieces consistent, and state in the hypothesis which template came from which parent.
