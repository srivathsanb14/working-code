# SYSTEM

see ../../../../system_prompt.md

# USER

Generation 1, island 1.  Operator: MUTATION.

### Parent: seed_cf_history_slab  [g000_i1_0001, island 1, score 2.5772]
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


Pre-mutation analysis of the parent (held-out buildings):
held-out rollout RMSE by horizon (model / persistence), K: 1h 2.339/1.464, 3h 2.498/3.005, 6h 2.326/4.722
one-step (5 min) RMSE 0.6604 K (persistence 0.2169); train buildings 3h 2.389 K
3h rollout error by group (rmse, bias), K:
  all rooms                              2.500  +1.129
  exterior rooms                         2.739  +1.250
  interior rooms                         1.645  +0.786
  most exposed third of exterior rooms   3.026  +1.402
  HVAC air on                            1.520  +0.434
  HVAC air off                           3.391  +2.057
  comfort hours                          1.520  +0.434
  night/weekend                          3.391  +2.057
  room being reheated                    1.814  +1.232
  room not reheated                      2.640  +1.104
  colder half of outside temps           3.489  +2.612
  warmer half                            1.288  -0.019
skill vs persistence by horizon (ratio <1 is better): 1h 1.60, 3h 0.83, 6h 0.49
worst rooms: building_train_5_synth0_mixed room #6 (rmse 3.76, bias +1.75, facade faces 74, area 35 m2, doors 0); building_train_5_synth1_mixed room #22 (rmse 3.56, bias +1.65, facade faces 79, area 55 m2, doors 0); building_train_5_synth0_mixed room #45 (rmse 3.54, bias +1.61, facade faces 93, area 78 m2, doors 0); building_train_5_synth1_mixed room #63 (rmse 3.51, bias +1.64, facade faces 111, area 74 m2, doors 0)
complexity 793 AST nodes, history 288 steps

Measured vs predicted (the parent's 3 h-ahead forecast) for two rooms:
episode building_train_5_synth0_mixed; room #57 = most exposed exterior room, room #10 = largest interior room
time  | room #57 measured / predicted | room #10 measured / predicted | T_out | HVAC air
   0h |   5.02 /   7.94 C |  14.00 /  15.41 C |  -2.0 C | off
   2h |   2.91 /   5.85 C |  12.74 /  14.52 C |  -3.9 C | off
   4h |   1.70 /   5.21 C |  11.39 /  13.75 C |  -4.0 C | off
   6h |   1.68 /   5.76 C |  10.12 /  13.04 C |  -2.2 C | off
   8h |   9.05 /  11.90 C |  13.73 /  15.70 C |   1.0 C | on
  10h |  12.64 /  13.39 C |  14.71 /  15.47 C |   4.6 C | on
  12h |  13.26 /  12.84 C |  15.85 /  15.41 C |   7.7 C | on
  14h |  13.96 /  13.36 C |  16.21 /  15.50 C |   9.5 C | on
  16h |  13.74 /  12.86 C |  16.70 /  15.96 C |   9.4 C | on
  18h |  13.52 /  12.57 C |  16.91 /  16.21 C |   7.7 C | on
  20h |  10.26 /   8.02 C |  16.11 /  14.82 C |   4.5 C | off
  22h |   7.57 /   5.55 C |  15.34 /  14.29 C |   0.8 C | off

Write 2 children of this parent.  Each must change ONE thing that the analysis points to (a template's formula, a new template, a different history summary, a history-inferred material value) and keep the rest. Say in the hypothesis which diagnostic line motivates the change.
