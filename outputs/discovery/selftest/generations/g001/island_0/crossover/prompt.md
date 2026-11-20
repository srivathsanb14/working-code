# SYSTEM

see ../../../../system_prompt.md

# USER

Generation 1, island 0.  Operator: CROSSOVER.

### Parent A: seed_cf_history_slab  [g000_i0_0000, island 0, score 2.5772]
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


### Parent B: seed_cf_one_node  [g000_i0_0002, island 0, score 3.3729]
Hypothesis: One temperature per room, nominal materials, no history beyond the last measurement.
```python
"""One temperature per room, nominal materials, no history beyond the last measurement.

Templates: room air capacity from floor area; facade loss = facade area x a nominal U-value; neighbour
exchange through shared walls and doors = contact area x nominal conductance per m2.  Every coefficient
is geometry x material; the material numbers are fixed universal guesses (nothing fitted, nothing
inferred from history), so this is the baseline the history-based candidates must beat.
"""
import numpy as np
import dlib

HISTORY = 1


def prepare(geo, U):
    face = U["CELL"] * U["HEIGHT"]                       # m2 of one 0.2 m boundary segment
    C = U["RHO_C_AIR"] * U["HEIGHT"] * geo["area"]       # J/K, room air
    G_ext = geo["facade_faces"] * face * 2.0             # W/K, facade + film, nominal 2 W/m2K
    G_nb = geo["wall_cells"] * face * 2.5 + geo["door_cells"] * face * 25.0
    G_nb = G_nb * (1 - np.eye(geo["V"]))
    return dict(C=C, G_ext=G_ext, G_nb=G_nb, G_nb_sum=G_nb.sum(1))


def step(coef, geo, hist, u, U):
    T = hist["T"][:, -1]                                  # (B, V) latest measured room temperatures
    C, dt = coef["C"], U["DT"]
    T_nb = (T @ coef["G_nb"].T) / np.maximum(coef["G_nb_sum"], 1e-9)
    Q = u["Q_reheat"] + u["Q_gain"] + U["OCC_W"] * u["occupants"][:, None] / geo["V"]
    return dlib.assemble(T, [(u["g"] * dt / C, u["T_supply"]),
                             (coef["G_nb_sum"] * dt / C, T_nb),
                             (coef["G_ext"] * dt / C, u["T_out"][:, None])], b=Q * dt / C)
```


Pre-crossover analysis:
Where each parent is better (A vs B, main-horizon rollout RMSE, K):
  all rooms                              2.500 vs 3.289  -> A
  exterior rooms                         2.739 vs 3.022  -> A
  interior rooms                         1.645 vs 3.950  -> A
  most exposed third of exterior rooms   3.026 vs 2.445  -> B
  HVAC air on                            1.520 vs 2.300  -> A
  HVAC air off                           3.391 vs 4.265  -> A
  comfort hours                          1.520 vs 2.300  -> A
  night/weekend                          3.391 vs 4.265  -> A
  room being reheated                    1.814 vs 2.523  -> A
  room not reheated                      2.640 vs 3.451  -> A
  colder half of outside temps           3.489 vs 4.488  -> A
  warmer half                            1.288 vs 1.896  -> A
A is clearly better on: all rooms, exterior rooms, interior rooms, HVAC air on, HVAC air off, comfort hours, night/weekend, room being reheated, room not reheated, colder half of outside temps, warmer half
B is clearly better on: most exposed third of exterior rooms
templates (top-level functions): same in both: none; different: prepare, step; only in A: _known, _slow, facade_u; only in B: none

Write 2 children that COMBINE the parents: take each template from the parent that is better where that template matters (e.g. the exterior-wall template from the parent better on exterior rooms), make the pieces consistent, and state in the hypothesis which template came from which parent.
