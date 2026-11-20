# SYSTEM

see ../../../../system_prompt.md

# USER

Generation 2, island 1.  Operator: CROSSOVER.

### Parent A: seed_cf_history_slab  [g000_i1_0001, island 1, score 2.5772]
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


### Parent B: mock_mut_g1_0  [g001_i1_0013, island 1, score 2.4253]
Hypothesis: [mock mutation, no LLM] rescaled 25 -> 20, 4 -> 2.4 in seed_cf_history_slab
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
HISTORY = 288
U_NOMINAL, U_LO, U_HI, SHRINK = (2.0, 0.3, 8.0, 5.0)
H_SLOW, TAU_SLOW = (2.4, 6 * 12)

def prepare(geo, U):
    face = U['CELL'] * U['HEIGHT']
    G_nb = (geo['wall_cells'] * face * 2.5 + geo['door_cells'] * face * 20.0) * (1 - np.eye(geo['V']))
    return dict(C=U['RHO_C_AIR'] * U['HEIGHT'] * geo['area'], A_fac=geo['facade_faces'] * face, G_nb=G_nb, G_nb_sum=G_nb.sum(1), G_slow=H_SLOW * geo['area'])

def _known(coef, geo, T, inp, U, T_slow):
    """Heat-balance terms other than the facade, as (terms, b) for dlib.assemble."""
    C, dt = (coef['C'], U['DT'])
    T_nb = T @ coef['G_nb'].T / np.maximum(coef['G_nb_sum'], 1e-09)
    Q = inp['Q_reheat'] + inp['Q_gain'] + U['OCC_W'] * inp['occupants'][..., None] / geo['V']
    return ([(inp['g'] * dt / C, inp['T_supply']), (coef['G_nb_sum'] * dt / C, T_nb), (coef['G_slow'] * dt / C, T_slow)], Q * dt / C)

def _slow(hist_T):
    w = np.exp(-np.arange(hist_T.shape[1])[::-1] / TAU_SLOW)
    return np.tensordot(hist_T, w / w.sum(), axes=([1], [0]))

def facade_u(coef, geo, hist, U):
    """Closed-form estimate of the facade U-value from the buffer (pooled over exterior rooms, per batch row)."""
    Th = hist['T']
    prev, nxt = (Th[:, :-1], Th[:, 1:])
    inp = {k: hist[k][:, 1:] for k in ('g', 'T_supply', 'Q_reheat', 'Q_gain', 'occupants', 'T_out')}
    T_slow = _slow(Th)[:, None, :]
    terms, b = _known(coef, geo, prev, inp, U, T_slow)
    base = dlib.assemble(prev, terms, b)
    den = 1 + sum((np.maximum(w, 0) for w, _ in terms))
    x = coef['A_fac'] * U['DT'] / coef['C'] * (inp['T_out'][..., None] - prev) / den
    r = nxt - base
    ext = coef['A_fac'] > 0
    sxy = (x * r)[..., ext].sum((1, 2))
    sxx = (x * x)[..., ext].sum((1, 2))
    return np.clip((sxy + SHRINK * U_NOMINAL) / (sxx + SHRINK), U_LO, U_HI)

def step(coef, geo, hist, u, U):
    T = hist['T'][:, -1]
    terms, b = _known(coef, geo, T, u, U, _slow(hist['T']))
    Uf = facade_u(coef, geo, hist, U)
    terms.append((coef['A_fac'] * Uf[:, None] * U['DT'] / coef['C'], u['T_out'][:, None]))
    return dlib.assemble(T, terms, b)
```


Pre-crossover analysis:
Where each parent is better (A vs B, main-horizon rollout RMSE, K):
  all rooms                              2.500 vs 2.345  -> B
  exterior rooms                         2.739 vs 2.583  -> B
  interior rooms                         1.645 vs 1.473  -> B
  most exposed third of exterior rooms   3.026 vs 2.863  -> B
  HVAC air on                            1.520 vs 1.703  -> A
  HVAC air off                           3.391 vs 2.994  -> B
  comfort hours                          1.520 vs 1.703  -> A
  night/weekend                          3.391 vs 2.994  -> B
  room being reheated                    1.814 vs 2.082  -> A
  room not reheated                      2.640 vs 2.405  -> B
  colder half of outside temps           3.489 vs 3.177  -> B
  warmer half                            1.288 vs 1.392  -> A
A is clearly better on: HVAC air on, comfort hours, room being reheated, warmer half
B is clearly better on: all rooms, exterior rooms, interior rooms, most exposed third of exterior rooms, HVAC air off, night/weekend, room not reheated, colder half of outside temps
templates (top-level functions): same in both: _known, _slow, facade_u, step; different: prepare; only in A: none; only in B: none

Recent failures on this island (do not repeat):
- mock_cro_g1_1 [rejected]: physics checks failed
    exception during checks: KeyError: 'G_slow'

Write 2 children that COMBINE the parents: take each template from the parent that is better where that template matters (e.g. the exterior-wall template from the parent better on exterior rooms), make the pieces consistent, and state in the hypothesis which template came from which parent.
