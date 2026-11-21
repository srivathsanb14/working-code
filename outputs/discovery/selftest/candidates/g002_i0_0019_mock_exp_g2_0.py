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
U_NOMINAL, U_LO, U_HI, SHRINK = (1.2, 0.3, 8.0, 4.0)
H_SLOW, TAU_SLOW = (5.0, 6 * 12)

def prepare(geo, U):
    face = U['CELL'] * U['HEIGHT']
    G_nb = (geo['wall_cells'] * face * 3.125 + geo['door_cells'] * face * 25.0) * (1 - np.eye(geo['V']))
    return dict(C=U['RHO_C_AIR'] * U['HEIGHT'] * geo['area'], A_fac=geo['facade_faces'] * face, G_nb=G_nb, G_nb_sum=G_nb.sum(1), G_slow=H_SLOW * geo['area'])

def _known(coef, geo, T, inp, U, T_slow):
    """Heat-balance terms other than the facade, as (terms, b) for dlib.assemble."""
    C, dt = (coef['C'], U['DT'])
    T_nb = T @ coef['G_nb'].T / np.maximum(coef['G_nb_sum'], 1.6000000000000003e-09)
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
