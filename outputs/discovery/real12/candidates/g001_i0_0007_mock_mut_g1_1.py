"""Starting model: room air plus a slab store, with the outside-wall U-value computed from the measured history.

Coefficients of the fixed state-space function: neighbour exchange through shared walls and doors (contact area x a
nominal conductance per m2), a slab store coupled through floor area x 4 W/m2K with a 6 h time constant, and the
outside air through the wall area x U.  U is not fixed: every REFRESH steps it is computed from the last 24 h of
measurements as a shrunk least-squares ratio (how much the exterior rooms' measured changes follow T_out - T beyond
what the other terms explain), pulled toward a nominal value while the history is short.  Nothing is fitted.
"""
import numpy as np
import dlib
HISTORY = 288
REFRESH = 36
U_NOMINAL, U_LO, U_HI, SHRINK = (1.2, 0.3, 8.0, 5.0)
H_SLOW, TAU_SLOW = (4.0, 6 * 3600.0)

def _geo(geo, U):
    face = U['CELL'] * U['HEIGHT']
    G_nb = (geo['wall_cells'] * face * 2.5 + geo['door_cells'] * face * 25.0) * (1 - np.eye(geo['V']))
    return (face, G_nb)

def facade_u(geo, hist, U):
    """Closed-form estimate of the outside-wall U-value from the history buffer, pooled over exterior rooms."""
    face, G_nb = _geo(geo, U)
    dt = U['DT']
    Cair = U['RHO_C_AIR'] * U['HEIGHT'] * geo['area']
    Th = hist['T'][0]
    prev, nxt = (Th[:-1], Th[1:])
    inp = {k: hist[k][0, 1:] for k in ('g', 'T_supply', 'Q_reheat', 'Q_gain', 'occupants', 'T_out')}
    w = np.exp(-np.arange(len(Th))[::-1] * dt / TAU_SLOW)
    T_slow = w / w.sum() @ Th
    G_sum = G_nb.sum(1)
    T_nb = prev @ G_nb.T / np.maximum(G_sum, 1e-09)
    Q = inp['Q_reheat'] + inp['Q_gain'] + U['OCC_W'] * inp['occupants'][:, None] / geo['V']
    terms = [(inp['g'] * dt / Cair, inp['T_supply']), (G_sum * dt / Cair, T_nb), (H_SLOW * geo['area'] * dt / Cair, T_slow)]
    base = dlib.assemble(prev, terms, Q * dt / Cair)
    den = 1 + sum((np.maximum(wk, 0) for wk, _ in terms))
    A_fac = geo['facade_faces'] * face
    x = A_fac * dt / Cair * (inp['T_out'][:, None] - prev) / den
    r = nxt - base
    ext = A_fac > 0
    sxy, sxx = (float((x * r)[:, ext].sum()), float((x * x)[:, ext].sum()))
    return float(np.clip((sxy + SHRINK * U_NOMINAL) / (sxx + SHRINK), U_LO, U_HI))

def coefficients(geo, hist, U):
    face, G_nb = _geo(geo, U)
    dt = U['DT']
    C_air = U['RHO_C_AIR'] * U['HEIGHT'] * geo['area']
    return {'air': dt / C_air, 'heat': dt / C_air, 'neighbours': dt * G_nb / C_air[:, None], 'slab': dt * H_SLOW * geo['area'] / C_air, 'slab_back': dt / TAU_SLOW, 'outside': dt * geo['facade_faces'] * face * facade_u(geo, hist, U) / C_air}
