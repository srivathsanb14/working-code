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
H_SLOW, TAU_SLOW = (4.0, 6 * 12)

def prepare(geo, U):
    V = geo['V']
    A = geo['area']
    H = U['HEIGHT']
    c = U['CELL']
    face = H * c
    wi, wj, wn = dlib.pairs(geo['wall_cells'])
    di, dj, dn = dlib.pairs(geo['door_cells'])
    N = 3 * V + len(wi)
    C = np.empty(N)
    G = np.zeros((N, N))
    G_amb = np.zeros(N)
    air, slab, fac = (np.arange(V), V + np.arange(V), 2 * V + np.arange(V))
    C[air] = U['RHO_C_AIR'] * H * A
    C[slab] = RC_MASS * H * A
    G[air, slab] = G[slab, air] = H_SLAB * A
    nf = geo['facade_faces']
    has_f = nf > 0
    C[fac] = np.where(has_f, RC_EXT * nf * face * c / 2, 1.0)
    g_in = nf * face / (c / (2 * K_AIR) + c / (4 * K_EXT))
    G[air, fac] = G[fac, air] = np.where(has_f, 1 / (1 / np.maximum(g_in, 1e-09) + _spread(A, nf * c)), 0.0)
    G_amb[fac] = np.where(has_f, nf * face / (c / (4 * K_EXT) + 1 / H_FILM), 0.0)
    for k in range(len(wi)):
        i, j, n = (wi[k], wj[k], wn[k])
        w = 3 * V + k
        C[w] = RC_IW * n * face * c
        g_half = n * face / (c / (2 * K_AIR) + c / (2 * K_IW))
        G[i, w] = G[w, i] = 1 / (1 / g_half + _spread(A[i], n * c))
        G[j, w] = G[w, j] = 1 / (1 / g_half + _spread(A[j], n * c))
    for i, j, n in zip(di, dj, dn):
        g_door = n * face / (2 * c / K_AIR)
        G[i, j] = G[j, i] = G[i, j] + 1 / (1 / g_door + _spread(A[i], n * c) + _spread(A[j], n * c))
    return dict(net=dlib.LinearNetwork(C, G, G_amb, V), V=V)

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
