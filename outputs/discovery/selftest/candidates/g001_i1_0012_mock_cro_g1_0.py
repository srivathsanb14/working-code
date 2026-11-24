"""Starting model: one temperature per room (its air), nominal materials, nothing read from history.

Coefficients: supply air and heat through the air capacity (floor area); outside wall = facade area x a nominal
2 W/m2K; neighbour exchange through shared walls and doors = contact area x a nominal conductance per m2.  No slab or
wall store.  This is the baseline the other starting models and every LLM candidate must beat.
"""
import numpy as np
HISTORY = 1
REFRESH = 288

def coefficients(geo, hist, U):
    V, A, H, c, dt = (geo['V'], geo['area'], U['HEIGHT'], U['CELL'], U['DT'])
    face = H * c
    C_air = U['RHO_C_AIR'] * H * A
    C_slab = RC_MASS * H * A
    G_slab = H_SLAB * A
    nf = geo['facade_faces']
    has = nf > 0
    g_fac = nf * face / (c / (2 * K_AIR) + c / (4 * K_EXT))
    G_wall = np.where(has, 1 / (1 / np.maximum(g_fac, 1e-09) + _spread(A, np.maximum(nf, 1) * c)), 0.0)
    C_wall = np.where(has, RC_EXT * nf * face * c / 2, 1.0)
    G_wall_out = nf * face / (c / (4 * K_EXT) + 1 / H_FILM)
    n = geo['wall_cells']
    g_half = n * face / (c / (2 * K_AIR) + c / (2 * K_IW))
    gi = 1 / (1 / np.maximum(g_half, 1e-12) + _spread(A[:, None], np.maximum(n, 1) * c))
    gi = np.where(n > 0, gi, 0.0)
    G_wall_ij = np.where(n > 0, 1 / (1 / np.maximum(gi, 1e-12) + 1 / np.maximum(gi.T, 1e-12)), 0.0)
    G_inner = gi.sum(1)
    C_inner = np.maximum(RC_IW * n.sum(1) * face * c / 2, 1.0)
    d = geo['door_cells']
    g_door = d * face / (2 * c / K_AIR)
    G_door = np.where(d > 0, 1 / (1 / np.maximum(g_door, 1e-12) + _spread(A[:, None], np.maximum(d, 1) * c) + _spread(A[None, :], np.maximum(d, 1) * c)), 0.0)
    return {'air': dt / C_air, 'heat': dt / C_air, 'neighbours': dt * (G_wall_ij + G_door) / C_air[:, None], 'slab': dt * G_slab / C_air, 'slab_back': dt * G_slab / C_slab, 'wall': dt * G_wall / C_air, 'wall_back': dt * G_wall / C_wall, 'wall_out': np.where(has, dt * G_wall_out / C_wall, 0.0), 'inner': dt * G_inner / C_air, 'inner_back': dt * G_inner / C_inner}
