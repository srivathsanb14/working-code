"""Starting model: the audited room network, written as coefficient formulas.

Nodes: every room's air, its slab and furniture, its outside wall (rooms that have one) and every shared wall.
Every coefficient is a floor-plan count x a material number.  The material numbers are universal constants: the
middle (geometric mean for log-drawn properties) of the kit generator's ranges (smart_control/utils/materials.py),
and the outside film is the middle of the stations' range.  Nothing is read from history and nothing is fitted;
the framework sets the room-air nodes to the measurements every step, so the hidden nodes follow the measured history.
"""
import numpy as np
import dlib

HISTORY = 1
REFRESH = 288                             # the coefficients do not depend on history, so one build a day is enough

K_EXT = (0.15 * 0.55) ** 0.5               # W/mK   outside-wall conductivity, log-middle of 0.15-0.55
RC_EXT = (350 * 2400) ** 0.5 * 920.0       # J/m3K  outside-wall density (log-middle 350-2400) x heat capacity (middle 840-1000)
K_IW = 0.525                               # W/mK   shared-wall conductivity, middle of 0.45-0.60
RC_IW = (110 * 700) ** 0.5 * 900.0         # J/m3K  shared-wall density (log-middle) x fixed heat capacity 900
RC_MASS = (10 * 40) ** 0.5 * 1000.0        # J/m3K  slab/furniture density (log-middle 10-40) x 1000
H_FILM = 17.75                             # W/m2K  outside film, middle of the stations' 8.8-26.7
H_SLAB, A_SPREAD, K_AIR = 5.0, 0.6, 50.0   # slab-air coupling per m2, spreading-resistance strength, mixed-air conductivity


def _spread(area, length):
    """Spreading resistance (K/W): heat entering through an opening of this length spreads through the room's air."""
    return A_SPREAD * np.log1p(np.sqrt(area) / np.maximum(length, 1e-9)) / (np.pi * K_AIR * dlib.HEIGHT)


def _layout(geo):
    V = geo["V"]
    ext = np.where(geo["facade_faces"] > 0)[0]
    wi, wj, wn = dlib.pairs(geo["wall_cells"]); di, dj, dn = dlib.pairs(geo["door_cells"])
    slab, fac, wall = V, 2 * V, 2 * V + len(ext)               # first index of each node group
    return V, ext, (wi, wj, wn), (di, dj, dn), slab, fac, wall


def nodes(geo, U):
    V, ext, (wi, wj, wn), (di, dj, dn), slab, fac, wall = _layout(geo)
    rooms = np.arange(V); w = wall + np.arange(len(wi))
    li = np.concatenate([rooms, ext, wi, wj, di])
    lj = np.concatenate([slab + rooms, fac + np.arange(len(ext)), w, w, dj])
    return {"N": wall + len(wi), "links": (li, lj), "outside": fac + np.arange(len(ext))}


def coefficients(geo, hist, U, net):
    V, ext, (wi, wj, wn), (di, dj, dn), slab, fac, wall = _layout(geo)
    A, H, c = geo["area"], U["HEIGHT"], U["CELL"]; face = H * c
    nf = geo["facade_faces"][ext]
    C = np.concatenate([U["RHO_C_AIR"] * H * A,                       # room air
                        RC_MASS * H * A,                              # slab and furniture
                        RC_EXT * nf * face * c / 2,                   # outside wall (half-thickness cells)
                        RC_IW * wn * face * c])                       # shared walls
    g_fac = nf * face / (c / (2 * K_AIR) + c / (4 * K_EXT))
    g_half_i = 1 / (1 / (wn * face / (c / (2 * K_AIR) + c / (2 * K_IW))) + _spread(A[wi], wn * c))
    g_half_j = 1 / (1 / (wn * face / (c / (2 * K_AIR) + c / (2 * K_IW))) + _spread(A[wj], wn * c))
    g_door = 1 / (1 / (dn * face / (2 * c / K_AIR)) + _spread(A[di], dn * c) + _spread(A[dj], dn * c))
    G = np.concatenate([H_SLAB * A,                                     # air - slab
                        1 / (1 / g_fac + _spread(A[ext], nf * c)),      # air - outside wall
                        g_half_i, g_half_j,                             # air - shared wall, both sides
                        g_door])                                        # air - air through doors
    G_out = nf * face / (c / (4 * K_EXT) + 1 / H_FILM)                  # outside wall - outside air
    return {"C": C, "G": G, "G_out": G_out}
