"""Starting model: one node per room (its air), nominal materials, nothing read from history.

Coefficients: air capacity from floor area; outside wall = facade area x a nominal 2 W/m2K; neighbour exchange
through shared walls and doors = contact area x a nominal conductance per m2.  This is the baseline the other
starting models and every LLM candidate must beat.
"""
import numpy as np
import dlib

HISTORY = 1
REFRESH = 288


def _nb(geo, U):
    face = U["CELL"] * U["HEIGHT"]
    return face, (geo["wall_cells"] * face * 2.5 + geo["door_cells"] * face * 25.0) * (1 - np.eye(geo["V"]))


def nodes(geo, U):
    _, G_nb = _nb(geo, U)
    i, j, _ = dlib.pairs(G_nb)
    return {"N": geo["V"], "links": (i, j), "outside": np.where(geo["facade_faces"] > 0)[0]}


def coefficients(geo, hist, U, net):
    face, G_nb = _nb(geo, U)
    _, _, g = dlib.pairs(G_nb)
    ext = np.where(geo["facade_faces"] > 0)[0]
    return {"C": U["RHO_C_AIR"] * U["HEIGHT"] * geo["area"], "G": g, "G_out": geo["facade_faces"][ext] * face * 2.0}
