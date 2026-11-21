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
