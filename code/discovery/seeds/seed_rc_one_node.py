"""One heat store per room: room air and everything in it share one capacity.

Each room exchanges heat with the outside through its facade (conductance per
facade segment), with neighbours through shared walls (per wall cell) and doors
(per door cell), and receives the supply-air stream, reheat, internal gains and
an equal share of the occupants.  The naive baseline every richer hypothesis
must beat.
"""
import numpy as np
import dlib

PARAMS = {
    "k_wall": (1.5, 0.1, 20.0),      # W/K per partition cell shared by two rooms
    "k_door": (30.0, 1.0, 500.0),    # W/K per door cell
}
THETA = {
    "cap": (60e3, 5e3, 500e3),       # J/K per m2 of floor
    "u_facade": (1.5, 0.05, 10.0),   # W/K per 0.2 m facade segment
}


def network(geo, inputs, theta, p):
    """The RC network and its per-step forcing (lets test_model.py run a Kalman filter)."""
    V = geo["V"]
    C = theta["cap"] * geo["area"]
    G = p["k_wall"] * geo["wall_cells"] + p["k_door"] * geo["door_cells"]
    G_amb = theta["u_facade"] * geo["facade_faces"]
    net = dlib.LinearNetwork(C, G, G_amb, V)
    Q = inputs["Q_reheat"] + inputs["Q_gain"] + 70.0 * inputs["occupants"][:, None] / V
    return net, dict(T_out=inputs["T_out"], g=inputs["g"], T_supply=inputs["T_supply"], Q=Q)


def simulate(geo, inputs, theta, p, T0):
    net, f = network(geo, inputs, theta, p)
    return net.simulate(f["T_out"], f["g"], f["T_supply"], f["Q"], T0)
