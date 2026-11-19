"""Explicit-Euler single-node model (expected to be rejected: unstable with large air streams)."""
import numpy as np

PARAMS = {"k_door": (30.0, 1.0, 500.0)}
THETA = {"cap": (60e3, 5e3, 500e3), "u": (1.5, 0.05, 10.0)}


def simulate(geo, inputs, theta, p, T0):
    V = geo["V"]; n = len(inputs["T_out"])
    C = theta["cap"] * geo["area"]; Ua = theta["u"] * geo["facade_faces"]; K = p["k_door"] * geo["door_cells"]
    T = np.full(V, T0, float); out = np.empty((n, V))
    for t in range(n):
        q = Ua * (inputs["T_out"][t] - T) + K @ T - K.sum(1) * T + inputs["g"][t] * (inputs["T_supply"][t] - T) + inputs["Q_gain"][t]
        T = T + 300.0 * q / C
        out[t] = T
    return out
