"""Coefficient form: the LLM writes the formulas for every coefficient of the state-space model; this module builds
the model from them and steps it.  Shared by the offline scorer (closedform.py) and the live policy
(runtime/closedform_tracker.py), so both compute exactly the same numbers.

A candidate in this form defines

    HISTORY = 288      # steps of measured history its coefficient formulas may read (1..288)
    REFRESH = 36       # the coefficients are recomputed from the history every REFRESH steps (1..288)

    def nodes(geo, U):
        # the model's nodes: the first V are the room-air nodes, in room order; any others are hidden
        # (slab, outside wall, shared wall, ...).  Returns
        #   {"N": int, "links": (i, j) int arrays, one entry per pair of nodes that exchange heat,
        #    "outside": int array of the nodes that exchange heat with the outside air}
        return net

    def coefficients(geo, hist, U, net):
        # every number of the model, written as formulas of geometry, universal constants and measured history
        #   {"C": (N,) heat capacities, J/K, > 0,
        #    "G": (len(links),) conductances between linked nodes, W/K, >= 0,
        #    "G_out": (len(outside),) conductances to the outside air, W/K, >= 0}
        return coef

    def heat(geo, u, U):          # optional: direct heat into each room's air, W, (B, V)
        return Q                  # default: reheat + lights/equipment + 70 W per person, people split evenly

From these the framework builds, every REFRESH steps,
    M(g) = diag(C)/dt + L + diag(g on the air nodes),   L = conductance matrix (links and outside links)
    x(t+1) = M(g)^-1 [ diag(C)/dt x(t) + G_out T_out + g * T_supply + Q ]       (backward Euler, as the simulator)
i.e. x(t+1) = A(g) x(t) + B(g) u(t) + E(g) d(t), and after every step it sets the room-air nodes to the measured
room temperatures (hidden nodes keep their predicted values).  Nothing is fitted anywhere.

Test-time information only: `geo` comes from the building metadata (floor plan, zone map, AHU map), `hist` holds
measured room temperatures and the step inputs a policy computes from its observations and its own commands
(closedform.ROOM_KEYS + building-wide observation keys), and `U` the universal constants.  `hist` at a refresh
step k contains only data up to the start of step k.
"""
from __future__ import annotations

import numpy as np

from . import dlib

DEFAULT_REFRESH = 36


def is_coefform(mod):
    """Any coefficient-formula candidate: the term form (coefficients only) or the network form (nodes + coefficients)."""
    return hasattr(mod, "coefficients")


def make_model(mod, geo, U):
    return CoefModel(mod, geo, U) if hasattr(mod, "nodes") else TermModel(mod, geo, U)


TERMS_ROOM = ("air", "heat", "outside", "slab", "slab_back", "wall", "wall_back", "wall_out", "inner", "inner_back")


class TermModel:
    """The term form: one fixed state-space function, with a formula for every coefficient written by the LLM.

    State per room i: its air T_i (measured), a slab store S_i, an outside-wall store W_i and an inner-wall store P_i
    (hidden).  Every 5 minutes

        T_i+ = ( T_i + air_i g_i T_supply_i + outside_i T_out + sum_j nb_ij T_j + slab_i S_i + wall_i W_i + inner_i P_i
                 + heat_i Q_i ) / ( 1 + air_i g_i + outside_i + sum_j nb_ij + slab_i + wall_i + inner_i )
        S_i+ = ( S_i + slab_back_i T_i+ ) / ( 1 + slab_back_i )
        W_i+ = ( W_i + wall_back_i T_i+ + wall_out_i T_out ) / ( 1 + wall_back_i + wall_out_i )
        P_i+ = ( P_i + inner_back_i T_i+ ) / ( 1 + inner_back_i )

    then the room air is set to the measurement.  Each line is the backward-Euler (end-of-step) form of
    x+ = x + sum_k a_k (source_k - x): a weighted average of the sources plus heat, stable for any coefficients >= 0.
    All coefficients are per 5-minute step and dimensionless, except air (per W/K of supply air) and heat (K per W).
    The candidate's coefficients(geo, hist, U) returns them: per room (V,) for TERMS_ROOM, "neighbours" (V, V);
    any term it leaves out is zero.  Nodes: x = (T, S, W, P), N = 4V."""

    def __init__(self, mod, geo, U):
        self.mod, self.geo, self.U = mod, geo, U
        self.V = int(geo["V"]); self.N = 4 * self.V
        self.refresh = int(getattr(mod, "REFRESH", DEFAULT_REFRESH))

    def build(self, hist):
        c = self.mod.coefficients(self.geo, hist, self.U)
        if not isinstance(c, dict):
            raise ValueError("coefficients() must return a dict of coefficient arrays")
        unknown = set(c) - set(TERMS_ROOM) - {"neighbours"}
        if unknown:
            raise ValueError(f"coefficients(): unknown terms {sorted(unknown)}; allowed: {list(TERMS_ROOM) + ['neighbours']}")
        V = self.V; out = {}
        for k in TERMS_ROOM:
            try:
                a = np.broadcast_to(np.asarray(c.get(k, 0.0), float), (V,)).copy()
            except ValueError:
                raise ValueError(f"coefficients(): '{k}' must be a number or a (V,) array") from None
            out[k] = a
        nb = np.asarray(c.get("neighbours", np.zeros((V, V))), float)
        if nb.shape != (V, V):
            raise ValueError(f"coefficients(): 'neighbours' must be (V, V), got {nb.shape}")
        out["neighbours"] = nb * (1 - np.eye(V))
        for k, a in out.items():
            if not np.all(np.isfinite(a)):
                raise ValueError(f"coefficients(): non-finite values in '{k}'")
            if np.any(a < 0):
                raise ValueError(f"coefficients(): need every coefficient >= 0 ('{k}' has a negative entry)")
        out["nb_sum"] = out["neighbours"].sum(1)
        return out

    def heat(self, u):
        f = getattr(self.mod, "heat", None)
        Q = f(self.geo, u, self.U) if f is not None else default_heat(self.geo, u, self.U)
        return np.asarray(Q, float)

    def init(self, T_first):
        T_first = np.asarray(T_first, float)
        x = np.full(self.N, float(np.mean(T_first)))
        x[: self.V] = T_first
        return x

    def predict(self, c, x, u):
        x = np.atleast_2d(np.asarray(x, float)); V = self.V
        T, S, W, P = x[:, :V], x[:, V:2 * V], x[:, 2 * V:3 * V], x[:, 3 * V:]
        g = np.asarray(u["g"], float); Ts = np.asarray(u["T_supply"], float)
        To = np.asarray(u["T_out"], float).reshape(-1, 1)
        wa = c["air"] * g
        num = T + wa * Ts + c["outside"] * To + T @ c["neighbours"].T + c["slab"] * S + c["wall"] * W + c["inner"] * P + c["heat"] * self.heat(u)
        Tn = num / (1 + wa + c["outside"] + c["nb_sum"] + c["slab"] + c["wall"] + c["inner"])
        Sn = (S + c["slab_back"] * Tn) / (1 + c["slab_back"])
        Wn = (W + c["wall_back"] * Tn + c["wall_out"] * To) / (1 + c["wall_back"] + c["wall_out"])
        Pn = (P + c["inner_back"] * Tn) / (1 + c["inner_back"])
        return np.concatenate([Tn, Sn, Wn, Pn], 1)

    def correct(self, x, T_measured):
        return dlib.correct_rooms(x, T_measured, self.V)


def default_heat(geo, u, U):
    occ = np.asarray(u["occupants"], float)
    return u["Q_reheat"] + u["Q_gain"] + U["OCC_W"] * occ[..., None] / geo["V"]


class CoefModel:
    """One candidate on one building: the node structure, and the network built from its coefficient formulas."""

    def __init__(self, mod, geo, U):
        self.mod, self.geo, self.U = mod, geo, U
        self.V = int(geo["V"])
        self.refresh = int(getattr(mod, "REFRESH", DEFAULT_REFRESH))
        net = mod.nodes(geo, U)
        try:
            self.N = int(net["N"])
            li, lj = (np.asarray(a, int).ravel() for a in net["links"])
            self.out = np.asarray(net["outside"], int).ravel()
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"nodes() must return {{'N': int, 'links': (i, j), 'outside': idx}}: {e}") from None
        if self.N < self.V:
            raise ValueError(f"nodes(): N = {self.N} is smaller than the number of rooms V = {self.V}")
        if li.shape != lj.shape:
            raise ValueError("nodes(): the two link index arrays differ in length")
        for name, a in (("links", np.r_[li, lj]), ("outside", self.out)):
            if a.size and (a.min() < 0 or a.max() >= self.N):
                raise ValueError(f"nodes(): {name} refers to a node outside 0..{self.N - 1}")
        if np.any(li == lj):
            raise ValueError("nodes(): a link joins a node to itself")
        self.net_spec = dict(N=self.N, links=(li, lj), outside=self.out)
        self.li, self.lj = li, lj

    def build(self, hist):
        """dlib.LinearNetwork from the candidate's coefficient formulas evaluated on `hist` (a batch of one window)."""
        c = self.mod.coefficients(self.geo, hist, self.U, self.net_spec)
        try:
            C = np.asarray(c["C"], float).reshape(-1)
            G = np.asarray(c["G"], float).reshape(-1)
            Go = np.asarray(c["G_out"], float).reshape(-1)
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"coefficients() must return {{'C', 'G', 'G_out'}} arrays: {e}") from None
        if C.shape != (self.N,) or G.shape != self.li.shape or Go.shape != self.out.shape:
            raise ValueError(f"coefficients(): shapes C {C.shape}, G {G.shape}, G_out {Go.shape}; expected "
                             f"({self.N},), {self.li.shape}, {self.out.shape}")
        if not (np.all(np.isfinite(C)) and np.all(np.isfinite(G)) and np.all(np.isfinite(Go))):
            raise ValueError("coefficients(): non-finite value")
        if np.any(C <= 0) or np.any(G < 0) or np.any(Go < 0):
            raise ValueError("coefficients(): need C > 0, G >= 0, G_out >= 0")
        Gm = np.zeros((self.N, self.N))
        np.add.at(Gm, (self.li, self.lj), G)
        np.add.at(Gm, (self.lj, self.li), G)
        Ga = np.zeros(self.N)
        np.add.at(Ga, self.out, Go)
        return dlib.LinearNetwork(C, Gm, Ga, self.V)

    def heat(self, u):
        f = getattr(self.mod, "heat", None)
        Q = f(self.geo, u, self.U) if f is not None else default_heat(self.geo, u, self.U)
        return np.asarray(Q, float)

    def init(self, T_first):
        """Every node starts at the mean of the first measured room temperatures; the room-air nodes at the measurement."""
        T_first = np.asarray(T_first, float)
        x = np.full(self.N, float(np.mean(T_first)))
        x[: self.V] = T_first
        return x

    def predict(self, net, x, u):
        """One backward-Euler step for a batch of states x (B, N) with the step inputs u (batched like closedform)."""
        return dlib.network_predict(net, x, u["T_out"], u["g"], u["T_supply"], self.heat(u))

    def correct(self, x, T_measured):
        return dlib.correct_rooms(x, T_measured, self.V)
