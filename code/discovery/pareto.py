"""Multi-objective selection (NSGA-II style): non-dominated sorting + crowding distance.

A candidate A dominates B when A is no worse on every objective and better on at least one.  Front 1 = candidates
nobody dominates; front 2 = those dominated only by front 1; and so on.  Inside a front, the crowding distance
(how far a candidate is from its neighbours along each objective, normalized by the objective's range) prefers
candidates in sparse parts of the trade-off, which keeps the island diverse.

Objectives come from closedform.objectives(); which ones are used is set by --objectives (closedform.OBJECTIVES lists
them with the coefficients each one isolates).  Lower is better for all.
"""
from __future__ import annotations

import math


def vec(rec, keys):
    o = rec.get("objectives") or {}
    return [float(o.get(k, rec.get("score", math.inf) if k == "main" else math.inf)) for k in keys]


def dominates(a, b):
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def fronts(V):
    """Indices of V (list of objective vectors) grouped into non-dominated fronts, best first."""
    n = len(V)
    dom_by = [0] * n; dom = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if dominates(V[i], V[j]):
                dom[i].append(j); dom_by[j] += 1
            elif dominates(V[j], V[i]):
                dom[j].append(i); dom_by[i] += 1
    F, cur = [], [i for i in range(n) if dom_by[i] == 0]
    while cur:
        F.append(cur); nxt = []
        for i in cur:
            for j in dom[i]:
                dom_by[j] -= 1
                if dom_by[j] == 0:
                    nxt.append(j)
        cur = nxt
    return F


def crowding(V, idx):
    d = {i: 0.0 for i in idx}
    if len(idx) <= 2:
        return {i: math.inf for i in idx}
    for m in range(len(V[idx[0]])):
        s = sorted(idx, key=lambda i: V[i][m])
        lo, hi = V[s[0]][m], V[s[-1]][m]
        d[s[0]] = d[s[-1]] = math.inf
        if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
            continue
        for a, i, b in zip(s, s[1:], s[2:]):
            d[i] += (V[b][m] - V[a][m]) / (hi - lo)
    return d


def rank(recs, keys):
    """{id: (front number, -crowding)}: sort ascending for best first."""
    V = [vec(r, keys) for r in recs]
    out = {}
    for f, idx in enumerate(fronts(V)):
        cd = crowding(V, idx)
        for i in idx:
            out[recs[i]["id"]] = (f, -cd[i])
    return out


def select(recs, k, keys):
    """The k best records by (front, crowding)."""
    R = rank(recs, keys)
    return sorted(recs, key=lambda r: R[r["id"]])[:k]


def front(recs, keys):
    """The non-dominated records."""
    R = rank(recs, keys)
    return [r for r in recs if R[r["id"]][0] == 0]
