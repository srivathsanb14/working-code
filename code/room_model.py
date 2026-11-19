"""Inference-time room thermal model built from the metadata arrays alone.

Everything here uses only what a policy is given at construction
(`floor_plan`, `zone_map`) plus constants that are fixed in the simulator code
(20 cm cells, 3 m floor height, air and slab constants) and a handful of
building-level material scalars `theta` that are NOT given and must be
estimated:

    theta = dict(k_iw, rc_iw, k_ext, rc_ext, rc_mass, h)

The module has three layers.

1. `grid_geometry`   - replicates the simulator's own preprocessing of the two
   arrays (exterior shell, enlarged exterior walls, partitions, doors, rooms,
   boundary-cell halving) so every control volume gets a material class and
   size.  No simulator import.
2. `cluster_model`   - lumps the grid into a small RC network: one air node and
   one slab node per room, one node per partition (cells grouped by the rooms
   they face), one facade node per room, one node per door.  Every capacitance
   and conductance is a sum over faces of a closed-form expression, so each
   room-level coefficient is (geometry count) x (function of theta).
3. `simulate`        - backward-Euler stepping of that network with the VAV
   air stream treated implicitly, exactly as the simulator treats it.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy import ndimage

# numpy on macOS Accelerate raises spurious FP flags from matmul on finite data
warnings.filterwarnings("ignore", message=".*encountered in matmul")

CV = 0.2          # m, control_volume_cm = 20 (hard-coded in scenario_generator)
Z = 3.0           # m, floor_height_cm = 300 (hard-coded)
K_AIR = 50.0      # W/mK, interior_cv_conductivity (class B, fixed)
RC_AIR = 1.2 * 1005.0
K_MASS = 15.0     # interior_mass_cv_conductivity (fixed)
C_MASS = 1000.0   # interior_mass_cv_heat_capacity (fixed); density varies
C_IW = 900.0      # interior_wall_cv_heat_capacity (fixed); density varies
DT = 300.0

EXT_SPACE, SHELL_OR_EXT, IWALL, AIR = 0, 1, 2, 3


def _pad(a):
    """guarantee_air_padding_in_frame: add a ring of exterior where walls touch the frame."""
    a = np.asarray(a)
    if np.any(a[0, :] == 1):
        a = np.vstack([np.full((1, a.shape[1]), 2), a])
    if np.any(a[:, 0] == 1):
        a = np.hstack([np.full((a.shape[0], 1), 2), a])
    if np.any(a[-1, :] == 1):
        a = np.vstack([a, np.full((1, a.shape[1]), 2)])
    if np.any(a[:, -1] == 1):
        a = np.hstack([a, np.full((a.shape[0], 1), 2)])
    return a


def grid_geometry(floor_plan, zone_map):
    """Per-cell material class, size and room label, as the simulator builds them."""
    fp = _pad(floor_plan)
    zm = _pad(zone_map)
    ext_space = fp == 2
    shell = ndimage.binary_dilation(ext_space, ndimage.generate_binary_structure(2, 1)) & ~ext_space
    iw0 = (fp == 1) & ~shell
    # enlarge_exterior_walls: wall cells within 2 cells (L2) of the shell take exterior material
    dist = ndimage.distance_transform_edt(~shell)
    ext_mat = shell | (iw0 & (dist <= 2.0))
    iw = iw0 & ~ext_mat
    cls = np.full(fp.shape, AIR, np.int8)
    cls[ext_space] = EXT_SPACE
    cls[ext_mat] = SHELL_OR_EXT
    cls[iw] = IWALL
    # rooms: connected components of zone_map == 0 (raster order, like cv2 connectedComponents)
    lab, nroom = ndimage.label(zm == 0, structure=ndimage.generate_binary_structure(2, 1))
    lab[ext_space] = 0
    door = (cls == AIR) & (lab == 0)          # open in floor_plan, closed in zone_map
    # neighbour bookkeeping identical to the solver: exterior space is never a neighbour
    nonext = ~ext_space
    up = np.zeros_like(nonext); up[1:] = nonext[:-1]
    dn = np.zeros_like(nonext); dn[:-1] = nonext[1:]
    lf = np.zeros_like(nonext); lf[:, 1:] = nonext[:, :-1]
    rt = np.zeros_like(nonext); rt[:, :-1] = nonext[:, 1:]
    nnb = (up.astype(int) + dn + lf + rt) * nonext
    solved = nonext & (nnb >= 2)              # 0-1 neighbours: pinned to ambient by the solver
    # boundary halving: missing up/down neighbour halves v, missing left/right halves u
    u = np.full(fp.shape, CV); v = np.full(fp.shape, CV)
    bnd = solved & (nnb < 4)
    v[bnd & (~up | ~dn)] = CV / 2
    u[bnd & (~lf | ~rt)] = CV / 2
    return dict(cls=cls, lab=lab, nroom=nroom, door=door, solved=solved, u=u, v=v,
                up=up, dn=dn, lf=lf, rt=rt, shape=fp.shape)


def cell_properties(G, theta):
    """Per-cell conductivity, volumetric heat capacity, slab terms, ambient conductance."""
    cls = G["cls"]
    k = np.where(cls == IWALL, theta["k_iw"], np.where(cls == SHELL_OR_EXT, theta["k_ext"], K_AIR))
    rc = np.where(cls == IWALL, theta["rc_iw"], np.where(cls == SHELL_OR_EXT, theta["rc_ext"], RC_AIR))
    u, v = G["u"], G["v"]
    C = Z * u * v * rc * G["solved"]
    mass = G["solved"] & (cls == AIR)
    Cm = np.where(mass, theta["rc_mass"] * u * v * Z, 0.0)
    Gm = np.where(mass, K_MASS * u * v / Z, 0.0)
    # film + half-thickness to ambient on every exposed face of a solved boundary cell
    h = theta["h"]
    s = G["solved"]
    amb = np.zeros(G["shape"])
    for miss, area, thick in ((~G["up"], Z * u, v), (~G["dn"], Z * u, v), (~G["lf"], Z * v, u), (~G["rt"], Z * v, u)):
        e = s & miss
        amb[e] += h * area[e] / (1.0 + h * 0.5 * thick[e] / k[e])
    return k, C, Cm, Gm, amb


def face_conductances(G, k):
    """Series half-cell conductance across every horizontal and vertical face (W/K)."""
    u, v, s = G["u"], G["v"], G["solved"]
    pinned_nonspace = (G["cls"] != EXT_SPACE) & ~s
    ok = s | pinned_nonspace                  # faces to pinned wall cells act as Dirichlet
    kk = np.where(G["cls"] == EXT_SPACE, 0.0, k)
    # (i,j)-(i,j+1)
    a = Z * np.minimum(v[:, :-1], v[:, 1:])
    r = u[:, :-1] / (2 * np.maximum(kk[:, :-1], 1e-30)) + u[:, 1:] / (2 * np.maximum(kk[:, 1:], 1e-30))
    gc = np.where((kk[:, :-1] > 0) & (kk[:, 1:] > 0) & ok[:, :-1] & ok[:, 1:] & (s[:, :-1] | s[:, 1:]), a / r, 0.0)
    a = Z * np.minimum(u[:-1, :], u[1:, :])
    r = v[:-1, :] / (2 * np.maximum(kk[:-1, :], 1e-30)) + v[1:, :] / (2 * np.maximum(kk[1:, :], 1e-30))
    gr = np.where((kk[:-1, :] > 0) & (kk[1:, :] > 0) & ok[:-1, :] & ok[1:, :] & (s[:-1, :] | s[1:, :]), a / r, 0.0)
    return gc, gr


def cluster_ids(G):
    """Assign every solved cell to a lumped node.

    room air -> one node per room; door cells -> one node per door component;
    partition and facade cells -> grouped by (material, set of rooms they face).
    Returns an int array (-1 for unsolved) and a list of node descriptors.
    """
    cls, lab, s = G["cls"], G["lab"], G["solved"]
    cid = np.full(G["shape"], -1, int)
    nodes = []
    for r in range(1, G["nroom"] + 1):
        nodes.append(("air", (r,)))
    cid[(lab > 0) & s] = lab[(lab > 0) & s] - 1
    dl, nd = ndimage.label(G["door"] & s)
    door_rooms = []
    for d in range(1, nd + 1):
        m = dl == d
        ring = ndimage.binary_dilation(m) & ~m
        rs = tuple(sorted(set(lab[ring][lab[ring] > 0].tolist())))
        cid[m] = len(nodes); nodes.append(("door", rs)); door_rooms.append(rs)
    # rooms faced by each wall cell: labels in the 3x3 neighbourhood, widened to 5x5 if none
    wall = s & ((cls == IWALL) | (cls == SHELL_OR_EXT)) & (lab == 0)
    rr, cc = np.nonzero(wall)
    keymap = {}
    H, W = G["shape"]
    for i, j in zip(rr, cc):
        for rad in (1, 2, 3):
            win = lab[max(0, i - rad):i + rad + 1, max(0, j - rad):j + rad + 1]
            rs = tuple(sorted(set(win[win > 0].tolist())))
            if rs:
                break
        kind = "partition" if cls[i, j] == IWALL else "facade"
        key = (kind, rs)
        if key not in keymap:
            keymap[key] = len(nodes); nodes.append(key)
        cid[i, j] = keymap[key]
    # any solved cell still unassigned (room cells of exterior material, stray air) -> its room or a wall node
    left = s & (cid < 0)
    if left.any():
        for i, j in zip(*np.nonzero(left)):
            key = ("other", (int(lab[i, j]),))
            if key not in keymap:
                keymap[key] = len(nodes); nodes.append(key)
            cid[i, j] = keymap[key]
    return cid, nodes


def cluster_model(G, theta, cid=None, nodes=None, spread=None):
    """Aggregate the grid into the lumped network.

    Returns dict with node capacitances C (n), slab capacitance/coupling per node
    (Cm, Gm), symmetric conductance matrix Gmat (n x n, zero diagonal), ambient
    conductance gamb (n), room count V, and the per-room cell counts.
    """
    if cid is None:
        cid, nodes = cluster_ids(G)
    n = len(nodes)
    k, C, Cm, Gm, amb = cell_properties(G, theta)
    gc, gr = face_conductances(G, k)
    s = G["solved"]
    Cn = np.bincount(cid[s], C[s], n)
    Cmn = np.bincount(cid[s], Cm[s], n)
    Gmn = np.bincount(cid[s], Gm[s], n)
    gamb = np.bincount(cid[s], amb[s], n)
    Gmat = np.zeros((n, n))
    for g, a, b in ((gc, cid[:, :-1], cid[:, 1:]), (gr, cid[:-1, :], cid[1:, :])):
        m = g > 0
        aa, bb, gg = a[m], b[m], g[m]
        # faces to pinned (non-solved, non-space) cells are conductance to ambient
        dir_a = (aa >= 0) & (bb < 0); dir_b = (bb >= 0) & (aa < 0)
        gamb += np.bincount(aa[dir_a], gg[dir_a], n) + np.bincount(bb[dir_b], gg[dir_b], n)
        both = (aa >= 0) & (bb >= 0) & (aa != bb)
        np.add.at(Gmat, (aa[both], bb[both]), gg[both])
        np.add.at(Gmat, (bb[both], aa[both]), gg[both])
    V = G["nroom"]
    area = np.bincount(G["lab"][G["lab"] > 0] - 1, minlength=V) * CV * CV
    M = dict(C=Cn, Cm=Cmn, Gm=Gmn, G=Gmat, gamb=gamb, V=V, nodes=nodes, cid=cid, area=area)
    if spread is not None:
        M["G"] = add_spreading(M, G, spread)
    return M


def link_faces(G, cid, n):
    """Number of grid faces joining each pair of lumped nodes."""
    F = np.zeros((n, n))
    s = G["solved"]
    for a, b, ok in ((cid[:, :-1], cid[:, 1:], s[:, :-1] & s[:, 1:]), (cid[:-1, :], cid[1:, :], s[:-1, :] & s[1:, :])):
        m = ok & (a >= 0) & (b >= 0) & (a != b)
        np.add.at(F, (a[m], b[m]), 1.0)
        np.add.at(F, (b[m], a[m]), 1.0)
    return F


def spreading_resistance(area, L, c, a):
    """Air-side resistance (K/W) between a room's mean temperature and a boundary
    segment of length L (m): a 1-D term for long segments plus a logarithmic
    constriction term for short ones (doors, short wall stubs).  k_air and z are
    fixed simulator constants; c and a are global, building-independent."""
    ell = np.sqrt(area)
    return (c * area / (3.0 * L * L) + a * np.log1p(ell / L) / np.pi) / (K_AIR * Z)


def add_spreading(M, G, spread):
    """Put the room-air spreading resistance in series with every link from a room
    air node to a wall, facade or door node."""
    Gm = M["G"].copy()
    V = M["V"]
    F = link_faces(G, M["cid"], len(Gm))
    for i in range(V):
        js = np.nonzero(Gm[i] > 0)[0]
        for j in js:
            if j < V and j < i:
                continue           # air-air link already handled from the other room
            L = F[i, j] * CV
            a = spread.get("af", spread["a"]) if M["nodes"][j][0] == "facade" else spread["a"]
            R = spreading_resistance(M["area"][i], L, spread["c"], a)
            if j < V:      # direct air-air contact between two rooms: both sides spread
                R = R + spreading_resistance(M["area"][j], L, spread["c"], spread["a"])
            g = 1.0 / (1.0 / Gm[i, j] + R)
            Gm[i, j] = g
            Gm[j, i] = g
    return Gm


def state_matrices(M):
    """Continuous-time network with slab nodes appended: C dx/dt = -L x + gamb*Tout + inputs."""
    n = len(M["C"])
    has = np.nonzero(M["Cm"] > 0)[0]
    N = n + len(has)
    Cfull = np.concatenate([M["C"], M["Cm"][has]])
    L = np.zeros((N, N))
    L[:n, :n] = -M["G"]
    L[np.arange(n), np.arange(n)] = M["G"].sum(1) + M["gamb"]
    for q, i in enumerate(has):
        j = n + q
        g = M["Gm"][i]
        L[i, i] += g; L[j, j] += g; L[i, j] -= g; L[j, i] -= g
    bamb = np.concatenate([M["gamb"], np.zeros(len(has))])
    return Cfull, L, bamb, has


def simulate(M, Tout, g, Tsup, Qroom, T0, dt=DT, x0=None):
    """Backward-Euler rollout.  Room r's air node receives Q_r and the implicit
    air stream g_r (Tsup_r - T_r); returns room-air temperatures after each step."""
    Cfull, L, bamb, _ = state_matrices(M)
    N = len(Cfull); V = M["V"]
    x = np.full(N, T0, float) if x0 is None else x0.copy()
    A0 = np.diag(Cfull / dt) + L
    out = np.empty((len(Tout), V))
    idx = np.arange(V)
    for t in range(len(Tout)):
        A = A0.copy()
        A[idx, idx] += g[t]
        rhs = Cfull / dt * x + bamb * Tout[t]
        rhs[:V] += Qroom[t] + g[t] * Tsup[t]
        x = np.linalg.solve(A, rhs)
        out[t] = x[:V]
    return out, x


def full_network(M):
    """Network over all lumped nodes (air, door, partition, facade, slab) as (C, L, bamb, kinds)."""
    Cfull, L, bamb, has = state_matrices(M)
    kinds = [nd[0] for nd in M["nodes"]] + ["slab"] * len(has)
    owner = [nd[1] for nd in M["nodes"]] + [M["nodes"][i][1] for i in has]
    return Cfull, L, bamb, kinds, owner


def guyan(Cfull, L, bamb, keep):
    """Static condensation onto the kept nodes.

    Eliminated nodes are slaved to their steady state given the kept nodes and
    the ambient: x_E = P x_K + q T_out.  The reduced network keeps the exact
    steady-state conductances (Kron reduction) and the eliminated capacitance is
    carried as P^T C_E P (Guyan mass), plus a P^T C_E q dT_out/dt forcing.
    """
    keep = np.asarray(keep)
    elim = np.setdiff1d(np.arange(len(Cfull)), keep)
    if len(elim) == 0:
        return np.diag(Cfull), L.copy(), bamb.copy(), np.zeros(len(Cfull))
    LKK, LKE, LEK, LEE = L[np.ix_(keep, keep)], L[np.ix_(keep, elim)], L[np.ix_(elim, keep)], L[np.ix_(elim, elim)]
    P = -np.linalg.solve(LEE, LEK)
    q = np.linalg.solve(LEE, bamb[elim])
    Lr = LKK + LKE @ P
    br = bamb[keep] - LKE @ q
    CE = Cfull[elim]
    Cr = np.diag(Cfull[keep]) + P.T @ (CE[:, None] * P)
    cr = P.T @ (CE * q)
    return Cr, Lr, br, cr


def simulate_reduced(Cr, Lr, br, cr, V, Tout, g, Tsup, Qroom, T0, dt=DT):
    """Backward Euler on a (possibly full-mass-matrix) network whose first V states are room air."""
    n = Cr.shape[0]
    x = np.full(n, T0, float)
    A0 = Cr / dt + Lr
    out = np.empty((len(Tout), V))
    idx = np.arange(V)
    prev = Tout[0]
    for t in range(len(Tout)):
        A = A0.copy(); A[idx, idx] += g[t]
        rhs = Cr @ x / dt + br * Tout[t] - cr * (Tout[t] - prev) / dt
        rhs[:V] += Qroom[t] + g[t] * Tsup[t]
        x = np.linalg.solve(A, rhs)
        out[t] = x[:V]; prev = Tout[t]
    return out


def simulate_fast(Cr, Lr, br, cr, V, Tout, g, Tsup, Qroom, T0, dt=DT):
    """Same as simulate_reduced, but the base matrix is inverted once and the
    per-step airflow conductances on the V air nodes enter by a Woodbury update
    (the solver's own trick), so a step costs one N x N matvec and a V x V solve."""
    n = Cr.shape[0]
    A0 = Cr / dt + Lr
    A0inv = np.linalg.inv(A0)
    W = A0inv[:, :V]
    S = W[:V, :]
    x = np.full(n, T0, float)
    out = np.empty((len(Tout), V))
    I = np.eye(V)
    prev = Tout[0]
    Cdt = Cr / dt
    for t in range(len(Tout)):
        rhs = Cdt @ x + br * Tout[t] - cr * (Tout[t] - prev) / dt
        rhs[:V] += Qroom[t] + g[t] * Tsup[t]
        y = A0inv @ rhs
        q = np.linalg.solve(I + g[t][:, None] * S, g[t] * y[:V])
        x = y - W @ q
        out[t] = x[:V]
        prev = Tout[t]
    return out
