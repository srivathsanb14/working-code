# SYSTEM

You are a scientist discovering the dynamics of rooms in office buildings from data, by writing small,
interpretable simulation programs.  Each program states a hypothesis about how a room's air temperature
evolves: what stores heat, what exchanges heat with what, and how every coefficient follows from the
building's geometry plus a few constants.  An optimizer fits the constants; a held-out evaluation decides.

Prefer physically meaningful structure (energy balances, conductances, capacities, sensible units) and
parsimony.  Use the diagnostics you are given: they say where the current best models are wrong.

GEOMETRY  geo (dict, rooms in a fixed order; all computable at run time from the floor-plan arrays):
  geo["V"]             int        number of rooms (one temperature sensor each)
  geo["area"]          (V,)       floor area, m2
  geo["facade_faces"]  (V,)       number of 0.2 m segments of the room's boundary that touch the exterior wall
  geo["wall_cells"]    (V,V)      number of 0.2 m partition-wall cells touching both rooms i and j (0 = no shared wall)
  geo["door_cells"]    (V,V)      number of 0.2 m door cells joining rooms i and j
  geo["n_doors"]       (V,V)      number of doors joining rooms i and j
  geo["bbox_x"], geo["bbox_y"] (V,)  bounding-box sides of each room, m
  geo["ahu"]           (V,)       index of the air handler serving the room
  geo["floor"]         (V,)       floor index (rooms on different floors never touch)

INPUTS  inputs (dict of arrays over n steps of 300 s; all computable at run time from observations and the policy's own actions):
  inputs["T_out"]     (n,)    outside air temperature, K
  inputs["g"]         (n,V)   capacity rate of the supply air stream entering each room, W/K (0 to ~2400)
  inputs["T_supply"]  (n,V)   temperature of that supply air, K
  inputs["Q_reheat"]  (n,V)   heat added by the room's reheat coil, W
  inputs["Q_gain"]    (n,V)   heat from lighting and equipment, W (known schedule)
  inputs["occupants"] (n,)    number of people in the whole building (which room they are in is unknown)
  inputs["comfort"]   (n,)    1 during occupied hours, 0 otherwise
  inputs["hour_sin"], inputs["hour_cos"], inputs["dow_sin"], inputs["dow_cos"]  (n,)  time of day / week

TARGET
  measured air temperature of every room after each step, (n,V), K.  Every episode starts from a uniform
  T0 = 294 K throughout the building (walls, floors and air alike).

CONSTANTS you may use: step 300 s; grid cell 0.2 m; floor-to-ceiling 3.0 m; air volumetric heat capacity 1206 J/m3K;
  each occupant gives about 70 W of heat to the room they are in.

Write one Python module:

    import numpy as np
    import dlib                                    # optional helper, API below

    PARAMS = {"name": (init, lo, hi), ...}         # GLOBAL constants: one value shared by every building
    THETA  = {"name": (init, lo, hi), ...}         # PER-BUILDING constants: re-fitted for each new building

    def simulate(geo, inputs, theta, p, T0):
        """Return predicted room air temperatures, array (n, V), after each step."""

Rules:
  - imports: numpy, math, dlib only; no files, no globals across calls; must be deterministic.
  - theta and p are dicts of floats keyed like THETA and PARAMS. Bounds must be finite; positive constants
    spanning more than a factor 3 are fitted in log space, so give physically sensible ranges.
  - at most 16 constants in PARAMS + THETA together. Every coefficient that differs between rooms must be
    computed from geo (and theta/p), never fitted per room - the model must apply unchanged to buildings
    with any number of rooms.
  - simulate must stay stable for 3 days of 300 s steps with air-stream conductances up to ~2400 W/K per room
    (implicit integration, e.g. dlib.LinearNetwork, is the safe choice).
  - candidates are checked before fitting: equilibrium (T_out = T0 and no heat -> no drift), bounded cooling
    toward a colder outside, and positive response of a room to heat put into it.

dlib API:
  dlib.LinearNetwork(C, G, G_amb, V)    RC network, backward Euler. Nodes 0..V-1 are the room-air outputs;
                                        further nodes are hidden states you define (any number).
      C (N,) J/K > 0;  G (N,N) symmetric W/K >= 0;  G_amb (N,) W/K to outside.
      .simulate(T_out, g, T_supply, Q, T0) -> (n,V): air node i gets heat Q[t,i] (W) and the
      implicit air stream g[t,i]*(T_supply[t,i] - T_i).
  dlib.pairs(M) -> (i, j, M[i,j]) over the upper triangle where M > 0
  dlib.softplus(x);  constants dlib.DT, dlib.CELL, dlib.HEIGHT, dlib.RHO_C_AIR

How a candidate is scored (you cannot see the data, only these results):
  1. checks as above; failing candidates are rejected with the reason.
  2. PARAMS and one THETA per TRAINING building are fitted jointly by least squares on open-loop simulation
     error over 3-day episodes (several behaviour policies per building: HVAC off, random actuator moves,
     a PID baseline).
  3. For each HELD-OUT building, PARAMS are frozen and THETA is fitted on DAY 1 ONLY - exactly what a
     deployed controller can do online.  Score = mean open-loop RMSE on DAYS 2-3 of held-out buildings
     (lower is better) + 0.0001 K per AST node of your code.
  So a structure wins only if it transfers to unseen buildings and its THETA is identifiable from one day.

Reply only in the JSON format requested: a list of candidates, each with a short name, a one-paragraph
hypothesis (what is new and why the diagnostics suggest it), and the complete module source in "code".

# USER

Generation 2.

The 3 best programs so far (best first):

### Program 1: seed_slab_wall_network  (score 0.3210; held-out days 2-3 RMSE 0.230 K)
Hypothesis: Room air, floor slab, facade wall and partition walls as separate heat stores.
```python
"""Room air, floor slab, facade wall and partition walls as separate heat stores.

Per room: an air node (air capacity only), a slab node coupled to the air in
proportion to floor area, and a facade node between the air and the outside
(inner and outer conductances from a wall conductivity and a surface film).  Each
shared wall is a node between its two rooms.  Doors couple rooms directly.  Heat
entering a room through a short boundary segment must spread through the room's
air, modelled as a series resistance growing with log(sqrt(area)/length).
This is the structure found by auditing the simulator code (STATE_SPACE.md §14),
written only in terms of the geometry a policy can compute.
"""
import numpy as np
import dlib

PARAMS = {
    "h_slab": (5.0, 0.5, 50.0),        # W/m2K, slab-to-air coupling per m2 of floor
    "a_spread": (0.6, 0.01, 5.0),      # strength of the spreading resistance
}
THETA = {
    "k_ext": (0.3, 0.1, 1.0),          # W/mK, facade wall conductivity
    "h_film": (15.0, 5.0, 40.0),       # W/m2K, outside surface film
    "rc_ext": (1.0e6, 2.0e5, 3.0e6),   # J/m3K, facade volumetric heat capacity
    "k_iw": (0.5, 0.2, 1.5),           # W/mK, partition conductivity
    "rc_iw": (3.0e5, 5.0e4, 8.0e5),    # J/m3K, partition volumetric heat capacity
    "rc_mass": (2.0e4, 5.0e3, 6.0e4),  # J/m3K, slab heat capacity per m3 of room
}
K_AIR = 50.0                           # W/mK, effective conductivity of mixed room air


def spread(area, length, a):
    return a * np.log1p(np.sqrt(area) / np.maximum(length, 1e-9)) / (np.pi * K_AIR * dlib.HEIGHT)


def simulate(geo, inputs, theta, p, T0):
    V = geo["V"]; A = geo["area"]; H = dlib.HEIGHT; c = dlib.CELL
    face = H * c                                                     # m2 of one 0.2 m segment
    wi, wj, wn = dlib.pairs(geo["wall_cells"])
    di, dj, dn = dlib.pairs(geo["door_cells"])
    nW = len(wi)
    N = 3 * V + nW                                                   # air, slab, facade, partitions
    C = np.empty(N); G = np.zeros((N, N)); G_amb = np.zeros(N)
    air, slab, fac = np.arange(V), V + np.arange(V), 2 * V + np.arange(V)
    C[air] = dlib.RHO_C_AIR * H * A
    C[slab] = theta["rc_mass"] * H * A
    G[air, slab] = G[slab, air] = p["h_slab"] * A
    nf = geo["facade_faces"]
    has_f = nf > 0
    C[fac] = np.where(has_f, theta["rc_ext"] * nf * face * c / 2, 1.0)
    g_in = nf * face / (c / (2 * K_AIR) + c / (4 * theta["k_ext"]))
    g_in = np.where(has_f, 1 / (1 / np.maximum(g_in, 1e-9) + spread(A, nf * c, p["a_spread"])), 0.0)
    G[air, fac] = G[fac, air] = g_in
    G_amb[fac] = np.where(has_f, nf * face / (c / (4 * theta["k_ext"]) + 1 / theta["h_film"]), 0.0)
    for k in range(nW):
        i, j, n = wi[k], wj[k], wn[k]; w = 3 * V + k
        C[w] = theta["rc_iw"] * n * face * c
        g_half = n * face / (c / (2 * K_AIR) + c / (2 * theta["k_iw"]))
        G[i, w] = G[w, i] = 1 / (1 / g_half + spread(A[i], n * c, p["a_spread"]))
        G[j, w] = G[w, j] = 1 / (1 / g_half + spread(A[j], n * c, p["a_spread"]))
    for i, j, n in zip(di, dj, dn):
        g_door = n * face / (2 * c / K_AIR)
        G[i, j] = G[j, i] = G[i, j] + 1 / (1 / g_door + spread(A[i], n * c, p["a_spread"]) + spread(A[j], n * c, p["a_spread"]))
    net = dlib.LinearNetwork(C, G, G_amb, V)
    Q = inputs["Q_reheat"] + inputs["Q_gain"] + 70.0 * inputs["occupants"][:, None] / V
    return net.simulate(inputs["T_out"], inputs["g"], inputs["T_supply"], Q, T0)
```
Diagnostics:
held-out open-loop RMSE, days 2-3 (theta fitted on day 1): mean 0.230 K
  building_train_7_phoenix_3_alloff: day1 0.272 K, days2-3 0.289 K, bias -0.139 K
  building_train_7_phoenix_3_pidwater: day1 0.276 K, days2-3 0.199 K, bias +0.086 K
  building_train_7_phoenix_3_random: day1 0.193 K, days2-3 0.203 K, bias -0.031 K
train RMSE (whole episodes, theta fitted jointly): mean 0.316 K
by facade length per m2 of floor (tertiles): low: rmse 0.239, bias -0.068; mid: rmse 0.173, bias -0.015; high: rmse 0.198, bias -0.000
by number of doors (tertiles): low: rmse 0.169, bias -0.039; mid: rmse 0.238, bias -0.027; high: rmse 0.203, bias -0.017
by floor area (tertiles): low: rmse 0.245, bias -0.038; mid: rmse 0.205, bias -0.036; high: rmse 0.160, bias -0.010
worst rooms: building_train_7_phoenix_3_alloff room 6 (rmse 0.85, bias -0.52, facade/m2 0.233, doors 1, area 81 m2); building_train_7_phoenix_3_random room 6 (rmse 0.62, bias -0.41, facade/m2 0.233, doors 1, area 81 m2); building_train_7_phoenix_3_pidwater room 6 (rmse 0.57, bias +0.42, facade/m2 0.233, doors 1, area 81 m2)
HVAC air stream active: rmse 0.212 bias +0.004; inactive: rmse 0.260 bias -0.036
by quarter of the day (00-06, 06-12, 12-18, 18-24): 0.226, 0.200, 0.263, 0.263
rmse by day 1/2/3: 0.247 / 0.196 / 0.251
theta across scenarios (train + held-out): k_ext 0.195-0.664; h_film 5-12.6; rc_ext 2.44e+05-8.23e+05; k_iw 0.328-0.453; rc_iw 1.26e+05-2.81e+05; rc_mass 1.21e+04-3.87e+04
complexity 906 AST nodes, 2 global + 6 per-scenario constants

### Program 2: cand_no_spread  (score 0.4289; held-out days 2-3 RMSE 0.338 K)
Hypothesis: Ablation: the slab-wall network without the spreading resistance (rooms perfectly mixed).
```python
"""Ablation: the slab-wall network without the spreading resistance (rooms perfectly mixed).

Per room: an air node (air capacity only), a slab node coupled to the air in
proportion to floor area, and a facade node between the air and the outside
(inner and outer conductances from a wall conductivity and a surface film).  Each
shared wall is a node between its two rooms.  Doors couple rooms directly.  Heat
entering a room through a short boundary segment must spread through the room's
air, modelled as a series resistance growing with log(sqrt(area)/length).
This is the structure found by auditing the simulator code (STATE_SPACE.md §14),
written only in terms of the geometry a policy can compute.
"""
import numpy as np
import dlib

PARAMS = {
    "h_slab": (5.0, 0.5, 50.0),        # W/m2K, slab-to-air coupling per m2 of floor
    "a_spread": (0.0011, 0.001, 0.0012),  # spreading switched off
}
THETA = {
    "k_ext": (0.3, 0.1, 1.0),          # W/mK, facade wall conductivity
    "h_film": (15.0, 5.0, 40.0),       # W/m2K, outside surface film
    "rc_ext": (1.0e6, 2.0e5, 3.0e6),   # J/m3K, facade volumetric heat capacity
    "k_iw": (0.5, 0.2, 1.5),           # W/mK, partition conductivity
    "rc_iw": (3.0e5, 5.0e4, 8.0e5),    # J/m3K, partition volumetric heat capacity
    "rc_mass": (2.0e4, 5.0e3, 6.0e4),  # J/m3K, slab heat capacity per m3 of room
}
K_AIR = 50.0                           # W/mK, effective conductivity of mixed room air


def spread(area, length, a):
    return a * np.log1p(np.sqrt(area) / np.maximum(length, 1e-9)) / (np.pi * K_AIR * dlib.HEIGHT)


def simulate(geo, inputs, theta, p, T0):
    V = geo["V"]; A = geo["area"]; H = dlib.HEIGHT; c = dlib.CELL
    face = H * c                                                     # m2 of one 0.2 m segment
    wi, wj, wn = dlib.pairs(geo["wall_cells"])
    di, dj, dn = dlib.pairs(geo["door_cells"])
    nW = len(wi)
    N = 3 * V + nW                                                   # air, slab, facade, partitions
    C = np.empty(N); G = np.zeros((N, N)); G_amb = np.zeros(N)
    air, slab, fac = np.arange(V), V + np.arange(V), 2 * V + np.arange(V)
    C[air] = dlib.RHO_C_AIR * H * A
    C[slab] = theta["rc_mass"] * H * A
    G[air, slab] = G[slab, air] = p["h_slab"] * A
    nf = geo["facade_faces"]
    has_f = nf > 0
    C[fac] = np.where(has_f, theta["rc_ext"] * nf * face * c / 2, 1.0)
    g_in = nf * face / (c / (2 * K_AIR) + c / (4 * theta["k_ext"]))
    g_in = np.where(has_f, 1 / (1 / np.maximum(g_in, 1e-9) + spread(A, nf * c, p["a_spread"])), 0.0)
    G[air, fac] = G[fac, air] = g_in
    G_amb[fac] = np.where(has_f, nf * face / (c / (4 * theta["k_ext"]) + 1 / theta["h_film"]), 0.0)
    for k in range(nW):
        i, j, n = wi[k], wj[k], wn[k]; w = 3 * V + k
        C[w] = theta["rc_iw"] * n * face * c
        g_half = n * face / (c / (2 * K_AIR) + c / (2 * theta["k_iw"]))
        G[i, w] = G[w, i] = 1 / (1 / g_half + spread(A[i], n * c, p["a_spread"]))
        G[j, w] = G[w, j] = 1 / (1 / g_half + spread(A[j], n * c, p["a_spread"]))
    for i, j, n in zip(di, dj, dn):
        g_door = n * face / (2 * c / K_AIR)
        G[i, j] = G[j, i] = G[i, j] + 1 / (1 / g_door + spread(A[i], n * c, p["a_spread"]) + spread(A[j], n * c, p["a_spread"]))
    net = dlib.LinearNetwork(C, G, G_amb, V)
    Q = inputs["Q_reheat"] + inputs["Q_gain"] + 70.0 * inputs["occupants"][:, None] / V
    return net.simulate(inputs["T_out"], inputs["g"], inputs["T_supply"], Q, T0)
```
Diagnostics:
held-out open-loop RMSE, days 2-3 (theta fitted on day 1): mean 0.338 K
  building_train_7_phoenix_3_alloff: day1 0.379 K, days2-3 0.482 K, bias -0.305 K
  building_train_7_phoenix_3_pidwater: day1 0.347 K, days2-3 0.239 K, bias +0.119 K
  building_train_7_phoenix_3_random: day1 0.288 K, days2-3 0.293 K, bias -0.074 K
train RMSE (whole episodes, theta fitted jointly): mean 0.591 K
by facade length per m2 of floor (tertiles): low: rmse 0.407, bias -0.153; mid: rmse 0.264, bias -0.089; high: rmse 0.241, bias -0.019
by number of doors (tertiles): low: rmse 0.244, bias -0.106; mid: rmse 0.314, bias -0.047; high: rmse 0.354, bias -0.107
by floor area (tertiles): low: rmse 0.337, bias -0.071; mid: rmse 0.289, bias -0.108; high: rmse 0.286, bias -0.081
worst rooms: building_train_7_phoenix_3_alloff room 30 (rmse 0.87, bias -0.43, facade/m2 0.000, doors 1, area 59 m2); building_train_7_phoenix_3_alloff room 24 (rmse 0.80, bias -0.79, facade/m2 0.000, doors 0, area 83 m2); building_train_7_phoenix_3_alloff room 6 (rmse 0.80, bias -0.57, facade/m2 0.233, doors 1, area 81 m2)
HVAC air stream active: rmse 0.294 bias -0.018; inactive: rmse 0.390 bias -0.093
by quarter of the day (00-06, 06-12, 12-18, 18-24): 0.346, 0.282, 0.372, 0.389
rmse by day 1/2/3: 0.338 / 0.290 / 0.367
theta across scenarios (train + held-out): k_ext 0.14-0.372; h_film 6.16-40; rc_ext 2.46e+05-8.61e+05; k_iw 0.2-0.255; rc_iw 1.15e+05-3.39e+05; rc_mass 1.15e+04-4.13e+04
complexity 906 AST nodes, 2 global + 6 per-scenario constants

### Program 3: seed_rc_one_node  (score 1.6819; held-out days 2-3 RMSE 1.667 K)
Hypothesis: One heat store per room: room air and everything in it share one capacity.
```python
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


def simulate(geo, inputs, theta, p, T0):
    V = geo["V"]
    C = theta["cap"] * geo["area"]
    G = p["k_wall"] * geo["wall_cells"] + p["k_door"] * geo["door_cells"]
    G_amb = theta["u_facade"] * geo["facade_faces"]
    net = dlib.LinearNetwork(C, G, G_amb, V)
    Q = inputs["Q_reheat"] + inputs["Q_gain"] + 70.0 * inputs["occupants"][:, None] / V
    return net.simulate(inputs["T_out"], inputs["g"], inputs["T_supply"], Q, T0)
```
Diagnostics:
held-out open-loop RMSE, days 2-3 (theta fitted on day 1): mean 1.667 K
  building_train_7_phoenix_3_alloff: day1 0.825 K, days2-3 1.689 K, bias +1.427 K
  building_train_7_phoenix_3_pidwater: day1 0.711 K, days2-3 1.243 K, bias -0.498 K
  building_train_7_phoenix_3_random: day1 2.141 K, days2-3 2.069 K, bias -0.224 K
train RMSE (whole episodes, theta fitted jointly): mean 1.408 K
by facade length per m2 of floor (tertiles): low: rmse 1.659, bias +0.428; mid: rmse 1.582, bias +0.238; high: rmse 1.620, bias +0.039
by number of doors (tertiles): low: rmse 1.727, bias +0.166; mid: rmse 1.575, bias +0.150; high: rmse 1.560, bias +0.389
by floor area (tertiles): low: rmse 1.733, bias +0.150; mid: rmse 1.641, bias +0.264; high: rmse 1.487, bias +0.292
worst rooms: building_train_7_phoenix_3_random room 30 (rmse 3.01, bias -1.03, facade/m2 0.000, doors 1, area 59 m2); building_train_7_phoenix_3_random room 24 (rmse 2.82, bias -0.41, facade/m2 0.000, doors 0, area 83 m2); building_train_7_phoenix_3_random room 19 (rmse 2.76, bias -0.39, facade/m2 0.290, doors 0, area 55 m2)
HVAC air stream active: rmse 1.891 bias -0.111; inactive: rmse 1.311 bias +0.378
by quarter of the day (00-06, 06-12, 12-18, 18-24): 1.674, 1.389, 1.523, 1.796
rmse by day 1/2/3: 1.226 / 1.743 / 1.558
theta across scenarios (train + held-out): cap 5.66e+04-1.06e+05; u_facade 0.825-1.54
complexity 150 AST nodes, 2 global + 2 per-scenario constants

Recent failures (do not repeat these mistakes):

- cand_explicit_euler: physics checks failed
  [lower bounds] stiff air stream: 2400 W/K of 290 K air per room gave -440700497317630845181376017438614739180269284782567999145329063400470909415676213542801773040653649855058430221076468186540672999917842621005824.00-781503123937121475166925800864952658501877841544587937068002742445001233475574420726749918457054871584774228811778548771864657403858588203483136.00 K (must stay within [290, 294] and approach 290 K); use implicit integration

Write 6 new candidates. At least 2 must explore a structurally different hypothesis (new state, new coupling law or new coefficient formula), not a re-tuning; the rest may refine the best programs to address their diagnostics.
