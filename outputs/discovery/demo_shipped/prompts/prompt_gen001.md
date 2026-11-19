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

    # optional but recommended: lets the tester run a Kalman filter and score multi-hour forecasts
    def network(geo, inputs, theta, p):
        """Return (dlib.LinearNetwork, dict(T_out=..., g=..., T_supply=..., Q=...)) with simulate() equal to
        net.simulate(T_out, g, T_supply, Q, T0)."""

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

Generation 1.

The 2 best programs so far (best first):

### Program 1: seed_slab_wall_network  (score 0.3827; held-out days 2-3 RMSE 0.286 K)
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


def network(geo, inputs, theta, p):
    """The RC network and its per-step forcing (lets test_model.py run a Kalman filter)."""
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
    return net, dict(T_out=inputs["T_out"], g=inputs["g"], T_supply=inputs["T_supply"], Q=Q)


def simulate(geo, inputs, theta, p, T0):
    net, f = network(geo, inputs, theta, p)
    return net.simulate(f["T_out"], f["g"], f["T_supply"], f["Q"], T0)
```
Diagnostics:
held-out open-loop RMSE, days 2-3 (theta fitted on day 1): mean 0.286 K
  building_train_7_phoenix_3_all_off: day1 0.281 K, days2-3 0.568 K, bias -0.435 K
  building_train_7_phoenix_3_bang_bang: day1 0.316 K, days2-3 0.217 K, bias -0.030 K
  building_train_7_phoenix_3_fixed_setpoint: day1 0.270 K, days2-3 0.214 K, bias +0.097 K
  building_train_7_phoenix_3_pid: day1 0.262 K, days2-3 0.215 K, bias +0.091 K
  building_train_7_phoenix_3_pid_water: day1 0.262 K, days2-3 0.218 K, bias +0.091 K
train RMSE (whole episodes, theta fitted jointly): mean 0.380 K
by facade length per m2 of floor (tertiles): low: rmse 0.304, bias -0.100; mid: rmse 0.245, bias -0.019; high: rmse 0.231, bias +0.007
by number of doors (tertiles): low: rmse 0.222, bias -0.069; mid: rmse 0.285, bias +0.000; high: rmse 0.273, bias -0.043
by floor area (tertiles): low: rmse 0.296, bias -0.031; mid: rmse 0.262, bias -0.061; high: rmse 0.222, bias -0.020
worst rooms: building_train_7_phoenix_3_all_off room 33 (rmse 0.99, bias -0.74, facade/m2 0.233, doors 1, area 81 m2); building_train_7_phoenix_3_all_off room 25 (rmse 0.97, bias -0.92, facade/m2 0.000, doors 0, area 82 m2); building_train_7_phoenix_3_all_off room 17 (rmse 0.95, bias -0.92, facade/m2 0.000, doors 0, area 83 m2)
HVAC air stream active: rmse 0.244 bias +0.106; inactive: rmse 0.340 bias -0.108
by quarter of the day (00-06, 06-12, 12-18, 18-24): 0.257, 0.252, 0.363, 0.337
rmse by day 1/2/3: 0.278 / 0.275 / 0.285
theta across scenarios (train + held-out): k_ext 0.165-0.528; h_film 5-40; rc_ext 2.11e+05-7.89e+05; k_iw 0.293-0.549; rc_iw 1.93e+05-2.88e+05; rc_mass 1.21e+04-3.91e+04
complexity 963 AST nodes, 2 global + 6 per-scenario constants

### Program 2: seed_rc_one_node  (score 1.7869; held-out days 2-3 RMSE 1.766 K)
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
```
Diagnostics:
held-out open-loop RMSE, days 2-3 (theta fitted on day 1): mean 1.766 K
  building_train_7_phoenix_3_all_off: day1 0.852 K, days2-3 0.712 K, bias +0.033 K
  building_train_7_phoenix_3_bang_bang: day1 1.407 K, days2-3 3.872 K, bias -0.285 K
  building_train_7_phoenix_3_fixed_setpoint: day1 0.731 K, days2-3 2.077 K, bias -0.066 K
  building_train_7_phoenix_3_pid: day1 0.739 K, days2-3 1.100 K, bias -0.190 K
  building_train_7_phoenix_3_pid_water: day1 0.739 K, days2-3 1.070 K, bias -0.233 K
train RMSE (whole episodes, theta fitted jointly): mean 1.275 K
by facade length per m2 of floor (tertiles): low: rmse 1.289, bias -0.128; mid: rmse 1.598, bias -0.113; high: rmse 1.883, bias -0.204
by number of doors (tertiles): low: rmse 1.589, bias -0.220; mid: rmse 1.724, bias -0.127; high: rmse 1.457, bias -0.098
by floor area (tertiles): low: rmse 1.839, bias -0.201; mid: rmse 1.631, bias -0.165; high: rmse 1.300, bias -0.078
worst rooms: building_train_7_phoenix_3_fixed_setpoint room 1 (rmse 5.24, bias -0.01, facade/m2 0.320, doors 0, area 55 m2); building_train_7_phoenix_3_bang_bang room 4 (rmse 4.79, bias -0.60, facade/m2 0.000, doors 2, area 51 m2); building_train_7_phoenix_3_bang_bang room 3 (rmse 4.75, bias -0.63, facade/m2 0.000, doors 2, area 53 m2)
HVAC air stream active: rmse 2.687 bias +0.082; inactive: rmse 0.831 bias -0.285
by quarter of the day (00-06, 06-12, 12-18, 18-24): 1.050, 1.981, 2.352, 1.550
rmse by day 1/2/3: 0.894 / 1.859 / 1.636  (growing: slow dynamics or a steady-state gain is off)
theta across scenarios (train + held-out): cap 6.05e+04-1.28e+05; u_facade 0.847-2.24
complexity 207 AST nodes, 2 global + 2 per-scenario constants

Write 6 new candidates. At least 2 must explore a structurally different hypothesis (new state, new coupling law or new coefficient formula), not a re-tuning; the rest may refine the best programs to address their diagnostics.
