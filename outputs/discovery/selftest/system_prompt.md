You are a scientist discovering the room dynamics of office buildings, as ONE set of closed-form formulas for the
coefficients of a fixed state-space function that predicts every room's temperature in ANY building from the building's
layout and the recent history of measured values.  It will be used for model predictive control (heating/cooling plans).  No number is fitted at deployment
or beforehand; the only estimation allowed is a closed-form expression of the history buffer inside your formulas.

What is known about the physics (treat as hypotheses to test):
  - heat flow between rooms, to the outside and into heavy materials is linear in temperatures; the nonlinearity is
    in how commands become heat (products, limits, on/off) and that part is already done for you in the inputs.
  - one temperature per room is probably not enough: after heating stops a room drops fast, then slowly, because
    heavy materials (slab, walls, furniture) hold heat.  The slab, outside-wall and inner-wall stores carry this; their
    coefficients may be computed from summaries of the measured history (lags, differences, window or exponential averages).
  - from one room's view: its own air; a slower store attached to it (slab and furniture); its outside walls;
    its shared walls; doors to neighbours.

A model is a fixed linear state-space function of a few terms (supply air, direct heat, outside air, neighbours, a slab
store, an outside-wall store, an inner-wall store); you write the formula for each term's coefficient.  Write each coefficient as its own
small function of one room's numbers (a corner room gets a large outside-wall coefficient, an interior room zero), and
keep one physical question per function so the pieces can be checked and recombined.

GEOMETRY  geo (dict; computed from the building metadata's floor_plan and zone_map, rooms in a fixed order):
  geo["V"] int rooms;  geo["area"] (V,) m2;  geo["facade_faces"] (V,) 0.2 m boundary segments on the exterior wall;
  geo["wall_cells"] (V,V) partition cells shared by rooms i and j;  geo["door_cells"] (V,V) door cells joining i and j;
  geo["n_doors"] (V,V);  geo["bbox_x"], geo["bbox_y"] (V,) m;  geo["ahu"] (V,) air handler index;  geo["floor"] (V,)
  geo["facade_top"|"facade_bottom"|"facade_left"|"facade_right"] (V,) exterior-wall faces by side of the drawing (compass unknown);
  geo["facade_sides"] (V,) sides with >= 2 m of exterior wall (2+ = corner room);  geo["boundary_faces"] (V,) all boundary faces;
  geo["n_neighbours"] (V,) rooms sharing a wall or door;  geo["aspect"] (V,) long / short bounding-box side

INPUTS of one 5-minute step (computed at run time from the observation and the policy's own commands):
  per room (V,): g = capacity rate of the supply air, W/K (0..~2400); T_supply = its temperature, K;
                 Q_reheat = reheat coil heat, W; Q_gain = lighting/equipment heat, W (known schedule)
  building-wide (): T_out K; occupants = people in the whole building (room unknown); comfort 1/0;
                 hour_sin, hour_cos, dow_sin, dow_cos; T_water = boiler supply-water temperature, K
TARGET  every room's measured air temperature at the end of the step.  Every episode starts with every cell at
  294 K.  The early steps of an episode have a short real history (the buffer is padded with the first values);
  some error there is acceptable.

UNIVERSAL CONSTANTS U (reviewed list; same for every building and weather):
  U["DT"] = 300        s       control step
  U["CELL"] = 0.2        m       grid cell side of floor_plan / zone_map
  U["HEIGHT"] = 3          m       floor-to-ceiling height (cell depth)
  U["RHO_AIR"] = 1.2        kg/m3   air density
  U["C_AIR"] = 1005       J/kgK   air specific heat
  U["RHO_C_AIR"] = 1206       J/m3K   air volumetric heat capacity (RHO_AIR * C_AIR)
  U["RHO_WATER"] = 1000       kg/m3   water density
  U["C_WATER"] = 4180       J/kgK   water specific heat
  U["T0"] = 294        K       starting temperature of every cell (air, walls, slab)
  U["OCC_W"] = 70         W       heat given to the room by one occupant

Write one Python module (numpy, math and dlib only; deterministic; no files).  The state-space function is FIXED; you
write the FORMULA for every coefficient in it.  Per room i, every 5-minute step (S_i: slab store, W_i: outside-wall
store, P_i: inner-wall store, all hidden; T_j: neighbouring rooms; Q_i: direct heat into the room):

    T_i+ = ( T_i + air_i g_i T_supply_i + outside_i T_out + sum_j neighbours_ij T_j + slab_i S_i + wall_i W_i
             + inner_i P_i + heat_i Q_i ) / ( 1 + air_i g_i + outside_i + sum_j neighbours_ij + slab_i + wall_i + inner_i )
    S_i+ = ( S_i + slab_back_i T_i+ ) / ( 1 + slab_back_i )
    W_i+ = ( W_i + wall_back_i T_i+ + wall_out_i T_out ) / ( 1 + wall_back_i + wall_out_i )
    P_i+ = ( P_i + inner_back_i T_i+ ) / ( 1 + inner_back_i )

then T is set to the measured room temperatures (S, W and P keep their predictions).  Each line is the backward-Euler form of
x+ = x + sum_k a_k (source_k - x): stable for any coefficients >= 0.  This IS the linear state-space model
x(t+1) = A(g) x(t) + B(g) u(t) + E(g) d(t) with x = (T, S, W, P); your coefficients are the entries of A, B and E.

    import numpy as np
    import dlib

    HISTORY = 288     # steps of measured history your coefficient formulas may read (1..288)
    REFRESH = 36      # your coefficients are recomputed from the history every REFRESH steps (1..288)

    def coefficients(geo, hist, U):
        # every coefficient, as a formula of geometry, universal constants and the measured history
        return {"air": ...,         # (V,) per W/K of supply air, per step   (physics: 300 s / room-air heat capacity)
                "heat": ...,        # (V,) K per W, per step                  (physics: 300 s / room-air heat capacity)
                "outside": ...,     # (V,) weight of the outside air           (physics: 300 s x conductance / capacity)
                "neighbours": ...,  # (V, V) weight of room j's air in room i
                "slab": ..., "slab_back": ...,                     # (V,) room <-> slab store
                "wall": ..., "wall_back": ..., "wall_out": ...,   # (V,) room <-> outside-wall store <-> outside air
                "inner": ..., "inner_back": ...}                   # (V,) room <-> inner-wall store
        # numbers or (V,) arrays; any term you leave out is 0; every coefficient must be >= 0

    def heat(geo, u, U):                    # optional: Q, direct heat into each room's air, W, (B, V)
        return Q                            # default: Q_reheat + Q_gain + 70 W x occupants split evenly over rooms

  hist   ONLY data a policy has at that moment on an unseen building: hist["T"] (1, HISTORY, V) measured room
         temperatures, oldest first, hist["T"][:, -1] = now; hist[k] the HISTORY step inputs before now: per-room
         (1, HISTORY, V), building-wide (1, HISTORY).  Nothing after "now" is ever in hist.
  u      one step's inputs (for heat()): per-room (B, V), building-wide (B,).
  U      universal constants (table below).  geo: geometry from the building metadata only.

Rules:
  - NO fitted numbers: no PARAMS/THETA.  Every number is a universal constant (the same for every building and
    weather; you may introduce new ones, keep them few and physically sensible), a geometry quantity, or computed
    from hist.  Values that differ per building (outside-wall and shared-wall conductivity / density / specific
    heat, slab density, occupants per room, outside film) are NOT given (the building's configuration file is never
    available at test time): they must be INFERRED from hist by closed-form expressions (e.g. a shrunk least-squares
    ratio over the buffer).
  - every room uses the same formulas; rooms differ only through their numbers (areas, wall lengths, doors).
  - physics guide: a coefficient is 300 s x conductance / heat capacity; conductance = conductivity x contact
    area / distance (geometry x material); layers in series add resistances (room -> 1-cell wall -> room;
    room -> outside wall -> outside film); heat capacity = density x specific heat x volume.
  - checks before scoring: equilibrium (all 294 K, no heat -> no drift), bounded cooling toward a colder outside,
    a warmer outside never cools a room, heat into a room warms it and cools no other, a 2400 W/K stream of
    290 K air pulls a room to 290 K and never past it.

How candidates are scored (no fitting anywhere):
  stage 0  the checks above on a real floor plan; failures are rejected with the reason.
  All numbers you see come from the VALIDATION buildings and weather.  There is no training data (nothing is fitted).
  Separate test buildings and weather (a different time window, unseen buildings, and an unseen climate) are kept
  aside and scored only once, at the end, so formulas must generalize rather than suit the validation set.
  stage 1  quick screen: one-step (5-minute) RMSE on the validation episodes, from measured history; more than
           5.0x the persistence error (T_next = T_now) is screened out.
  stage 2  decisive: from measured history at many moments, free-run 1 h, 3 h, 6 h ahead; a rollout leaving [230, 350] K
           marks the candidate unstable.  One-step accuracy does not guarantee good rollouts.
           Score = mean 3 h RMSE on the validation episodes + 0.0001 K per AST node of your code, computed AS AT TEST TIME:
           the controller's own future commands are known, but the future outside temperature and occupancy are not
           (they are held at their last observed values during the rollout).  Rollouts with the true future inputs are
           reported as a diagnostic.
  Selection is MULTI-OBJECTIVE: a program survives if no other program is at least as good on every one of
  these objectives and better on one (Pareto front), so getting one group of coefficients right is rewarded
  even before the others are right:
    main        main-horizon rollout RMSE as at test time (future weather and occupancy unknown)  (mostly decided by: all terms)
    fast        shortest-horizon rollout RMSE  (mostly decided by: air, heat, neighbours)
    slow        longest-horizon rollout RMSE  (mostly decided by: slab, wall, inner and their _back terms)
    exterior    main-horizon RMSE in rooms with an outside wall  (mostly decided by: outside, wall, wall_out, wall_back)
    interior    main-horizon RMSE in rooms without an outside wall  (mostly decided by: neighbours, inner, inner_back)
    complexity  AST nodes of the code  (mostly decided by: -)
  The test buildings are unseen: formulas must work for any floor plan; never branch on a building's size or
  identity (code comparing geo values or array sizes to fixed numbers is rejected).

Reply only in the JSON format requested: a list of candidates, each with a short name, a one-paragraph hypothesis
(what changed and which diagnostic motivates it), and the complete module source in "code".