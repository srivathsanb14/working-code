You are a scientist discovering the room dynamics of office buildings, as ONE set of closed-form formulas that
predicts every room's temperature in ANY building from the building's layout and the recent history of measured
values.  It will be used for model predictive control (heating/cooling plans).  No number is fitted at deployment
or beforehand; the only estimation allowed is a closed-form expression of the history buffer inside your formulas.

What is known about the physics (treat as hypotheses to test):
  - heat flow between rooms, to the outside and into heavy materials is linear in temperatures; the nonlinearity is
    in how commands become heat (products, limits, on/off) and that part is already done for you in the inputs.
  - one temperature per room is probably not enough: after heating stops a room drops fast, then slowly, because
    heavy materials (slab, walls, furniture) hold heat.  Instead of hidden states you may summarise the history
    (lags, differences, window or exponential averages): averaging a linear system gives a linear system.
  - from one room's view: its own air; a slower store attached to it (slab and furniture); its outside walls;
    its shared walls; doors to neighbours.

A model is a set of TEMPLATES: small formulas with blanks filled by one room's numbers; a room's equation is the
sum of its filled templates (a corner room gets an outside-wall term, an interior room does not).  Roles:
building physics (passive), controls/HVAC, disturbances (people, equipment, weather), bookkeeping.  Keep one
physical question per template so the pieces can be checked and recombined.

GEOMETRY  geo (dict; computed from the building metadata's floor_plan and zone_map, rooms in a fixed order):
  geo["V"] int rooms;  geo["area"] (V,) m2;  geo["facade_faces"] (V,) 0.2 m boundary segments on the exterior wall;
  geo["wall_cells"] (V,V) partition cells shared by rooms i and j;  geo["door_cells"] (V,V) door cells joining i and j;
  geo["n_doors"] (V,V);  geo["bbox_x"], geo["bbox_y"] (V,) m;  geo["ahu"] (V,) air handler index;  geo["floor"] (V,)

INPUTS of one 5-minute step (computed at run time from the observation and the policy's own commands):
  per room (V,): g = capacity rate of the supply air, W/K (0..~2400); T_supply = its temperature, K;
                 Q_reheat = reheat coil heat, W; Q_gain = lighting/equipment heat, W (known schedule)
  building-wide (): T_out K; occupants = people in the whole building (room unknown); comfort 1/0;
                 hour_sin, hour_cos, dow_sin, dow_cos
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

Write one Python module (numpy, math and dlib only; deterministic; no files):

    import numpy as np
    import dlib

    HISTORY = 36          # steps of measured history your formulas read (1..288)

    def prepare(geo, U):                  # optional: geometry-only templates, computed once per building
        return coef                       # anything (dict of arrays is typical)

    def step(coef, geo, hist, u, U):      # one 5-minute step for a BATCH of B independent moments
        return T_next                     # (B, V) room air temperatures at the end of this step

  hist["T"]  (B, HISTORY, V) measured room temperatures, oldest first, hist["T"][:, -1] = now.
  hist[k]    the HISTORY inputs before this step: per-room (B, HISTORY, V), building-wide (B, HISTORY);
             input hist[k][:, j] drove hist["T"][:, j-1] -> hist["T"][:, j].
  u[k]       this step's inputs: per-room (B, V), building-wide (B,).
  U          universal constants (table below).  geo matrices are (V, V): use T @ M.T for neighbour sums.
  In rollouts your own predictions are fed back into hist["T"] (free run), so the map must stay stable.

Rules:
  - NO fitted numbers: no PARAMS/THETA.  Every number in the code is either a universal constant (the same for
    every building and weather; you may introduce new ones, keep them few and physically sensible), a geometry
    quantity, or computed from the history buffer.  Values that differ per building (facade and partition
    conductivity / density / specific heat, slab density, occupants per room, outdoor film coefficient) must be
    INFERRED from hist by closed-form expressions (e.g. a shrunk least-squares ratio over the buffer).
  - every room uses the same templates; rooms differ only through their numbers (areas, wall lengths, doors).
  - conductance between cells = conductivity x contact area / distance (geometry x material); layers in series
    add resistances (room -> 1-cell wall -> room; room -> facade -> outside film).
  - assemble each room with dlib.assemble(T, [(w_k, T_k), ...], b): T_next = (T + sum w_k T_k + b)/(1 + sum w_k),
    w_k = DT * conductance / C >= 0, b = DT * direct heat / C.  It is stable for any weights; the supply air must
    be one of the terms (implicit), never an explicit Euler step.
  - checks before scoring: equilibrium (all 294 K, no heat -> no drift), bounded cooling toward a colder outside,
    positive response to heat, a 2400 W/K stream of 290 K air must pull a room to 290 K and never past it.

How candidates are scored (no fitting anywhere):
  stage 0  the checks above on a real floor plan; failures are rejected with the reason.
  stage 1  quick screen: one-step (5-minute) RMSE on held-out buildings, from measured history; more than
           5.0x the persistence error (T_next = T_now) is screened out.
  stage 2  decisive: from measured history at many moments, free-run 1 h, 3 h, 6 h ahead with the actual inputs;
           a rollout leaving [230, 350] K marks the candidate unstable.  One-step accuracy does not guarantee good
           rollouts.  Score = mean 3 h RMSE on held-out buildings + 0.0001 K per AST node of your code.

Reply only in the JSON format requested: a list of candidates, each with a short name, a one-paragraph hypothesis
(what changed and which diagnostic motivates it), and the complete module source in "code".