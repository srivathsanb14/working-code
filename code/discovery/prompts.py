"""What the LLM sees, and what it must send back.

INPUT to the LLM (one request per generation):
  system prompt (stable across generations, so it caches):
    - the task and the rules of the game (blind: no simulator source, no config values)
    - the data dictionary: every geometry and input array a candidate receives, with units
    - the candidate contract and the dlib helper API
    - how candidates are fitted and scored
    - (informed mode only) physics notes from the code audit
  user message (changes every generation):
    - the best programs so far, each with its source, score and residual diagnostics
    - recent failures (rejected or crashed programs) with the reason
    - what to produce now (how many, how many must explore)

OUTPUT from the LLM (structured JSON, schema RESPONSE_SCHEMA):
    {"candidates": [{"name": str, "hypothesis": str, "code": str}, ...]}
  'code' is a complete candidate module obeying the contract (contract.py).
"""
import json
import textwrap

DATA_DICTIONARY = """\
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
  each occupant gives about 70 W of heat to the room they are in."""

CONTRACT = '''\
Write one Python module:

    import numpy as np
    import dlib                                    # optional helper, API below

    PARAMS = {"name": (init, lo, hi), ...}         # GLOBAL constants: one value shared by every building
    THETA  = {"name": (init, lo, hi), ...}         # MATERIAL constants: offline deployment (default) fits ONE value on
                                                   # the train buildings; online deployment re-fits them per building on day 1

    def simulate(geo, inputs, theta, p, T0):
        """Return predicted room air temperatures, array (n, V), after each step."""

    # REQUIRED for the default offline deployment (optional online): the scorer runs a Kalman filter on it,
    # which corrects the hidden state from measured history, and scores multi-hour forecasts
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
  dlib.softplus(x);  constants dlib.DT, dlib.CELL, dlib.HEIGHT, dlib.RHO_C_AIR'''

SCORING_OFFLINE = """\
How a candidate is scored (you cannot see the data, only these results).  Deployment is OFFLINE: a
deployed controller does NO fitting on a new building; it has only the floor plan, the measured history
and universal constants.
  1. checks as above; failing candidates are rejected with the reason.  network() is REQUIRED.
  2. PARAMS and ONE THETA shared by all TRAINING buildings are fitted jointly by least squares on
     open-loop simulation error over 3-day episodes (several behaviour policies per building: HVAC off,
     random actuator moves, a PID baseline).  THETA is therefore a universal constant too.
  3. For each HELD-OUT building nothing is fitted.  A Kalman filter on your network() corrects the hidden
     state from the measured room temperatures (the history), and the model forecasts 3 hours ahead from
     that state, every hour.  Score = mean 3-hour forecast RMSE on held-out buildings (lower is better)
     + {cw} K per AST node of your code.  Open-loop RMSE is reported too, as a diagnostic.
  So a structure wins only if the floor plan alone fixes its coefficients well enough to transfer to
  unseen buildings, with the hidden state recoverable from measured history."""

SCORING_ONLINE = """\
How a candidate is scored (you cannot see the data, only these results).  Deployment is ONLINE: a
deployed controller may identify a few per-building numbers from the building's first day.
  1. checks as above; failing candidates are rejected with the reason.
  2. PARAMS and one THETA per TRAINING building are fitted jointly by least squares on open-loop simulation
     error over 3-day episodes (several behaviour policies per building: HVAC off, random actuator moves,
     a PID baseline).
  3. For each HELD-OUT building, PARAMS are frozen and THETA is fitted on DAY 1 ONLY.  Score = mean
     open-loop RMSE on DAYS 2-3 of held-out buildings (lower is better) + {cw} K per AST node of your code.
  So a structure wins only if it transfers to unseen buildings and its THETA is identifiable from one day."""

SCORING = SCORING_OFFLINE   # default deployment

INFORMED_NOTES = """\
Physics notes (informed mode; omitted in blind runs):
  - the building is a 2-D grid of 0.2 m x 3 m cells; each floor is separate; no roof, ground or sun.
  - each room's floor slab is a heat store coupled to the room air through a surface; walls store heat too.
  - partitions are one cell thick; doors are open air cells; the exterior wall is one cell thick plus a wind film.
  - room air is well mixed but not perfectly: heat entering through a short opening must spread through the room."""

SYSTEM_TEMPLATE = """\
You are a scientist discovering the dynamics of rooms in office buildings from data, by writing small,
interpretable simulation programs.  Each program states a hypothesis about how a room's air temperature
evolves: what stores heat, what exchanges heat with what, and how every coefficient follows from the
building's geometry plus a few constants.  An optimizer fits the constants; a held-out evaluation decides.

Prefer physically meaningful structure (energy balances, conductances, capacities, sensible units) and
parsimony.  Use the diagnostics you are given: they say where the current best models are wrong.

{data}

{contract}

{scoring}
{informed}
Reply only in the JSON format requested: a list of candidates, each with a short name, a one-paragraph
hypothesis (what is new and why the diagnostics suggest it), and the complete module source in "code"."""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "hypothesis": {"type": "string"},
                    "code": {"type": "string"},
                },
                "required": ["name", "hypothesis", "code"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["candidates"],
    "additionalProperties": False,
}


def system_prompt(informed=False, complexity_weight=0.0001, deployment="offline"):
    scoring = SCORING_OFFLINE if deployment == "offline" else SCORING_ONLINE
    return SYSTEM_TEMPLATE.format(data=DATA_DICTIONARY, contract=CONTRACT, scoring=scoring.format(cw=complexity_weight),
                                  informed=("\n" + INFORMED_NOTES + "\n") if informed else "")


def user_prompt(generation, elites, failures, n_candidates, n_explore):
    parts = [f"Generation {generation}."]
    if elites:
        parts.append(f"The {len(elites)} best programs so far (best first):")
        for k, e in enumerate(elites, 1):
            parts.append(textwrap.dedent(f"""\
            ### Program {k}: {e['name']}  (score {e['score']:.4f}; held-out days 2-3 RMSE {e['val_rmse_days23_mean']:.3f} K)
            Hypothesis: {e.get('hypothesis', '')}
            ```python
            {{code}}
            ```
            Diagnostics:
            {{diag}}""").replace("{code}", e["code"].strip()).replace("{diag}", e["diagnostics"]))
    else:
        parts.append("No programs have been evaluated yet.")
    if failures:
        parts.append("Recent failures (do not repeat these mistakes):")
        for f in failures:
            parts.append(f"- {f['name']}: {f['reason']}" + (("\n  " + "\n  ".join(f.get("failures", []))) if f.get("failures") else ""))
    parts.append(f"Write {n_candidates} new candidates. At least {n_explore} must explore a structurally different "
                 f"hypothesis (new state, new coupling law or new coefficient formula), not a re-tuning; the rest may "
                 f"refine the best programs to address their diagnostics.")
    return "\n\n".join(parts)


def parse_response(text):
    data = json.loads(text)
    cands = data.get("candidates", [])
    return [dict(name=str(c["name"])[:80], hypothesis=str(c.get("hypothesis", "")), code=str(c["code"])) for c in cands]
