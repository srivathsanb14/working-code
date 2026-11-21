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

    # optional: the per-step forcing alone, so a deployed policy can step the model without rebuilding it
    def forcing(geo, inputs, theta, p):
        """Return dict(T_out=..., g=..., T_supply=..., Q=...) exactly as network() returns it."""

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


class ParseError(ValueError):
    pass


def _json_blob(text):
    """The JSON object in an answer: the whole text, a ```json fenced block, or the outermost {...}."""
    import re
    t = (text or "").strip()
    for cand in [t] + re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", t, re.S):
        try:
            return json.loads(cand, strict=False)
        except Exception:  # noqa: BLE001
            pass
    i, j = t.find("{"), t.rfind("}")
    if i >= 0 and j > i:
        try:
            return json.loads(t[i:j + 1], strict=False)
        except Exception as e:  # noqa: BLE001
            raise ParseError(f"no valid JSON object in the answer ({e})") from None
    raise ParseError("no JSON object in the answer")


def _strip_fence(code):
    import re
    m = re.match(r"^\s*```(?:python)?\s*\n(.*?)\n?```\s*$", code, re.S)
    return m.group(1) if m else code


def parse_response(text):
    """Robust parse of the LLM answer (backlog E5): tolerates code fences, prose around the JSON, a bare list,
    missing hypotheses and duplicate names.  Raises ParseError with a reason the proposer can send back."""
    data = _json_blob(text)
    cands = data.get("candidates", []) if isinstance(data, dict) else data if isinstance(data, list) else []
    if not isinstance(cands, list) or not cands:
        raise ParseError('expected {"candidates": [{"name", "hypothesis", "code"}, ...]} with at least one candidate')
    out, seen = [], {}
    for k, c in enumerate(cands):
        if not isinstance(c, dict) or not str(c.get("code", "")).strip():
            continue
        name = str(c.get("name") or f"candidate_{k}")[:80]
        if name in seen:
            seen[name] += 1; name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        out.append(dict(name=name, hypothesis=str(c.get("hypothesis", "")), code=_strip_fence(str(c["code"]))))
    if not out:
        raise ParseError("no candidate had non-empty code")
    return out


# ============================================================================= closed-form mode (default)
# No number is fitted anywhere.  One LLM request per (island, operator); the operator decides the user message.

CF_DATA = """\
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
  some error there is acceptable."""

CF_CONTRACT = '''\
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

Optional STATEFUL form (instead of step): keep hidden stores (slab, walls) as a state corrected by measurements.
    def init_state(coef, geo, U): return x0            # (N,), the first V entries are the room-air nodes
    def predict(coef, geo, state, u, U): return state  # (B, N) -> (B, N), one 5-minute step
    def correct(coef, geo, state, T_measured, U): return state   # fold in measured room temps (B, V); a FIXED rule
  The scorer runs predict/correct over the measured history, then free-runs predict from the corrected state.
  dlib.network_predict(net, state, T_out, g, T_supply, Q) and dlib.correct_rooms(state, T_measured, V) help.

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
    positive response to heat, a 2400 W/K stream of 290 K air must pull a room to 290 K and never past it.'''

CF_SCORING = """\
How candidates are scored (no fitting anywhere):
  stage 0  the checks above on a real floor plan; failures are rejected with the reason.
  stage 1  quick screen: one-step (5-minute) RMSE on held-out buildings, from measured history; more than
           {screen}x the persistence error (T_next = T_now) is screened out.
  stage 2  decisive: from measured history at many moments, free-run {horizons} ahead; a rollout leaving [230, 350] K
           marks the candidate unstable.  One-step accuracy does not guarantee good rollouts.
           Score = mean {main} RMSE on held-out buildings + {cw} K per AST node of your code, computed AS AT TEST TIME:
           the controller's own future commands are known, but the future outside temperature and occupancy are not
           (they are held at their last observed values during the rollout).  Rollouts with the true future inputs are
           reported as a diagnostic.
  The evaluation buildings are unseen: formulas must work for any floor plan; never branch on a building's size or
  identity (code comparing geo values or array sizes to fixed numbers is rejected)."""

CF_SYSTEM = """\
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

{data}

UNIVERSAL CONSTANTS U (reviewed list; same for every building and weather):
{constants}

{contract}

{scoring}

Reply only in the JSON format requested: a list of candidates, each with a short name, a one-paragraph hypothesis
(what changed and which diagnostic motivates it), and the complete module source in "code"."""


def cf_system_prompt(constants_raw, screen=5.0, horizons=(12, 36, 72), main_h=36, cw=0.0001):
    rows = "\n".join(f"  U[\"{k}\"] = {v['value']:<10g} {v['unit']:<7} {v['what']}" for k, v in constants_raw.items() if not k.startswith("_"))
    hz = ", ".join(f"{h / 12:g} h" for h in horizons)
    return CF_SYSTEM.format(data=CF_DATA, constants=rows, contract=CF_CONTRACT,
                            scoring=CF_SCORING.format(screen=screen, horizons=hz, main=f"{main_h / 12:g} h", cw=cw))


def _prog(rec, label):
    return (f"### {label}: {rec['name']}  [{rec['id']}, island {rec['island']}, score {rec['score']:.4f}]\n"
            f"Hypothesis: {rec.get('hypothesis', '')}\n```python\n{rec['code'].strip()}\n```\n")


def _fails(failures):
    if not failures:
        return ""
    return "Recent failures on this island (do not repeat):\n" + "\n".join(
        f"- {f['name']} [{f['status']}]: {f.get('reason', '')}" + ("".join(f"\n    {x}" for x in f.get("failures", [])[:3])) for f in failures)


def cf_user_mutation(gen, island, parent, analysis_text, profile_text, failures, n):
    return "\n\n".join(x for x in [
        f"Generation {gen}, island {island}.  Operator: MUTATION.",
        _prog(parent, "Parent"),
        "Pre-mutation analysis of the parent (held-out buildings):\n" + analysis_text,
        ("Measured vs predicted (the parent's 3 h-ahead forecast) for two rooms:\n" + profile_text) if profile_text else "",
        _fails(failures),
        f"Write {n} children of this parent.  Each must change ONE thing that the analysis points to (a template's "
        "formula, a new template, a different history summary, a history-inferred material value) and keep the rest. "
        "Say in the hypothesis which diagnostic line motivates the change."] if x)


def cf_user_crossover(gen, island, a, b, cx_text, failures, n):
    return "\n\n".join(x for x in [
        f"Generation {gen}, island {island}.  Operator: CROSSOVER.",
        _prog(a, "Parent A"), _prog(b, "Parent B"),
        "Pre-crossover analysis:\n" + cx_text,
        _fails(failures),
        f"Write {n} children that COMBINE the parents: take each template from the parent that is better where that "
        "template matters (e.g. the exterior-wall template from the parent better on exterior rooms), make the pieces "
        "consistent, and state in the hypothesis which template came from which parent."] if x)


def cf_user_explore(gen, island, elites, failures, n):
    lb = "\n".join(f"- {e['name']} [{e['id']}]: score {e['score']:.4f}; {e.get('hypothesis', '')[:300]}" for e in elites) or "- none yet"
    return "\n\n".join(x for x in [
        f"Generation {gen}, island {island}.  Operator: EXPLORE.",
        "Programs on this island so far (best first):\n" + lb,
        _fails(failures),
        f"Write {n} candidates with a STRUCTURALLY different hypothesis from all of the above (a different slow-store "
        "summary, a different way to infer per-building materials from history, different coupling laws), not a "
        "re-tuning."] if x)
