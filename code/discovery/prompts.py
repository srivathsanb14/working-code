"""What the LLM sees, and what it must send back.

INPUT to the LLM (closed-form discovery, evolve.py; one stateless request per operator):
  system prompt (the same in every request, so it caches): the fixed state-space function and the 11 coefficients to
    write, the data dictionary (names, shapes, units; no values), the universal constants, the contract and rules,
    and how a candidate is scored on validation
  user message, by operator:
    - init:      how many first candidates to write
    - mutation:  the parent (code, hypothesis, five errors), its analysis on all validation episodes (when, where,
                 which term), one real validation day hourly for one room of each type
    - crossover: both parents and the analysis of who is better where and for which term
    - explore:   the island's best programs with their errors and hypotheses
    - every operator: the island's recent rejected or crashed programs with the reason, and its recent children
      that ran but scored WORSE than their parent (what they tried and how much each error moved)

OUTPUT from the LLM (structured JSON, schema RESPONSE_SCHEMA):
    {"candidates": [{"name": str, "hypothesis": str, "code": str}, ...]}
  'code' is a complete candidate module obeying the contract (contract.py).
"""
import json

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
    """Robust parse of the LLM answer: tolerates code fences, prose around the JSON, a bare list,
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
    290 K air pulls a room to 290 K and never past it.'''

CF_SCORING = """\
How candidates are scored (no fitting anywhere):
  stage 0  the checks above on a real floor plan; failures are rejected with the reason.
  All numbers you see come from the VALIDATION buildings and weather.  There is no training data (nothing is fitted).
  Separate test buildings and weather (a different time window, unseen buildings, and an unseen climate) are kept
  aside and scored only once, at the end, so formulas must generalize rather than suit the validation set.
  stage 1  quick screen: one-step (5-minute) RMSE on the validation episodes, from measured history; more than
           {screen}x the persistence error (T_next = T_now) is screened out.
  stage 2  decisive: from measured history at many moments, free-run {horizons} ahead; a rollout leaving [230, 350] K
           marks the candidate unstable.  One-step accuracy does not guarantee good rollouts.
           Score = mean {main} RMSE on the validation episodes + {cw} K per AST node of your code, computed AS AT TEST TIME:
           the controller's own future commands are known, but the future outside temperature and occupancy are not
           (they are held at their last observed values during the rollout).  Rollouts with the true future inputs are
           reported as a diagnostic.
{selection}
  The test buildings are unseen: formulas must work for any floor plan; never branch on a building's size or
  identity (code comparing geo values or array sizes to fixed numbers is rejected)."""

CF_SYSTEM = """\
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

{data}

UNIVERSAL CONSTANTS U (reviewed list; same for every building and weather):
{constants}

{contract}

{scoring}

Reply only in the JSON format requested: a list of candidates, each with a short name, a one-paragraph hypothesis
(what changed and which diagnostic motivates it), and the complete module source in "code"."""


def selection_text(keys):
    from .closedform import OBJECTIVES
    if not keys:
        return "  Selection: islands keep the programs with the lowest score."
    rows = "\n".join(f"    {k:<11} {OBJECTIVES[k][0]}  (mostly decided by: {OBJECTIVES[k][1]})" for k in keys)
    return ("  Selection is MULTI-OBJECTIVE: a program survives if no other program is at least as good on every one of\n"
            "  these objectives and better on one (Pareto front), so getting one group of coefficients right is rewarded\n"
            "  even before the others are right:\n" + rows)


def _objs(rec):
    o = rec.get("objectives") or {}
    return ("; objectives " + ", ".join(f"{k} {v:.3f}" if k != "complexity" else f"{k} {v:.0f}" for k, v in o.items())) if o else ""


def cf_system_prompt(constants_raw, screen=5.0, horizons=(12, 36, 72), main_h=36, cw=0.0001, keys=None):
    rows = "\n".join(f"  U[\"{k}\"] = {v['value']:<10g} {v['unit']:<7} {v['what']}" for k, v in constants_raw.items() if not k.startswith("_"))
    hz = ", ".join(f"{h / 12:g} h" for h in horizons)
    return CF_SYSTEM.format(data=CF_DATA, constants=rows, contract=CF_CONTRACT,
                            scoring=CF_SCORING.format(screen=screen, horizons=hz, main=f"{main_h / 12:g} h", cw=cw,
                                                      selection=selection_text(keys)))


def _prog(rec, label):
    return (f"### {label}: {rec['name']}  [{rec['id']}, island {rec['island']}, score {rec['score']:.4f}{_objs(rec)}]\n"
            f"Hypothesis: {rec.get('hypothesis', '')}\n```python\n{rec['code'].strip()}\n```\n")


def _fails(failures):
    if not failures:
        return ""
    return "Recent failures on this island (do not repeat):\n" + "\n".join(
        f"- {f['name']} [{f['status']}]: {f.get('reason', '')}" + ("".join(f"\n    {x}" for x in f.get("failures", [])[:3])) for f in failures)


def _worse(setbacks):
    """Children that ran but scored worse than their best parent: the edit tried and how much each error moved (K; + = worse)."""
    if not setbacks:
        return ""
    return ("Recent children on this island that RAN but scored WORSE than their parent (do not repeat these edits; if the idea "
            "still seems right, change it in a different way):\n" + "\n".join(
                f"- {w['name']} [{w['id']}, {w['operator']} of {w['parent']}]: score {w['score']:.4f} vs parent {w['parent_score']:.4f} "
                f"({w['score'] - w['parent_score']:+.4f})" + ("; " + ", ".join(f"{k} {v:+.3f}" for k, v in w["deltas"].items()) if w["deltas"] else "")
                + f"\n    tried: {w['hypothesis'][:250]}" for w in setbacks))


def cf_user_mutation(gen, island, parent, analysis_text, profile_text, failures, n, keys=None, worse=None):
    return "\n\n".join(x for x in [
        f"Generation {gen}, island {island}.  Operator: MUTATION.",
        _prog(parent, "Parent"),
        "Pre-mutation analysis of the parent (validation buildings): where, when and through which term it errs\n" + analysis_text,
        ("A real trajectory excerpt (one validation day, one room of each type, the parent's 3 h-ahead forecast):\n" + profile_text) if profile_text else "",
        _fails(failures),
        _worse(worse),
        f"Write {n} children of this parent.  Each must change ONE thing that the analysis points to (one "
        "coefficient formula, a different history summary, a history-inferred material value) and keep the rest. "
        "Use the TERMS slopes, the SPATIAL correlations and the TEMPORAL breakdown to decide which coefficient to change "
        "and in which direction, and say in the hypothesis which diagnostic line motivates the change and which "
        "objective it should improve."] if x)


def cf_user_crossover(gen, island, a, b, cx_text, failures, n, keys=None, worse=None):
    return "\n\n".join(x for x in [
        f"Generation {gen}, island {island}.  Operator: CROSSOVER.",
        _prog(a, "Parent A"), _prog(b, "Parent B"),
        "Pre-crossover analysis:\n" + cx_text,
        _fails(failures),
        _worse(worse),
        f"Write {n} children that COMBINE the parents: take each coefficient formula from the parent that is better "
        "where that coefficient matters (the TERMS lines say which parent's formula is closer to right for each term; the "
        "SPATIAL and TEMPORAL lines say where and when each parent wins), make "
        "the formulas consistent, and state in the hypothesis which formula came from which parent."] if x)


def cf_user_init(gen, island, failures, n):
    return "\n\n".join(x for x in [
        f"Generation {gen}, island {island}.  Operator: INIT (no programs exist yet on this island).",
        _fails(failures),
        f"Write {n} first candidates from the physics and the contract above.  Make them differ from each other in "
        "structure (which stores they use, how the per-building materials are read from the measured history, how the "
        "neighbour exchange is written), and state in each hypothesis which physical assumption it tests."] if x)


def cf_user_explore(gen, island, elites, failures, n, keys=None, worse=None):
    lb = "\n".join(f"- {e['name']} [{e['id']}]: score {e['score']:.4f}{_objs(e)}; {e.get('hypothesis', '')[:300]}" for e in elites) or "- none yet"
    return "\n\n".join(x for x in [
        f"Generation {gen}, island {island}.  Operator: EXPLORE.",
        "Programs on this island so far (best first):\n" + lb,
        _fails(failures),
        _worse(worse),
        f"Write {n} candidates with a STRUCTURALLY different hypothesis from all of the above (a different "
        "use of the stores, a different way to infer per-building materials from history, different coupling laws), not a "
        "re-tuning."] if x)
