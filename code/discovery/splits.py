"""Data splits for closed-form discovery: one validation split and three held-out test tiers.  No training split.

Nothing is fitted, so there is nothing to train.  The search (mutation, crossover, islands) only ever sees the
VALIDATION split: every candidate is scored there and every analysis the LLM reads comes from there.  The three
TEST tiers are never looked at during the search; the final candidates are scored on them once, at the end.

  test_easy    seen building, seen weather station, unseen weather window
               (e.g. atlanta_3 is in validation through another building, this building's atlanta_2 window is not)
  test_medium  unseen building, seen weather stations
  test_hard    unseen building AND unseen station: the station whose outside temperatures are farthest, in
               Wasserstein-1 distance, from all the other stations pooled (on the shipped data: Fairbanks)
  test_station seen building, unseen station (kept out of validation so the hard station stays unseen;
               reported as an extra diagnostic)

Which buildings are unseen: the most atypical floor plans (farthest from the others in standardized geometry:
floor area, number of rooms, mean room area, exterior wall per floor area), so "unseen building" means a building
unlike the ones the search scored on.

The split is computed from scenario folders only (config.yaml weather window + the station's weather CSV, and the
floor plan), so it is fixed before any data is simulated and works for the shipped scenarios and generated ones alike.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import yaml

DAY_STEPS = 288


def scan_scenarios(roots, kit):
    """Every scenario folder (<root>/<building>/<weather>/config.yaml) as a dict."""
    out = []
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for cfg in sorted(root.glob("*/*/config.yaml")):
            c = yaml.safe_load(cfg.read_text())
            w = c.get("weather", {})
            if "weather_csv" not in w:
                continue
            csv = Path(w["weather_csv"])
            if not csv.is_absolute():
                cand = [Path(kit) / "smart_buildings" / "weather" / csv, cfg.parent / csv, Path(kit) / csv]
                csv = next((p for p in cand if p.exists()), cand[0])
            b, wname = cfg.parent.parent.name, cfg.parent.name
            out.append(dict(building=b, weather=wname, scenario=f"{b}/{wname}", station=wname.rsplit("_", 1)[0],
                            start=str(w.get("start_timestamp", "")), csv=str(csv), dir=str(cfg.parent)))
    return out


def weather_window(csv, start, days=30):
    """Hourly outside temperatures (deg C) of the scenario's window, from the station's weather CSV."""
    import pandas as pd
    d = pd.read_csv(csv, usecols=["Time", "TempC"])
    t = pd.to_datetime(d["Time"], format="%Y%m%d-%H%M")
    t0 = pd.Timestamp(start) if start else t.iloc[0]
    m = (t >= t0) & (t < t0 + pd.Timedelta(days=days))
    return d["TempC"].to_numpy(float)[m.to_numpy()]


def station_distances(scen, days=30):
    """Wasserstein-1 distance (K) of each station's pooled window temperatures to every other station, and to the
    rest pooled.  Returns (to_rest {station: d}, pairwise {station: {station: d}})."""
    from scipy.stats import wasserstein_distance as w1
    cache, pool = {}, {}
    for s in scen:
        key = (s["csv"], s["start"])
        if key not in cache:
            cache[key] = weather_window(s["csv"], s["start"], days)
        pool.setdefault(s["station"], []).append(cache[key])
    pool = {k: np.concatenate(v) for k, v in pool.items()}
    st = sorted(pool)
    pair = {a: {b: float(w1(pool[a], pool[b])) for b in st} for a in st}
    rest = {a: float(w1(pool[a], np.concatenate([pool[b] for b in st if b != a]))) if len(st) > 1 else 0.0 for a in st}
    return rest, pair


def building_features(scen_dir, buildings_root=None):
    """Standardizable geometry of one building from its floor plan: floor area (m2), rooms, mean room area (m2),
    exterior wall faces per m2 of floor."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import room_model as rm
    d = Path(scen_dir)
    cands = [d, d.parent] + ([Path(buildings_root) / d.parent.name] if buildings_root else [])
    src = next((c for c in cands if (c / "floor_plan.npy").exists() and (c / "zone_map.npy").exists()), None)
    if src is None:
        return None
    G = rm.grid_geometry(np.load(src / "floor_plan.npy"), np.load(src / "zone_map.npy", allow_pickle=True))
    cls, n = G["cls"], int(G["nroom"])
    air = float((cls == rm.AIR).sum()) * 0.2 * 0.2
    shell = float((cls == rm.SHELL_OR_EXT).sum())
    return np.array([math.log(max(air, 1.0)), n, air / max(n, 1), shell / max(air, 1.0)])


def atypical_buildings(feats, k):
    """Greedy farthest-point choice of k buildings in standardized feature space (first: farthest from the mean)."""
    names = sorted(feats)
    X = np.array([feats[b] for b in names], float)
    Z = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1.0)
    chosen = [int(np.argmax(np.linalg.norm(Z, axis=1)))]
    while len(chosen) < min(k, len(names)):
        d = np.min([np.linalg.norm(Z - Z[c], axis=1) for c in chosen], axis=0)
        d[chosen] = -1
        chosen.append(int(np.argmax(d)))
    score = np.linalg.norm(Z, axis=1)
    return [names[i] for i in chosen], {b: float(s) for b, s in zip(names, score)}


def make_split(scen, hard_station=None, hard_buildings=None, hard_frac=0.25, easy_frac=0.25, buildings_root=None, days=30):
    """From scenario folders (before any simulation): the split as {val, test_easy, test_medium, test_hard,
    test_station, test, _why} of exact building/weather names."""
    rest, pair = station_distances(scen, days)
    feats = {}
    if not hard_buildings:
        for b in sorted({s["building"] for s in scen}):
            f = building_features(next(s["dir"] for s in scen if s["building"] == b), buildings_root)
            if f is not None:
                feats[b] = f
    for s in scen:
        s.setdefault("window", (s["csv"], s["start"]))
    return assign(scen, rest, pair, feats, hard_station, hard_buildings, hard_frac, easy_frac, source="scenario folders")


def split_from_episodes(eps, hard_station=None, hard_buildings=None, hard_frac=0.25, easy_frac=0.25):
    """From whatever episodes are generated and available (the default inside discovery): stations compared by the
    Wasserstein distance of the outside temperatures actually in the episodes, buildings by the geometry a policy
    computes from the metadata (floor area, rooms, mean room area, outside wall per floor area)."""
    from scipy.stats import wasserstein_distance as w1
    scen, pool, feats = {}, {}, {}
    for e in eps:
        if e.scenario in scen:
            continue
        b, w = e.scenario.split("/", 1)
        To = np.asarray(e.inputs["T_out"], float)
        scen[e.scenario] = dict(building=b, weather=w, scenario=e.scenario, station=w.rsplit("_", 1)[0],
                                window=hash(np.round(To[:: 12], 2).tobytes()))
        pool.setdefault(w.rsplit("_", 1)[0], []).append(To)
        g = e.geo
        if b not in feats:
            area = float(np.sum(g["area"]))
            feats[b] = np.array([math.log(max(area, 1.0)), int(g["V"]), area / max(int(g["V"]), 1),
                                 float(np.sum(g["facade_faces"])) / max(area, 1.0)])
    pool = {k: np.concatenate(v) for k, v in pool.items()}
    st = sorted(pool)
    pair = {a: {b: float(w1(pool[a], pool[b])) for b in st} for a in st}
    rest = {a: float(w1(pool[a], np.concatenate([pool[b] for b in st if b != a]))) if len(st) > 1 else 0.0 for a in st}
    return assign(list(scen.values()), rest, pair, feats, hard_station, hard_buildings, hard_frac, easy_frac,
                  source="available episodes")


def assign(scen, rest, pair, feats, hard_station=None, hard_buildings=None, hard_frac=0.25, easy_frac=0.25, source=""):
    """The tiers, given station distances and building features.  Nothing here depends on particular names."""
    hs = hard_station or (max(rest, key=rest.get) if len(rest) > 1 else None)
    blds = sorted({s["building"] for s in scen})
    if hard_buildings:
        hb, atyp = list(hard_buildings), {}
    else:
        k = max(1, math.ceil(hard_frac * len(blds))) if len(blds) > 1 else 0
        hb, atyp = atypical_buildings(feats, k) if len(feats) > k and k else (blds[:k], {})
    seen = [b for b in blds if b not in hb]
    out = {"val": [], "test_easy": [], "test_medium": [], "test_hard": [], "test_station": []}
    for s in scen:
        if s["building"] in hb:
            out["test_hard" if s["station"] == hs else "test_medium"].append(s["scenario"])
        elif s["station"] == hs:
            out["test_station"].append(s["scenario"])
    # easy: per seen building (round robin over stations) one window whose building and station stay in validation
    pool = [s for s in scen if s["building"] in seen and s["station"] != hs]
    n_easy = int(round(easy_frac * len(pool)))
    by_b = {b: sorted([s for s in pool if s["building"] == b], key=lambda s: s["weather"]) for b in seen}
    easy, i = set(), 0
    while len(easy) < n_easy and i < 10 * max(len(seen), 1) * max((len(v) for v in by_b.values()), default=1):
        b = seen[i % len(seen)]
        cand = by_b[b][(seen.index(b) + i // len(seen)) % len(by_b[b])] if by_b[b] else None
        i += 1
        if cand is None or cand["scenario"] in easy:
            continue
        left_b = [s for s in by_b[b] if s["scenario"] not in easy and s is not cand]
        left_st = [s for s in pool if s["station"] == cand["station"] and s["scenario"] not in easy and s is not cand
                   and s["window"] != cand["window"]]
        if left_b and left_st:
            easy.add(cand["scenario"])
    for s in pool:
        out["test_easy" if s["scenario"] in easy else "val"].append(s["scenario"])
    out["test"] = out["test_easy"] + out["test_medium"] + out["test_hard"]
    out["_why"] = {
        "rule": "no training split: the search scores on val only; test_* are scored once at the end",
        "computed_from": source,
        "hard_station": hs, "station_wasserstein_to_rest_K": {k: round(v, 2) for k, v in sorted(rest.items(), key=lambda x: -x[1])},
        "station_wasserstein_pairwise_K": {a: {b: round(v, 1) for b, v in d.items()} for a, d in pair.items()},
        "held_out_buildings": hb, "building_atypicality_z": {k: round(v, 2) for k, v in sorted(atyp.items(), key=lambda x: -x[1])},
        "counts": {k: len(v) for k, v in out.items() if isinstance(v, list)}}
    return out


def check_whole_episodes(S, tiers=("val", "test_easy", "test_medium", "test_hard", "test_station")):
    """Every episode belongs to at most one split, and every scenario (building + weather window, all its controllers)
    to exactly one: no episode and no scenario is ever cut between validation and test.  Raises ValueError otherwise;
    returns {split: [episode names]} for the record."""
    where_ep, where_sc = {}, {}
    for k in tiers:
        for e in S.get(k, []):
            where_ep.setdefault(e.name, set()).add(k)
            where_sc.setdefault(e.scenario, set()).add(k)
    bad_ep = {n: sorted(v) for n, v in where_ep.items() if len(v) > 1}
    bad_sc = {n: sorted(v) for n, v in where_sc.items() if len(v) > 1}
    if bad_ep or bad_sc:
        raise ValueError(f"split cuts data across splits: episodes {bad_ep}, scenarios {bad_sc}")
    return {k: sorted(e.name for e in S.get(k, [])) for k in tiers if S.get(k)}


def split_episodes(eps, split):
    """Assign loaded episodes to the split's parts.  split: {"val": [...], "test_easy": [...], ...} of names or glob
    patterns.  A pattern with "/" is matched against "building/weather", one without against the building name; a pattern
    starting with "!" excludes what it matches.  Keys starting with "_" are comments.  Every part except the test tiers
    must match at least one episode."""
    from fnmatch import fnmatch

    def hit(e, pat):
        return fnmatch(e.scenario, pat) if "/" in pat else fnmatch(e.building, pat)

    def match(e, pats):
        if any(hit(e, p[1:]) for p in pats if p.startswith("!")):
            return False
        return any(hit(e, p) for p in pats if not p.startswith("!"))
    out = {k: [e for e in eps if match(e, v)] for k, v in split.items() if not k.startswith("_")}
    for k, v in out.items():
        if not v and not k.startswith("test"):
            raise SystemExit(f"split '{k}' matched no episodes; adjust --split")
    return out


def stratified_sample(eps, k, seed=0):
    """Pick k episodes so that every (building, station, controller) group is represented before any group gets a
    second one; within that, deterministic random order.  k <= 0 or k >= len(eps) returns everything."""
    if not k or k <= 0 or len(eps) <= k:
        return list(eps)
    rng = np.random.default_rng(seed)
    groups = {}
    for e in eps:
        groups.setdefault((e.building, e.weather.rsplit("_", 1)[0], e.policy), []).append(e)
    order = [groups[g] for g in sorted(groups)]
    for g in order:
        rng.shuffle(g)
    rng.shuffle(order)
    out, i = [], 0
    while len(out) < k:
        for g in order:
            if i < len(g):
                out.append(g[i])
                if len(out) == k:
                    break
        i += 1
    return sorted(out, key=lambda e: e.name)


def coverage(eps):
    """How many buildings, stations and controllers a list of episodes covers."""
    return dict(episodes=len(eps), buildings=len({e.building for e in eps}),
                stations=len({e.weather.rsplit("_", 1)[0] for e in eps}), controllers=len({e.policy for e in eps}))


def light_per_scenario(paths):
    """One light episode (geometry + outside temperature) per scenario, from episode files: what the split needs."""
    from .episodes import load_light
    seen, out = set(), []
    for p in paths:
        e = load_light(p)
        if e.scenario not in seen:
            seen.add(e.scenario); out.append(e)
    return out
