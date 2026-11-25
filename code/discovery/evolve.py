"""Closed-form discovery with islands, mutation and crossover (the default discovery mode; no fitting anywhere).

Input: a folder of finished episodes (index.json + one .npz per episode) produced elsewhere, by the data repository
that runs the starter kit.  Nothing is simulated here.  The episodes are split, whole, into validation (all the search
sees) and held-out test tiers (splits.py).  Every candidate, generation 0 included, is written by the Claude API.

Generation 0, for every island:
   INIT       the system prompt alone -> one LLM request for --n-init first candidates
Each later generation, for every island:
   MUTATION   one parent (tournament from the island's best) + its PRE-MUTATION analysis
              -> one LLM request for --n-mut children
   CROSSOVER  two parents + the PRE-CROSSOVER analysis (who is better where; which templates differ)
              -> one LLM request for --n-cx children
   EXPLORE    the island's leaderboard -> one LLM request for --n-exp structurally new candidates
 every child: stage 0 checks -> stage 1 one-step screen -> stage 2 rollouts (closedform.evaluate)
 every --migrate-every generations the best of island k is copied into island k+1 (ring)

Everything a person needs to follow the search is written to outputs/discovery/<run>/:
  REPORT.md                    start here: leaderboard, lineage, operator statistics, plots, where everything is
  log.md                       timeline, one line per request and per child (with parents and the change in score)
  database.jsonl               one JSON record per candidate (island, operator, parents, scores, analysis)
  candidates/<id>_<name>.py    the source of every candidate
  generations/gNNN/island_K/<operator>/prompt.md, analysis.md, response.json, candidates.json
  plots/<id>_profile.png       measured vs predicted temperatures (one-step and 3 h ahead), two rooms
  plots/<id>_horizon.png       rollout error vs horizon, against persistence (and the fitted ceiling if computed)
  plots/progress.png           best score per island per generation

    .venv/bin/python code/run_discovery.py --dataset /path/to/dataset --run cf --generations 20 --islands 3 --test --max-usd 50
"""
from __future__ import annotations

import argparse
import hashlib
import json
import signal
import time
from pathlib import Path

import numpy as np

from . import closedform as cf
from . import pareto, prompts, proposers
from .episodes import _load_cached
from .starterkit_data import DATA

ROOT = Path(__file__).resolve().parents[2]
OUTD = ROOT / "outputs" / "discovery"
SEEDS_CF = Path(__file__).resolve().parent / "seeds_cf"


# ----------------------------------------------------------------------------- data

TEST_TIERS = ("test_easy", "test_medium", "test_hard", "test_station")


def load_split(args):
    """Validation episodes (all the search ever sees) and the held-out test tiers (scored once, at the end).
    There is no training split: nothing is fitted.  A split file that still has "train" is accepted, and its
    train episodes are ignored.

    Memory: episodes are chosen by name from index.json first; only the validation sample is loaded here (one
    light episode per scenario if the split must be computed).  Test tiers stay as file references and are loaded in
    chunks by score_tests."""
    from types import SimpleNamespace

    from . import splits
    ds = Path(args.dataset)
    if not (ds / "index.json").exists():
        raise SystemExit(f"no dataset at {ds} (index.json missing): run code/build_dataset.py first (docs/RUNBOOK.md), "
                         "or try the synthetic self-test: code/run_discovery.py --selftest")
    idx = json.loads((ds / "index.json").read_text())
    sp_path = Path(args.split) if args.split else ds / "split.json"
    split = json.loads(sp_path.read_text()) if sp_path.exists() else {}
    refs = []
    for name, v in idx.items():
        if v.get("status") != "ok":
            continue
        path = ds / v["file"]
        m = v if all(k in v for k in ("building", "weather", "policy")) else json.loads(str(np.load(path)["meta"]))
        refs.append(SimpleNamespace(name=name, building=m["building"], weather=m["weather"], policy=m["policy"],
                                    scenario=m.get("scenario") or f"{m['building']}/{m['weather']}", path=path))
        if args.policies and refs[-1].policy not in args.policies:
            refs.pop()
    if args.auto_split or not any(k in split for k in TEST_TIERS):
        first = {}
        for r in refs:
            first.setdefault(r.scenario, r.path)
        split = splits.split_from_episodes(splits.light_per_scenario(list(first.values())))
        w = split["_why"]
        print(f"split computed from the {len(refs)} available episodes: hard station {w['hard_station']} "
              f"(Wasserstein to the rest {w['station_wasserstein_to_rest_K']}), held-out buildings {w['held_out_buildings']}, "
              f"counts {w['counts']}")
    S = splits.split_episodes(refs, {k: v for k, v in split.items() if k != "train" and not k.startswith("_")})
    if split.get("train"):
        print(f"note: the split has a 'train' list ({len(split['train'])} patterns); it is ignored, discovery uses no training data")
    split["_episodes"] = splits.check_whole_episodes(S)         # whole episodes only; raises if any is shared
    pick = splits.stratified_sample(S["val"], args.max_val)
    cv, ca = splits.coverage(pick), splits.coverage(S["val"])
    print(f"validation sample: {cv['episodes']} of {ca['episodes']} episodes, covering {cv['buildings']}/{ca['buildings']} buildings, "
          f"{cv['stations']}/{ca['stations']} stations, {cv['controllers']}/{ca['controllers']} controllers")
    val = [_load_cached(r.path) for r in pick]
    tests = {k: S[k] for k in TEST_TIERS if S.get(k)}
    if not tests and S.get("test"):
        tests = {"test": S["test"]}
    if getattr(args, "max_test", 0):
        tests = {k: splits.stratified_sample(v, args.max_test) for k, v in tests.items()}
    return val, tests, split, len(refs)


# ----------------------------------------------------------------------------- database

class DB:
    def __init__(self, d: Path):
        self.dir = d; d.mkdir(parents=True, exist_ok=True); (d / "candidates").mkdir(exist_ok=True)
        self.path = d / "database.jsonl"
        self.recs = [json.loads(x) for x in self.path.read_text().splitlines()] if self.path.exists() else []

    def add(self, r):
        self.recs.append(r)
        with self.path.open("a") as f:
            f.write(json.dumps(r, default=float) + "\n")

    def get(self, i):
        return next(r for r in self.recs if r["id"] == i)

    def ok(self, island=None):
        return [r for r in self.recs if r["status"] == "ok" and (island is None or r["island"] == island)]

    def elites(self, island, k, keys=None):
        """The island's population: its best k programs, by Pareto rank + crowding over `keys` (multi-objective),
        or by the scalar score when keys is None."""
        best = {}
        for r in sorted(self.ok(island), key=lambda r: r["score"]):
            best.setdefault(r["hash"], r)               # a migrated copy and its original count once
        pop = list(best.values())
        return pareto.select(pop, k, keys) if keys else pop[:k]

    def failures(self, island, k):
        return [r for r in self.recs if r["island"] == island and r["status"] not in ("ok",) and r["operator"] != "migration"][-k:]

    def setbacks(self, island, k):
        """The island's last k children that ran and were scored but did WORSE than their best parent: what was tried,
        and how much each objective moved.  Shown to the LLM so it does not repeat an edit that already hurt."""
        out = []
        for r in self.recs:
            if r["island"] != island or r["status"] != "ok" or r["operator"] not in ("mutation", "crossover") or not r.get("parents"):
                continue
            ps = [p for p in (self._by_id(i) for i in r["parents"]) if p is not None and p.get("status") == "ok"]
            if not ps:
                continue
            best = min(ps, key=lambda p: p["score"])
            if r["score"] > best["score"]:
                po, ro = best.get("objectives") or {}, r.get("objectives") or {}
                out.append(dict(id=r["id"], name=r["name"], operator=r["operator"], parent=best["id"], hypothesis=r.get("hypothesis", ""),
                                score=r["score"], parent_score=best["score"],
                                deltas={k: ro[k] - po[k] for k in ro if k in po and k != "complexity"}))
        return out[-k:]

    def _by_id(self, i):
        return next((r for r in self.recs if r["id"] == i), None)

    def generation(self):
        return max((r["generation"] for r in self.recs), default=-1)

    def next_id(self, gen, island):
        return f"g{gen:03d}_i{island}_{len(self.recs):04d}"


def _hash(code):
    return hashlib.sha1(code.strip().encode()).hexdigest()[:12]


# ----------------------------------------------------------------------------- evaluation

class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def evaluate(code, val, U, args):
    signal.signal(signal.SIGALRM, _alarm); signal.alarm(int(args.timeout))
    try:
        return cf.evaluate(code, val, U, complexity_weight=args.complexity_weight, horizons=tuple(args.horizons),
                           main_h=args.main_horizon, every=args.every, warmup=args.warmup, screen_factor=args.screen,
                           deploy_score=(args.score == "deploy"))
    except _Timeout:
        return {"status": "error", "reason": f"timeout after {args.timeout} s"}
    finally:
        signal.alarm(0)


_G = None          # (val, U, args) shared with forked workers


def _eval_worker(code):
    va, U, args = _G
    return evaluate(code, va, U, args)


def evaluate_many(codes, val, U, args):
    """Evaluate candidates serially, or in --workers forked processes; same results either way."""
    global _G
    if args.workers <= 1 or len(codes) <= 1:
        return [evaluate(c, val, U, args) for c in codes]
    import multiprocessing as mp
    _G = (val, U, args)
    with mp.get_context("fork").Pool(min(args.workers, len(codes))) as pool:
        return pool.map(_eval_worker, codes)


METRICS = ("score", "score_metric", "warnings", "complexity", "history", "stage", "reason", "failures", "traceback",
           "stage1_one_step_mean", "stage1_persistence_mean", "val_one_step_mean", "unstable_episodes", "horizons", "main_horizon",
           "objectives")


def record(db, code, name, hyp, gen, island, op, parents, req_dir, res, val, wall):
    r = {"id": db.next_id(gen, island), "generation": gen, "island": island, "operator": op, "parents": parents,
         "name": name, "hypothesis": hyp, "hash": _hash(code), "status": res["status"],
         "request": str(req_dir.relative_to(db.dir)) if req_dir else None, "wall_sec": round(wall, 1)}
    r.update({k: res[k] for k in METRICS if k in res})
    r.update({k: v for k, v in res.items() if k.startswith("val_") and k != "val_by_episode"})
    an = cf.analysis(res, val)
    r["analysis"] = an["text"]; r["groups"] = an["groups"]; r["terms"] = an["terms"]; r["rooms"] = an["rooms"]
    path = db.dir / "candidates" / f"{r['id']}_{name}.py"
    path.write_text(code); r["file"] = str(path.relative_to(db.dir))
    db.add(r)
    return dict(r, code=code)


def code_of(db, r):
    return (db.dir / r["file"]).read_text()


# ----------------------------------------------------------------------------- logs

class Log:
    def __init__(self, path):
        self.path = path
        if not path.exists():
            path.write_text("# Discovery log\n\nOne line per LLM request and per candidate. Everything is measured on the "
                            "VALIDATION split (no training split; the test tiers are scored once at the end). Score = "
                            "validation rollout RMSE (K) at the main horizon + complexity price; selection is by Pareto rank "
                            "over the objectives unless --selection scalar. Δ = change in score against the best parent.\n\n")

    def __call__(self, s):
        print(s, flush=True)
        with self.path.open("a") as f:
            f.write(s + "\n")


def _fmt(r):
    if r["status"] == "ok":
        return f"score {r['score']:.4f} (1 h {r.get('val_1h_mean', float('nan')):.3f}, 3 h {r.get('val_3h_mean', float('nan')):.3f}, 6 h {r.get('val_6h_mean', float('nan')):.3f} K)"
    return f"{r['status'].upper()} at stage {r.get('stage', '?')}: {r.get('reason', '')}"


def _excerpt_episode(val):
    """The validation episode to show as a trajectory: the first whose HVAC both runs and stops (switching is where
    the stores show), else the first."""
    for k, e in enumerate(val):
        on = (e.inputs["g"].mean(1) > cf.HVAC_ON).mean()
        if 0.1 < on < 0.9:
            return k, e
    return 0, val[0]


def _excerpt_rooms(geo):
    """One room of each type: the most exposed corner room, the median one-side exterior room, the largest interior room."""
    expo = geo["facade_faces"] * 0.2 / np.maximum(geo["area"], 1e-9)
    corner = np.where((geo["facade_faces"] > 0) & (geo.get("facade_sides", np.zeros(geo["V"])) >= 2))[0]
    side = np.where((geo["facade_faces"] > 0) & ~np.isin(np.arange(geo["V"]), corner))[0]
    inter = np.where(geo["facade_faces"] == 0)[0]
    out = []
    if len(corner):
        out.append(("corner", int(corner[np.argmax(expo[corner])])))
    if len(side):
        out.append(("one-side exterior", int(side[np.argsort(expo[side])[len(side) // 2]])))
    if len(inter):
        out.append(("interior", int(inter[np.argmax(geo["area"][inter])])))
    return out


def profile_text(code, val, U, horizon):
    """A real trajectory excerpt for the prompt: one validation episode, one room of each type, every hour of its
    second day: measured temperature, the parent's forecast made `horizon` steps earlier from measured history, the
    room's HVAC air and reheat, and the outside temperature.  About 2 KB."""
    k, ep = _excerpt_episode(val)
    rooms = _excerpt_rooms(ep.geo)
    try:
        P = cf.profile(code, ep, U, horizon, rooms=[r for _, r in rooms])
    except Exception as e:  # noqa: BLE001
        return f"(profile failed: {e})"
    X, hz, ht, inp, geo = P["X"], P["hz"], P["hz_t"], ep.inputs, ep.geo
    head = [f"validation episode {k + 1} of {len(val)} ({geo['V']} rooms), day 2, hourly; T = measured / forecast made "
            f"{horizon * 5 / 60:g} h earlier (C); air = supply-air W/K into the room; rh = reheat W"]
    for name, i in rooms:
        head.append(f"  {name} room #{i}: area {geo['area'][i]:.0f} m2, outside-wall faces {int(geo['facade_faces'][i])}, "
                    f"doors {int(geo['n_doors'][i].sum())}, neighbours {int(geo['n_neighbours'][i])}")
    rows = ["hour | " + " | ".join(f"{n[:8]:>8} T meas/fcst   air    rh" for n, _ in rooms) + " | T_out"]
    for t in range(288, min(576, len(X) - 1), 12):
        j = np.searchsorted(ht, t)
        if j >= len(ht) or ht[j] != t:
            continue
        u = min(t, len(inp["T_out"]) - 1)
        hr = cf._hour(inp, np.array([u]))
        hr = f"{hr[0]:4.0f}" if hr is not None else f"{(t % 288) / 12:4.0f}"
        rows.append(f"{hr} | " + " | ".join(f"{X[t, i] - 273.15:6.2f}/{hz[j, i] - 273.15:6.2f} {inp['g'][u, i]:5.0f} {inp['Q_reheat'][u, i]:5.0f}"
                                            for _, i in rooms) + f" | {inp['T_out'][u] - 273.15:5.1f}")
    return "\n".join(head + rows)


# ----------------------------------------------------------------------------- the loop

def tournament(rng, pop, k=2, keys=None):
    pick = [pop[i] for i in rng.choice(len(pop), size=min(k, len(pop)), replace=False)]
    if keys:
        R = pareto.rank(pop, keys)
        return min(pick, key=lambda r: R[r["id"]])
    return min(pick, key=lambda r: r["score"])


def make_proposer(a):
    return proposers.ClaudeProposer(model=a.model, effort=a.effort, max_usd=a.max_usd)


def api_proposer(a):
    """The API client, created before any data is loaded so a missing library or key stops the run at once.  None when
    no request will be made (--generations 0, --dry-run)."""
    if a.generations <= 0 or a.dry_run:
        return None
    try:
        return make_proposer(a)
    except ImportError:
        raise SystemExit("the Claude API library is not installed: .venv/bin/pip install -r requirements-llm.txt "
                         "(or run with --generations 0 / --dry-run, which make no API request)") from None
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"the Claude API client could not be created ({type(e).__name__}: {e}); set ANTHROPIC_API_KEY "
                         "(or run with --generations 0 / --dry-run, which make no API request)") from None


def dry_run(a, val, U, Uraw, keys, run_dir, log):
    """--dry-run: write every kind of request exactly as a real run would build it, and stop before any API call.
    Generation 0: one init request per island.  Then, with the hand-written models in seeds_cf/ as stand-in parents
    (scored on the validation sample, recorded only in this run's folder): one mutation, one crossover and one explore
    request.  Read system_prompt.md and generations/*/*/*/prompt.md."""
    for k in range(a.islands):
        rq = run_dir / "generations" / "g000" / f"island_{k}" / "init"; rq.mkdir(parents=True, exist_ok=True)
        (rq / "prompt.md").write_text(f"# SYSTEM\n\nsee ../../../../system_prompt.md\n\n# USER\n\n{prompts.cf_user_init(0, k, [], a.n_init)}\n")
    db = DB(run_dir)
    seeds = sorted(SEEDS_CF.glob("*.py"))
    res = evaluate_many([s.read_text() for s in seeds], val, U, a)
    for s, r in zip(seeds, res):
        record(db, s.read_text(), s.stem, "hand-written model (stand-in parent for the dry run)", 0, 0, "reference", [], None, r, val, 0.0)
    pop = [dict(r, code=code_of(db, r)) for r in db.elites(0, a.elites, keys)]
    if not pop:
        log("dry run: no hand-written model passed the checks on this data; only the init prompts were written")
        return
    rq = run_dir / "generations" / "g001" / "island_0"
    p = pop[0]
    pt = profile_text(p["code"], val, U, a.main_horizon) if val else ""
    reqs = {"mutation": prompts.cf_user_mutation(1, 0, p, p["analysis"], pt, [], a.n_mut, keys, [])}
    if len(pop) >= 2:
        cx = cf.crossover_analysis(*({"analysis": {"groups": q.get("groups", {}), "terms": q.get("terms", {}), "rooms": q.get("rooms", {})},
                                      "code": q["code"]} for q in pop[:2]), ep=val[0] if val else None, U=U)
        reqs["crossover"] = prompts.cf_user_crossover(1, 0, pop[0], pop[1], cx["text"], [], a.n_cx, keys, [])
    reqs["explore"] = prompts.cf_user_explore(1, 0, pop, [], a.n_exp, keys, [])
    for op, user in reqs.items():
        (rq / op).mkdir(parents=True, exist_ok=True)
        (rq / op / "prompt.md").write_text(f"# SYSTEM\n\nsee ../../../../system_prompt.md\n\n# USER\n\n{user}\n")
    sizes = {op: len(u) for op, u in reqs.items()}
    log(f"dry run: wrote {a.islands} init prompts and {', '.join(reqs)} prompts (characters: {sizes}, system prompt "
        f"{len(prompts.cf_system_prompt(Uraw, a.screen, tuple(a.horizons), a.main_horizon, a.complexity_weight, keys))}); "
        "no API request was made")


def score_baselines(val, U, a, log, run_dir):
    """--baselines: the hand-written models, scored once on validation as REFERENCE rows; never parents."""
    out = {}
    for s in sorted(SEEDS_CF.glob("*.py")):
        res = evaluate(s.read_text(), val, U, a)
        out[s.stem] = {k: res.get(k) for k in ("status", "score", "objectives", "reason")}
        log(f"- reference (hand-written, not in the population) {s.stem} -> " + (
            f"score {res['score']:.4f}" if res["status"] == "ok" else f"{res['status']}: {res.get('reason', '')}"))
    (run_dir / "baselines.json").write_text(json.dumps(out, indent=1, default=float))


def run(a):
    run_dir = OUTD / a.run
    proposer = api_proposer(a)                            # before any data is loaded: fails at once without library or key
    db = DB(run_dir); log = Log(run_dir / "log.md")
    U, Uraw = cf.universal_constants()
    if not Uraw.get("_reviewed"):
        print("note: code/discovery/universal_constants.json is not marked reviewed yet (set _reviewed: true after checking it)")
    val, tests, split, n_runs = load_split(a)
    keys = list(a.objectives) if a.selection == "pareto" else None
    (run_dir / "config.json").write_text(json.dumps({**vars(a), "mode": "closedform", "split": split, "n_runs": n_runs,
                                                     "n_val_used": len(val), "n_test": {k: len(v) for k, v in tests.items()}},
                                                    indent=1, default=str))
    log(f"validation: {len(val)} episodes (the only data the search sees); held out until the end: "
        + (", ".join(f"{k} {len(v)}" for k, v in tests.items()) or "none"))
    system = prompts.cf_system_prompt(Uraw, a.screen, tuple(a.horizons), a.main_horizon, a.complexity_weight, keys)
    (run_dir / "system_prompt.md").write_text(system)
    rng = np.random.default_rng(a.seed)
    if a.dry_run:
        dry_run(a, val, U, Uraw, keys, run_dir, log)
        return None
    if a.ceiling and not (run_dir / "ceiling.json").exists():
        log("ceiling (diagnostic): per-room linear model fitted on the first half of each held-out episode")
        C = {e.name: cf.ceiling(e, U, tuple(a.horizons), a.every, a.warmup) for e in val}
        keys = next(iter(C.values())).keys()
        (run_dir / "ceiling.json").write_text(json.dumps({"episodes": C, "mean": {k: float(np.mean([c[k] for c in C.values() if k in c])) for k in keys}}, indent=1))

    if a.baselines and not (run_dir / "baselines.json").exists():
        log("\n## Reference models (hand-written; scored for comparison only, never parents)\n")
        score_baselines(val, U, a, log, run_dir)

    start = db.generation() + 1
    flag = run_dir / "in_progress.json"
    if flag.exists():                                     # an interrupted generation is finished first
        start = json.loads(flag.read_text())["generation"]
        log(f"\nresuming unfinished generation {start} (saved LLM answers are reused; evaluated candidates are skipped)")
    for gen in range(start, start + a.generations):
        flag.write_text(json.dumps({"generation": gen}))
        log(f"\n## Generation {gen}\n")
        made = 0
        pending = []                                      # (candidate, island, operator, parent ids, request dir, best parent score)
        for k in range(a.islands):
            pop = [dict(r, code=code_of(db, r)) for r in db.elites(k, a.elites, keys)]
            fails = db.failures(k, 4)
            worse = db.setbacks(k, 4)
            reqs = []
            if not pop:                                   # generation 0 (or an island with nothing that passed yet)
                reqs.append(("init", [], a.n_init))
            else:
                for j in range(a.mut_requests):
                    reqs.append(("mutation", [tournament(rng, pop, keys=keys)], a.n_mut))
                if len(pop) >= 2:
                    pa = tournament(rng, pop, keys=keys)
                    pb = tournament(rng, [p for p in pop if p["hash"] != pa["hash"]] or pop, keys=keys)
                    reqs.append(("crossover", [pa, pb], a.n_cx))
            if pop:
                reqs.append(("explore", pop, a.n_exp))
            seen_op = {}
            for op, parents, n in reqs:
                if n <= 0:
                    continue
                seen_op[op] = seen_op.get(op, 0) + 1
                rq = run_dir / "generations" / f"g{gen:03d}" / f"island_{k}" / (op if seen_op[op] == 1 else f"{op}_{seen_op[op]}")
                rq.mkdir(parents=True, exist_ok=True)
                saved = rq / "candidates.json"
                if saved.exists():                         # resume: reuse the answer this request already got
                    S = json.loads(saved.read_text()); cands, pids, best_parent = S["candidates"], S["parents"], S["best_parent"]
                    log(f"- island {k} **{op}**: reusing saved answer ({len(cands)} candidates)")
                    pending += [(c, k, op, pids, rq, best_parent) for c in cands]
                    continue
                if op == "mutation":
                    p = parents[0]
                    an = p["analysis"]
                    pt = profile_text(p["code"], val, U, a.main_horizon) if val else ""
                    (rq / "analysis.md").write_text(f"# Pre-mutation analysis of {p['id']} {p['name']}\n\n```\n{an}\n```\n\n## Measured vs predicted\n\n```\n{pt}\n```\n")
                    user = prompts.cf_user_mutation(gen, k, p, an, pt, fails, n, keys, worse)
                elif op == "crossover":
                    cx = cf.crossover_analysis(*({"analysis": {"groups": p.get("groups", {}), "terms": p.get("terms", {}),
                                                               "rooms": p.get("rooms", {})}, "code": p["code"]} for p in parents),
                                               ep=val[0] if val else None, U=U)
                    (rq / "analysis.md").write_text(f"# Pre-crossover analysis: A = {parents[0]['id']} {parents[0]['name']}, "
                                                    f"B = {parents[1]['id']} {parents[1]['name']}\n\n```\n{cx['text']}\n```\n")
                    user = prompts.cf_user_crossover(gen, k, parents[0], parents[1], cx["text"], fails, n, keys, worse)
                elif op == "init":
                    user = prompts.cf_user_init(gen, k, fails, n)
                else:
                    user = prompts.cf_user_explore(gen, k, pop, fails, n, keys, worse)
                (rq / "prompt.md").write_text(f"# SYSTEM\n\nsee ../../../../system_prompt.md\n\n# USER\n\n{user}\n")
                ctx = {"operator": op, "parents": parents, "dir": rq}
                usage = None
                try:
                    cands = proposer.propose(system, user, n, gen, ctx)
                    usage = getattr(proposer, "last_usage", None)
                except Exception as e:  # noqa: BLE001
                    log(f"- island {k} {op}: API request failed: {type(e).__name__}: {e}"); cands = []
                pids = [p["id"] for p in parents] if op not in ("explore", "init") else []
                best_parent = min((p["score"] for p in parents), default=None) if op not in ("explore", "init") else None
                if cands:                                  # saved BEFORE evaluation, so a crash never costs a second LLM call
                    saved.write_text(json.dumps({"candidates": cands, "parents": pids, "best_parent": best_parent, "usage": usage}, indent=1))
                cost = f"; {usage['input']}+{usage['output']} tokens, ${usage['usd']:.3f}" if usage else ""
                log(f"- island {k} **{op}**" + (f" of {', '.join(pids)}" if pids else "") + f": asked for {n}, got {len(cands)}{cost} "
                    f"([prompt]({rq.relative_to(run_dir)}/prompt.md)" + (f", [analysis]({rq.relative_to(run_dir)}/analysis.md)" if op not in ("explore", "init") else "") + ")")
                pending += [(c, k, op, pids, rq, best_parent) for c in cands]
        todo, seen_h = [], {r["hash"] for r in db.recs}
        for item in pending:
            h = _hash(item[0]["code"])
            if h in seen_h:
                log(f"  - skip duplicate {item[0]['name']}"); continue
            seen_h.add(h); todo.append(item)
        t0 = time.time()
        results = evaluate_many([c["code"] for c, *_ in todo], val, U, a)
        per = (time.time() - t0) / max(len(todo), 1)
        for (c, k, op, pids, rq, best_parent), res in zip(todo, results):
            r = record(db, c["code"], c["name"], c["hypothesis"], gen, k, op, pids, rq, res, val, per)
            d = f", Δ {r['score'] - best_parent:+.4f}" if (best_parent is not None and r["status"] == "ok") else ""
            log(f"  - [{r['id']}] island {k} {op} {c['name']} -> {_fmt(r)}{d}")
            made += 1
        if todo:
            log(f"  ({len(todo)} candidates evaluated in {time.time() - t0:.0f} s with {a.workers} worker(s))")
        if a.islands > 1 and a.migrate_every and gen % a.migrate_every == 0:
            for k in range(a.islands):
                el = db.elites(k, 1)                      # the island's best main score migrates
                if el:
                    b = el[0]; dst = (k + 1) % a.islands
                    if any(r["hash"] == b["hash"] for r in db.ok(dst)):
                        continue
                    m = {**b, "id": db.next_id(gen, dst), "island": dst, "operator": "migration", "parents": [b["id"]], "generation": gen}
                    db.add(m)
                    log(f"- migration: [{b['id']}] {b['name']} (score {b['score']:.4f}) island {k} -> island {dst} as [{m['id']}]")
        from . import report
        report.write(run_dir, db, a, val, U)
        flag.unlink(missing_ok=True)
        if made == 0:
            log(f"\nno candidates came back in generation {gen}: every prompt and response is under generations/g{gen:03d}/; stopping")
            break

    from . import report
    best = report.write(run_dir, db, a, val, U, final=True)
    if best and a.test and tests:
        score_tests(db, best, tests, U, a, keys, log, run_dir)
    if getattr(proposer, "usage", None):
        u = proposer.usage
        log(f"\nLLM usage this run: {u.get('input', 0)} input + {u.get('output', 0)} output tokens "
            f"({u.get('cache_read', 0)} cached), about ${u.get('usd', 0):.2f}")
    print(f"\nreport: {run_dir / 'REPORT.md'}")


TEST_CHUNK = 16


def _test_chunk(job):
    """One (model, tier, chunk) of the final test scoring: load these episodes, score, return (n, record)."""
    code, paths, U, kw = job
    eps = [_load_cached(p) for p in paths]
    t = cf.evaluate(code, eps, U, **kw)
    return len(eps), {k: v for k, v in t.items() if (k.startswith("val_") and k != "val_by_episode") or k in ("status", "objectives")}


def _merge_chunks(parts):
    """Episode-weighted means of the chunk records (every val_*_mean is a mean over episodes); objectives likewise."""
    ok = [(n, t) for n, t in parts if t.get("status") == "ok"]
    out = {"status": "ok" if len(ok) == len(parts) else next(t["status"] for _, t in parts if t.get("status") != "ok"),
           "n_episodes": sum(n for n, _ in parts)}
    if not ok:
        return out
    N = sum(n for n, _ in ok)
    for k in {k for _, t in ok for k in t if k.startswith("val_") and isinstance(t[k], (int, float))}:
        vals = [(n, t[k]) for n, t in ok if k in t and np.isfinite(t[k])]
        if vals:
            out[k] = float(sum(n * x for n, x in vals) / sum(n for n, _ in vals))
    obj = {}
    for k in ok[0][1].get("objectives", {}):
        vals = [(n, t["objectives"][k]) for n, t in ok if np.isfinite(t["objectives"].get(k, np.nan))]
        obj[k] = float(sum(n * x for n, x in vals) / sum(n for n, _ in vals)) if vals else float("inf")
    out["objectives"] = obj
    out["merged_chunks"] = len(ok); out["episodes_scored"] = N
    return out


def score_tests(db, best, tests, U, a, keys, log, run_dir):
    """Once, at the end: the chosen model and the rest of the validation Pareto front on every held-out test tier.
    Each tier is scored in chunks of TEST_CHUNK episodes, in --workers forked processes, so memory stays at a few
    chunks however large the tier is."""
    uniq = {}
    for r in db.ok():
        uniq.setdefault(r["hash"], r)
    front = pareto.front(list(uniq.values()), keys) if keys else []
    pick = [best] + [r for r in sorted(front, key=lambda r: r["score"]) if r["hash"] != best["hash"]][: max(a.test_front - 1, 0)]
    kw = dict(complexity_weight=a.complexity_weight, horizons=tuple(a.horizons), main_h=a.main_horizon, every=a.every,
              warmup=a.warmup, screen_factor=1e9, deploy_score=(a.score == "deploy"))
    jobs, where = [], []
    for r in pick:
        code = code_of(db, r)
        for tier, refs in tests.items():
            paths = [getattr(e, "path", None) for e in refs]
            if any(p is None for p in paths):             # already-loaded episodes (tests, self-test): one job per tier
                jobs.append(("loaded", code, refs)); where.append((r["id"], tier)); continue
            for i in range(0, len(paths), TEST_CHUNK):
                jobs.append((code, paths[i:i + TEST_CHUNK], U, kw)); where.append((r["id"], tier))
    loaded = [(i, j) for i, j in enumerate(jobs) if j[0] == "loaded"]
    file_jobs = [(i, j) for i, j in enumerate(jobs) if j[0] != "loaded"]
    results = [None] * len(jobs)
    for i, (_, code, eps) in loaded:
        t = cf.evaluate(code, eps, U, **kw)
        results[i] = (len(eps), {k: v for k, v in t.items() if (k.startswith("val_") and k != "val_by_episode") or k in ("status", "objectives")})
    if file_jobs:
        if a.workers > 1 and len(file_jobs) > 1:
            import multiprocessing as mp
            with mp.get_context("fork").Pool(min(a.workers, len(file_jobs))) as pool:
                for (i, _), res in zip(file_jobs, pool.map(_test_chunk, [j for _, j in file_jobs])):
                    results[i] = res
        else:
            for i, j in file_jobs:
                results[i] = _test_chunk(j)
    out = {}
    log("\n## Held-out test tiers (scored once; never seen by the search)\n")
    log("| model | " + " | ".join(f"{t} {_hname_main(a)} (persistence)" for t in tests) + " |")
    log("|---|" + "---|" * len(tests))
    mh = _hname_main(a)
    for r in pick:
        row = {tier: _merge_chunks([results[i] for i, w in enumerate(where) if w == (r["id"], tier)]) for tier in tests}
        out[r["id"]] = {"name": r["name"], "chosen": r is best, "tiers": row}
        log(f"| {'**' if r is best else ''}{r['name']} [{r['id']}]{'**' if r is best else ''} | " + " | ".join(
            f"{row[t].get(f'val_{mh}_deploy_mean', float('nan')):.3f} ({row[t].get(f'val_persistence_{mh}_mean', float('nan')):.3f})"
            for t in tests) + " |")
    s = json.loads((run_dir / "summary.json").read_text())
    s["test"] = out
    s["best"]["test"] = out[best["id"]]["tiers"]
    (run_dir / "summary.json").write_text(json.dumps(s, indent=1, default=float))


def _hname_main(a):
    return cf._hname(a.main_horizon)


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="cf")
    ap.add_argument("--dataset", type=Path, default=DATA / "dataset")
    ap.add_argument("--split", help="split JSON with val + test_easy/test_medium/test_hard (default: the dataset's split.json; "
                    "computed from the available episodes when it has no test tiers)")
    ap.add_argument("--auto-split", action="store_true", help="always compute the split from the available episodes (splits.py)")
    ap.add_argument("--policies", nargs="*")
    ap.add_argument("--max-usd", type=float, default=None, help="stop asking the API once the run has spent this much (USD)")
    ap.add_argument("--n-init", type=int, default=3, help="generation 0: candidates the LLM writes per island from the system prompt alone")
    ap.add_argument("--baselines", action="store_true", help="also score the hand-written models as reference rows (never parents)")
    ap.add_argument("--selection", choices=["pareto", "scalar"], default="pareto",
                    help="pareto (default): island populations and parents by non-dominated rank + crowding over --objectives; "
                         "scalar: by the single score")
    ap.add_argument("--objectives", nargs="+", default=list(cf.DEFAULT_OBJECTIVES), choices=list(cf.OBJECTIVES),
                    help="objectives for --selection pareto (closedform.OBJECTIVES: what each measures and which coefficients it isolates)")
    ap.add_argument("--test-front", type=int, default=5, help="at the end, score the chosen model and up to this many Pareto-front models on the test tiers")
    ap.add_argument("--generations", type=int, default=0)
    ap.add_argument("--islands", type=int, default=3)
    ap.add_argument("--migrate-every", type=int, default=3, help="copy each island's best into the next island every k generations")
    ap.add_argument("--mut-requests", type=int, default=1, help="mutation requests per island per generation (each a different parent)")
    ap.add_argument("--n-mut", type=int, default=2); ap.add_argument("--n-cx", type=int, default=2); ap.add_argument("--n-exp", type=int, default=1)
    ap.add_argument("--elites", type=int, default=6, help="population of an island = its best k programs")
    ap.add_argument("--horizons", type=int, nargs="+", default=[12, 36, 72], help="rollout horizons in 5-minute steps")
    ap.add_argument("--main-horizon", type=int, default=36, help="horizon that decides the score (MPC horizon not chosen yet)")
    ap.add_argument("--every", type=int, default=6, help="a rollout starts every k steps")
    ap.add_argument("--warmup", type=int, default=36, help="steps at the start of an episode not scored (history buffer filling)")
    ap.add_argument("--screen", type=float, default=5.0, help="stage-1 screen: reject if one-step RMSE > this x persistence")
    ap.add_argument("--complexity-weight", type=float, default=0.0001)
    ap.add_argument("--score", choices=["deploy", "oracle"], default="deploy",
                    help="deploy (default): rollouts hold the future outside temperature and occupancy at their last observed "
                         "values, as a policy must at test time; oracle: the true future inputs (dynamics only, a diagnostic)")
    ap.add_argument("--max-val", type=int, default=30, help="validation episodes used for scoring (0 = all; stratified sample: every building x station x controller group first)")
    ap.add_argument("--dry-run", action="store_true", help="write every kind of request (init, mutation, crossover, explore) without "
                    "calling the API, using the hand-written models as stand-in parents; then stop")
    ap.add_argument("--max-test", type=int, default=0, help="episodes per test tier at the end (0 = all; stratified sample)")
    ap.add_argument("--ceiling", action="store_true", help="also compute the per-building fitted ceiling (diagnostic only)")
    ap.add_argument("--test", action="store_true", help="at the end, score the chosen model (and the Pareto front) on the test tiers")
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--workers", type=int, default=1, help="evaluate candidates in this many forked processes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="claude-opus-5-5"); ap.add_argument("--effort", default="high")
    return ap


def main_args(argv=None):
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main_args()
