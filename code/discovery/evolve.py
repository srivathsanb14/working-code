"""Closed-form discovery with islands, mutation and crossover (the default discovery mode; no fitting anywhere).

Each generation, for every island:
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
  generations/gNNN/island_K/<operator>/prompt.md, analysis.md, response.json (Claude only), inbox/ (dir proposer)
  plots/<id>_profile.png       measured vs predicted temperatures (one-step and 3 h ahead), two rooms
  plots/<id>_horizon.png       rollout error vs horizon, against persistence (and the fitted ceiling if computed)
  plots/progress.png           best score per island per generation

    .venv/bin/python code/run_discovery.py --run cf --proposer dry --generations 1     # writes every prompt, calls nothing
    .venv/bin/python code/run_discovery.py --run cf --proposer mock --generations 3    # no LLM: mechanical operators
    .venv/bin/python code/run_discovery.py --run cf --proposer claude --generations 20 --islands 3
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
    train episodes are ignored."""
    from .loop import split_episodes
    ds = Path(args.dataset)
    if not (ds / "index.json").exists():
        raise SystemExit(f"no dataset at {ds} (index.json missing): run code/build_dataset.py first (docs/RUNBOOK.md), "
                         "or try the synthetic self-test: code/run_discovery.py --selftest")
    idx = json.loads((ds / "index.json").read_text())
    sp_path = Path(args.split) if args.split else ds / "split.json"
    split = json.loads(sp_path.read_text()) if sp_path.exists() else {}
    paths = [ds / v["file"] for v in idx.values() if v.get("status") == "ok" and (not args.policies or v["policy"] in args.policies)]
    eps = [_load_cached(p) for p in paths]
    if args.auto_split or not any(k in split for k in TEST_TIERS):
        from . import splits
        split = splits.split_from_episodes(eps)
        w = split["_why"]
        print(f"split computed from the {len(eps)} available episodes: hard station {w['hard_station']} "
              f"(Wasserstein to the rest {w['station_wasserstein_to_rest_K']}), held-out buildings {w['held_out_buildings']}, "
              f"counts {w['counts']}")
    S = split_episodes(eps, {k: v for k, v in split.items() if k != "train" and k != "_why"})
    if split.get("train"):
        print(f"note: the split has a 'train' list ({len(split['train'])} patterns); it is ignored, discovery uses no training data")
    rng = np.random.default_rng(0)

    def cap(L, k):
        return [L[i] for i in sorted(rng.choice(len(L), size=k, replace=False))] if k and len(L) > k else L
    tests = {k: S[k] for k in TEST_TIERS if S.get(k)}
    if not tests and S.get("test"):
        tests = {"test": S["test"]}
    return cap(S["val"], args.max_val), tests, split, len(paths)


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
    """Evaluate candidates serially, or in --workers forked processes (backlog F3); same results either way."""
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


def profile_text(code, ep, U, horizon):
    """Small measured-vs-predicted table for the mutation prompt (every 2 h over the second day)."""
    try:
        P = cf.profile(code, ep, U, horizon)
    except Exception as e:  # noqa: BLE001
        return f"(profile failed: {e})"
    X, hz, ht = P["X"], P["hz"], P["hz_t"]
    rows = ["time  | " + " | ".join(f"room #{i} measured / predicted" for i in P["rooms"]) + " | T_out | HVAC air"]
    for t in range(288, min(576, len(X) - 1), 24):
        j = np.searchsorted(ht, t)
        if j >= len(ht) or ht[j] != t:
            continue
        rows.append(f"{(t % 288) / 12:4.0f}h | " + " | ".join(f"{X[t, i] - 273.15:6.2f} / {hz[j, i] - 273.15:6.2f} C" for i in P["rooms"])
                    + f" | {P['T_out'][min(t, len(P['T_out']) - 1)] - 273.15:5.1f} C | {'on' if P['hvac'][min(t, len(P['hvac']) - 1)] > 50 else 'off'}")
    return f"a validation episode; room #{P['rooms'][0]} = most exposed exterior room, room #{P['rooms'][1]} = largest interior room\n" + "\n".join(rows)


# ----------------------------------------------------------------------------- the loop

def tournament(rng, pop, k=2, keys=None):
    pick = [pop[i] for i in rng.choice(len(pop), size=min(k, len(pop)), replace=False)]
    if keys:
        R = pareto.rank(pop, keys)
        return min(pick, key=lambda r: R[r["id"]])
    return min(pick, key=lambda r: r["score"])


def make_proposer(a):
    if a.proposer == "mock":
        return proposers.MockProposer(a.seed)
    if a.proposer in ("api", "claude"):
        return proposers.ClaudeProposer(model=a.model, effort=a.effort, max_usd=a.max_usd)
    if a.proposer == "agent":
        return proposers.AgentCLIProposer(a.agent_cmd)
    return None                                                 # dry / dir handled per request


def run(a):
    run_dir = OUTD / a.run
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
    proposer = make_proposer(a)
    if a.ceiling and not (run_dir / "ceiling.json").exists():
        log("ceiling (diagnostic): per-room linear model fitted on the first half of each held-out episode")
        C = {e.name: cf.ceiling(e, U, tuple(a.horizons), a.every, a.warmup) for e in val}
        keys = next(iter(C.values())).keys()
        (run_dir / "ceiling.json").write_text(json.dumps({"episodes": C, "mean": {k: float(np.mean([c[k] for c in C.values() if k in c])) for k in keys}}, indent=1))

    if not db.recs:
        log(f"\n## Generation 0: seeds ({len(list(SEEDS_CF.glob('*.py')))} programs, copied into each of {a.islands} islands)\n")
        for s in sorted(SEEDS_CF.glob("*.py")):
            code = s.read_text(); t0 = time.time()
            res = evaluate(code, val, U, a)
            doc = proposers._from_file(s)["hypothesis"]
            for k in range(a.islands):
                r = record(db, code, s.stem, doc, 0, k, "seed", [], None, res, val, time.time() - t0)
                log(f"- [{r['id']}] seed {s.stem} -> {_fmt(r)}")

    start = db.generation() + 1
    flag = run_dir / "in_progress.json"
    if flag.exists():                                     # backlog G6: an interrupted generation is finished first
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
            reqs = []
            if pop:
                for j in range(a.mut_requests):
                    reqs.append(("mutation", [tournament(rng, pop, keys=keys)], a.n_mut))
                if len(pop) >= 2:
                    pa = tournament(rng, pop, keys=keys)
                    pb = tournament(rng, [p for p in pop if p["hash"] != pa["hash"]] or pop, keys=keys)
                    reqs.append(("crossover", [pa, pb], a.n_cx))
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
                    pt = profile_text(p["code"], val[0], U, a.main_horizon) if val else ""
                    (rq / "analysis.md").write_text(f"# Pre-mutation analysis of {p['id']} {p['name']}\n\n```\n{an}\n```\n\n## Measured vs predicted\n\n```\n{pt}\n```\n")
                    user = prompts.cf_user_mutation(gen, k, p, an, pt, fails, n, keys)
                elif op == "crossover":
                    cx = cf.crossover_analysis(*({"analysis": {"groups": p.get("groups", {}), "terms": p.get("terms", {}),
                                                               "rooms": p.get("rooms", {})}, "code": p["code"]} for p in parents),
                                               ep=val[0] if val else None, U=U)
                    (rq / "analysis.md").write_text(f"# Pre-crossover analysis: A = {parents[0]['id']} {parents[0]['name']}, "
                                                    f"B = {parents[1]['id']} {parents[1]['name']}\n\n```\n{cx['text']}\n```\n")
                    user = prompts.cf_user_crossover(gen, k, parents[0], parents[1], cx["text"], fails, n, keys)
                else:
                    user = prompts.cf_user_explore(gen, k, pop, fails, n, keys)
                (rq / "prompt.md").write_text(f"# SYSTEM\n\nsee ../../../../system_prompt.md\n\n# USER\n\n{user}\n")
                ctx = {"operator": op, "parents": parents, "dir": rq}
                usage = None
                if a.proposer == "dry":
                    cands = []
                elif a.proposer == "dir":
                    cands = proposers.DirectoryProposer(rq / "inbox").propose(system, user, n, gen, ctx)
                else:
                    try:
                        cands = proposer.propose(system, user, n, gen, ctx)
                        usage = getattr(proposer, "last_usage", None)
                    except Exception as e:  # noqa: BLE001
                        log(f"- island {k} {op}: proposer failed: {type(e).__name__}: {e}"); cands = []
                pids = [p["id"] for p in parents] if op != "explore" else []
                best_parent = min((p["score"] for p in parents), default=None) if op != "explore" else None
                if cands:                                  # saved BEFORE evaluation, so a crash never costs a second LLM call
                    saved.write_text(json.dumps({"candidates": cands, "parents": pids, "best_parent": best_parent, "usage": usage}, indent=1))
                cost = f"; {usage['input']}+{usage['output']} tokens, ${usage['usd']:.3f}" if usage else ""
                log(f"- island {k} **{op}**" + (f" of {', '.join(pids)}" if pids else "") + f": asked for {n}, got {len(cands)}{cost} "
                    f"([prompt]({rq.relative_to(run_dir)}/prompt.md)" + (f", [analysis]({rq.relative_to(run_dir)}/analysis.md)" if op != "explore" else "") + ")")
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
            log(f"\nno candidates came back in generation {gen} ({a.proposer} proposer): every prompt is written under "
                f"generations/g{gen:03d}/; " + ("drop candidate .py files into each request's inbox/ and re-run" if a.proposer == "dir" else "stopping"))
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


def score_tests(db, best, tests, U, a, keys, log, run_dir):
    """Once, at the end: the chosen model and the rest of the validation Pareto front on every held-out test tier."""
    uniq = {}
    for r in db.ok():
        uniq.setdefault(r["hash"], r)
    front = pareto.front(list(uniq.values()), keys) if keys else []
    pick = [best] + [r for r in sorted(front, key=lambda r: r["score"]) if r["hash"] != best["hash"]][: max(a.test_front - 1, 0)]
    out = {}
    log("\n## Held-out test tiers (scored once; never seen by the search)\n")
    log("| model | " + " | ".join(f"{t} {_hname_main(a)} (persistence)" for t in tests) + " |")
    log("|---|" + "---|" * len(tests))
    for r in pick:
        row = {}
        for tier, eps in tests.items():
            t = cf.evaluate(code_of(db, r), eps, U, complexity_weight=a.complexity_weight, horizons=tuple(a.horizons),
                            main_h=a.main_horizon, every=a.every, warmup=a.warmup, screen_factor=1e9,
                            deploy_score=(a.score == "deploy"))
            row[tier] = {k: v for k, v in t.items() if k.startswith("val_") and k != "val_by_episode"}
            row[tier].update(status=t["status"], objectives=t.get("objectives"), n_episodes=len(eps))
        out[r["id"]] = {"name": r["name"], "chosen": r is best, "tiers": row}
        mh = _hname_main(a)
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
    ap.add_argument("--proposer", default="dry", choices=["dry", "api", "claude", "agent", "dir", "mock"],
                    help="api (= claude): one stateless Claude API request per operator (system prompt + this request only); "
                         "agent: a coding-agent CLI run headless, a fresh session per request (--agent-cmd); "
                         "dir: prompts written to inbox folders, candidates dropped in by hand or by any tool; "
                         "dry: write every prompt, call nothing; mock: mechanical operators, no LLM")
    ap.add_argument("--agent-cmd", default="claude -p",
                    help="command for --proposer agent; the prompt goes to its stdin, the JSON answer is read from stdout")
    ap.add_argument("--max-usd", type=float, default=None, help="api proposer: stop asking once the run has spent this much")
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
    ap.add_argument("--max-val", type=int, default=30, help="validation episodes used for scoring (0 = all; deterministic sample)")
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
