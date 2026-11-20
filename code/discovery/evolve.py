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
from . import prompts, proposers
from .episodes import _load_cached
from .starterkit_data import DATA

ROOT = Path(__file__).resolve().parents[2]
OUTD = ROOT / "outputs" / "discovery"
SEEDS_CF = Path(__file__).resolve().parent / "seeds_cf"


# ----------------------------------------------------------------------------- data

def load_split(args):
    from .loop import split_episodes
    ds = Path(args.dataset)
    if not (ds / "index.json").exists():
        raise SystemExit(f"no dataset at {ds} (index.json missing): run code/build_dataset.py first (docs/RUNBOOK.md), "
                         "or try the synthetic self-test: code/run_discovery.py --selftest")
    idx = json.loads((ds / "index.json").read_text())
    split = json.loads((Path(args.split) if args.split else ds / "split.json").read_text())
    paths = [ds / v["file"] for v in idx.values() if v.get("status") == "ok" and (not args.policies or v["policy"] in args.policies)]
    eps = [_load_cached(p) for p in paths]
    S = split_episodes(eps, split)
    rng = np.random.default_rng(0)

    def cap(L, k):
        return [L[i] for i in sorted(rng.choice(len(L), size=k, replace=False))] if k and len(L) > k else L
    return cap(S["train"], args.max_train), cap(S["val"], args.max_val), S.get("test", []), split, len(paths)


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

    def elites(self, island, k):
        best = {}
        for r in sorted(self.ok(island), key=lambda r: r["score"]):
            best.setdefault(r["hash"], r)               # a migrated copy and its original count once
        return list(best.values())[:k]

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


def evaluate(code, train, val, U, args):
    signal.signal(signal.SIGALRM, _alarm); signal.alarm(int(args.timeout))
    try:
        return cf.evaluate(code, train, val, U, complexity_weight=args.complexity_weight, horizons=tuple(args.horizons),
                           main_h=args.main_horizon, every=args.every, warmup=args.warmup, screen_factor=args.screen)
    except _Timeout:
        return {"status": "error", "reason": f"timeout after {args.timeout} s"}
    finally:
        signal.alarm(0)


METRICS = ("score", "score_metric", "complexity", "history", "stage", "reason", "failures", "traceback",
           "stage1_one_step_mean", "stage1_persistence_mean", "val_one_step_mean", "unstable_episodes", "horizons", "main_horizon")


def record(db, code, name, hyp, gen, island, op, parents, req_dir, res, val, wall):
    r = {"id": db.next_id(gen, island), "generation": gen, "island": island, "operator": op, "parents": parents,
         "name": name, "hypothesis": hyp, "hash": _hash(code), "status": res["status"],
         "request": str(req_dir.relative_to(db.dir)) if req_dir else None, "wall_sec": round(wall, 1)}
    r.update({k: res[k] for k in METRICS if k in res})
    r.update({k: v for k, v in res.items() if k.startswith(("val_", "train_")) and k != "val_by_episode"})
    an = cf.analysis(res, val)
    r["analysis"] = an["text"]; r["groups"] = an["groups"]
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
            path.write_text("# Discovery log\n\nOne line per LLM request and per candidate. Score = held-out rollout RMSE (K) "
                            "at the main horizon + complexity price; lower is better. Δ = change against the best parent.\n\n")

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
    return f"episode {ep.name}; room #{P['rooms'][0]} = most exposed exterior room, room #{P['rooms'][1]} = largest interior room\n" + "\n".join(rows)


# ----------------------------------------------------------------------------- the loop

def tournament(rng, pop, k=2):
    pick = [pop[i] for i in rng.choice(len(pop), size=min(k, len(pop)), replace=False)]
    return min(pick, key=lambda r: r["score"])


def make_proposer(a):
    if a.proposer == "mock":
        return proposers.MockProposer(a.seed)
    if a.proposer == "claude":
        return proposers.ClaudeProposer(model=a.model, effort=a.effort)
    return None                                                 # dry / dir handled per request


def run(a):
    run_dir = OUTD / a.run
    db = DB(run_dir); log = Log(run_dir / "log.md")
    U, Uraw = cf.universal_constants()
    if not Uraw.get("_reviewed"):
        print("note: code/discovery/universal_constants.json is not marked reviewed yet (set _reviewed: true after checking it)")
    train, val, test, split, n_runs = load_split(a)
    (run_dir / "config.json").write_text(json.dumps({**vars(a), "mode": "closedform", "split": split, "n_runs": n_runs,
                                                     "n_train_used": len(train), "n_val_used": len(val)}, indent=1, default=str))
    system = prompts.cf_system_prompt(Uraw, a.screen, tuple(a.horizons), a.main_horizon, a.complexity_weight)
    (run_dir / "system_prompt.md").write_text(system)
    rng = np.random.default_rng(a.seed)
    proposer = make_proposer(a)
    if a.ceiling and not (run_dir / "ceiling.json").exists():
        log(f"ceiling (diagnostic): per-room linear model fitted on the first half of each held-out episode")
        C = {e.name: cf.ceiling(e, U, tuple(a.horizons), a.every, a.warmup) for e in val}
        keys = next(iter(C.values())).keys()
        (run_dir / "ceiling.json").write_text(json.dumps({"episodes": C, "mean": {k: float(np.mean([c[k] for c in C.values() if k in c])) for k in keys}}, indent=1))

    if not db.recs:
        log(f"\n## Generation 0: seeds ({len(list(SEEDS_CF.glob('*.py')))} programs, copied into each of {a.islands} islands)\n")
        for s in sorted(SEEDS_CF.glob("*.py")):
            code = s.read_text(); t0 = time.time()
            res = evaluate(code, train, val, U, a)
            doc = proposers._from_file(s)["hypothesis"]
            for k in range(a.islands):
                r = record(db, code, s.stem, doc, 0, k, "seed", [], None, res, val, time.time() - t0)
                log(f"- [{r['id']}] seed {s.stem} -> {_fmt(r)}")

    start = db.generation() + 1
    for gen in range(start, start + a.generations):
        log(f"\n## Generation {gen}\n")
        made = 0
        for k in range(a.islands):
            pop = [dict(r, code=code_of(db, r)) for r in db.elites(k, a.elites)]
            fails = db.failures(k, 4)
            reqs = []
            if pop:
                for j in range(a.mut_requests):
                    reqs.append(("mutation", [tournament(rng, pop)], a.n_mut))
                if len(pop) >= 2:
                    pa = tournament(rng, pop); pb = tournament(rng, [p for p in pop if p["hash"] != pa["hash"]] or pop)
                    reqs.append(("crossover", [pa, pb], a.n_cx))
            reqs.append(("explore", pop, a.n_exp))
            seen_op = {}
            for op, parents, n in reqs:
                if n <= 0:
                    continue
                seen_op[op] = seen_op.get(op, 0) + 1
                rq = run_dir / "generations" / f"g{gen:03d}" / f"island_{k}" / (op if seen_op[op] == 1 else f"{op}_{seen_op[op]}")
                rq.mkdir(parents=True, exist_ok=True)
                if op == "mutation":
                    p = parents[0]
                    an = p["analysis"]
                    pt = profile_text(p["code"], val[0], U, a.main_horizon) if val else ""
                    (rq / "analysis.md").write_text(f"# Pre-mutation analysis of {p['id']} {p['name']}\n\n```\n{an}\n```\n\n## Measured vs predicted\n\n```\n{pt}\n```\n")
                    user = prompts.cf_user_mutation(gen, k, p, an, pt, fails, n)
                elif op == "crossover":
                    cx = cf.crossover_analysis({"analysis": {"groups": parents[0]["groups"]}, "code": parents[0]["code"]},
                                               {"analysis": {"groups": parents[1]["groups"]}, "code": parents[1]["code"]})
                    (rq / "analysis.md").write_text(f"# Pre-crossover analysis: A = {parents[0]['id']} {parents[0]['name']}, "
                                                    f"B = {parents[1]['id']} {parents[1]['name']}\n\n```\n{cx['text']}\n```\n")
                    user = prompts.cf_user_crossover(gen, k, parents[0], parents[1], cx["text"], fails, n)
                else:
                    user = prompts.cf_user_explore(gen, k, pop, fails, n)
                (rq / "prompt.md").write_text(f"# SYSTEM\n\nsee ../../../../system_prompt.md\n\n# USER\n\n{user}\n")
                ctx = {"operator": op, "parents": parents, "dir": rq}
                if a.proposer == "dry":
                    cands = []
                elif a.proposer == "dir":
                    cands = proposers.DirectoryProposer(rq / "inbox").propose(system, user, n, gen, ctx)
                else:
                    try:
                        cands = proposer.propose(system, user, n, gen, ctx)
                    except Exception as e:  # noqa: BLE001
                        log(f"- island {k} {op}: proposer failed: {type(e).__name__}: {e}"); cands = []
                pids = [p["id"] for p in parents] if op != "explore" else []
                log(f"- island {k} **{op}**" + (f" of {', '.join(pids)}" if pids else "") + f": asked for {n}, got {len(cands)} "
                    f"([prompt]({rq.relative_to(run_dir)}/prompt.md)" + (f", [analysis]({rq.relative_to(run_dir)}/analysis.md)" if op != "explore" else "") + ")")
                best_parent = min((p["score"] for p in parents), default=None) if op != "explore" else None
                for c in cands:
                    if any(r["hash"] == _hash(c["code"]) for r in db.recs):
                        log(f"  - skip duplicate {c['name']}"); continue
                    t0 = time.time()
                    res = evaluate(c["code"], train, val, U, a)
                    r = record(db, c["code"], c["name"], c["hypothesis"], gen, k, op, pids, rq, res, val, time.time() - t0)
                    d = f", Δ {r['score'] - best_parent:+.4f}" if (best_parent is not None and r["status"] == "ok") else ""
                    log(f"  - [{r['id']}] {c['name']} -> {_fmt(r)}{d}  [{r['wall_sec']} s]")
                    made += 1
        if a.islands > 1 and a.migrate_every and gen % a.migrate_every == 0:
            for k in range(a.islands):
                el = db.elites(k, 1)
                if el:
                    b = el[0]; dst = (k + 1) % a.islands
                    if any(r["hash"] == b["hash"] for r in db.ok(dst)):
                        continue
                    m = {**b, "id": db.next_id(gen, dst), "island": dst, "operator": "migration", "parents": [b["id"]], "generation": gen}
                    db.add(m)
                    log(f"- migration: [{b['id']}] {b['name']} (score {b['score']:.4f}) island {k} -> island {dst} as [{m['id']}]")
        from . import report
        report.write(run_dir, db, a, val, U)
        if made == 0:
            log(f"\nno candidates came back in generation {gen} ({a.proposer} proposer): every prompt is written under "
                f"generations/g{gen:03d}/; " + ("drop candidate .py files into each request's inbox/ and re-run" if a.proposer == "dir" else "stopping"))
            break

    from . import report
    best = report.write(run_dir, db, a, val, U, final=True)
    if best and a.test and test:
        t = cf.evaluate(code_of(db, best), train, test, U, complexity_weight=a.complexity_weight, horizons=tuple(a.horizons),
                        main_h=a.main_horizon, every=a.every, warmup=a.warmup, screen_factor=1e9)
        s = json.loads((run_dir / "summary.json").read_text())
        s["best"]["test"] = {k: v for k, v in t.items() if k.startswith("val_") and k != "val_by_episode"}
        (run_dir / "summary.json").write_text(json.dumps(s, indent=1, default=float))
        log(f"\nbest on the test split: 3 h {t.get('val_3h_mean', float('nan')):.3f} K ({t['status']})")
    print(f"\nreport: {run_dir / 'REPORT.md'}")


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="cf")
    ap.add_argument("--dataset", type=Path, default=DATA / "dataset")
    ap.add_argument("--split", help="split JSON (default: the dataset's split.json)")
    ap.add_argument("--policies", nargs="*")
    ap.add_argument("--proposer", default="dry", choices=["dry", "dir", "mock", "claude"])
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
    ap.add_argument("--max-train", type=int, default=8, help="train episodes used for the diagnostics (0 = all)")
    ap.add_argument("--max-val", type=int, default=30, help="held-out episodes used for scoring (0 = all; deterministic sample)")
    ap.add_argument("--ceiling", action="store_true", help="also compute the per-building fitted ceiling (diagnostic only)")
    ap.add_argument("--test", action="store_true", help="score the final best on the test split")
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="claude-opus-5-5"); ap.add_argument("--effort", default="high")
    return ap


def main_args(argv=None):
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main_args()
