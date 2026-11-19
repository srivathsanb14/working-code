"""LLM-driven symbolic discovery loop for the room dynamics.

    propose (LLM or other proposer)  ->  check + fit + score on held-out buildings  ->  diagnostics
          ^                                                                               |
          '--------------- best programs, their diagnostics, recent failures ------------'

Run from working-code/ (no LLM is called unless --proposer claude):

  # seeds only, end-to-end check of the pipeline
  .venv/bin/python code/run_discovery.py --run demo --generations 0

  # write the prompt the LLM would get for generation 1, without calling anything
  .venv/bin/python code/run_discovery.py --run demo --proposer dry --generations 1

  # let any LLM / person answer: drop candidate .py files into outputs/discovery/demo/inbox
  .venv/bin/python code/run_discovery.py --run demo --proposer dir --generations 1

  # the real thing (needs `pip install anthropic` and credentials)
  .venv/bin/python code/run_discovery.py --run demo --proposer claude --generations 20

State lives in outputs/discovery/<run>/: database.jsonl (one record per candidate),
candidates/<id>.py, prompts/prompt_genNNN.md, best.py, summary.json.  Re-running
with the same --run continues from the database.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import time
from pathlib import Path

from . import diagnostics, evaluate, prompts, proposers
from .episodes import _load_cached
from .starterkit_data import KIT, load_all

ROOT = Path(__file__).resolve().parents[2]
OUTD = ROOT / "outputs"
DEFAULT_SPLIT = {"train": ["building_train_8", "building_train_2", "building_train_4"],
                 "val": ["building_train_7"], "test": ["building_train_1"]}


# ----------------------------------------------------------------------------- data

def split_episodes(eps, split):
    """split: {"train": [...], "val": [...], "test": [...]} of glob patterns.

    A pattern with "/" is matched against "building/weather" (e.g. "building_07[01]*/phoenix*");
    one without is matched against the building name (e.g. "building_05*", "building_train_8").
    A pattern starting with "!" excludes what it matches (e.g. "!*/minneapolis*").
    Keys starting with "_" are comments."""
    from fnmatch import fnmatch

    def hit(e, pat):
        return fnmatch(e.scenario, pat) if "/" in pat else fnmatch(e.building, pat)

    def match(e, pats):
        if any(hit(e, p[1:]) for p in pats if p.startswith("!")):
            return False
        return any(hit(e, p) for p in pats if not p.startswith("!"))
    out = {k: [e for e in eps if match(e, v)] for k, v in split.items() if not k.startswith("_")}
    for k, v in out.items():
        if not v and k != "test":
            raise SystemExit(f"split '{k}' matched no episodes; adjust --split")
    return out


# ----------------------------------------------------------------------------- isolated evaluation

def _worker(src, paths, cache, split, kw, q):
    try:
        eps = [_load_cached(p) for p in paths]
        S = split_episodes(eps, split)
        rec = evaluate.evaluate(src, S["train"], S["val"], **kw)
        rec["diagnostics"] = diagnostics.summarize(rec, S["val"])
        rec.pop("val_detail", None)
        q.put(rec)
    except Exception as e:  # noqa: BLE001
        q.put({"status": "error", "reason": f"{type(e).__name__}: {e}"})


def evaluate_isolated(src, paths, cache, split, timeout, **kw):
    """Run one evaluation in a child process with a wall-clock limit."""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_worker, args=(src, paths, cache, split, kw, q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.kill(); p.join()
        return {"status": "error", "reason": f"timeout after {timeout} s", "diagnostics": f"status: error - timeout after {timeout} s"}
    try:
        return q.get(timeout=5)
    except Exception:  # noqa: BLE001
        return {"status": "error", "reason": f"child exited with code {p.exitcode}", "diagnostics": "status: error - child crashed"}


# ----------------------------------------------------------------------------- database

class Database:
    def __init__(self, run_dir: Path):
        self.dir = run_dir; self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "candidates").mkdir(exist_ok=True)
        self.path = self.dir / "database.jsonl"
        self.recs = [json.loads(l) for l in self.path.read_text().splitlines()] if self.path.exists() else []

    def seen(self, code):
        h = hashlib.sha1(code.strip().encode()).hexdigest()[:12]
        return h, any(r["hash"] == h for r in self.recs)

    def add(self, rec):
        self.recs.append(rec)
        with self.path.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    def elites(self, k):
        ok = [r for r in self.recs if r["status"] == "ok"]
        return sorted(ok, key=lambda r: r["score"])[:k]

    def failures(self, k):
        return [r for r in self.recs if r["status"] != "ok"][-k:]

    def generation(self):
        return max((r["generation"] for r in self.recs), default=-1)


# ----------------------------------------------------------------------------- loop

def make_proposer(kind, run_dir, model, effort):
    if kind == "dry":
        return proposers.DryRunProposer(run_dir / "prompts")
    if kind == "dir":
        return proposers.DirectoryProposer(run_dir / "inbox")
    if kind == "claude":
        return proposers.ClaudeProposer(model=model, effort=effort, log_dir=run_dir / "responses")
    raise SystemExit(f"unknown proposer {kind}")


def run(args):
    run_dir = OUTD / "discovery" / args.run
    db = Database(run_dir)
    split = json.loads(Path(args.split).read_text()) if args.split else DEFAULT_SPLIT
    cache = run_dir / "episodes"
    print(f"loading starter-kit runs from {args.results_dir} (scenarios: {args.scenarios_dir})")
    eps, report = load_all(args.scenarios_dir, args.results_dir, args.policies, cache)
    (run_dir / "data_report.json").write_text(json.dumps(report, indent=1))
    if not eps:
        raise SystemExit("no usable runs; generate them with the starter kit (see GUIDE.md, Part A)")
    paths = [cache / f"episode_{e.name}.npz" for e in eps]
    (run_dir / "config.json").write_text(json.dumps({**vars(args), "split": split, "n_runs": len(paths)}, indent=1, default=str))
    kw = dict(max_nfev=args.max_nfev, stride=args.stride, complexity_weight=args.complexity_weight)

    def consider(cands, gen):
        for c in cands:
            h, dup = db.seen(c["code"])
            if dup:
                print(f"  skip duplicate {c['name']}"); continue
            t0 = time.time()
            rec = evaluate_isolated(c["code"], paths, cache, split, args.timeout, **kw)
            rec.update(id=f"g{gen:03d}_{len(db.recs):04d}", generation=gen, name=c["name"], hypothesis=c["hypothesis"],
                       code=c["code"], hash=h, wall_sec=round(time.time() - t0, 1))
            (db.dir / "candidates" / f"{rec['id']}_{c['name']}.py").write_text(c["code"])
            db.add(rec)
            s = f"score {rec['score']:.4f} (held-out {rec['val_rmse_days23_mean']:.3f} K, train {rec['train_rmse_mean']:.3f} K)" if rec["status"] == "ok" else f"{rec['status']}: {rec.get('reason')}"
            print(f"  [{rec['id']}] {c['name']}: {s}  [{rec['wall_sec']} s]")

    if not db.recs and args.seeds != "none":
        print(f"generation 0: seeds ({args.seeds})")
        seeds = proposers.SeedProposer().propose("", "", 99, 0)
        if args.seeds == "naive":                      # blind runs: no structure from the code audit
            seeds = [c for c in seeds if c["name"] == "seed_rc_one_node"]
        consider(seeds, 0)

    proposer = make_proposer(args.proposer, run_dir, args.model, args.effort) if args.generations > 0 else None
    system = prompts.system_prompt(informed=args.informed, complexity_weight=args.complexity_weight)
    start = db.generation() + 1
    for gen in range(start, start + args.generations):
        user = prompts.user_prompt(gen, db.elites(args.elites), db.failures(args.failures), args.n, args.explore)
        (run_dir / "prompts").mkdir(exist_ok=True)
        (run_dir / "prompts" / f"prompt_gen{gen:03d}.md").write_text(f"# SYSTEM\n\n{system}\n\n# USER\n\n{user}\n")
        print(f"generation {gen}: asking {args.proposer} for {args.n} candidates")
        cands = proposer.propose(system, user, args.n, gen)
        if not cands:
            print("  no candidates returned (dry run or empty inbox); stopping"); break
        consider(cands, gen)

    best = db.elites(1)
    summary = {"n_candidates": len(db.recs), "n_ok": sum(r["status"] == "ok" for r in db.recs)}
    if best:
        b = best[0]
        (run_dir / "best.py").write_text(b["code"])
        summary["best"] = {k: b[k] for k in ("id", "name", "score", "val_rmse_days23_mean", "train_rmse_mean", "complexity",
                                             "params", "val_theta", "val_rmse_days23")}
        if args.test:
            S = split_episodes([_load_cached(p) for p in paths], split)
            if S.get("test"):
                from . import contract
                mod, _ = contract.load(b["code"])
                t = evaluate.score_heldout(mod, S["test"], b["params"], args.max_nfev)
                summary["best"]["test_rmse_days23"] = {k: v["rmse_days23"] for k, v in t["episodes"].items()}
                summary["best"]["test_rmse_days23_mean"] = t["rmse_days23_mean"]
    summary["leaderboard"] = [{k: r.get(k) for k in ("id", "name", "score", "val_rmse_days23_mean", "complexity")} for r in db.elites(10)]
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary.get("best", {}), indent=1, default=str)[:2000])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="default")
    ap.add_argument("--scenarios-dir", type=Path, default=KIT / "smart_buildings/training_scenarios",
                    help="scenario folders <building>/<weather>/config.gin (run_scenario_generator --output-dir)")
    ap.add_argument("--results-dir", type=Path, default=ROOT / "data/results",
                    help="evaluate.py --results-dir: <building>/<weather>/<policy>/timeseries.csv, ahu_log.csv, reward_components.csv")
    ap.add_argument("--policies", nargs="*", default=None, help="only these policy folders (default: all found)")
    ap.add_argument("--split", help="JSON file {train: [...], val: [...], test: [...]} of glob patterns (see split_episodes, split_example.json)")
    ap.add_argument("--proposer", default="dry", choices=["dry", "dir", "claude"])
    ap.add_argument("--generations", type=int, default=0)
    ap.add_argument("--n", type=int, default=6, help="candidates per generation")
    ap.add_argument("--explore", type=int, default=2)
    ap.add_argument("--elites", type=int, default=4)
    ap.add_argument("--failures", type=int, default=6)
    ap.add_argument("--informed", action="store_true", help="add physics notes from the code audit to the prompt")
    ap.add_argument("--seeds", default="all", choices=["all", "naive", "none"],
                    help="generation-0 seeds: all (incl. the audited physics network), naive (blind runs), none")
    ap.add_argument("--timeout", type=float, default=1200)
    ap.add_argument("--max-nfev", type=int, default=30)
    ap.add_argument("--stride", type=int, default=2, help="score every k-th step when fitting (speed)")
    ap.add_argument("--complexity-weight", type=float, default=0.0001, help="K added to the score per AST node")
    ap.add_argument("--test", action="store_true", help="also score the best program on the test split at the end")
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="high")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
