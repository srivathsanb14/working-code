"""REPORT.md for a closed-form discovery run: what was proposed, by which operator, on which island, from which
parents, how it scored, and how the best models' predicted temperatures compare with the measured ones."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import closedform as cf


def _h(r, k):
    v = r.get(k)
    return f"{v:.3f}" if isinstance(v, (int, float)) and np.isfinite(v) else "-"


def _plots(run_dir, rec, code, val, U, a, ceiling):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pd = run_dir / "plots"; pd.mkdir(exist_ok=True)
    out = {}
    if not val:
        return out
    ep = val[0]
    try:
        P = cf.profile(code, ep, U, a.main_horizon)
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}
    X, n = P["X"], len(P["X"]) - 1
    tt = np.arange(n + 1) / 288
    fig, axs = plt.subplots(len(P["rooms"]), 1, figsize=(11, 3.3 * len(P["rooms"])), sharex=True)
    labels = ["most exposed exterior room", "largest interior room"]
    for ax, i, lab in zip(np.atleast_1d(axs), P["rooms"], labels):
        on = P["hvac"] > 50
        for s, e in _runs(on):
            ax.axvspan(s / 288, e / 288, color="#f2c14e", alpha=0.15, lw=0)
        ax.plot(tt, X[:, i] - 273.15, color="black", lw=1.6, label="measured")
        ax.plot(tt[1:], P["one"][:, i] - 273.15, color="#4a90d9", lw=0.9, ls=":", label="predicted 5 min ahead")
        ax.plot(P["hz_t"] / 288, P["hz"][:, i] - 273.15, color="#d9534f", lw=1.1,
                label=f"predicted {a.main_horizon / 12:g} h ahead (free run from measured history)")
        ax.set_ylabel("°C"); ax.set_title(f"room #{i} ({lab}) — {ep.name}", fontsize=10, loc="left")
        ax2 = ax.twinx(); ax2.plot(tt[:-1], P["T_out"] - 273.15, color="#999999", lw=0.7, alpha=0.6); ax2.set_ylabel("outside °C", color="#999999")
    np.atleast_1d(axs)[0].legend(loc="upper left", fontsize=8, ncol=3)
    np.atleast_1d(axs)[-1].set_xlabel("days (shaded: HVAC supply air on)")
    fig.suptitle(f"{rec['id']} {rec['name']}: measured vs predicted", fontsize=11)
    fig.tight_layout(); f1 = pd / f"{rec['id']}_profile.png"; fig.savefig(f1, dpi=110); plt.close(fig)
    out["profile"] = f1.name
    hs = rec.get("horizons", a.horizons)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    hh = [h / 12 for h in hs]
    ax.plot(hh, [rec.get(f"val_{cf._hname(h)}_mean", np.nan) for h in hs], "o-", color="#d9534f", label=rec["name"])
    ax.plot(hh, [rec.get(f"val_persistence_{cf._hname(h)}_mean", np.nan) for h in hs], "s--", color="#777777", label="persistence (T stays)")
    if ceiling:
        ax.plot(hh, [ceiling["mean"].get(cf._hname(h), np.nan) for h in hs], "^:", color="#2e8b57", label="fitted ceiling (diagnostic, 2nd half)")
    ax.set_xlabel("horizon, h"); ax.set_ylabel("held-out RMSE, K"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title(f"{rec['id']}: error vs horizon", fontsize=10)
    fig.tight_layout(); f2 = pd / f"{rec['id']}_horizon.png"; fig.savefig(f2, dpi=110); plt.close(fig)
    out["horizon"] = f2.name
    return out


def _runs(mask):
    out, s = [], None
    for i, m in enumerate(mask):
        if m and s is None:
            s = i
        if not m and s is not None:
            out.append((s, i)); s = None
    if s is not None:
        out.append((s, len(mask)))
    return out


def _progress(run_dir, recs, islands):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    G = max(r["generation"] for r in recs)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for k in range(islands):
        best, ys = np.inf, []
        for g in range(G + 1):
            s = [r["score"] for r in recs if r["island"] == k and r["generation"] == g and r["status"] == "ok"]
            best = min([best] + s); ys.append(best)
        ax.step(range(G + 1), ys, where="post", label=f"island {k}")
    ax.set_xticks(range(G + 1)); ax.set_xlabel("generation"); ax.set_ylabel("best score so far"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(run_dir / "plots" / "progress.png", dpi=110); plt.close(fig)


def _lineage(db, rid, depth=0, seen=None, maxd=8):
    seen = seen if seen is not None else set()
    r = db.get(rid)
    pre = "  " * depth + "- "
    s = [f"{pre}[{r['id']}] {r['operator']} `{r['name']}` (island {r['island']}): " + (f"score {r['score']:.4f}" if r["status"] == "ok" else r["status"])]
    if rid in seen or depth >= maxd:
        return s
    seen.add(rid)
    for p in r.get("parents", []):
        s += _lineage(db, p, depth + 1, seen, maxd)
    return s


def _mermaid(db, ids):
    E, N = set(), {}

    def walk(i, d=0):
        if i in N or d > 10:
            return
        r = db.get(i)
        N[i] = f'{i}["{r["operator"]}<br/>{r["name"][:24]}<br/>' + (f'{r["score"]:.3f}' if r["status"] == "ok" else r["status"]) + '"]'
        for p in r.get("parents", []):
            E.add((p, i)); walk(p, d + 1)
    for i in ids:
        walk(i)
    return "```mermaid\ngraph LR\n" + "\n".join(f"  {v}" for v in N.values()) + "\n" + "\n".join(f"  {p} --> {c}" for p, c in sorted(E)) + "\n```"


def write(run_dir: Path, db, a, val, U, final=False):
    recs = db.recs
    if not recs:
        return None
    (run_dir / "plots").mkdir(exist_ok=True)
    cfg = json.loads((run_dir / "config.json").read_text()) if (run_dir / "config.json").exists() else {}
    ceiling = json.loads((run_dir / "ceiling.json").read_text()) if (run_dir / "ceiling.json").exists() else None
    ok = sorted([r for r in recs if r["status"] == "ok" and r["operator"] != "migration"], key=lambda r: r["score"])
    best = ok[0] if ok else None
    synthetic = "SYNTHETIC" in json.dumps(cfg.get("split", {}))
    L = [f"# Discovery run `{run_dir.name}`" + (" — SYNTHETIC self-test (planted truth, not simulator data)" if synthetic else ""), ""]
    L += [f"Mode: closed-form, **no fitting anywhere**. Proposer: `{a.proposer}`"
          + (" (mechanical operators, **no LLM**)" if a.proposer == "mock" else "") +
          f". Islands: {a.islands}, migration every {a.migrate_every} generation(s). Generations so far: {db.generation()}. "
          f"Candidates: {len(recs)} records ({sum(r['operator'] != 'migration' for r in recs)} evaluated programs incl. seed copies). "
          f"Held-out episodes used for the score: {cfg.get('n_val_used', '?')}. Score = held-out "
          f"{a.main_horizon / 12:g} h rollout RMSE from measured history + {a.complexity_weight} K per AST node.", ""]
    L += ["## Where to look", "",
          "| what | where |", "|---|---|",
          "| the system prompt every request shares | [system_prompt.md](system_prompt.md) |",
          "| every LLM request: the prompt it got | `generations/gNNN/island_K/<operator>/prompt.md` |",
          "| the pre-mutation / pre-crossover analysis it was given | `generations/gNNN/island_K/<operator>/analysis.md` |",
          "| the raw LLM response (Claude proposer) | `generations/gNNN/island_K/<operator>/response.json` |",
          "| candidates for the `dir` proposer (any LLM or a person) | `generations/gNNN/island_K/<operator>/inbox/*.py` |",
          "| every candidate's source | `candidates/<id>_<name>.py` |",
          "| every candidate's scores, parents, operator, analysis | `database.jsonl` (one JSON per line) |",
          "| the timeline (requests, children, Δ vs parent, migrations) | [log.md](log.md) |",
          "| measured vs predicted temperatures, error vs horizon | `plots/<id>_profile.png`, `plots/<id>_horizon.png` |", "",
          "Operators: **seed** (hand-written start), **mutation** (one parent changed where its analysis points), "
          "**crossover** (two parents' templates combined), **explore** (structurally new), **migration** (an island's best "
          "copied into the next island; not re-evaluated).", ""]
    if best:
        L += ["## Best so far", "", f"**[{best['id']}] `{best['name']}`** — island {best['island']}, {best['operator']}, score {best['score']:.4f}", "",
              f"Hypothesis: {best.get('hypothesis', '')}", "", f"Source: [{best['file']}]({best['file']})", "",
              "```", best.get("analysis", ""), "```", ""]
    # leaderboard
    L += ["## Leaderboard (held-out, K)", "", "| id | island | operator | parents | name | score | 5 min | 1 h | 3 h | 6 h | persistence 3 h |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in ok[:15]:
        L.append(f"| {r['id']} | {r['island']} | {r['operator']} | {', '.join(r.get('parents', [])) or '-'} | `{r['name']}` | {r['score']:.4f} | "
                 f"{_h(r, 'val_one_step_mean')} | {_h(r, 'val_1h_mean')} | {_h(r, 'val_3h_mean')} | {_h(r, 'val_6h_mean')} | {_h(r, 'val_persistence_3h_mean')} |")
    if ceiling:
        m = ceiling["mean"]
        L.append(f"| ceiling | - | fitted per room (diagnostic) | - | second half of each episode | - | - | {m.get('1h', float('nan')):.3f} | "
                 f"{m.get('3h', float('nan')):.3f} | {m.get('6h', float('nan')):.3f} | {m.get('persistence_3h', float('nan')):.3f} |")
    L.append("")
    # operators
    st = defaultdict(lambda: defaultdict(int)); gains = defaultdict(list)
    for r in recs:
        if r["operator"] in ("migration",):
            continue
        st[r["operator"]]["proposed"] += 1; st[r["operator"]][r["status"]] += 1
        if r["status"] == "ok" and r.get("parents") and r["operator"] in ("mutation", "crossover"):
            ps = [db.get(p) for p in r["parents"]]
            bp = min(p["score"] for p in ps if p["status"] == "ok")
            gains[r["operator"]].append(r["score"] - bp)
    L += ["## Operators", "", "| operator | proposed | ok | screened (stage 1) | rejected (stage 0) | unstable (stage 2) | error | better than best parent | median Δ |",
          "|---|---|---|---|---|---|---|---|---|"]
    for op in ("seed", "mutation", "crossover", "explore"):
        if op in st:
            s = st[op]; g = gains.get(op, [])
            L.append(f"| {op} | {s['proposed']} | {s['ok']} | {s['screened']} | {s['rejected']} | {s['unstable']} | {s['error']} | "
                     + (f"{sum(x < 0 for x in g)} of {len(g)} | {np.median(g):+.4f} |" if g else "- | - |"))
    L.append("")
    # islands
    L += ["## Islands", "", "![progress](plots/progress.png)", ""]
    _progress(run_dir, recs, a.islands)
    tops = []
    for k in range(a.islands):
        el = db.elites(k, 1)
        if el:
            tops.append(el[0]["id"])
            L.append(f"- island {k}: best [{el[0]['id']}] `{el[0]['name']}` score {el[0]['score']:.4f} ({el[0]['operator']})")
    L.append("")
    # lineage
    if best:
        L += ["## Lineage of the best", "", *_lineage(db, best["id"]), "", "Family tree of each island's best (mermaid):", "", _mermaid(db, list(dict.fromkeys(tops + [best["id"]]))), ""]
    # plots
    L += ["## Measured vs predicted", ""]
    shown = list(dict.fromkeys(([best["id"]] if best else []) + [i for i in tops]))
    for i in shown[: (4 if final else 1)]:
        r = db.get(i)
        if r["operator"] == "migration":
            r = db.get(r["parents"][0])
        code = (run_dir / r["file"]).read_text()
        P = _plots(run_dir, r, code, val, U, a, ceiling)
        if "profile" in P:
            L += [f"**[{r['id']}] `{r['name']}`**", "", f"![profile](plots/{P['profile']})", "", f"![horizon](plots/{P['horizon']})", ""]
    # per generation
    L += ["## Generation by generation", "", "Each line: island, operator, parents → child, outcome. Open the request folder for its prompt and analysis.", ""]
    for g in range(db.generation() + 1):
        L.append(f"### Generation {g}")
        for r in [x for x in recs if x["generation"] == g]:
            par = f" of {', '.join(r['parents'])}" if r.get("parents") else ""
            req = f" — [request]({r['request']}/prompt.md)" if r.get("request") else ""
            out = f"score {r['score']:.4f}" if r["status"] == "ok" else f"{r['status']}: {r.get('reason', '')}"
            L.append(f"- island {r['island']} {r['operator']}{par} → [{r['id']}] `{r['name']}`: {out}{req}")
        L.append("")
    (run_dir / "REPORT.md").write_text("\n".join(L) + "\n")
    summ = {"mode": "closedform", "n_records": len(recs), "n_ok": len(ok),
            "best": ({k: best.get(k) for k in ("id", "name", "island", "operator", "parents", "score", "score_metric", "file",
                                               "val_one_step_mean", "val_1h_mean", "val_3h_mean", "val_6h_mean", "val_persistence_3h_mean",
                                               "complexity", "history")} if best else None),
            "leaderboard": [{k: r.get(k) for k in ("id", "name", "island", "operator", "score")} for r in ok[:10]]}
    (run_dir / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
    if best:
        (run_dir / "best.py").write_text((run_dir / best["file"]).read_text())
    return best
