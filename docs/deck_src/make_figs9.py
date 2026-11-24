"""Evidence for the coefficient-form starting models on the 12 thirty-day runs (6 shipped buildings x pid_water / HVAC off):
seeds_compare.png, ep30_by_episode.png, ep30_profile_pred.png, and the pre-mutation analysis + measured-vs-predicted text
of the history model.  Nothing is fitted except the diagnostic ceiling.  Usage: make_figs9.py <scratch> <out img dir>"""
import sys, json, pickle
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
S = Path(sys.argv[1]); OUT = Path(sys.argv[2])
WC = Path("/Users/srivathsan/Desktop/smart-buildings/working-code"); sys.path.insert(0, str(WC / "code"))
from discovery import closedform as cf, evolve
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
ORA, BLU, GRY, INK, GRN, PUR = "#D9692A", "#2F72B5", "#8A93A3", "#13213A", "#3a9a5b", "#8e44ad"
eps = pickle.load(open(S / "ep30_eps.pkl", "rb")); U, _ = cf.universal_constants(); H = (12, 36, 72)
SEEDS = {"network": "seed_cf_network", "history": "seed_cf_history", "one_node": "seed_cf_one_node"}
src = {k: (WC / f"code/discovery/seeds_cf/{f}.py").read_text() for k, f in SEEDS.items()}
rec = {k: cf.evaluate(s, eps[:1], eps, U, every=12, screen_factor=1e9, max_train=1) for k, s in src.items()}
ceil = {e.name: cf.ceiling(e, U, H, 12) for e in eps}
C3 = {h: float(np.mean([c[f"{h / 12:g}h"] for c in ceil.values()])) for h in H}
res = {k: {kk: v for kk, v in r.items() if kk.startswith("val_") and isinstance(v, float)} | {"status": r["status"]} for k, r in rec.items()}
res["ceiling"] = C3

# seeds compared
fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw=dict(width_ratios=[1.4, 1]))
hh = [1, 3, 6]; p0 = rec["network"]
axs[0].plot(hh, [p0[f"val_persistence_{h}h_mean"] for h in hh], "s--", color=GRY, label="persistence ('temperature stays')")
for k, lab, c in (("network", "network model", ORA), ("history", "history model", BLU)):
    axs[0].plot(hh, [rec[k][f"val_{h}h_mean"] for h in hh], "o-", color=c, label=lab)
axs[0].plot(hh, [C3[h * 12] for h in hh], "^:", color=GRN, label="ceiling: fitted per room (diagnostic)")
axs[0].set_ylim(0, 2.2); axs[0].set_xticks(hh); axs[0].set_xlabel("forecast horizon, h"); axs[0].set_ylabel("RMSE, K (true future inputs)")
axs[0].legend(frameon=False, fontsize=9)
o = rec["one_node"]
axs[0].set_title(f"12 thirty-day runs (one-node model off the chart: {o['val_1h_mean']:.1f} / {o['val_3h_mean']:.1f} / {o['val_6h_mean']:.1f} K)", fontsize=10, loc="left")
names = ["network", "history"]; x = np.arange(2)
axs[1].bar(x - 0.2, [rec[k]["val_3h_mean"] for k in names], 0.4, color=BLU, label="true future inputs (diagnostic)")
axs[1].bar(x + 0.2, [rec[k]["val_3h_deploy_mean"] for k in names], 0.4, color=ORA, label="future weather and occupancy unknown (the score)")
axs[1].axhline(p0["val_persistence_3h_mean"], color=GRY, ls="--", lw=1)
axs[1].set_xticks(x); axs[1].set_xticklabels(["network model", "history model"]); axs[1].set_ylabel("3 h RMSE, K")
axs[1].legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(OUT / "seeds_compare.png", dpi=130); plt.close()

# per episode, history model
r = rec["history"]; names = [e.name for e in eps]
fig, ax = plt.subplots(figsize=(13, 4.2)); x = np.arange(len(names)); w = .27
ax.bar(x - w, [r["val_by_episode"][n]["persistence_36"] for n in names], w, color=GRY, label="persistence")
ax.bar(x, [r["val_by_episode"][n]["36"] for n in names], w, color=BLU, label="history model")
ax.bar(x + w, [ceil[n]["3h"] for n in names], w, color=GRN, label="ceiling (second half of each run)")
ax.set_xticks(x); ax.set_xticklabels([n.replace("building_train_", "b").replace("_pid_water", " pid").replace("_all_off", " off") for n in names], rotation=30, ha="right", fontsize=9)
ax.set_ylabel("3 h forecast RMSE, K"); ax.legend(frameon=False, ncol=3)
fig.tight_layout(); fig.savefig(OUT / "ep30_by_episode.png", dpi=130); plt.close()
res["history_by_episode"] = {n: (r["val_by_episode"][n]["36"], r["val_by_episode"][n]["persistence_36"], ceil[n]["3h"], ceil[n]["persistence_3h"]) for n in names}
res["network_loses"] = sum(v["36"] > v["persistence_36"] for v in rec["network"]["val_by_episode"].values())
res["history_loses"] = sum(v["36"] > v["persistence_36"] for v in r["val_by_episode"].values())
res["history_loses_off"] = sum(v["36"] > v["persistence_36"] for n, v in r["val_by_episode"].items() if n.endswith("all_off"))
res["ceiling_wins"] = sum(c["3h"] < c["persistence_3h"] for c in ceil.values())

# measured vs predicted, one week, network model (the better starting model)
e = [x for x in eps if x.name == "building_train_2_atlanta_3_pid_water"][0]
P = cf.profile(src["network"], e, U, 36); X = P["X"]; t = np.arange(len(X)) / 288; a, b = 7 * 288, 14 * 288
fig, axs = plt.subplots(2, 1, figsize=(12, 6.4), sharex=True)
for ax, i, lab in zip(axs, P["rooms"], ["most exposed exterior room", "largest interior room"]):
    ax.plot(t[a:b], X[a:b, i] - 273.15, color=INK, lw=1.6, label="simulator")
    m = (P["hz_t"] >= a) & (P["hz_t"] < b); ax.plot(P["hz_t"][m] / 288, P["hz"][m, i] - 273.15, color=ORA, lw=1.1, label="network model, 3 h ahead")
    ax.set_title(f"room #{i}: {lab}", fontsize=10, loc="left"); ax.set_ylabel("°C")
axs[0].legend(frameon=False, fontsize=9, ncol=2, loc="upper left"); axs[1].set_xlabel("day of the episode")
fig.tight_layout(); fig.savefig(OUT / "ep30_profile_pred.png", dpi=130); plt.close()

# the analysis text a mutation request would carry, for the network model
res["analysis_network"] = cf.analysis(rec["network"], eps)["text"]
res["profile_network"] = evolve.profile_text(src["network"], eps[2], U, 36)
json.dump(res, open(S / "figs9_results.json", "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in res.items() if not k.startswith(("analysis", "profile", "history_by"))}, indent=1, default=float))
print(res["analysis_network"]); print(res["profile_network"])
