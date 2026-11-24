"""Controller chart from the rehearsal reports (outputs/rehearsal/check_{pid_water,ff,mpc}/report.json): 2-day checks
through the kit's harness with the coefficient-form network model.  Usage: make_figs10.py <working-code> <out img dir>"""
import sys, json
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
W, OUT = Path(sys.argv[1]), Path(sys.argv[2])
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
sc = ["building_train_8/atlanta_1", "building_train_1/fairbanks_1"]
vals = {}
for tag, lab in (("pid_water", "pid_water"), ("ff", "feed-forward"), ("mpc", "MPC")):
    r = {x["scenario"]: x["score"] for x in json.load(open(W / f"outputs/rehearsal/check_{tag}/report.json"))["runs"]}
    vals[lab] = [r[s] for s in sc]
fig, ax = plt.subplots(figsize=(8, 4.2)); x = np.arange(2)
for i, (k, v) in enumerate(vals.items()):
    ax.bar(x + (i - 1) * 0.26, v, 0.26, color=["#8A93A3", "#2F72B5", "#D9692A"][i], label=k)
    for j, y in enumerate(v):
        ax.text(x[j] + (i - 1) * 0.26, y + 0.0003, f"{y:.5f}", ha="center", fontsize=8, rotation=90)
ax.set_xticks(x); ax.set_xticklabels([s.replace("building_", "").replace("/", " · ") for s in sc])
ax.set_ylabel("score (mean loss per step, lower is better)"); ax.set_ylim(0, 0.031)
ax.legend(frameon=False, ncol=3, loc="upper center"); ax.set_title("2-day checks through the kit's harness (576 steps), network model", fontsize=10, loc="left")
fig.tight_layout(); fig.savefig(OUT / "controllers.png", dpi=130); plt.close()
print(json.dumps(vals))
