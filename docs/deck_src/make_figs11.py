"""Figures for the meeting deck's main slides, all from real outputs (no made-up numbers):

  split_grid.png      the computed split of the 30 shipped scenarios (outputs/split_shipped.json) as a building x station
                      grid, and each station's Wasserstein distance to the others (the same file's _why)
  term_heatmap.png    the network model's per-term residual slopes by room type and condition (closedform.analysis
                      on the 6 validation runs, saved as outputs/term_slopes_network.json)
  tiers_bars.png      3 h forecast error of persistence and the hand-written models on validation and held-out runs
                      (closedform.evaluate on the 12 thirty-day runs, as on the 'tiers' slide)

    python make_figs11.py <working-code> <working-code>/outputs/term_slopes_network.json <out img dir>
"""
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

WC, ANALYSIS, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
INK, INK2, GRID, SURF = "#13213A", "#52514E", "#D9D6CE", "#FBFAF7"
TIER = {"val": ("#2a78d6", "validation"), "test_hard": ("#eb6834", "test · hard"), "test_easy": ("#1baf7a", "test · easy"),
        "test_medium": ("#eda100", "test · medium"), "test_station": ("#B9B6AE", "extra (seen building)")}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 18, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "figure.facecolor": SURF, "axes.facecolor": SURF})

# ---------------------------------------------------------------- A. split grid + Wasserstein
sp = json.loads((WC / "outputs" / "split_shipped.json").read_text())
where = {sc: k for k in TIER for sc in sp.get(k, [])}
blds = sorted({sc.split("/")[0] for sc in where}, key=lambda b: int(b.rsplit("_", 1)[1]))
w = sp["_why"]["station_wasserstein_to_rest_K"]
stations = sorted(w, key=lambda s: w[s])                                   # least to most different
fig = plt.figure(figsize=(18, 5.6))
ax = fig.add_axes([0.09, 0.04, 0.58, 0.8]); bx = fig.add_axes([0.76, 0.12, 0.22, 0.72])
for i, b in enumerate(blds):
    for j, st in enumerate(stations):
        sc = next((s for s in where if s.startswith(b + "/") and s.split("/")[1].rsplit("_", 1)[0] == st), None)
        k = where.get(sc)
        col, lab = TIER[k] if k else ("#EEE", "")
        ax.add_patch(plt.Rectangle((j + 0.03, i + 0.05), 0.94, 0.9, color=col, lw=0))
        ax.text(j + 0.5, i + 0.36, sc.split("/")[1] if sc else "", ha="center", va="center", fontsize=14, color=INK, weight="bold")
        ax.text(j + 0.5, i + 0.7, lab.replace("extra (seen building)", "extra"), ha="center", va="center", fontsize=12.5, color=INK)
ax.set_xlim(0, len(stations)); ax.set_ylim(len(blds), 0)
ax.set_xticks(np.arange(len(stations)) + 0.5, [s.capitalize() for s in stations]); ax.xaxis.tick_top()
ax.set_yticks(np.arange(len(blds)) + 0.5, [b.replace("building_train_", "building ") for b in blds])
ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
ax.set_title("The 30 shipped scenarios: every cell is one whole scenario (all its runs)", loc="left", fontsize=17, pad=36, color=INK)
y = np.arange(len(stations))
bx.barh(y, [w[s] for s in stations], color=["#eb6834" if s == sp["_why"]["hard_station"] else "#8A93A3" for s in stations], height=0.6)
for yi, s in zip(y, stations):
    bx.text(w[s] + 0.4, yi, f"{w[s]:.1f} K", va="center", fontsize=16, color=INK)
bx.set_yticks(y, [s.capitalize() for s in stations]); bx.invert_yaxis(); bx.set_xlim(0, max(w.values()) * 1.3)
bx.spines[["top", "right"]].set_visible(False); bx.tick_params(length=0); bx.xaxis.grid(True, color=GRID, lw=0.8); bx.set_axisbelow(True)
bx.set_title("How different each climate is\n(Wasserstein distance to the rest)", loc="left", fontsize=16, color=INK)
fig.savefig(OUT / "split_grid.png", dpi=110); plt.close(fig)

# ---------------------------------------------------------------- B. term heatmap
T = json.loads(ANALYSIS.read_text())                                      # outputs/term_slopes_network.json
terms = ["air", "neighbours", "slab", "wall", "inner", "heat"]
short = {"corner rooms (outside wall, 2+ sides)": "corner rooms", "one-side exterior rooms": "one-side rooms",
         "interior rooms": "interior rooms", "larger half of rooms (by floor area)": "larger rooms", "smaller half of rooms": "smaller rooms",
         "HVAC air on (room mean > 50 W/K)": "HVAC air on", "HVAC air off": "HVAC air off", "occupied (comfort) hours": "occupied hours",
         "colder half of outside temperatures": "colder outside"}
rows = [short[g] for g in short if g in T]
M = [[T[g].get(t, np.nan) for t in terms] for g in short if g in T]
M = np.array(M)
cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#ECEAE4", "#eb6834"])
fig, ax = plt.subplots(figsize=(14, 6.4))
ax.imshow(np.nan_to_num(M, nan=0.0), cmap=cmap, vmin=-1.2, vmax=1.2, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v = M[i, j]
        ax.text(j, i, "n/a" if np.isnan(v) else f"{v:+.2f}", ha="center", va="center", fontsize=18,
                color=INK2 if np.isnan(v) else INK, weight="bold" if not np.isnan(v) and abs(v) >= 0.4 else "normal")
ax.set_xticks(range(len(terms)), terms); ax.xaxis.tick_top(); ax.set_yticks(range(len(rows)), rows)
ax.axhline(4.5, color=SURF, lw=6)
ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
ax.text(-0.5, M.shape[0] + 0.2, "orange: too weak (should be larger)     blue: too strong (should be smaller)     near 0: about right",
        fontsize=15, color=INK2, va="top")
fig.subplots_adjust(left=0.2, right=0.99, top=0.9, bottom=0.11)
fig.savefig(OUT / "term_heatmap.png", dpi=110); plt.close(fig)

# ---------------------------------------------------------------- C. tiers bars
groups = ["Validation\n(6 runs)", "Unseen buildings 2, 8\n(4 runs)", "Building 1 at Fairbanks\n(2 runs)"]
models = {"Persistence ('temperature stays')": ([0.889, 0.948, 1.274], "#8A93A3"),
          "Network model (hand-written)": ([0.740, 0.538, 2.114], "#2a78d6"),
          "History model (hand-written)": ([1.099, 1.270, 1.732], "#eb6834")}
fig, ax = plt.subplots(figsize=(13, 5.0))
x = np.arange(len(groups)); wd = 0.26
for k, (name, (v, c)) in enumerate(models.items()):
    xs = x + (k - 1) * (wd + 0.02)
    ax.bar(xs, v, wd, color=c, label=name)
    for xi, vi in zip(xs, v):
        ax.text(xi, vi + 0.03, f"{vi:.2f}", ha="center", fontsize=17, color=INK)
ax.set_xticks(x, groups); ax.set_ylabel("3 h error (K), lower = better"); ax.set_ylim(0, 2.45)
ax.spines[["top", "right"]].set_visible(False); ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)
ax.legend(frameon=False, loc="upper left", fontsize=16)
fig.tight_layout(); fig.savefig(OUT / "tiers_bars.png", dpi=110); plt.close(fig)
print("wrote split_grid.png, term_heatmap.png, tiers_bars.png")
