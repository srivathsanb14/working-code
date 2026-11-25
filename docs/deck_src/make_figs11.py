"""Figures for the meeting deck's main slides, all from real outputs (no made-up numbers):

  split_grid.png      the computed split of the 30 shipped scenarios (outputs/split_shipped.json) as a building x station
                      grid, and each station's Wasserstein distance to the others (the same file's _why)
  term_heatmap.png    the network model's per-term residual slopes by room type and condition (closedform.analysis
                      on the 6 validation runs, saved as outputs/term_slopes_network.json)
  buildings.png       the six shipped buildings' real floor plans at one scale, rooms coloured by type (kit zone_map.npy;
                      room types from outputs/building_train_*_rooms.csv, written by code/explore_zone_map.py)
  weather.png         the 30 simulated 30-day episodes at the 5 stations, against the rest of 2023 that is in the weather
                      file but in no simulated run (kit weather/csv/*.csv and weather/<scenario>.yaml)
  tiers_bars.png      3 h forecast error of persistence and the hand-written models on validation and held-out runs
                      (closedform.evaluate on the 12 thirty-day runs, as on the 'tiers' slide)

  episode.png         one real episode (building 4 x denver_2 x pid_water, a validation run) and every column it holds,
                      from the episode file build_dataset.py writes from the kit's run (optional 4th argument)

    python make_figs11.py <working-code> <working-code>/outputs/term_slopes_network.json <out img dir> [episode.npz]
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
import matplotlib.patches  # noqa: E402,F401

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

# ---------------------------------------------------------------- D. building floor plans
import csv  # noqa: E402
import datetime as dt  # noqa: E402
from scipy import ndimage  # noqa: E402
KIT = WC.parent / "smart-buildings-starter-kit" / "smart_buildings"
ZS = json.loads((WC / "outputs" / "zone_map_summary.json").read_text())
FA = json.loads((WC / "outputs" / "floors_and_ahus.json").read_text())
TYPE = {"corner": ("#2a78d6", "corner room"), "edge": ("#1baf7a", "one outside wall"), "interior": ("#eda100", "interior room")}
fig, axs = plt.subplots(2, 3, figsize=(16, 6.8))
for ax, b in zip(axs.ravel(), sorted(ZS, key=lambda b: int(b.rsplit("_", 1)[1]))):
    zm = np.load(KIT / "training_scenarios" / b / ZS[b]["scenario"] / "zone_map.npy")
    lab, n = ndimage.label(zm == 0)
    rows_ = list(csv.DictReader(open(WC / "outputs" / f"{b}_rooms.csv")))
    img = np.ones(zm.shape + (3,)); img[zm == 1] = matplotlib.colors.to_rgb(INK2); img[zm == 2] = matplotlib.colors.to_rgb(SURF)
    for r in rows_:
        l = lab[int(round(float(r["cy"]))), int(round(float(r["cx"])))]
        if l > 0:
            img[lab == l] = matplotlib.colors.to_rgb(TYPE[r["type"]][0])
    fl, nf = ndimage.label(ndimage.binary_dilation(zm != 2, iterations=3))      # one component per floor
    r0, r1 = np.nonzero((fl == fl[np.nonzero(zm != 2)[0][0], np.nonzero(zm != 2)[1][np.argmin(np.nonzero(zm != 2)[0])]]).any(1))[0][[0, -1]]
    c0, c1 = np.nonzero((zm != 2).any(0))[0][[0, -1]]
    img = img[r0:r1 + 1, c0:c1 + 1]
    cv = ZS[b]["cv_m"]; H, W = img.shape[:2]
    ax.imshow(img, extent=[0, W * cv, H * cv, 0], interpolation="nearest")
    ax.set_xlim(0, 90); ax.set_ylim(64, 0); ax.set_aspect("equal"); ax.axis("off")
    t = ZS[b]["types"]
    ax.set_title(f"Building {b.rsplit('_', 1)[1]}: {ZS[b]['rooms']} rooms, {FA[b]['floors']} floors\n"
                 f"{t['interior']} interior of {ZS[b]['rooms']}", fontsize=17, loc="left", color=INK)
axs[1, 2].plot([60, 80], [62, 62], color=INK, lw=3); axs[1, 2].text(70, 60.5, "20 m", ha="center", fontsize=15)
fig.legend(handles=[matplotlib.patches.Patch(color=c, label=l) for c, l in TYPE.values()], loc="lower center", ncol=3,
           frameon=False, fontsize=17)
fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.09, wspace=0.03, hspace=0.3)
fig.savefig(OUT / "buildings.png", dpi=100); plt.close(fig)

# ---------------------------------------------------------------- E. weather at the five stations
STN = ["singapore", "phoenix", "atlanta", "denver", "fairbanks"]
win = {}
for bd in sorted((KIT / "training_scenarios").iterdir()):
    for sd in bd.iterdir():
        st = sd.name.rsplit("_", 1)[0]
        start = re.search(r"start_timestamp:\s*(\S+)", (KIT / "weather" / f"{sd.name}.yaml").read_text()).group(1)
        win.setdefault(st, []).append(dt.datetime.fromisoformat(start))
fig, axs = plt.subplots(len(STN), 1, figsize=(16, 6.6), sharex=True, sharey=True)
for ax, st in zip(axs, STN):
    rr = list(csv.DictReader(open(KIT / "weather" / "csv" / f"{st}.csv")))
    t = np.array([dt.datetime.strptime(r["Time"], "%Y%m%d-%H%M") for r in rr]); T = np.array([float(r["TempC"]) for r in rr])
    day = np.array([x.timetuple().tm_yday for x in t])
    d = np.arange(1, day.max() + 1)
    lo = np.array([T[day == k].min() for k in d]); hi = np.array([T[day == k].max() for k in d])
    ax.fill_between(d, lo, hi, color="#E2DFD8", lw=0, label="rest of 2023: in the weather file, never simulated")
    inw = []
    for s0 in win[st]:
        m = (t >= s0) & (t < s0 + dt.timedelta(days=30)); inw.append(T[m])
        ax.plot([x.timetuple().tm_yday + x.hour / 24 for x in t[m]], T[m], color="#eb6834", lw=1.0)
    a = np.concatenate(inw)
    ax.text(1.01, 0.5, f"{st.capitalize()}\nepisodes: {a.min():.0f} to {a.max():.0f} °C", transform=ax.transAxes, va="center", fontsize=17, color=INK)
    ax.axhline(21, color=INK2, lw=0.8, ls=":")
    ax.spines[["top", "right"]].set_visible(False); ax.tick_params(length=0); ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_yticks([-40, 0, 40])
axs[0].plot([], [], color="#eb6834", lw=2, label="the 30 simulated episodes (6 per station): all we have")
axs[0].legend(loc="lower left", ncol=2, frameon=False, fontsize=15, bbox_to_anchor=(0, 1.0))
mo = [dt.date(2023, m, 1).timetuple().tm_yday for m in range(1, 13)]
axs[-1].set_xticks(mo, ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]); axs[-1].set_xlim(1, 365)
fig.text(0.01, 0.5, "outside °C (dotted: 21 °C)", rotation=90, va="center", fontsize=16, color=INK2)
fig.subplots_adjust(left=0.08, right=0.82, top=0.92, bottom=0.06, hspace=0.15)
fig.savefig(OUT / "weather.png", dpi=100); plt.close(fig)

# ---------------------------------------------------------------- F. one episode = one kit run, and its columns
if len(sys.argv) > 4:
    EP = np.load(sys.argv[4]); meta = json.loads(str(EP["meta"]))
    b, st = meta["building"], meta["weather"]
    n, V = EP["T_meas"].shape; days = np.arange(n) / 288
    fig = plt.figure(figsize=(13, 7.0))
    # left: the three ingredients
    zm = np.load(KIT / "training_scenarios" / b / st / "zone_map.npy"); lab, _ = ndimage.label(zm == 0)
    img = np.ones(zm.shape + (3,)); img[zm == 1] = matplotlib.colors.to_rgb(INK2); img[zm == 2] = matplotlib.colors.to_rgb(SURF)
    for r in csv.DictReader(open(WC / "outputs" / f"{b}_rooms.csv")):
        l = lab[int(round(float(r["cy"]))), int(round(float(r["cx"])))]
        if l > 0:
            img[lab == l] = matplotlib.colors.to_rgb(TYPE[r["type"]][0])
    fl, _ = ndimage.label(ndimage.binary_dilation(zm != 2, iterations=3)); rr, cc = np.nonzero(zm != 2)
    r0, r1 = np.nonzero((fl == fl[rr[0], cc[0]]).any(1))[0][[0, -1]]; c0, c1 = cc.min(), cc.max()
    a1 = fig.add_axes([0.01, 0.5, 0.22, 0.34]); a1.imshow(img[r0:r1 + 1, c0:c1 + 1], interpolation="nearest"); a1.axis("off")
    a1.set_title(f"1 building\n{b.replace('building_train_', 'building ')}: {V} rooms", loc="left", fontsize=16, color=INK)
    a2 = fig.add_axes([0.04, 0.2, 0.19, 0.18]); a2.plot(days, EP["in__T_out"] - 273.15, color="#eb6834", lw=0.8)
    a2.set_title(f"× 1 weather window\n{st}: 30 days", loc="left", fontsize=16, color=INK); a2.set_xticks([0, 10, 20, 30])
    a2.set_ylabel("°C", fontsize=13); a2.tick_params(labelsize=12, length=0); a2.spines[["top", "right"]].set_visible(False)
    fig.text(0.03, 0.06, f"× 1 controller\n{meta['policy']}", fontsize=16, color=INK, va="bottom")
    # right: every column, over the 30 days
    per_room = [("T_meas", "room temperature (target)", "°C", -273.15), ("in__g", "supply air", "W/K", 0),
                ("in__T_supply", "supply-air temperature", "°C", -273.15), ("in__Q_reheat", "reheat heat", "W", 0),
                ("in__Q_gain", "lights & equipment", "W", 0)]
    bw = [("in__T_out", "outside temperature", "°C", -273.15), ("in__occupants", "people in building", "", 0),
          ("in__comfort", "occupied hours (1/0)", "", 0), ("in__T_water", "boiler water temp.", "°C", -273.15)]
    H = len(per_room) + len(bw); top, bot, gap = 0.9, 0.08, 0.012; h = (top - bot - gap * (H - 1)) / H
    fig.text(0.29, 0.95, f"= 1 episode: these columns at every 5-minute step for 30 days ({n:,} rows)", fontsize=17, color=INK)
    for k, (key, name, unit, off) in enumerate(per_room + bw):
        ax = fig.add_axes([0.29, top - (k + 1) * h - k * gap, 0.46, h])
        A = EP[key] + off
        if A.ndim == 2:
            ax.imshow(A.T, aspect="auto", cmap="Blues", interpolation="nearest", extent=[0, n / 288, V, 0])
            ax.set_yticks([])
        else:
            ax.plot(days, A, color=INK2, lw=0.7); ax.set_xlim(0, n / 288); ax.set_yticks([])
        ax.set_xticks([0, 7, 14, 21, 28] if k == H - 1 else []); ax.tick_params(labelsize=12, length=0)
        [sp.set_visible(False) for sp in ax.spines.values()]
        shape = f"per room ({V})" if A.ndim == 2 else "building-wide"
        ax.text(1.01, 0.5, f"{name}\n{shape}" + (f", {unit}" if unit else ""), transform=ax.transAxes, va="center", fontsize=14.5, color=INK)
    fig.text(0.52, 0.015, "day   (per-room columns: one row per room, darker = higher)", ha="center", fontsize=14, color=INK2)
    fig.savefig(OUT / "episode.png", dpi=100); plt.close(fig)
print("wrote split_grid.png, term_heatmap.png, tiers_bars.png, buildings.png, weather.png" + (", episode.png" if len(sys.argv) > 4 else ""))
