"""Clearer 30-day figures for part 3 (same 12 kit runs as make_figs4: 6 shipped buildings x pid_water / HVAC off,
8,640 steps). Nothing new is simulated: episodes come from the ep30_eps.pkl cache, reward terms from the kit's
reward_components.csv. Usage: make_figs7.py <scratch> <out img dir>"""
import sys, json, pickle
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
S = Path(sys.argv[1]); OUT = Path(sys.argv[2]); RES = S / "ep30/results"
WC = Path("/Users/srivathsan/Desktop/smart-buildings/working-code"); sys.path.insert(0, str(WC / "code"))
plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False})
ORA, BLU, GRY, INK, GRN, PUR = "#D9692A", "#2F72B5", "#8A93A3", "#13213A", "#3a9a5b", "#8e44ad"
BAND = "#bfe3c8"; WKND = "#ecebe6"
eps = {e.name: e for e in pickle.load(open(S / "ep30_eps.pkl", "rb"))}
SC = [("building_train_1", "fairbanks_1"), ("building_train_2", "atlanta_3"), ("building_train_4", "denver_2"),
      ("building_train_5", "phoenix_1"), ("building_train_7", "singapore_6"), ("building_train_8", "atlanta_1")]
NICE = {"fairbanks_1": "Fairbanks, Jan", "atlanta_3": "Atlanta, Apr", "denver_2": "Denver, Feb", "phoenix_1": "Phoenix, Jan",
        "singapore_6": "Singapore, Oct", "atlanta_1": "Atlanta, Jan"}
COL = dict(zip([w for _, w in SC], [BLU, GRN, GRY, ORA, PUR, INK]))
day = lambda n: np.arange(n) / 288
C = lambda x: x - 273.15
score = {f"{b}/{w}": -json.load(open(RES / b / w / "pid_water/summary.json"))["avg_reward"] for b, w in SC}


def E(b, w, p="pid_water"):
    return eps[f"{b}_{w}_{p}"]


def hour(e):
    return (np.degrees(np.arctan2(e.inputs["hour_sin"], e.inputs["hour_cos"])) / 15) % 24


def workday(e):
    """a day counts as a working day if its comfort flag is ever on"""
    c = e.inputs["comfort"] > 0.5; d = np.arange(e.n) // 288
    return np.array([c[d == k].any() for k in range(d.max() + 1)])


def shade(ax, e, lo=None, hi=None):
    """grey = weekend/holiday; green = comfort band, drawn only during comfort hours"""
    for k, wd in enumerate(workday(e)):
        if not wd: ax.axvspan(k, k + 1, color=WKND, lw=0, zorder=0)
    c = e.inputs["comfort"] > 0.5; t = day(e.n)
    ax.fill_between(t, 294 - 273.15, 297 - 273.15, where=c, color=BAND, lw=0, step="post", zorder=1)


def spread(ax, t, X, col, lab, lw=1.6):
    ax.fill_between(t, np.percentile(X, 10, 1), np.percentile(X, 90, 1), color=col, alpha=.22, lw=0, zorder=2)
    ax.plot(t, np.median(X, 1), color=col, lw=lw, label=lab, zorder=3)


def smooth(X, k=6):
    return pd.DataFrame(X).rolling(k, min_periods=1, center=True).mean().values


# 1 ---------------------------------------------------------------- 30 days, six buildings, rooms summarised
fig, axs = plt.subplots(3, 2, figsize=(15, 10.5), sharex=True)
for ax, (b, w) in zip(axs.flat, SC):
    e = E(b, w); X = smooth(C(e.T_meas)); t = day(e.n); ext = e.geo["facade_faces"] > 0
    shade(ax, e)
    spread(ax, t, X[:, ext], ORA, "rooms with an outside wall"); spread(ax, t, X[:, ~ext], BLU, "interior rooms")
    ax.set_ylim(min(np.percentile(X, 0.5), 20) - 0.5, max(np.percentile(X, 99.5), 25) + 0.5)
    ax.set_title(f"{b.replace('building_', '')} · {NICE[w]} · {e.V} rooms · pid_water score {score[b + '/' + w]:.3f}", fontsize=12, loc="left")
    ax.set_ylabel("°C"); ax.set_xlim(0, 30)
for ax in axs[-1]: ax.set_xlabel("day of the 30-day episode")
h, l = axs[0, 1].get_legend_handles_labels()
h += [plt.Rectangle((0, 0), 1, 1, color=BAND), plt.Rectangle((0, 0), 1, 1, color=WKND)]; l += ["comfort band (occupied hours)", "weekend / holiday"]
fig.legend(h, l, loc="upper center", ncol=4, frameon=False, fontsize=12, bbox_to_anchor=(.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, .96)); fig.savefig(OUT / "ep30_profiles.png", dpi=110); plt.close()

# 2 ---------------------------------------------------------------- one week, annotated
b, w = "building_train_2", "atlanta_3"; e = E(b, w); X = C(e.T_meas); ext = e.geo["facade_faces"] > 0
a, z = 7 * 288, 14 * 288; t = day(e.n)
fig, (ax, axo) = plt.subplots(2, 1, figsize=(13, 6.2), sharex=True, gridspec_kw=dict(height_ratios=[3, 1]))
shade(ax, e); Xs = smooth(X)
spread(ax, t[a:z], Xs[a:z][:, ext], ORA, "rooms with an outside wall (median, 10–90 %)")
spread(ax, t[a:z], Xs[a:z][:, ~ext], BLU, "interior rooms (median, 10–90 %)")
ax.set_ylim(17.5, 27.5); ax.set_xlim(7, 14); ax.set_ylabel("room °C")
start = pd.Timestamp("2023-04-16 06:00")
ticks = np.arange(7, 15); axo.set_xticks(ticks); axo.set_xticklabels([(start + pd.Timedelta(days=int(k))).strftime("%a %d %b") for k in ticks], fontsize=11)
ax.annotate("06:00 comfort hours start:\nrooms pulled into the band", xy=(8.0, 21.0), xytext=(7.05, 18.0), fontsize=11, arrowprops=dict(arrowstyle="->", color=INK))
ax.annotate("19:00 setback: cooling stops; interior rooms\nwarm up, rooms with outside walls cool off", xy=(8.55, 23.6), xytext=(8.65, 18.0), fontsize=11, arrowprops=dict(arrowstyle="->", color=INK))
ax.annotate("weekend: no one in,\nrooms drift freely", xy=(13.4, 25.0), xytext=(12.75, 18.0), fontsize=11, arrowprops=dict(arrowstyle="->", color=INK))
ax.legend(frameon=False, fontsize=11, loc="upper left", ncol=2)
axo.plot(t[a:z], C(e.inputs["T_out"][a:z]), color=GRY, lw=1.2); axo.set_ylabel("outside °C"); axo.set_xlabel("each tick = 06:00 on that day")
fig.tight_layout(); fig.savefig(OUT / "ep30_week.png", dpi=120); plt.close()

# 3 ---------------------------------------------------------------- HVAC off vs pid_water
fig, axs = plt.subplots(1, 2, figsize=(14, 4.6))
for ax, (b, w) in zip(axs, [SC[1], SC[0]]):
    shade(ax, E(b, w))
    for p, c, l in (("all_off", GRY, "HVAC off"), ("pid_water", ORA, "pid_water (best baseline)")):
        e = E(b, w, p); spread(ax, day(e.n), smooth(C(e.T_meas)), c, l)
    ax.set_title(f"{b.replace('building_', '')} · {NICE[w]}", fontsize=12, loc="left"); ax.set_xlabel("day"); ax.set_ylabel("room °C (median, 10–90 %)"); ax.set_xlim(0, 30)
axs[0].legend(frameon=False, fontsize=11, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "ep30_policies.png", dpi=125); plt.close()

# 4 ---------------------------------------------------------------- geometry predicts free cooling
fig, ax = plt.subplots(figsize=(8, 5)); res = {}
for b, w in SC:
    e = E(b, w, "all_off"); X = np.vstack([np.full((1, e.V), e.T0), e.T_meas]); gap = X[0] - e.inputs["T_out"][:144].mean()
    rate = (X[0] - X[144]) / 12 / np.where(np.abs(gap) < 1, np.nan, gap) * 100
    x = e.geo["facade_faces"] * 0.2 / e.geo["area"]; ok = np.isfinite(rate) & (x > 0); rate = np.where(x > 0, rate, np.nan)
    ax.scatter(x, rate, s=16, color=COL[w], alpha=.75, label=f"{b.replace('building_train_', 'b')} {NICE[w]}")
    if ok.sum() > 3:
        k, c0 = np.polyfit(x[ok], rate[ok], 1); xx = np.linspace(0, x.max(), 10); ax.plot(xx, k * xx + c0, color=COL[w], lw=1, alpha=.8)
        res[w] = dict(r=float(np.corrcoef(x[ok], rate[ok])[0, 1]), slope=float(k))
ax.set_xlabel("outside-wall length per m² of floor, 1/m  (read from the floor plan)")
ax.set_ylabel("temperature drop per hour,\n% of the indoor–outdoor gap")
ax.set_title("HVAC off, first 12 hours: one dot per room with an outside wall", fontsize=12, loc="left"); ax.set_xlim(0, None)
ax.legend(frameon=False, fontsize=9.5); fig.tight_layout(); fig.savefig(OUT / "ep30_geometry.png", dpi=125); plt.close()

# 5 ---------------------------------------------------------------- where the loss comes from
fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw=dict(width_ratios=[1, 1.35]))
names = [f"{b}/{w}" for b, w in SC]; terms = {}
for b, w in SC:
    rc = pd.read_csv(RES / b / w / "pid_water/reward_components.csv")
    terms[w] = dict(comfort=float((-0.6 * rc["normalized_productivity_regret"]).mean()), energy=float((0.2 * rc["normalized_energy_cost"]).mean()),
                    carbon=float((0.2 * rc["normalized_carbon_emission"]).mean()))
y = np.arange(len(SC)); left = np.zeros(len(SC))
for k, c in (("comfort", ORA), ("energy", BLU), ("carbon", GRY)):
    v = np.array([terms[w][k] for _, w in SC]); a1.barh(y, v, left=left, color=c, label=k); left += v
for i, v in enumerate(left): a1.text(v + .003, i, f"{v:.3f}", va="center", fontsize=11)
a1.set_yticks(y); a1.set_yticklabels([f"{b.replace('building_train_', 'b')} {NICE[w]}" for b, w in SC]); a1.invert_yaxis()
a1.set_xlabel("score = mean loss per step"); a1.set_title("Each scenario's score, split into its three terms", fontsize=12, loc="left"); a1.legend(frameon=False, fontsize=11, loc="lower right")
a1.set_xlim(0, left.max() * 1.2)
b, w = "building_train_4", "denver_2"; rc = pd.read_csv(RES / b / w / "pid_water/reward_components.csv"); d = np.arange(len(rc)) // 288
cm = (-0.6 * rc["normalized_productivity_regret"]).groupby(d).mean(); en = (0.2 * rc["normalized_energy_cost"]).groupby(d).mean(); ca = (0.2 * rc["normalized_carbon_emission"]).groupby(d).mean()
wd = workday(E(b, w))
for k in range(len(wd)):
    if not wd[k]: a2.axvspan(k - .5, k + .5, color=WKND, lw=0, zorder=0)
a2.bar(cm.index, cm, color=ORA); a2.bar(en.index, en, bottom=cm, color=BLU); a2.bar(ca.index, ca, bottom=cm + en, color=GRY)
a2.set_xlabel("day (grey = weekend / holiday)"); a2.set_ylabel("mean loss per step that day"); a2.set_title(f"Day by day: {b.replace('building_', '')} · {NICE[w]}", fontsize=12, loc="left")
fig.tight_layout(); fig.savefig(OUT / "ep30_reward.png", dpi=115); plt.close()

# 6 ---------------------------------------------------------------- comfort deviation by hour of day, too cold vs too warm
fig, axs = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True); prof = {}
for b, w in SC:
    e = E(b, w); T = e.T_meas; c = e.inputs["comfort"] > 0.5; hr = np.floor(hour(e) * 2) / 2
    for ax, dev, k in ((axs[0], np.maximum(294 - T, 0).mean(1), "cold"), (axs[1], np.maximum(T - 297, 0).mean(1), "warm")):
        sr = pd.Series(np.where(c, dev, np.nan)).groupby(hr).mean().dropna(); prof.setdefault(w, {})[k] = sr.round(3).to_dict()
        ax.plot(sr.index, sr.values, "o-", ms=3, lw=1.4, color=COL[w], label=f"{b.replace('building_train_', 'b')} {NICE[w]}")
for ax, t in zip(axs, ["Too cold: K below the band", "Too warm: K above the band"]):
    ax.set_xlim(5.5, 19.5); ax.set_xticks(range(6, 20, 2)); ax.set_xticklabels([f"{h:02d}:00" for h in range(6, 20, 2)], fontsize=11)
    ax.set_xlabel("time of day (working days, comfort hours)"); ax.set_title(t, fontsize=12, loc="left")
axs[0].set_ylabel("mean over rooms, K"); axs[1].legend(frameon=False, fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "ep30_comfort.png", dpi=125); plt.close()

json.dump(dict(geometry_fit=res, terms=terms, comfort_by_hour=prof, scores=score), open(S / "figs7_results.json", "w"), indent=1)
print("done")
