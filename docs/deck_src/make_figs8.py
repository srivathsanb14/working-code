"""Real-building figures for the physics and discovery parts (no synthetic data, no drawings):
  phys_cells.png      building_train_8 floor plan, every 20 cm cell coloured by the class the simulator gives it
  phys_room.png       one real room of building_train_8, its neighbours, outside wall and doors, with the conductances
                      computed by the network seed's formulas (mid-range universal materials)
  phys_matrix.png     the conductance matrix of building_train_8's room network (air, slab, facade, partition nodes)
and a JSON of the worked example and of a real pre-mutation analysis of the history seed on the 12 thirty-day episodes.
Usage: make_figs8.py <scratch> <out img dir>"""
import sys, json, pickle, importlib.util
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
S = Path(sys.argv[1]); OUT = Path(sys.argv[2])
WC = Path("/Users/srivathsan/Desktop/smart-buildings/working-code"); KIT = WC.parent / "smart-buildings-starter-kit"
sys.path.insert(0, str(WC / "code")); sys.path.insert(0, str(WC / "code/discovery"))
import room_model as rm
from discovery import episodes, closedform as cf, evolve
import dlib
plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False})
INK, ORA, BLU, GRY = "#13213A", "#D9692A", "#2F72B5", "#8A93A3"
res = {}

d = KIT / "smart_buildings/training_scenarios/building_train_8/atlanta_1"
fp, zm = np.load(d / "floor_plan.npy"), np.load(d / "zone_map.npy")
G = rm.grid_geometry(fp, zm)
V = int(G["nroom"]); rooms = [f"room_{i + 1}" for i in range(V)]
geo = episodes.geometry(fp, zm, rooms, {"all": rooms})
U, _ = cf.universal_constants()

# ---------------------------------------------------------------- 1 cell classes
cls = G["cls"].copy().astype(int)
img = np.zeros(cls.shape, int)                    # 0 outside, 1 facade, 2 partition, 3 room air, 4 door
img[cls == rm.EXT_SPACE] = 0; img[cls == rm.SHELL_OR_EXT] = 1; img[cls == rm.IWALL] = 2
img[(cls == rm.AIR) & (G["lab"] > 0)] = 3; img[G["door"]] = 4
cmap = ListedColormap(["#ffffff", "#5b6676", "#c9b8a0", "#e6eef8", ORA])
counts = {n: int((img == k).sum()) for k, n in enumerate(["outside", "facade", "partition", "room air", "door"])}
res["cells_train8"] = counts
fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6.6), gridspec_kw=dict(width_ratios=[1.25, 1]))
ys, xs = np.where(img > 0); y0, y1, x0, x1 = ys.min() - 3, ys.max() + 4, xs.min() - 3, xs.max() + 4
a1.imshow(img[y0:y1, x0:x1], cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
a1.set_title(f"building_train_8: both floors in one plan, {img.shape[0]} × {img.shape[1]} cells of 20 cm", fontsize=12, loc="left"); a1.axis("off")
# zoom on a corner with a door
dy, dx = np.where(G["door"]); k = len(dy) // 3; cy, cx = dy[k], dx[k]
zy0, zx0 = max(cy - 22, 0), max(cx - 30, 0); Z = img[zy0:zy0 + 44, zx0:zx0 + 60]
a2.imshow(Z, cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
a2.set_xticks(np.arange(-.5, Z.shape[1], 1), minor=True); a2.set_yticks(np.arange(-.5, Z.shape[0], 1), minor=True)
a2.grid(which="minor", color="#ffffff", lw=.3); a2.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
a2.set_title("Zoom: 60 × 44 cells (12 m × 8.8 m); each square is one control volume", fontsize=12, loc="left")
a1.add_patch(plt.Rectangle((zx0 - x0 - .5, zy0 - y0 - .5), 60, 44, fill=False, ec=INK, lw=1.5))
h = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ["#5b6676", "#c9b8a0", "#e6eef8", ORA]]
fig.legend(h, [f"facade ({counts['facade']:,})", f"partition ({counts['partition']:,})", f"room air ({counts['room air']:,})", f"door ({counts['door']:,})"],
           loc="lower center", ncol=4, frameon=False, fontsize=12)
fig.tight_layout(rect=(0, .06, 1, 1)); fig.savefig(OUT / "phys_cells.png", dpi=115); plt.close()

# ---------------------------------------------------------------- 2 one room, its numbers
spec = importlib.util.spec_from_file_location("obs", WC / "code/discovery/seeds_cf_legacy/seed_cf_network_nodes.py")
obs = importlib.util.module_from_spec(spec); sys.modules["dlib"] = dlib; spec.loader.exec_module(obs)
from discovery import coefform
cm = coefform.CoefModel(obs, geo, U); net = cm.build({"T": np.full((1, 1, V), 294.0)})
cand = [i for i in range(V) if geo["facade_faces"][i] > 0 and geo["n_doors"][i].sum() >= 1 and (geo["wall_cells"][i] > 0).sum() >= 2 and 30 < geo["area"][i] < 90]
i = cand[0]
lab = G["lab"]; my = lab == i + 1
yy, xx = np.where(my); pad = 18
ry0, ry1, rx0, rx1 = max(yy.min() - pad, 0), yy.max() + pad, max(xx.min() - pad, 0), xx.max() + pad
crop = img[ry0:ry1, rx0:rx1].copy(); hl = my[ry0:ry1, rx0:rx1]
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(crop, cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
ax.imshow(np.ma.masked_where(~hl, hl), cmap=ListedColormap([BLU]), alpha=.45, interpolation="nearest")
ax.axis("off")
A = geo["area"][i]; nf = geo["facade_faces"][i]; face = 0.6
C_air = U["RHO_C_AIR"] * 3 * A
G_full = net.A0inv  # not used; read conductances back from the seed's own formulas
K_EXT, K_IW, H = obs.K_EXT, obs.K_IW, obs.H_FILM
g_in = nf * face / (0.2 / (2 * 50) + 0.2 / (4 * K_EXT)); kW = 1 / (1 / g_in + obs._spread(A, nf * 0.2))
gam = nf * face / (0.2 / (4 * K_EXT) + 1 / H)
nbrs = []
for j in range(V):
    w, dc = geo["wall_cells"][i, j], geo["door_cells"][i, j]
    if w > 0 or dc > 0:
        gw = 1 / (1 / (w * face / (0.2 / 100 + 0.2 / (2 * K_IW))) + obs._spread(A, w * 0.2)) if w > 0 else 0.0
        gd = 1 / (1 / (dc * face / (2 * 0.2 / 50)) + obs._spread(A, dc * 0.2) + obs._spread(geo["area"][j], dc * 0.2)) if dc > 0 else 0.0
        nbrs.append(dict(room=j + 1, wall_m=float(w * 0.2), door_cells=int(dc), G_wall_half=float(gw), G_door=float(gd)))
        jy, jx = np.where(lab[ry0:ry1, rx0:rx1] == j + 1)
        if len(jy):
            ax.text(jx.mean(), jy.mean(), f"room {j + 1}" + (f"\ndoor {gd:.0f} W/K" if dc else "") + (f"\nwall {w * 0.2:.1f} m" if w else ""),
                    ha="center", va="center", fontsize=10, color=INK)
ax.text(xx.mean() - rx0, yy.mean() - ry0, f"room {i + 1}\n{A:.0f} m²\nair C = {C_air / 1e3:.0f} kJ/K", ha="center", va="center", fontsize=12, color="white", weight="bold")
ax.set_title(f"building_train_8, room {i + 1}, and the heat paths of the network seed", fontsize=12, loc="left")
ax.text(crop.shape[1] - 1, 1, f"outside wall {nf * 0.2:.1f} m\nair → wall {kW:.0f} W/K\nwall → outside {gam:.0f} W/K", ha="right", va="top", fontsize=11, color="white",
        bbox=dict(boxstyle="round", fc=INK, ec="none"))
h = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ["#5b6676", "#c9b8a0", ORA]]
ax.legend(h, ["facade", "partition", "door"], loc="lower right", frameon=False, fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "phys_room.png", dpi=120, bbox_inches="tight"); plt.close()
res["worked_room"] = dict(building="building_train_8", room=i + 1, area_m2=float(A), facade_faces=float(nf), facade_m=float(nf * 0.2),
                          C_air_J_per_K=float(C_air), C_slab_J_per_K=float(obs.RC_MASS * 3 * A), G_slab_W_per_K=float(5 * A),
                          kappa_W_air_to_facade=float(kW), gamma_facade_to_outside=float(gam), neighbours=nbrs,
                          materials=dict(k_ext=K_EXT, rc_ext=obs.RC_EXT, k_iw=K_IW, rc_iw=obs.RC_IW, rc_mass=obs.RC_MASS, h=H),
                          states=int(net.N), V=V, tau_air_slab_h=float(C_air / (5 * A) / 3600),
                          tau_air_supply_max_s=float(C_air / (U["RHO_C_AIR"] * 2.0)))

# ---------------------------------------------------------------- 3 the network matrix
N = net.N; L = np.linalg.inv(net.A0inv) - np.diag(net.Cdt)       # = diag(G sum + G_amb) - G
fig, ax = plt.subplots(figsize=(7.6, 7.2))
M = np.abs(L) > 1e-9
ax.spy(M, markersize=1.2, color=INK)
NE = int((geo["facade_faces"] > 0).sum())
for b, nm in ((V, "air"), (2 * V, "slab"), (2 * V + NE, "facade")):
    ax.axhline(b - .5, color=ORA, lw=.8); ax.axvline(b - .5, color=ORA, lw=.8)
for c, nm in ((V / 2, "air"), (1.5 * V, "slab"), (2 * V + NE / 2, "outside\nwall"), ((2 * V + NE + N) / 2, "shared walls")):
    ax.text(c, N + 8, nm, ha="center", fontsize=11, color=ORA)
ax.set_title(f"building_train_8: {N} nodes for {V} rooms; a dot = a conductance between two nodes", fontsize=11, loc="left", pad=28)
fig.tight_layout(); fig.savefig(OUT / "phys_matrix.png", dpi=120, bbox_inches="tight"); plt.close()
res["matrix"] = dict(N=int(N), V=V, nonzeros=int(M.sum()), offdiag=int(M.sum() - np.count_nonzero(np.diag(M))), exterior=NE, partitions=int(N - 2 * V - NE))
json.dump(res, open(S / "figs8b_results.json", "w"), indent=1, default=float); print(res["matrix"])

# ---------------------------------------------------------------- 4 a real pre-mutation analysis
if "--figs-only" in sys.argv:
    print("figures only"); sys.exit()
eps = pickle.load(open(S / "ep30_eps.pkl", "rb"))
src = (WC / "code/discovery/seeds_cf/seed_cf_history_slab.py").read_text()
rec = cf.evaluate(src, eps[:1], eps, U, every=12, screen_factor=1e9, max_train=1)
an = cf.analysis(rec, eps)
prof = evolve.profile_text(src, eps[2], U, 36)
res["analysis_history_seed"] = an["text"]; res["profile_text"] = prof; res["score"] = rec.get("score")
json.dump(res, open(S / "figs8_results.json", "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in res.items() if k not in ("analysis_history_seed", "profile_text")}, indent=1, default=float)[:3000])
print(an["text"]); print(prof)
