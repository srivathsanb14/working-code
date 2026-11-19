"""Turn a candidate's held-out residuals into short text the LLM can act on.

The point is to say WHERE the model is wrong in terms of things the model could
depend on (exposure, doors, size, HVAC activity, time of day, elapsed time),
not just how wrong it is.
"""
import numpy as np

from .evaluate import DAY


def _tertiles(x):
    q = np.quantile(x, [1 / 3, 2 / 3])
    return np.digitize(x, q)


def summarize(rec, val_eps, max_lines=40):
    if rec.get("status") != "ok":
        lines = [f"status: {rec.get('status')} - {rec.get('reason', '')}"]
        lines += [f"  - {f}" for f in rec.get("failures", [])]
        if rec.get("traceback"):
            lines.append("  traceback (last lines):\n    " + rec["traceback"].replace("\n", "\n    "))
        return "\n".join(lines)
    det = rec["val_detail"]["episodes"]
    L = []
    L.append(f"held-out open-loop RMSE, days 2-3 (theta fitted on day 1): mean {rec['val_rmse_days23_mean']:.3f} K")
    for e in val_eps:
        d = det[e.name]
        L.append(f"  {e.name}: day1 {d['rmse_day1']:.3f} K, days2-3 {d['rmse_days23']:.3f} K, bias {d['bias_days23']:+.3f} K")
    L.append(f"train RMSE (whole episodes, theta fitted jointly): mean {rec['train_rmse_mean']:.3f} K")
    # pooled per-room errors on days 2-3
    rows = []
    for e in val_eps:
        err = det[e.name]["err"][DAY:]
        g = e.geo
        rms = np.sqrt((err ** 2).mean(0)); bias = err.mean(0)
        exposure = g["facade_faces"] * 0.2 / np.maximum(g["area"], 1e-9)
        doors = g["n_doors"].sum(1)
        for i in range(e.V):
            rows.append((rms[i], bias[i], exposure[i], doors[i], g["area"][i], e.name, i))
    R = np.array([r[:5] for r in rows], float)
    for col, label in ((2, "facade length per m2 of floor"), (3, "number of doors"), (4, "floor area")):
        t = _tertiles(R[:, col]) if np.ptp(R[:, col]) > 0 else np.zeros(len(R), int)
        parts = [f"{['low', 'mid', 'high'][k]}: rmse {R[t == k, 0].mean():.3f}, bias {R[t == k, 1].mean():+.3f}" for k in range(3) if (t == k).any()]
        L.append(f"by {label} (tertiles): " + "; ".join(parts))
    worst = sorted(rows, key=lambda r: -r[0])[:3]
    L.append("worst rooms: " + "; ".join(f"{w[5]} room {w[6] + 1} (rmse {w[0]:.2f}, bias {w[1]:+.2f}, facade/m2 {w[2]:.3f}, doors {int(w[3])}, area {w[4]:.0f} m2)" for w in worst))
    # time structure
    hv, nohv, hour_err = [], [], np.zeros(4); hour_n = np.zeros(4)
    growth = []
    for e in val_eps:
        err = det[e.name]["err"]
        active = e.inputs["g"].mean(1) > 50.0
        hv.append(err[active]); nohv.append(err[~active])
        ang = np.arctan2(e.inputs["hour_sin"], e.inputs["hour_cos"]) % (2 * np.pi)
        b = np.minimum((ang / (2 * np.pi) * 4).astype(int), 3)
        for k in range(4):
            m = b == k
            hour_err[k] += (err[m] ** 2).sum(); hour_n[k] += m.sum() * e.V
        growth.append([np.sqrt((err[a:b] ** 2).mean()) for a, b in ((0, DAY), (DAY, 2 * DAY), (2 * DAY, len(err))) if b > a])
    hv = np.concatenate([h.ravel() for h in hv if h.size]) if any(h.size for h in hv) else np.zeros(0)
    nohv = np.concatenate([h.ravel() for h in nohv if h.size]) if any(h.size for h in nohv) else np.zeros(0)
    if hv.size and nohv.size:
        L.append(f"HVAC air stream active: rmse {np.sqrt((hv ** 2).mean()):.3f} bias {hv.mean():+.3f}; inactive: rmse {np.sqrt((nohv ** 2).mean()):.3f} bias {nohv.mean():+.3f}")
    L.append("by quarter of the day (00-06, 06-12, 12-18, 18-24): " + ", ".join(f"{np.sqrt(hour_err[k] / max(hour_n[k], 1)):.3f}" for k in range(4)))
    gm = np.array(growth).mean(0)
    L.append("rmse by day 1/2/3: " + " / ".join(f"{x:.3f}" for x in gm) + ("  (growing: slow dynamics or a steady-state gain is off)" if gm[-1] > 1.5 * gm[0] else ""))
    # identifiability of theta
    th = rec.get("theta_train", {})
    if th:
        names = list(next(iter(th.values())).keys())
        spread = []
        for k in names:
            vals = np.array([t[k] for t in list(th.values()) + list(rec.get("val_theta", {}).values())])
            spread.append(f"{k} {vals.min():.3g}-{vals.max():.3g}")
        L.append("theta across scenarios (train + held-out): " + "; ".join(spread))
    L.append(f"complexity {rec['complexity']} AST nodes, {rec['n_params']} global + {rec['n_theta']} per-scenario constants")
    return "\n".join(L[:max_lines])
