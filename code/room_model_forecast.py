"""k-step forecast error of the room model from a Kalman-filtered state.

theta is the one identified on day 1 (room_model_identify.json).  A Kalman
filter runs on the full lumped network (air, slab, wall, facade and door nodes)
using only the observed zone temperatures (one step stale) and the inputs rebuilt
from observations.  From every hour of days 2-3 the filtered state is rolled
forward open loop (with the actual future inputs, so this scores the building
model, not weather or occupancy forecasting) and compared with the observed
temperatures 1, 3 and 6 hours later.  Persistence (no change) is the baseline.
"""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; OUT = HERE.parent / "outputs"
sys.path.insert(0, str(HERE))
import room_model as rm
from room_model_identify import obs_inputs, SPREAD, KEYS
from room_model_eval import theta_from

H = (12, 36, 72); DAY = 288


def main():
    ident = json.loads((OUT / "room_model_identify.json").read_text())
    res = {}
    for stem, rec in ident.items():
        D = np.load(OUT / f"{stem}.npz")
        G, Tout, g, Ts, Q, meas = obs_inputs(D)
        tt = theta_from(D); th = {k: tt[k] * rec["theta_ratio_identified_over_true"][k] for k in KEYS}
        M = rm.cluster_model(G, th, spread=SPREAD); C, L, b, _, _ = rm.full_network(M)
        V = M["V"]; N = len(C); dt = rm.DT
        Cdt = C / dt; A0inv = np.linalg.inv(np.diag(Cdt) + L); W = A0inv[:, :V]; S = W[:V, :]
        I = np.eye(V)

        def Minv(gt):                       # (A0 + U G U^T)^-1
            K = np.linalg.solve(I + gt[:, None] * S, gt[:, None] * W.T)
            return A0inv - W @ K

        def step(x, t, Mi=None):
            Mi = Minv(g[t]) if Mi is None else Mi
            rhs = Cdt * x + b * Tout[t]; rhs[:V] += Q[t] + g[t] * Ts[t]
            return Mi @ rhs, Mi

        Qn = np.r_[np.full(V, 0.05 ** 2), np.full(N - V, 0.01 ** 2)]; Rn = 0.01 ** 2
        x = np.full(N, 294.0); P = np.diag(np.full(N, 1e-6))
        n = len(meas); filt = np.zeros((n, N))
        for t in range(n):                  # predict x[t+1], then update with its (stale) measurement meas[t]
            x, Mi = step(x, t)
            A = Mi * Cdt[None, :]
            P = A @ P @ A.T + np.diag(Qn)
            Pyy = P[:V, :V] + Rn * I
            Kg = np.linalg.solve(Pyy, P[:V, :]).T
            x = x + Kg @ (meas[t] - x[:V]); P = P - Kg @ P[:V, :]
            filt[t] = x
        errs = {h: [] for h in H}; pers = {h: [] for h in H}
        for t0 in range(DAY, n - max(H), 12):
            xx = filt[t0].copy()
            for k in range(1, max(H) + 1):
                xx, _ = step(xx, t0 + k)
                if k in H:
                    errs[k].append(xx[:V] - meas[t0 + k]); pers[k].append(meas[t0] - meas[t0 + k])
        r = {f"{h // 12}h": dict(model=float(np.sqrt(np.mean(np.square(errs[h])))), persistence=float(np.sqrt(np.mean(np.square(pers[h]))))) for h in H}
        res[stem] = r
        print(stem.replace("roomdata_building_", ""), " ".join(f"{k}: {v['model']:.3f} vs {v['persistence']:.3f}" for k, v in r.items()))
    (OUT / "room_model_forecast.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
