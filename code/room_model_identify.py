"""Inference-only test of the room model: inputs rebuilt from observations
(room_inputs), the six building scalars theta unknown.

For every rollout:
  * true theta            - model error alone (structure + occupancy split)
  * prior theta           - geometric centre of the generator's ranges, no data
  * identified theta      - fitted on the first 24 h by open-loop simulation
                            error from the known reset state, then tested
                            open loop on hours 24-72
Error is against the zone temperatures the policy observes (one step stale,
so prediction of x[n+1] is scored against obs[n+2]).
Usage: <venv>/python <this file> [glob]
"""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
HERE = Path(__file__).resolve().parent; OUT = HERE.parent / "outputs"
sys.path.insert(0, str(HERE))
import room_model as rm, room_inputs as ri
from room_model_eval import theta_from
from room_coeff_targets import RANGES

KEYS = ["k_iw", "rc_iw", "k_ext", "rc_ext", "rc_mass", "h"]
LO = np.log([RANGES[k][0] for k in KEYS]); HI = np.log([RANGES[k][1] for k in KEYS])
PRIOR = dict(zip(KEYS, np.exp((LO + HI) / 2)))
SPREAD = dict(zip(("c", "a", "af"), json.loads((OUT / "room_spread_fit.json").read_text())["c_a_af"]))
DAY = 288


def obs_inputs(D):
    rooms = list(D["rooms"]); ahu = json.loads(str(D["ahu_json"]))
    G = rm.grid_geometry(D["floor_plan"], D["zone_map"])
    area = np.bincount(G["lab"][G["lab"] > 0] - 1) * rm.CV ** 2
    IM = ri.InputModel(list(D["names"]), rooms, ahu, area)
    O = D["obs"]; n = len(O) - 1
    Tout = np.zeros(n); g = np.zeros((n, len(rooms))); Ts = np.zeros_like(g); Q = np.zeros_like(g)
    for t in range(n):
        Tstart = IM.zone_T(O[t + 1])            # start-of-step temperatures, as the policy sees them
        Tout[t], g[t], Ts[t], Q[t], _ = IM.step_inputs(O[t], O[t + 1], Tstart)
    meas = np.array([IM.zone_T(O[t + 2]) for t in range(n - 1)])   # x[t+1] as observed
    return G, Tout, g, Ts, Q, meas


def run(G, cid, nodes, th, Tout, g, Ts, Q):
    M = rm.cluster_model(G, th, cid, nodes, spread=SPREAD)
    C, L, b, _, _ = rm.full_network(M)
    return rm.simulate_fast(np.diag(C), L, b, np.zeros(len(C)), M["V"], Tout, g, Ts, Q, 294.0)


def rmse(P, meas, a, b_):
    e = P[a:b_] - meas[a:b_]
    return float(np.sqrt((e ** 2).mean()))


def main():
    pat = sys.argv[1] if len(sys.argv) > 1 else "roomdata_*.npz"
    res = {}
    for f in sorted(OUT.glob(pat)):
        D = np.load(f)
        if "ahu_json" not in D.files:
            continue
        t0 = time.time()
        G, Tout, g, Ts, Q, meas = obs_inputs(D)
        cid, nodes = rm.cluster_ids(G)
        n = len(meas)
        sim = lambda th: run(G, cid, nodes, th, Tout, g, Ts, Q)[:n]
        tt = theta_from(D)
        r = {"V": int(G["nroom"])}
        Ptrue = sim(tt); Pprior = sim(PRIOR)
        r["true_theta"] = dict(day1=rmse(Ptrue, meas, 0, DAY), days2_3=rmse(Ptrue, meas, DAY, n))
        r["prior_theta"] = dict(day1=rmse(Pprior, meas, 0, DAY), days2_3=rmse(Pprior, meas, DAY, n))
        x0 = (LO + HI) / 2
        fit = least_squares(lambda z: (sim(dict(zip(KEYS, np.exp(z))))[:DAY] - meas[:DAY]).ravel(), x0, bounds=(LO - 0.3, HI + 0.3),
                            diff_step=0.02, max_nfev=60, x_scale=1.0)
        th_hat = dict(zip(KEYS, np.exp(fit.x)))
        Pid = sim(th_hat)
        r["identified_theta"] = dict(day1=rmse(Pid, meas, 0, DAY), days2_3=rmse(Pid, meas, DAY, n), nfev=int(fit.nfev))
        r["theta_ratio_identified_over_true"] = {k: round(float(th_hat[k] / tt[k]), 3) for k in KEYS}
        J = fit.jac; s = np.linalg.svd(J, compute_uv=False)
        r["jacobian_singular_values"] = [float(x) for x in s]
        r["persistence_baseline_days2_3"] = float(np.sqrt(((meas[DAY + 1:n] - meas[DAY:n - 1]) ** 2).mean()))
        r["measured_std_K"] = float(meas.std()); r["sec"] = round(time.time() - t0, 1)
        res[f.stem] = r
        print(f.stem.replace("roomdata_building_", ""), json.dumps({k: v for k, v in r.items() if k != "jacobian_singular_values"}))
    (OUT / "room_model_identify.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
