"""Check the cost model (control/costs.py) against the kit's logged costs (backlog K4).

    python code/control/validate_costs.py --results <kit results dir> --runs building_train_2/atlanta_3/pid_water ...

For each run: replay it (discovery/starterkit_data), rebuild the equipment quantities of every step, estimate electric
and gas power from commands, and compare with the power implied by the kit's reward_components.csv (cost / price).
Also compares the normalized energy and carbon terms.  Prints correlation and median ratio per run.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from common import KIT  # noqa: E402
from control import costs  # noqa: E402
from discovery import starterkit_data as skd  # noqa: E402


def check(results, run):
    b, w, pol = run.split("/")
    ep, chk, F = skd.load_starterkit_episode(KIT / "smart_buildings" / "training_scenarios", results, b, w, pol, {}, full=True)
    ctx, act, R = F["ctx"], F["act"], F["R"]
    rc = pd.read_csv(results / b / w / pol / "reward_components.csv")
    n = min(ep.n, len(rc)); K = len(ctx.ahu_names); area = ctx.geo["area"]; ra = ctx.room_ahu
    temps = R["temps"]
    est_e, est_g, kit_e, kit_g, nE, nC, kE, kC = [], [], [], [], [], [], [], []
    for k in range(n):
        on = act["ahu_on"][k] > 0.5
        Tret = np.bincount(ra, area * temps[k], K) / np.bincount(ra, area, K)
        rec = act["ahu_recirculation"][k]; mixed = rec * Tret + (1 - rec) * ep.inputs["T_out"][k]
        sp = act["ahu_sat_setpoint"][k]; Tsa = np.where(on & (mixed > sp), sp, mixed)
        dp = act["hws_dp"][k]
        aux = dict(ahu_flow=act["ahu_flow"][k], ahu_pressure=act["ahu_pressure"][k], recirculation=rec, mixed=mixed, T_supply_ahu=Tsa,
                   water_flow=float((act["reheat"][k] * 0.108 * np.sqrt(max(dp, 0.0)) / 3600.0).sum()), pump_dp=dp,
                   Q_reheat_total=float(ep.inputs["Q_reheat"][k].sum()), T_water=act["T_water"][k],
                   T_water_setpoint=act["hws_water_setpoint"][k], hws_on=bool(act["hws_on"][k]))
        p = costs.power(aux)
        t = pd.Timestamp(rc["timestamp"].iloc[k]) - pd.Timedelta(minutes=5)
        e, ec, g, gc = costs.prices(t.hour, t.dayofweek < 5, t.month)
        est_e.append(p["electric"]); est_g.append(p["gas"])
        kit_e.append(rc["electricity_energy_cost"].iloc[k] / (e * 300)); kit_g.append(rc["natural_gas_energy_cost"].iloc[k] / (g * 300))
        E, C = costs.normalized_energy_carbon(p["electric"], p["gas"], K, t.hour, t.dayofweek < 5, t.month)
        nE.append(E); nC.append(C); kE.append(rc["normalized_energy_cost"].iloc[k]); kC.append(rc["normalized_carbon_emission"].iloc[k])
    out = {}
    for name, a, b_ in (("electric W", est_e, kit_e), ("gas W", est_g, kit_g), ("normalized energy", nE, kE), ("normalized carbon", nC, kC)):
        a, b_ = np.asarray(a), np.asarray(b_); m = (a > 0) | (b_ > 0)
        r = float(np.corrcoef(a[m], b_[m])[0, 1]) if m.sum() > 2 and a[m].std() > 0 and b_[m].std() > 0 else float("nan")
        ratio = float(np.median(a[m & (b_ > 0)] / b_[m & (b_ > 0)])) if (m & (b_ > 0)).any() else float("nan")
        out[name] = (r, ratio, float(a.mean()), float(b_.mean()))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    a = ap.parse_args()
    for run in a.runs:
        print(run)
        for k, (r, ratio, ma, mb) in check(a.results, run).items():
            print(f"  {k:<18} corr {r:6.3f}   median estimate/kit {ratio:6.3f}   mean estimate {ma:10.4g}   mean kit {mb:10.4g}")


if __name__ == "__main__":
    main()
