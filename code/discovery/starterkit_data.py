"""Episodes from data generated ONLY with the starter kit's own commands.

Assumed layout (all written by unmodified starter-kit CLIs; see DATA.md):

  <scenarios_dir>/<building>/<weather>/config.gin, floor_plan.npy, zone_map.npy
        from `python -m smart_buildings.run_scenario_generator --output-dir <scenarios_dir>`
        (or the shipped smart_buildings/training_scenarios)
  <results_dir>/<building>/<weather>/<policy>/timeseries.csv, ahu_log.csv, reward_components.csv, summary.json
        from `python -m smart_buildings.evaluate --scenarios-dir <scenarios_dir> --results-dir <results_dir>
              --policies <keys> --steps <n>`

What those files hold per step: every room's temperature sensor, outside air, per-AHU supply
temperature, run command and total airflow, building occupancy, energy and carbon.  What they do
not hold: the per-room damper and reheat commands and the plant setpoints.  Those are recovered by
REPLAY: the simulator and every shipped policy are deterministic, and the policies read only zone
temperatures, the clock, the comfort-band sensors and the outside temperature - all present in
the files above.  So the unmodified policy class is re-run on observations rebuilt from the CSVs
and returns exactly the actions it took.  The replay is checked step by step against the saved
per-AHU airflow and run commands; an episode whose replay does not match is rejected.

The starter kit is only imported (environment metadata, policy classes, time-feature helpers);
nothing is stepped, generated or written there.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WC = HERE.parents[1]
KIT = Path(os.environ.get("STARTER_KIT", WC.parent / "smart-buildings-starter-kit"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(WC / ".numba_cache"))
sys.path.insert(0, str(HERE.parent))
from discovery.episodes import Episode, geometry  # noqa: E402

RHO_C_AIR = 1.2 * 1006.0
RHO_C_WATER = 1000.0 * 4180.0
VAV_MAX_FLOW, P_DESIGN, AHU_MAX_FLOW, REHEAT_KV, T_DISCHARGE_MAX = 2.0, 2000.0, 20.0, 0.108, 308.15
GAIN_W_M2, STANDBY = 10.0, 0.2
T_RESET, TW_RESET = 294.0, 340.0
TW_UP, TW_DOWN = 2.5, 0.5          # K per step: boiler tank ramp (0.5 and 0.1 K/min)


def _kit():
    if str(KIT) not in sys.path:
        sys.path.insert(0, str(KIT))
    cwd = os.getcwd()
    os.chdir(KIT)                   # gin configs use paths relative to the kit root
    try:
        from smart_buildings import evaluate as ev
        from smart_buildings.smart_control.utils import conversion_utils, regression_building_utils
    finally:
        os.chdir(cwd)
    return ev, conversion_utils, regression_building_utils


class ScenarioContext:
    """Everything static about one scenario, read through the starter kit (no stepping)."""

    def __init__(self, scenario_dir: Path):
        ev, self.cu, self.rbu = _kit()
        sdir = Path(scenario_dir).resolve()             # resolve before changing into the kit
        cwd = os.getcwd(); os.chdir(KIT)
        try:
            self.env = ev.load_hybrid_environment(sdir / "config.gin")
            self.meta = ev.create_building_meta_data(self.env, sdir)
            reset = self.env.reset()
            self.obs0 = np.asarray(reset[0] if isinstance(reset, tuple) else reset.observation, float)
        finally:
            os.chdir(cwd)
        self.ev = ev
        self.names = list(self.meta["observation_field_names"])
        self.ix = {n: i for i, n in enumerate(self.names)}
        sim = self.env.building.simulator
        self.vav_ids = list(sim._hvac._vavs.keys())                  # = timeseries zone_k order
        self.rooms = [f"room_{v.split('_')[1]}" for v in self.vav_ids]
        self.schedule = sim._hvac.schedule
        self.start = sim._current_timestamp
        self.dt = pd.Timedelta(sim._time_step_sec, unit="s")
        ahu = {k: list(v) for k, v in self.meta["ahu_zone_assignments"].items()}
        self.ahu_names = [a.device_id() for a in sim._hvac.air_handler._ahus]     # = ahu_log column order
        assert set(self.ahu_names) == set(ahu), (self.ahu_names, list(ahu))
        self.geo = geometry(self.meta["floor_plan"], self.meta["zone_map"], self.rooms, ahu, ahu_order=self.ahu_names)
        self.room_ahu = self.geo["ahu"]
        tuples = list(self.env._device_action_tuples)
        self.cont = [nm for dv, nm in tuples if "run_command" not in nm]
        self.disc = [(dv, nm) for dv, nm in tuples if "run_command" in nm]
        anm = self.meta["action_normalization_metadata"]
        self.lo = np.array([anm[nm]["min_native_value"] for nm in self.cont])
        self.hi = np.array([anm[nm]["max_native_value"] for nm in self.cont])
        self.za = {v: self.meta["zone_action_indices"][v] for v in self.vav_ids}
        ci = {nm: i for i, nm in enumerate(self.cont)}
        di = {nm: i for i, (dv, nm) in enumerate(self.disc)}
        self.ahu_temp_ix = np.array([ci[f"{a}_supply_air_temperature_setpoint"] for a in self.ahu_names])
        self.ahu_press_ix = np.array([ci[f"{a}_supply_air_static_pressure_setpoint"] for a in self.ahu_names])
        self.ahu_rec_ix = np.array([ci[f"{a}_recirculation_percentage_command"] for a in self.ahu_names])
        self.ahu_run_ix = np.array([di[f"{a}_supervisor_run_command"] for a in self.ahu_names])
        self.water_ix = ci["supply_water_temperature_setpoint"]
        self.dp_ix = ci["differential_pressure_setpoint"]
        self.hws_run_ix = [i for i, (dv, nm) in enumerate(self.disc) if dv == "hws"][0]
        self.n_hod = sum(n.startswith("hod_cos_") for n in self.names)
        self.n_dow = sum(n.startswith("dow_cos_") for n in self.names)

    # ------------------------------------------------------------------ observations for replay
    def rebuild_obs(self, k, ts_rows, temps, oat):
        """Observation the policy saw before step k (k = 0 is the reset observation)."""
        if k == 0:
            return self.obs0.copy()
        o = self.obs0.copy()
        for j, v in enumerate(self.vav_ids):
            o[self.ix[f"{v}_zone_air_temperature_sensor"]] = (temps[k - 1, j] - 295.0) / 5.0
        for a in self.ahu_names:
            o[self.ix[f"ahs_{a}_outside_air_temperature_sensor"]] = oat[k - 1]
        now = self.start + k * self.dt
        lo, hi = self.schedule.get_temperature_window(now - self.dt)  # sensors trail the schedule by one step
        for v in self.vav_ids:
            o[self.ix[f"{v}_zone_air_heating_temperature_setpoint_sensor"]] = lo
            o[self.ix[f"{v}_zone_air_cooling_temperature_setpoint_sensor"]] = hi
        for label, n, kind in (("hod", self.n_hod, self.cu.TimeIntervalEnum.HOUR_OF_DAY), ("dow", self.n_dow, self.cu.TimeIntervalEnum.DAY_OF_WEEK)):
            feats = self.rbu.expand_time_features(n, self.cu.get_radian_time(now, kind), label)
            for (a, b), val in feats.items():
                o[self.ix[f"{a}_{b}"]] = val
        o[self.ix["comfort_mode_now"]] = float(self.schedule.is_comfort_mode(now))
        o[self.ix["comfort_mode_soon"]] = float(self.schedule.is_comfort_mode(now + pd.Timedelta(60, unit="minute")))
        return o


def read_results(policy_dir: Path):
    ts = pd.read_csv(policy_dir / "timeseries.csv")
    ahu = pd.read_csv(policy_dir / "ahu_log.csv")
    rc = pd.read_csv(policy_dir / "reward_components.csv")
    zc = sorted([c for c in ts.columns if c.startswith("zone_") and c.endswith("_temp")], key=lambda c: int(c.split("_")[1]))
    return dict(temps=ts[zc].to_numpy(float), oat=ahu["outside_air"].to_numpy(float),
                flow=ahu[sorted([c for c in ahu.columns if c.startswith("ahu_flow_rates_")], key=lambda c: int(c.rsplit("_", 1)[1]))].to_numpy(float),
                run=ahu[sorted([c for c in ahu.columns if c.startswith("ahu_run_cmds_")], key=lambda c: int(c.rsplit("_", 1)[1]))].to_numpy(float),
                occupancy=rc["total_occupancy"].to_numpy(float) if "total_occupancy" in rc else ts["occupancy"].to_numpy(float),
                timestamps=ts["timestamp"].to_numpy())


def replay(ctx: ScenarioContext, policy_key: str, R: dict):
    """Re-run the unmodified policy on rebuilt observations; return native actions per step."""
    ev = ctx.ev
    policy = ev.initialize_policy(ev.POLICY_MAP[policy_key], ctx.meta, ctx.env.action_spec())
    n = len(R["temps"])
    C = np.zeros((n, len(ctx.cont))); Dd = np.zeros((n, len(ctx.disc)))
    for k in range(n):
        a = ev.validate_action(ev.get_policy_action(policy, ctx.rebuild_obs(k, R["timestamps"], R["temps"], R["oat"])), ctx.env.action_spec())
        c = np.clip(a["continuous_action"].astype(float), -1, 1)
        C[k] = ctx.lo + (c + 1) / 2 * (ctx.hi - ctx.lo)
        Dd[k] = a["discrete_action"]
    return C, Dd


def episode_inputs(ctx: ScenarioContext, R: dict, C: np.ndarray, Dd: np.ndarray):
    """Per-step room inputs from replayed native actions + logged sensors (same physics as room_inputs.py)."""
    m = ctx.meta; V = ctx.geo["V"]; K = len(ctx.ahu_names); area = ctx.geo["area"]
    n = len(R["temps"]) - 1                      # step k's result is seen in row k+1
    temps_start = np.vstack([np.full(V, T_RESET), R["temps"][:-1]])     # sensor rows are start-of-step values
    oat_step = np.concatenate([[ctx.obs0[ctx.ix[f"ahs_{ctx.ahu_names[0]}_outside_air_temperature_sensor"]]], R["oat"][:-1]])
    damper_ix = np.array([ctx.za[v]["supply_air_damper_percentage_command"] for v in ctx.vav_ids])
    reheat_ix = np.array([ctx.za[v]["reheat_valve_percentage_command"] for v in ctx.vav_ids])
    temp_ix, press_ix, rec_ix = ctx.ahu_temp_ix, ctx.ahu_press_ix, ctx.ahu_rec_ix
    water_ix, dp_ix, ahu_disc, hws_disc = ctx.water_ix, ctx.dp_ix, ctx.ahu_run_ix, ctx.hws_run_ix
    out = {k: np.zeros(n) for k in ("T_out", "occupants", "comfort", "hour_sin", "hour_cos", "dow_sin", "dow_cos")}
    for k in ("g", "T_supply", "Q_reheat", "Q_gain"):
        out[k] = np.zeros((n, V))
    check = np.zeros((n, K))
    tw = TW_RESET
    for k in range(n):
        on = R["run"][k] > 0.5            # logged AHU run state (the first command takes effect one step late)
        P = np.where(on, np.maximum(C[k, press_ix], 200.0), 0.0)
        rec = np.minimum(C[k, rec_ix], 0.7)
        sp = C[k, temp_ix]
        damper = np.maximum(C[k, damper_ix], 0.1); reheat = C[k, reheat_ix]
        fu = np.maximum(damper * VAV_MAX_FLOW * np.sqrt(P[ctx.room_ahu] / P_DESIGN), 1e-5)
        dem = np.bincount(ctx.room_ahu, fu, K)
        flow = fu * np.minimum(1.0, AHU_MAX_FLOW / np.maximum(dem, 1e-12))[ctx.room_ahu]
        check[k] = np.bincount(ctx.room_ahu, flow, K)
        g = flow * RHO_C_AIR
        Tret = np.bincount(ctx.room_ahu, area * temps_start[k], K) / np.bincount(ctx.room_ahu, area, K)
        mixed = rec * Tret + (1 - rec) * oat_step[k]
        Tsa = np.where(on & (mixed > sp), sp, mixed)[ctx.room_ahu]
        hws_on = Dd[k, hws_disc] > 0.5
        dp = C[k, dp_ix] if hws_on else 0.0
        qw = reheat * REHEAT_KV * np.sqrt(max(dp, 0.0)) / 3600.0
        eff = np.minimum(1.0, RHO_C_WATER * qw / np.maximum(g, 1e-12))
        gap = tw - Tsa; head = T_DISCHARGE_MAX - Tsa
        eff = np.where(gap > 0, np.where(head <= 0, 0.0, np.minimum(eff, head / np.where(gap > 0, gap, 1))), eff)
        out["Q_reheat"][k] = eff * g * gap
        start_ts = ctx.start + k * ctx.dt
        comfort = bool(ctx.schedule.is_comfort_mode(start_ts))
        out["Q_gain"][k] = GAIN_W_M2 * area * (1.0 if comfort else STANDBY)
        out["g"][k] = g; out["T_supply"][k] = Tsa; out["T_out"][k] = oat_step[k]
        out["occupants"][k] = R["occupancy"][k]; out["comfort"][k] = comfort
        o = ctx.rebuild_obs(k, None, R["temps"], R["oat"]) if k else ctx.obs0
        for a, b in (("hour_sin", "hod_sin_000"), ("hour_cos", "hod_cos_000"), ("dow_sin", "dow_sin_000"), ("dow_cos", "dow_cos_000")):
            out[a][k] = o[ctx.ix[b]]
        tw = tw + np.clip(C[k, water_ix] - tw, -TW_DOWN, TW_UP)     # tank temperature seen by the next step
    T_meas = R["temps"][1:n + 1]
    logged = R["flow"][:n]
    check = np.where(logged > 0, check, 0.0)       # an AHU that is off reports 0 flow (the 1e-5 floor is not logged)
    rel = np.abs(check - logged) / np.maximum(logged, 0.05)
    run_ok = float(np.mean((Dd[1:n][:, ahu_disc] > 0.5) == (R["run"][1:n] > 0.5)))   # step 0 excluded, see above
    return out, T_meas, dict(flow_max_rel_err=float(rel.max()), flow_median_rel_err=float(np.median(rel)), run_cmd_agreement=run_ok)


def load_starterkit_episode(scenarios_dir: Path, results_dir: Path, building: str, weather: str, policy: str, ctx_cache=None):
    sdir = Path(scenarios_dir) / building / weather
    key = str(sdir)
    ctx = (ctx_cache or {}).get(key) or ScenarioContext(sdir)
    if ctx_cache is not None:
        ctx_cache[key] = ctx
    R = read_results(Path(results_dir) / building / weather / policy)
    C, Dd = replay(ctx, policy, R)
    inputs, T_meas, chk = episode_inputs(ctx, R, C, Dd)
    ep = Episode(name=f"{building}_{weather}_{policy}", building=building, weather=weather, policy=policy,
                 scenario=f"{building}/{weather}", geo=ctx.geo, inputs=inputs, T_meas=T_meas)
    return ep, chk


def discover_runs(results_dir: Path, policies=None):
    """(building, weather, policy) for every results folder that has the three CSVs."""
    out = []
    for p in sorted(Path(results_dir).glob("*/*/*/timeseries.csv")):
        pol = p.parent.name
        if policies and pol not in policies:
            continue
        if (p.parent / "ahu_log.csv").exists() and (p.parent / "reward_components.csv").exists():
            out.append((p.parent.parent.parent.name, p.parent.parent.name, pol))
    return out


def load_all(scenarios_dir, results_dir, policies=None, cache_dir: Path | None = None, max_flow_err=0.02, verbose=True):
    """Load every run; cache derived Episodes; drop runs whose replay does not reproduce the logged airflow."""
    from discovery.episodes import _load_cached, _save_cached
    eps, report, ctxs = [], {}, {}
    for b, w, pol in discover_runs(results_dir, policies):
        name = f"{b}_{w}_{pol}"
        cp = cache_dir / f"episode_{name}.npz" if cache_dir else None
        src = Path(results_dir) / b / w / pol / "timeseries.csv"
        if cp is not None and cp.exists() and cp.stat().st_mtime > src.stat().st_mtime:
            eps.append(_load_cached(cp)); report[name] = json.loads((cp.with_suffix(".json")).read_text()); continue
        try:
            ep, chk = load_starterkit_episode(scenarios_dir, results_dir, b, w, pol, ctxs)
        except Exception as e:  # noqa: BLE001
            report[name] = {"status": "error", "reason": f"{type(e).__name__}: {e}"}
            if verbose:
                print(f"  {name}: ERROR {e}")
            continue
        ok = chk["flow_max_rel_err"] <= max_flow_err and chk["run_cmd_agreement"] >= 0.995
        chk["status"] = "ok" if ok else "replay mismatch"
        report[name] = chk
        if verbose:
            print(f"  {name}: {chk['status']} (airflow max rel err {chk['flow_max_rel_err']:.2e}, run agreement {chk['run_cmd_agreement']:.3f})")
        if ok:
            eps.append(ep)
            if cp is not None:
                cp.parent.mkdir(parents=True, exist_ok=True)
                _save_cached(ep, cp); cp.with_suffix(".json").write_text(json.dumps(chk))
    return eps, report
