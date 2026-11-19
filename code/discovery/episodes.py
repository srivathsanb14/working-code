"""Episodes: the data a candidate dynamics model is fitted and scored on.

Everything in an Episode is something a submitted policy can compute at run
time from the observation vector, its own previous action and the building
metadata.  Nothing comes from the YAML/gin config or from simulator internals,
so a model discovered on Episodes is deployable as it stands.

An Episode is built from a rollout written by code/room_model_data.py
(outputs/roomdata_<building>_<weather>_<policy>.npz).  Generating more rollouts
(more buildings, stations, behaviour policies) only means running that script
again; nothing here changes.
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import room_inputs as ri  # noqa: E402
import room_model as rm  # noqa: E402
from room_coeff_targets import features as _pair_features  # noqa: E402

DT = 300.0
T_RESET = 294.0


@dataclasses.dataclass
class Episode:
    name: str                 # e.g. building_train_8_atlanta_1_random
    building: str
    weather: str
    policy: str
    scenario: str             # building/weather: the unit that gets its own theta
    geo: dict                 # per-room and per-pair geometry (see geometry())
    inputs: dict              # per-step inputs (see build_inputs())
    T_meas: np.ndarray        # (n, V) measured room temperature after each step
    T0: float = T_RESET       # every episode starts from the simulator's reset state

    @property
    def n(self):
        return self.T_meas.shape[0]

    @property
    def V(self):
        return self.T_meas.shape[1]


def geometry(floor_plan, zone_map, rooms, ahu_rooms):
    """Geometry a policy can compute from the metadata arrays.

    Returns a dict of numpy arrays, rooms in observation order (room_1..room_V):
      V                  number of rooms
      area   (V)         floor area, m2
      facade_faces (V)   20-cm faces between the room's air and exterior wall
      wall_cells (V,V)   partition cells touching both rooms (shared wall length / 0.2 m)
      door_cells (V,V)   door cells touching both rooms (a door is usually 2 cells)
      n_doors (V,V)      number of doors between the two rooms
      bbox_x, bbox_y (V) bounding-box sides of the room, m
      ahu (V)            index of the air handler serving the room
      floor (V)          floor index
    """
    G = rm.grid_geometry(floor_plan, zone_map)
    f = _pair_features(G)
    V = int(G["nroom"])
    names = list(rooms)
    pos = {r: k for k, r in enumerate(names)}
    ahu = np.zeros(V, int)
    for a, ah in enumerate(sorted(ahu_rooms)):
        for r in ahu_rooms[ah]:
            ahu[pos[r]] = a
    lab = G["lab"]
    fl, nf = ndimage.label(G["cls"] != rm.EXT_SPACE)
    floor = np.array([int(np.bincount(fl[lab == i + 1]).argmax()) - 1 for i in range(V)])
    return dict(V=V, area=f["area"], facade_faces=f["F"], wall_cells=f["P"], door_cells=f["D"], n_doors=f["ndoor"],
                bbox_x=f["Lx"], bbox_y=f["Ly"], ahu=ahu, floor=floor)


def build_inputs(D, geo):
    """Per-step inputs, rebuilt from observations (room_inputs.InputModel).

    Returns a dict (n = number of scored steps, V rooms):
      T_out    (n)      outside temperature, K
      g        (n,V)    VAV air-stream capacity rate rho*c*flow, W/K
      T_supply (n,V)    supply-air temperature reaching the room, K
      Q_reheat (n,V)    reheat coil heat, W
      Q_gain   (n,V)    internal gain from lighting and equipment, W (known schedule)
      occupants(n)      building-total occupant count (per-room split unknown)
      comfort  (n)      1 in comfort mode
      hour_sin, hour_cos, dow_sin, dow_cos (n)
    The heat a room receives in step t is g*(T_supply - T) + Q_reheat + Q_gain + 70 W per occupant in it.
    """
    rooms = list(D["rooms"])
    IM = ri.InputModel(list(D["names"]), rooms, json.loads(str(D["ahu_json"])), geo["area"])
    O = D["obs"]
    n = len(O) - 2                           # last step has no measurement of its result
    V = len(rooms)
    out = {k: np.zeros(n) for k in ("T_out", "occupants", "comfort", "hour_sin", "hour_cos", "dow_sin", "dow_cos")}
    for k in ("g", "T_supply", "Q_reheat", "Q_gain"):
        out[k] = np.zeros((n, V))
    ix = {nm: i for i, nm in enumerate(D["names"])}
    for t in range(n):
        Tstart = IM.zone_T(O[t + 1])
        Tout, g, Ts, Q, parts = IM.step_inputs(O[t], O[t + 1], Tstart)
        out["T_out"][t] = Tout; out["g"][t] = g; out["T_supply"][t] = Ts
        out["Q_reheat"][t] = parts["Qre"]; out["Q_gain"][t] = parts["Qgain"]
        out["occupants"][t] = O[t + 1][IM.occ]; out["comfort"][t] = O[t][IM.comfort]
        out["hour_sin"][t] = O[t][ix["hod_sin_000"]]; out["hour_cos"][t] = O[t][ix["hod_cos_000"]]
        out["dow_sin"][t] = O[t][ix["dow_sin_000"]]; out["dow_cos"][t] = O[t][ix["dow_cos_000"]]
    T_meas = np.array([IM.zone_T(O[t + 2]) for t in range(n)])
    return out, T_meas


def load_episode(path: Path) -> Episode:
    D = np.load(path)
    stem = path.stem.replace("roomdata_", "")
    parts = stem.split("_")
    building = "_".join(parts[:3])
    policy = parts[-1]
    weather = "_".join(parts[3:-1])
    geo = geometry(D["floor_plan"], D["zone_map"], list(D["rooms"]), json.loads(str(D["ahu_json"])))
    inputs, T_meas = build_inputs(D, geo)
    return Episode(name=stem, building=building, weather=weather, policy=policy, scenario=f"{building}/{weather}",
                   geo=geo, inputs=inputs, T_meas=T_meas)


def load_episodes(paths, cache_dir: Path | None = None):
    """Load rollouts, caching the derived Episode as .npz next to the run."""
    eps = []
    for p in sorted(map(Path, paths)):
        if cache_dir is not None:
            cp = cache_dir / f"episode_{p.stem}.npz"
            if cp.exists() and cp.stat().st_mtime > p.stat().st_mtime:
                eps.append(_load_cached(cp)); continue
        ep = load_episode(p)
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            _save_cached(ep, cache_dir / f"episode_{p.stem}.npz")
        eps.append(ep)
    return eps


def _save_cached(ep, path):
    flat = {f"geo__{k}": np.asarray(v) for k, v in ep.geo.items()}
    flat.update({f"in__{k}": v for k, v in ep.inputs.items()})
    np.savez_compressed(path, T_meas=ep.T_meas, meta=np.array(json.dumps(dict(name=ep.name, building=ep.building, weather=ep.weather,
                                                                                  policy=ep.policy, scenario=ep.scenario, T0=ep.T0))), **flat)


def _load_cached(path):
    Z = np.load(path)
    m = json.loads(str(Z["meta"]))
    geo = {k[5:]: (int(Z[k]) if k == "geo__V" else Z[k]) for k in Z.files if k.startswith("geo__")}
    inputs = {k[4:]: Z[k] for k in Z.files if k.startswith("in__")}
    return Episode(name=m["name"], building=m["building"], weather=m["weather"], policy=m["policy"], scenario=m["scenario"],
                   geo=geo, inputs=inputs, T_meas=Z["T_meas"], T0=m["T0"])
