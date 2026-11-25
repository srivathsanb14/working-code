"""Episodes: the data a candidate dynamics model is fitted and scored on.

Everything in an Episode is something a submitted policy can compute at run
time from the observation vector, its own previous action and the building
metadata.  Nothing comes from the YAML/gin config or from simulator internals,
so a model discovered on Episodes is deployable as it stands.

Episodes are built by starterkit_data.py from the starter kit's own outputs
(evaluate.py results folders and the scenario folders); this module defines the
container, the geometry a policy can compute from the metadata arrays, and the
on-disk cache.
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
import room_model as rm  # noqa: E402
from room_coeff_targets import features as _pair_features  # noqa: E402

DT = 300.0
T_RESET = 294.0


@dataclasses.dataclass
class Episode:
    name: str                 # e.g. building_train_8_atlanta_1_pid_water
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


def natural_key(s):
    import re
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(s))]


def geometry(floor_plan, zone_map, rooms, ahu_rooms, ahu_order=None):
    """Geometry a policy can compute from the metadata arrays.

    `rooms` fixes the order of every per-room axis (e.g. the VAV order of the
    observation and of the starter kit's CSVs: room_1, room_10, room_11, ...).
    Returns a dict of numpy arrays:
      V                  number of rooms
      area   (V)         floor area, m2
      facade_faces (V)   20-cm faces between the room's air and exterior wall
      wall_cells (V,V)   partition cells touching both rooms (shared wall length / 0.2 m)
      door_cells (V,V)   door cells touching both rooms (a door is usually 2 cells)
      n_doors (V,V)      number of doors between the two rooms
      bbox_x, bbox_y (V) bounding-box sides of the room, m
      ahu (V)            index of the air handler serving the room (in ahu_order, default natural sort)
      floor (V)          floor index
      facade_top/bottom/left/right (V)  exterior-wall faces by side of the drawing; facade_sides (V) sides with >= 2 m of exterior wall (2+ = corner)
      boundary_faces (V) all 0.2 m faces on the room boundary;  n_neighbours (V);  aspect (V) long / short bbox side
    """
    G = rm.grid_geometry(floor_plan, zone_map)
    f = _pair_features(G)
    V = int(G["nroom"])
    names = list(rooms)
    pos = {r: k for k, r in enumerate(names)}
    ahu = np.zeros(V, int)
    for a, ah in enumerate(ahu_order or sorted(ahu_rooms, key=natural_key)):
        for r in ahu_rooms[ah]:
            ahu[pos[r]] = a
    lab = G["lab"]
    fl, nf = ndimage.label(G["cls"] != rm.EXT_SPACE)
    floor = np.array([int(np.bincount(fl[lab == i + 1]).argmax()) - 1 for i in range(V)])
    # feature arrays come in label order (room_1, room_2, ...); permute to the requested order
    perm = np.array([int(str(r).split("_")[-1]) - 1 for r in names])
    if sorted(perm.tolist()) != list(range(V)):
        raise ValueError(f"rooms {names[:5]}... do not match the {V} rooms of the floor plan")
    pm = np.ix_(perm, perm)
    x = extra_geometry(G)
    geo = dict(V=V, area=f["area"][perm], facade_faces=f["F"][perm], wall_cells=f["P"][pm], door_cells=f["D"][pm],
               n_doors=f["ndoor"][pm], bbox_x=f["Lx"][perm], bbox_y=f["Ly"][perm], ahu=ahu, floor=floor[perm])
    geo.update({k: v[perm] for k, v in x.items()})
    geo["n_neighbours"] = ((geo["wall_cells"] + geo["door_cells"]) > 0).sum(1)
    geo["aspect"] = np.maximum(geo["bbox_x"], geo["bbox_y"]) / np.maximum(np.minimum(geo["bbox_x"], geo["bbox_y"]), 1e-9)
    return geo


def extra_geometry(G):
    """Extra per-room quantities from the grid (label order): exterior-wall faces by side of the drawing (top =
    row 0; the compass is not given), the number of sides with exterior wall (2+ = corner room), and boundary faces."""
    lab, cls = G["lab"], G["cls"]; V = int(G["nroom"])
    ext = cls == rm.SHELL_OR_EXT
    out = {}
    for name, sh in (("facade_top", (1, 0)), ("facade_bottom", (-1, 0)), ("facade_left", (0, 1)), ("facade_right", (0, -1))):
        nb = np.roll(ext, sh, axis=(0, 1)); m = (lab > 0) & nb
        out[name] = np.bincount(lab[m] - 1, minlength=V).astype(float)
    out["facade_sides"] = sum((out[k] >= 10).astype(float) for k in ("facade_top", "facade_bottom", "facade_left", "facade_right"))   # a side counts from 2 m of exterior wall
    room = lab > 0; per = np.zeros(V)
    for sh in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nb = np.roll(lab, sh, axis=(0, 1)); m = room & (nb != lab)
        per += np.bincount(lab[m] - 1, minlength=V)
    out["boundary_faces"] = per
    return out


def _save_cached(ep, path):
    flat = {f"geo__{k}": np.asarray(v) for k, v in ep.geo.items()}
    flat.update({f"in__{k}": v for k, v in ep.inputs.items()})
    np.savez_compressed(path, T_meas=ep.T_meas, meta=np.array(json.dumps(dict(name=ep.name, building=ep.building, weather=ep.weather,
                                                                                  policy=ep.policy, scenario=ep.scenario, T0=ep.T0))), **flat)


def load_light(path):
    """Metadata, geometry and the outside temperature of one episode file, without the large per-step, per-room arrays
    (np.load reads each member of the .npz on demand).  Enough for the split; a few hundred KB instead of tens of MB."""
    Z = np.load(path)
    m = json.loads(str(Z["meta"]))
    geo = {k[5:]: (int(Z[k]) if k == "geo__V" else Z[k]) for k in Z.files if k.startswith("geo__")}
    return Episode(name=m["name"], building=m["building"], weather=m["weather"], policy=m["policy"], scenario=m["scenario"],
                   geo=geo, inputs={"T_out": Z["in__T_out"]}, T_meas=np.empty((0, int(geo["V"]))), T0=m["T0"])


def _load_cached(path):
    Z = np.load(path)
    m = json.loads(str(Z["meta"]))
    geo = {k[5:]: (int(Z[k]) if k == "geo__V" else Z[k]) for k in Z.files if k.startswith("geo__")}
    inputs = {k[4:]: Z[k] for k in Z.files if k.startswith("in__")}
    return Episode(name=m["name"], building=m["building"], weather=m["weather"], policy=m["policy"], scenario=m["scenario"],
                   geo=geo, inputs=inputs, T_meas=Z["T_meas"], T0=m["T0"])
