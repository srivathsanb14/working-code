"""Splits computed from the data, Pareto selection, and the spatial / temporal / term analysis."""
import dataclasses
from pathlib import Path

import numpy as np

from discovery import closedform as cf
from discovery import coefform, pareto, splits

D = Path(__file__).resolve().parents[1] / "code" / "discovery"


def _fake(b, w, To, geo):
    @dataclasses.dataclass
    class E:
        scenario: str
        inputs: dict
        geo: dict
    return E(f"{b}/{w}", {"T_out": np.asarray(To, float)}, geo)


def test_split_tiers_follow_the_data(toy_geo):
    rng = np.random.default_rng(0)
    clim = {"mild": 15, "warm": 25, "hot": 30, "polar": -20}
    eps = []
    for bi in range(6):
        g = dict(toy_geo); g["area"] = toy_geo["area"] * (4.0 if bi == 5 else 1.0 + 0.05 * bi)
        for si, (st, m) in enumerate(clim.items()):
            eps.append(_fake(f"b{bi}", f"{st}_{bi + si}", m + rng.normal(0, 3, 2000), g))
    sp = splits.split_from_episodes(eps)
    assert sp["_why"]["hard_station"] == "polar"                 # farthest in Wasserstein distance
    assert "b5" in sp["_why"]["held_out_buildings"]               # the atypical floor plan
    hb = set(sp["_why"]["held_out_buildings"])
    val_b = {s.split("/")[0] for s in sp["val"]}; val_st = {s.split("/")[1].rsplit("_", 1)[0] for s in sp["val"]}
    assert not (val_b & hb) and "polar" not in val_st
    assert all(s.split("/")[0] in hb and s.split("/")[1].startswith("polar") for s in sp["test_hard"])
    assert all(s.split("/")[0] in hb and not s.split("/")[1].startswith("polar") for s in sp["test_medium"])
    for s in sp["test_easy"]:                                     # seen building and station, unseen window
        b, w = s.split("/"); st = w.rsplit("_", 1)[0]
        assert b in val_b and st in val_st and s not in sp["val"]
    assert sp["test_easy"]


def test_pareto_fronts_and_selection():
    R = [{"id": "a", "objectives": {"x": 1, "y": 5}}, {"id": "b", "objectives": {"x": 5, "y": 1}},
         {"id": "c", "objectives": {"x": 3, "y": 3}}, {"id": "d", "objectives": {"x": 4, "y": 4}}]
    assert {r["id"] for r in pareto.front(R, ["x", "y"])} == {"a", "b", "c"}
    assert "d" not in {r["id"] for r in pareto.select(R, 3, ["x", "y"])}


def test_term_contributions_reproduce_the_model(toy_episodes, U):
    mod, _ = __import__("discovery.contract", fromlist=["x"]).load_closedform((D / "seeds_cf" / "seed_cf_network.py").read_text())
    ep = toy_episodes[1]
    cm = coefform.make_model(mod, ep.geo, U); prep = cf.Prepared(ep, mod.HISTORY)
    S, nets, idx = cf.filter_coef(mod, cm, prep, U)
    ts = np.arange(40, 80)
    C, R, I = cf.term_contributions(cm, prep, ts)
    pred = np.concatenate([cm.predict(nets[idx[t]], S[t][None], cf._u_at(prep, np.array([t])))[:, :ep.V] for t in ts])
    assert np.allclose((prep.X[ts] + C.reshape(len(ts), ep.V, -1).sum(-1)), pred, atol=1e-9)


def test_analysis_has_spatial_temporal_and_terms(toy_episodes, U):
    r = cf.evaluate((D / "seeds_cf" / "seed_cf_network.py").read_text(), toy_episodes[1:], U, every=24, screen_factor=1e9)
    assert r["status"] == "ok" and set(cf.DEFAULT_OBJECTIVES) <= set(r["objectives"])
    an = cf.analysis(r, toy_episodes[1:])
    for part in ("TEMPORAL", "SPATIAL", "TERMS", "error vs horizon", "correlation of a room's bias"):
        assert part in an["text"]
    assert "air" in an["terms"]["all rooms"] and "_r2" in an["terms"]["all rooms"]
