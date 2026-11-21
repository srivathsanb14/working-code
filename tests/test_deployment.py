"""Deployment parity: discovery may only use what a policy has on an unseen building at Kaggle test time."""
import numpy as np
import pytest

from discovery import closedform as cf
from discovery import contract
from discovery import selftest as st

ALLOWED_GEO = {"V", "ahu", "area", "aspect", "bbox_x", "bbox_y", "boundary_faces", "door_cells", "facade_bottom", "facade_faces",
               "facade_left", "facade_right", "facade_sides", "facade_top", "floor", "n_doors", "n_neighbours", "wall_cells"}
ALLOWED_INPUTS = {"T_out", "g", "T_supply", "Q_reheat", "Q_gain", "occupants", "comfort", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "T_water"}


def test_candidates_only_see_metadata_geometry_and_observable_inputs(toy_episodes):
    e = toy_episodes[0]
    assert set(e.geo) <= ALLOWED_GEO                 # all computed from floor_plan + zone_map + AHU assignment (metadata)
    assert set(e.inputs) <= ALLOWED_INPUTS           # observations, own commands through the kit's equipment laws, clock


@pytest.mark.parametrize("body", ['if geo["V"] == 85:\n        return T', 'k = 1 if len(geo["area"]) in (30, 85) else 2',
                                  'k = 1 if T.shape[1] != 39 else 2'])
def test_building_specific_branching_is_rejected(body):
    src = f'HISTORY = 1\ndef step(coef, geo, hist, u, U):\n    T = hist["T"][:, -1]\n    {body}\n    return T\n'
    with pytest.raises(contract.ContractError, match="building-specific"):
        contract.load_closedform(src)


def test_test_time_score_does_not_peek_at_future_weather(toy_episodes, U):
    """The planted truth is exact with the true future inputs, but not when the future outside temperature and occupancy
    are unknown: the score must use the latter."""
    r = cf.evaluate(st.TRUTH_SRC, toy_episodes[:1], toy_episodes[1:], U, every=24)
    assert r["val_3h_mean"] < 1e-9 and r["val_3h_deploy_mean"] > 1e-3
    assert abs(r["score"] - r["complexity"] * 1e-4 - r["val_3h_deploy_mean"]) < 1e-9


def test_freeze_holds_exogenous_inputs(toy_episodes, U):
    mod, _ = contract.load_closedform(st.TRUTH_SRC)
    prep = cf.Prepared(toy_episodes[0], mod.HISTORY)
    a = cf.rollout(mod, mod.prepare(prep.ep.geo, U), prep, U, np.array([100]), 36)
    b = cf.rollout(mod, mod.prepare(prep.ep.geo, U), prep, U, np.array([100]), 36, freeze=cf.EXOGENOUS)
    assert np.allclose(a[:, 0], b[:, 0]) and not np.allclose(a[:, -1], b[:, -1])
