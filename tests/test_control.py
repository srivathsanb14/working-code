"""Controller building blocks: cost model, schedule decoding, occupancy forecast, stateful candidates."""
from pathlib import Path

import numpy as np

from control import costs
from control.occupancy import OccupancyForecaster, prior_profile
from control.schedule import Schedule

SEEDS = Path(__file__).resolve().parents[1] / "code" / "discovery" / "seeds_cf"


def test_shortfall_matches_the_kit_definition():
    assert np.allclose(costs.shortfall(np.array([0.0, 2.5, 5.0, 10.0])), [0.0, 0.5, 1.0, 2.0])
    assert costs.comfort_regret(np.array([290.0, 295.0]), 294, 297, np.array([1, 1])) == -0.4
    assert costs.comfort_regret(np.array([280.0]), 294, 297, np.array([0])) == 0.0


def test_energy_normalisation_is_one_at_the_nominal_maximum():
    E, C = costs.normalized_energy_carbon(80000.0 * 3, 205000.0 * 3, 3, 10, True, 1)
    assert abs(E - 1) < 1e-9 and abs(C - 1) < 1e-9


def test_schedule_decodes_time_and_comfort():
    names = ["hod_sin_000", "hod_cos_000", "dow_sin_000", "dow_cos_000"]
    ix = {n: i for i, n in enumerate(names)}
    s = Schedule(dict(morning_start_hour=6, evening_start_hour=19, comfort_temp_window=(294, 297), eco_temp_window=(289, 298)), ix)
    th, td = 2 * np.pi * 8 / 24, 2 * np.pi * 2 / 7                    # Wednesday 08:00
    obs = np.array([np.sin(th), np.cos(th), np.sin(td), np.cos(td)])
    h, d = s.time(obs); assert abs(h - 8) < 1e-6 and d == 2 and s.is_comfort(h, d)
    h, d = s.time(obs, 12 * 17); assert d == 3 and abs(h - 1) < 1e-6 and not s.is_comfort(h, d)   # 17 h later: Thu 01:00


def test_occupancy_learns_the_profile():
    f = OccupancyForecaster(10)
    true = prior_profile(30)
    for day in range(8):
        for slot in range(288):
            f.observe(true[slot], slot, True)
    assert np.abs(f.total(0, True, 288) - true).max() < 1.0
    assert np.isclose(f.rooms(150, True, 1).sum(), f.total(150, True, 1)[0])


def test_stateful_seed_scores_and_contract(toy_episodes, U):
    from discovery import closedform as cf
    src = (SEEDS / "seed_cf_network.py").read_text()
    r = cf.evaluate(src, toy_episodes[1:], U, every=24, screen_factor=1e9)
    assert r["status"] == "ok" and np.isfinite(r["val_3h_mean"])
