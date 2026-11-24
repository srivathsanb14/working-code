"""The scorer must recognise a right answer: the planted truth scores 0, seeds score worse, bad physics is rejected."""
from pathlib import Path

from discovery import closedform as cf
from discovery import selftest as st

SEEDS = Path(__file__).resolve().parents[1] / "code" / "discovery" / "seeds_cf"


def test_truth_scores_zero(toy_episodes, U):
    r = cf.evaluate(st.TRUTH_SRC, toy_episodes[1:], U, every=24)
    assert r["status"] == "ok" and r["val_one_step_mean"] < 1e-9 and r["val_3h_mean"] < 1e-9 and r["val_24h_mean"] < 1e-9


def test_seeds_are_worse_than_truth(toy_episodes, U):
    for s in SEEDS.glob("*.py"):
        r = cf.evaluate(s.read_text(), toy_episodes[1:], U, every=24, screen_factor=1e9)
        assert r["status"] in ("ok", "unstable") and r["val_3h_mean"] > 1e-3, s.name


EXPLICIT_AIR = '''import numpy as np
HISTORY = 1
def step(coef, geo, hist, u, U):
    T = hist["T"][:, -1]
    C = U["RHO_C_AIR"] * U["HEIGHT"] * geo["area"]
    return T + U["DT"] / C * u["g"] * (u["T_supply"] - T)
'''

HEAT_FROM_NOWHERE = '''import numpy as np
HISTORY = 1
def step(coef, geo, hist, u, U):
    return hist["T"][:, -1] + 0.01
'''


def test_explicit_supply_air_is_rejected(toy_episodes, U):
    r = cf.evaluate(EXPLICIT_AIR, toy_episodes[1:], U)
    assert r["status"] == "rejected" and any("stiff air stream" in f for f in r["failures"])


def test_heat_from_nowhere_is_rejected(toy_episodes, U):
    r = cf.evaluate(HEAT_FROM_NOWHERE, toy_episodes[1:], U)
    assert r["status"] == "rejected" and any("equilibrium" in f for f in r["failures"])


def test_ceiling_is_stable(toy_episodes, U):
    c = cf.ceiling(toy_episodes[0], U)
    assert all(v < 5 for k, v in c.items() if not k.startswith("persistence"))
