"""Coefficient forms: the LLM writes a formula for every coefficient of the fixed state-space function (term form), or
for every capacity and conductance of a node network (network form); the framework builds and steps the model."""
from pathlib import Path

import numpy as np
import pytest

from discovery import closedform as cf
from discovery import coefform, contract

D = Path(__file__).resolve().parents[1] / "code" / "discovery"
ALLOWED_INPUTS = {"T_out", "g", "T_supply", "Q_reheat", "Q_gain", "occupants", "comfort", "hour_sin", "hour_cos",
                  "dow_sin", "dow_cos", "T_water"}


def _load(path):
    return contract.load_closedform(Path(path).read_text())[0]


def test_network_form_matches_the_stateful_network(toy_episodes, U):
    e = toy_episodes[1]; starts = np.arange(36, e.n - 36, 24)
    old = _load(D / "seeds_cf_legacy" / "seed_cf_network_observer.py")
    new = _load(D / "seeds_cf_legacy" / "seed_cf_network_nodes.py")
    a = cf.rollout(old, cf.prepare_model(old, e.geo, U), cf.Prepared(e, old.HISTORY), U, starts, 36)
    b = cf.rollout(new, cf.prepare_model(new, e.geo, U), cf.Prepared(e, new.HISTORY), U, starts, 36)
    assert np.abs(a - b).max() < 1e-8


@pytest.mark.parametrize("seed", ["seed_cf_network.py", "seed_cf_history.py", "seed_cf_one_node.py"])
def test_starting_models_pass_checks_and_score(seed, toy_episodes, U):
    r = cf.evaluate((D / "seeds_cf" / seed).read_text(), toy_episodes[:1], toy_episodes[1:], U, every=24, screen_factor=1e9)
    assert r["status"] in ("ok", "unstable"), r.get("reason")
    assert "failures" not in r


def test_coefficients_see_only_test_time_data_up_to_now(toy_episodes, U):
    """hist holds only measured temperatures and observable step inputs, and only up to the refresh step."""
    seen = []

    class Spy:
        HISTORY, REFRESH = 36, 12

        @staticmethod
        def coefficients(geo, hist, U):
            seen.append((set(hist), hist["T"].copy(), hist["T_out"].copy()))
            return {"air": 1e-3, "heat": 1e-3, "outside": 0.01, "slab": 0.05, "slab_back": 0.01}

    e = toy_episodes[0]; k0 = 120
    cm = coefform.make_model(Spy, e.geo, U)
    cf.filter_coef(Spy, cm, cf.Prepared(e, Spy.HISTORY), U)
    before = list(seen); seen.clear()
    import copy
    e2 = copy.deepcopy(e)
    e2.T_meas[k0:] += 7.0                                # change everything measured from step k0 on
    e2.inputs["T_out"][k0:] += 9.0
    cf.filter_coef(Spy, coefform.make_model(Spy, e2.geo, U), cf.Prepared(e2, Spy.HISTORY), U)
    for i, ((keys, T, To), (_, T2, To2)) in enumerate(zip(before, seen)):
        assert keys <= ALLOWED_INPUTS | {"T"}
        if i * Spy.REFRESH <= k0:                        # refreshes at or before k0 saw nothing from after k0
            assert np.array_equal(T, T2) and np.array_equal(To, To2)


@pytest.mark.parametrize("bad, msg", [('{"air": a, "heat": a, "outside": -a}', ">= 0"),
                                      ('{"air": a, "heat": a, "outside": np.ones(3)}', "(V,) array"),
                                      ('{"air": a, "heat": a, "neighbours": np.ones((2, 2))}', "(V, V)"),
                                      ('{"air": a, "heat": a, "sun": a}', "unknown terms")])
def test_bad_coefficients_are_rejected_with_a_reason(bad, msg, toy_episodes, U):
    src = ("import numpy as np\nHISTORY = 1\nREFRESH = 12\n"
           "def coefficients(geo, hist, U):\n    a = 300.0 / (1206.0 * 3.0 * geo['area'])\n"
           f"    return {bad}\n")
    r = cf.evaluate(src, toy_episodes[:1], toy_episodes[1:], U, every=24, screen_factor=1e9)
    assert r["status"] in ("error", "rejected") and msg in (r.get("reason", "") + " ".join(r.get("failures", [])))


def test_term_form_is_the_backward_euler_room_balance(toy_geo, U):
    """One room, supply air only: T+ = (T + a g Ts) / (1 + a g), always between T and Ts."""
    class M:
        HISTORY, REFRESH = 1, 1
        coefficients = staticmethod(lambda geo, hist, U: {"air": 1e-3, "heat": 0.0})
    tm = coefform.make_model(M, toy_geo, U); V = tm.V
    c = tm.build({"T": np.full((1, 1, V), 294.0)})
    x = np.full((1, 4 * V), 294.0)
    u = {"g": np.full((1, V), 2400.0), "T_supply": np.full((1, V), 290.0), "T_out": np.array([294.0]),
         "Q_reheat": np.zeros((1, V)), "Q_gain": np.zeros((1, V)), "occupants": np.zeros(1)}
    T1 = tm.predict(c, x, u)[0, :V]
    assert np.allclose(T1, (294.0 + 2.4 * 290.0) / (1 + 2.4))


def test_refresh_must_be_in_range():
    src = "import numpy as np\nHISTORY = 1\nREFRESH = 0\ndef coefficients(geo, hist, U): pass\n"
    with pytest.raises(contract.ContractError, match="REFRESH"):
        contract.load_closedform(src)


def test_crossover_analysis_compares_coefficient_values(toy_episodes, U):
    a = (D / "seeds_cf" / "seed_cf_network.py").read_text()
    b = (D / "seeds_cf" / "seed_cf_one_node.py").read_text()
    g = {"all rooms": {"rmse": 1.0, "bias": 0.0}}
    r = cf.crossover_analysis({"analysis": {"groups": g}, "code": a}, {"analysis": {"groups": g}, "code": b}, ep=toy_episodes[0], U=U)
    assert "coefficient values on a held-out building" in r["text"]
    assert "slab" in r["text"] and "outside" in r["text"]
