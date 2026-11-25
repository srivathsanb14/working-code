"""Whatever the controller returns, the gateway gets a valid action."""
import numpy as np

from submission.safety import SafeAction


def valid(a, nc, nd):
    c, d = a["continuous_action"], a["discrete_action"]
    return (c.shape == (nc,) and d.shape == (nd,) and c.dtype == np.float32 and d.dtype == np.int32
            and np.all(np.isfinite(c)) and np.all(np.abs(c) <= 1) and set(np.unique(d)) <= {0, 1})


def test_fuzz_sanitize():
    rng = np.random.default_rng(0); nc, nd = 17, 4; s = SafeAction(nc, nd)
    for _ in range(500):
        kind = rng.integers(6)
        c = rng.normal(0, 5, nc); d = rng.normal(0, 2, nd)
        if kind == 1: c[rng.integers(nc)] = np.nan
        if kind == 2: c[rng.integers(nc)] = np.inf
        if kind == 3: c = c[:-1]
        if kind == 4: d = np.array([np.nan] * nd)
        a = {"continuous_action": c, "discrete_action": d} if kind != 5 else {"oops": 1}
        assert valid(s.sanitize(a), nc, nd)


def test_exception_falls_back():
    s = SafeAction(5, 2)

    def boom(obs):
        raise RuntimeError("controller crashed")
    a = s.call(boom, None)
    assert valid(a, 5, 2) and s.repairs == 1
