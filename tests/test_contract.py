import pytest

from discovery import contract

OK = '''import numpy as np
HISTORY = 3
def step(coef, geo, hist, u, U):
    return hist["T"].mean(1)
'''


def test_numpy_methods_work_inside_the_sandbox():
    mod, cx = contract.load_closedform(OK)
    import numpy as np
    assert mod.step(None, None, {"T": np.ones((2, 3, 4))}, None, None).shape == (2, 4) and cx > 0


@pytest.mark.parametrize("src,msg", [
    ("import os\ndef step(c, g, h, u, U): pass\n", "not allowed"),
    ("PARAMS = {'a': (1, 0, 2)}\ndef step(c, g, h, u, U): pass\n", "absent or empty"),
    ("HISTORY = 999\ndef step(c, g, h, u, U): pass\n", "HISTORY"),
    ("def prepare(g, U): pass\n", "missing the model"),
    ("x = open('f')\ndef step(c, g, h, u, U): pass\n", "not allowed"),
])
def test_contract_rejects(src, msg):
    with pytest.raises(contract.ContractError, match=msg):
        contract.load_closedform(src)
