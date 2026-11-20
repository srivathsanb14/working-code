import numpy as np

import generate_data as g


def test_facade_sides_sum_to_facade_faces(toy_geo):
    s = toy_geo["facade_top"] + toy_geo["facade_bottom"] + toy_geo["facade_left"] + toy_geo["facade_right"]
    assert np.allclose(s, toy_geo["facade_faces"])
    assert (toy_geo["facade_sides"] == 2).sum() == 4          # the four corner rooms of a 3 x 2 block
    assert toy_geo["n_neighbours"].min() >= 2


def test_seed_topup():
    assert g.seeds("5:8,20:22") == [5, 6, 7, 20, 21]
    assert g.with_topup("500:600", 1.3) == "500:600,10500:10530"
    assert g.with_topup("700:720", 1.0) == "700:720"
