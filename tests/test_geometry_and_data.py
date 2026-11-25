import numpy as np


def test_facade_sides_sum_to_facade_faces(toy_geo):
    s = toy_geo["facade_top"] + toy_geo["facade_bottom"] + toy_geo["facade_left"] + toy_geo["facade_right"]
    assert np.allclose(s, toy_geo["facade_faces"])
    assert (toy_geo["facade_sides"] == 2).sum() == 4          # the four corner rooms of a 3 x 2 block
    assert toy_geo["n_neighbours"].min() >= 2

