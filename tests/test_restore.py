"""Option « netteté » (niveaux 1 et 2) : conversions et répartition des entrées du modèle, sans le modèle lui-même."""
from types import SimpleNamespace

import numpy as np

from src import restore


def test_to_input_then_from_output_round_trips_a_flat_color():
    crop = np.full((128, 128, 3), (40, 120, 200), np.uint8)   # BGR
    x = restore.to_input(crop)
    assert x.shape == (1, 3, restore.SIZE, restore.SIZE)
    assert x.min() >= -1.0 - 1e-6 and x.max() <= 1.0 + 1e-6
    y = np.tile(x[0], (1, 1, 1))[None]   # le modèle renverrait la même image (identité)
    out = restore.from_output(y)
    assert out.shape == (restore.SIZE, restore.SIZE, 3)
    assert np.abs(out.astype(int) - np.array([40, 120, 200])).max() <= 2


def _input(name, shape, dtype):
    return SimpleNamespace(name=name, shape=shape, type=dtype)


def test_feed_routes_the_image_to_the_4d_input_and_the_fidelity_to_the_other():
    sess = SimpleNamespace(get_inputs=lambda: [
        _input("input", [1, 3, 512, 512], "tensor(float)"),
        _input("weight", [1], "tensor(double)"),
    ])
    x = np.zeros((1, 3, 512, 512), np.float32)
    feed = restore._feed(sess, x, 0.7)
    assert feed["input"].shape == (1, 3, 512, 512) and feed["input"].dtype == np.float32
    assert feed["weight"].dtype == np.float64
    assert float(feed["weight"][0]) == 0.7


def test_feed_works_whatever_order_the_model_declares_its_inputs():
    sess = SimpleNamespace(get_inputs=lambda: [
        _input("w", [1], "tensor(float)"),
        _input("x", [1, 3, 512, 512], "tensor(float)"),
    ])
    x = np.ones((1, 3, 512, 512), np.float32)
    feed = restore._feed(sess, x, 0.3)
    assert np.array_equal(feed["x"], x)
    assert feed["w"] == np.float32(0.3)
