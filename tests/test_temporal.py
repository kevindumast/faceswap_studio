"""Suivi du visage cible : lissage des points clés et de la boîte, sans retard quand la tête bouge vite."""
from types import SimpleNamespace

import numpy as np

from src.temporal import TargetTracker

KPS = np.array([[130, 140], [170, 140], [150, 160], [135, 180], [165, 180]], np.float32)
BOX = np.array([100, 100, 200, 220], np.float32)


def _face(dx: float, dy: float = 0.0):
    return SimpleNamespace(bbox=BOX + [dx, dy, dx, dy], kps=KPS + [dx, dy])


def test_box_jitter_is_smoothed_at_rest():
    rng = np.random.default_rng(0)
    tracker = TargetTracker(np.zeros(512, np.float32), threshold=0.3, smoothing=0.6)
    raw, smooth = [], []
    for _ in range(40):
        face = _face(*rng.uniform(-1.5, 1.5, 2))
        raw.append(face.bbox.copy())
        tracker.smooth(face)
        smooth.append(np.asarray(face.bbox, np.float32))
    raw_jump = np.abs(np.diff(np.array(raw), axis=0)).mean()
    smooth_jump = np.abs(np.diff(np.array(smooth), axis=0)).mean()
    assert smooth_jump < raw_jump * 0.75


def test_no_lag_when_the_head_moves_fast():
    tracker = TargetTracker(np.zeros(512, np.float32), threshold=0.3, smoothing=0.6)
    for i in range(10):
        face = _face(30.0 * i)                  # 30 px par image : bien au-delà du seuil de mouvement
        tracker.smooth(face)
        assert abs(face.bbox[0] - (BOX[0] + 30.0 * i)) < 1.0
        assert np.abs(face.kps - (KPS + [30.0 * i, 0])).max() < 1.0
