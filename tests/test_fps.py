"""Cadence plafonnée (option « Limiter à 30 i/s ») et réutilisation des images copiées."""
import numpy as np
import pytest

from src import media
from src.pipeline import DuplicateDetector


@pytest.mark.parametrize("fps, expected", [
    ("60/1", "30/1"),
    ("60000/1001", "30000/1001"),   # 59,94 → 29,97 : une image sur deux, pas un 30 approché
    ("50/1", "25/1"),
    ("120/1", "30/1"),
    ("30/1", "30/1"),               # déjà sous le plafond : inchangé
    ("25/1", "25/1"),
])
def test_capped_fps_divides_by_an_integer(fps, expected):
    assert media.capped_fps(fps, 30) == expected


def test_capped_fps_without_cap_is_unchanged():
    assert media.capped_fps("60/1", None) == "60/1"


def _frame(seed=0):
    return np.random.default_rng(seed).integers(0, 255, (360, 640, 3), dtype=np.uint8)


def test_identical_frame_is_a_duplicate():
    det = DuplicateDetector(4.0)
    a = _frame()
    assert det.is_duplicate(a) is False            # première image : rien à comparer
    assert det.is_duplicate(a.copy()) is True


def test_small_local_change_is_not_a_duplicate():
    """Un petit mouvement (lèvres, yeux) dans un plan fixe ne doit jamais être pris pour une copie."""
    det = DuplicateDetector(4.0)
    a = np.full((360, 640, 3), 120, np.uint8)
    b = a.copy()
    b[150:170, 300:330] = 170                       # zone de 20×30 px qui change
    det.is_duplicate(a)
    assert det.is_duplicate(b) is False


def test_threshold_zero_disables_detection():
    det = DuplicateDetector(0.0)
    a = _frame()
    det.is_duplicate(a)
    assert det.is_duplicate(a.copy()) is False


def test_cut_with_capped_fps(sample_video, tmp_path):
    info = media.probe(sample_video)                # 25 i/s
    out = tmp_path / "cut.mp4"
    media.cut_segment(sample_video, out, 1.0, 6.0, (640, 360), info.has_audio, media.capped_fps(info.fps_str, 15))
    cut = media.probe(out)
    assert cut.fps == pytest.approx(12.5)
    assert cut.duration == pytest.approx(5.0, abs=0.1)
