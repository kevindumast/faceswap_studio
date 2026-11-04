"""Photos de la personne source : cadrage serré et photos en noir et blanc."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from src import tone
from src.identity import analyze_photo, detect_photo, pad_tight

MODELS = Path(__file__).resolve().parent.parent / "models" / "buffalo_l"
needs_models = pytest.mark.skipif(not any(MODELS.glob("*.onnx")), reason="modèles absents (scripts/download_models.py)")


def _sample(name: str) -> Path:
    insightface = pytest.importorskip("insightface")
    path = Path(insightface.__file__).parent / "data" / "images" / name
    if not path.is_file():
        pytest.skip(f"image d'exemple {name} absente")
    return path


def test_pad_tight_adds_replicated_margin():
    img = np.full((100, 60, 3), 7, np.uint8)
    out = pad_tight(img)
    assert out.shape == (200, 160, 3) and (out == 7).all()


@needs_models
def test_tight_crop_is_detected(tmp_path):
    """Visage cadré au ras (112 px aligné) : rien sans marge, trouvé avec."""
    path = _sample("Tom_Hanks_54745.png")
    from src.faces import detect

    img = cv2.imread(str(path))
    assert detect(img) == []
    used, faces = detect_photo(img)
    assert faces and used.shape[0] > img.shape[0]
    result = analyze_photo(path)
    assert result.ok and result.image is not None and result.embedding is not None


def test_grayscale_photo_is_flagged():
    rng = np.random.default_rng(0)
    gray = cv2.cvtColor(rng.integers(0, 255, (80, 80), np.uint8), cv2.COLOR_GRAY2BGR)
    color = gray.copy()
    color[..., 2] = np.clip(color[..., 2].astype(int) + 40, 0, 255)   # teinte chaude
    assert tone.is_grayscale(gray)
    assert not tone.is_grayscale(color)


def test_grayscale_photos_give_no_tone():
    """Une photo en noir et blanc ne dicte jamais le teint (le visage virait au gris)."""
    from insightface.app.common import Face

    img = np.full((200, 200, 3), 128, np.uint8)
    face = Face(bbox=np.array([50, 40, 150, 160], np.float32),
                kps=np.array([[80, 85], [120, 85], [100, 110], [85, 135], [115, 135]], np.float32), det_score=0.9)
    assert tone.source_tone([img], [face]) is None
