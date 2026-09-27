"""Segmentation du visage (BiSeNet, 19 classes CelebAMask-HQ) : quelle zone est de la peau, des cheveux, le cou…"""
from __future__ import annotations

import threading
from functools import lru_cache

import cv2
import numpy as np

from .config import load_config, providers
from .faces import ModelsMissing

# Classes du modèle (ordre face-parsing.PyTorch / CelebAMask-HQ).
BACKGROUND, SKIN, L_BROW, R_BROW, L_EYE, R_EYE, GLASSES, L_EAR, R_EAR, EARRING, NOSE, MOUTH, U_LIP, L_LIP, \
    NECK, NECKLACE, CLOTH, HAIR, HAT = range(19)

# Peau candidate au niveau 2. HAT en fait partie : de profil, BiSeNet classe souvent le bas du visage en « chapeau » ;
# la zone géométrique et le filtre couleur de tone.py écartent les vrais chapeaux et les vêtements.
TONE_CLASSES = (SKIN, NOSE, L_EAR, R_EAR, NECK, HAT)
# Jamais recolorés.
EXCLUDE_TONE = (L_BROW, R_BROW, L_EYE, R_EYE, GLASSES, MOUTH, U_LIP, L_LIP, HAIR, NECKLACE, EARRING)

SIZE = 512
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)
_lock = threading.Lock()


@lru_cache(maxsize=1)
def _session():
    import onnxruntime as ort

    cfg = load_config()
    path = cfg.path("models") / cfg.models.parser
    if not path.is_file():
        raise ModelsMissing("Segmentation du visage absente : python scripts/download_models.py --level tone")
    return ort.InferenceSession(str(path), providers=providers(cfg, role="analysis"))


def parse(img_bgr: np.ndarray) -> np.ndarray:
    """Carte des classes (uint8, même taille que l'image). L'image doit être centrée sur un visage."""
    h, w = img_bgr.shape[:2]
    rgb = cv2.cvtColor(cv2.resize(img_bgr, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
    x = ((rgb.astype(np.float32) / 255.0 - _MEAN) / _STD).transpose(2, 0, 1)[None]
    sess = _session()
    with _lock:
        logits = sess.run([sess.get_outputs()[0].name], {sess.get_inputs()[0].name: x})[0][0]
    labels = logits.argmax(0).astype(np.uint8)
    return cv2.resize(labels, (w, h), interpolation=cv2.INTER_NEAREST)


def mask(labels: np.ndarray, classes: tuple[int, ...]) -> np.ndarray:
    return np.isin(labels, classes)


def face_crop_box(bbox, frame_shape, scale: float = 2.2, down: float = 0.12) -> tuple[int, int, int, int]:
    """Carré autour du visage, un peu décalé vers le bas pour inclure le cou. Renvoie (x1, y1, x2, y2) entiers."""
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    size = max(x2 - x1, y2 - y1) * scale
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2 + (y2 - y1) * down
    h, w = frame_shape[:2]
    l, t = int(round(cx - size / 2)), int(round(cy - size / 2))
    r, b = int(round(cx + size / 2)), int(round(cy + size / 2))
    return max(0, l), max(0, t), min(w, r), min(h, b)
