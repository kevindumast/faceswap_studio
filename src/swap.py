"""Swap d'un visage sur une frame avec inswapper_128, recollé par nos soins.

Le recollage d'insightface est remplacé par le nôtre :
- masque doux construit dans l'espace aligné 128 (érosion et flou réglables), recollé sur la seule zone du visage ;
- couleur du visage généré ramenée sur celle du visage d'origine (joues + arête du nez, à la même place quel que soit
  le visage grâce à l'alignement), écart lissé d'une image à l'autre : inswapper grise ou teinte parfois le visage
  (profils), ce qui faisait « sauter » la couleur et marquait le bord du visage collé.
"""
from __future__ import annotations

import threading
from functools import lru_cache

import cv2
import numpy as np

from .config import load_config, providers
from .faces import ModelsMissing

_lock = threading.Lock()
SIZE = 128
MAX_SHIFT = np.array([20.0, 12.0, 12.0], np.float32)   # écart de couleur corrigé au plus (L, a, b en LAB 8 bits)


@lru_cache(maxsize=1)
def swapper():
    from insightface.model_zoo import get_model

    cfg = load_config()
    path = cfg.path("models") / cfg.models.inswapper
    if not path.is_file():
        raise ModelsMissing(f"{path.name} absent. Lancez : python scripts/download_models.py")
    return get_model(str(path), providers=providers(cfg))


@lru_cache(maxsize=8)
def paste_mask(erode: float, blur: float, size: int = SIZE) -> np.ndarray:
    """Masque doux dans l'espace aligné : carré érodé (fraction du côté) puis flouté (sigma en fraction du côté)."""
    m = np.zeros((size, size), np.float32)
    e = int(round(erode * size))
    m[e:size - e, e:size - e] = 1.0
    if blur > 0:
        m = cv2.GaussianBlur(m, (0, 0), blur * size)
    return m


@lru_cache(maxsize=1)
def color_zone() -> np.ndarray:
    """Joues et arête du nez dans l'espace aligné (gabarit ArcFace 128 : yeux y≈52, nez y≈72, bouche y≈92)."""
    m = np.zeros((SIZE, SIZE), np.uint8)
    for center, axes in (((44, 78), (8, 8)), ((84, 78), (8, 8)), ((64, 62), (5, 8))):
        cv2.ellipse(m, center, axes, 0, 0, 360, 1, -1)
    return m.astype(bool)


class SwapState:
    """Ce qu'un visage suivi garde d'une image à l'autre : l'écart de couleur lissé entre visage généré et d'origine."""

    def __init__(self):
        self.shift: np.ndarray | None = None


def color_shift(fake: np.ndarray, original: np.ndarray) -> np.ndarray | None:
    """Écart LAB (médianes) qui ramène le visage généré sur la couleur du visage d'origine, borné."""
    zone = color_zone()
    lab_f = cv2.cvtColor(fake, cv2.COLOR_BGR2LAB).astype(np.float32)[zone]
    lab_o = cv2.cvtColor(original, cv2.COLOR_BGR2LAB).astype(np.float32)[zone]
    usable = (lab_o[:, 0] > 25) & (lab_o[:, 0] < 245)    # ni noir (hors cadre) ni reflet
    if usable.sum() < zone.sum() * 0.5:
        return None
    shift = np.median(lab_o[usable], axis=0) - np.median(lab_f[usable], axis=0)
    return np.clip(shift, -MAX_SHIFT, MAX_SHIFT)


@lru_cache(maxsize=1)
def face_zone() -> np.ndarray:
    """Forme douce du visage dans l'espace aligné : la correction de couleur s'y limite. Appliquée à tout le carré,
    elle éclaircissait aussi le bout de décor qu'il contient (rectangle visible sur un fond uni)."""
    m = np.zeros((SIZE, SIZE), np.float32)
    cv2.ellipse(m, (64, 70), (40, 54), 0, 0, 360, 1.0, -1)
    return cv2.GaussianBlur(m, (0, 0), 8)[..., None]


def _shift_lab(img: np.ndarray, shift: np.ndarray, weight: np.ndarray | None = None) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab += shift if weight is None else weight * shift
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def paste(frame: np.ndarray, fake: np.ndarray, M: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Recolle l'image alignée `fake` (carrée, M : image → espace aligné) dans une copie de frame, selon `mask` (0–1),
    sur le seul rectangle qu'elle couvre."""
    IM = cv2.invertAffineTransform(M)
    n = fake.shape[0]
    corners = np.array([[0, 0, 1], [n, 0, 1], [0, n, 1], [n, n, 1]], np.float64) @ IM.T
    h, w = frame.shape[:2]
    x0, y0 = (int(max(0, np.floor(v))) for v in corners.min(axis=0))
    x1, y1 = int(min(w, np.ceil(corners[:, 0].max()))), int(min(h, np.ceil(corners[:, 1].max())))
    out = frame.copy()
    if x1 <= x0 or y1 <= y0:
        return out
    IMr = IM.copy()
    IMr[:, 2] -= (x0, y0)
    size = (x1 - x0, y1 - y0)
    face = cv2.warpAffine(fake, IMr, size, borderValue=0.0).astype(np.float32)
    alpha = cv2.warpAffine(mask, IMr, size, borderValue=0.0)[..., None]
    roi = out[y0:y1, x0:x1].astype(np.float32)
    out[y0:y1, x0:x1] = (face * alpha + roi * (1 - alpha)).astype(np.uint8)
    return out


def swap_face(frame: np.ndarray, target_face, source_face, state: SwapState | None = None,
             restore: bool = False) -> np.ndarray:
    """Remplace target_face (détecté dans frame) par l'identité source. Renvoie une nouvelle frame.

    `restore` : option « netteté » (CodeFormer) sur le visage généré, avant recollage — voir src/restore.py.
    """
    rcfg = load_config().render
    with _lock:
        fake, M = swapper().get(frame, target_face, source_face, paste_back=False)
    strength = float(rcfg.get("swap_color_match", 1.0))
    if strength > 0:
        original = cv2.warpAffine(frame, M, (SIZE, SIZE), borderValue=0.0)
        shift = color_shift(fake, original)
        if state is not None:
            if shift is None:
                shift = state.shift
            elif state.shift is not None:
                s = float(rcfg.get("swap_color_smoothing", 0.8))
                shift = s * state.shift + (1 - s) * shift
            state.shift = shift
        if shift is not None:
            fake = _shift_lab(fake, strength * shift, face_zone())
    if restore:
        from . import restore as _restore

        fake = _restore.restore(fake, float(rcfg.get("restore_fidelity", 0.5)))
        M = M * (fake.shape[0] / SIZE)   # le collage se fait maintenant depuis un visage plus grand (même cadrage)
    mask = paste_mask(float(rcfg.get("swap_mask_erode", 0.1)), float(rcfg.get("swap_mask_blur", 0.02)), fake.shape[0])
    return paste(frame, fake, M, mask)
