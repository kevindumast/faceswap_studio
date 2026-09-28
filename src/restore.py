"""Niveaux 1 et 2 (option « netteté ») : restauration du visage collé par inswapper avant recollage.

inswapper_128 génère toujours un visage à 128×128 px, quelle que soit la résolution de la vidéo : le recollage
(swap.paste) l'agrandit ensuite sans ajouter de détail, d'où un visage plus flou que le reste de l'image. CodeFormer
(modèle de restauration de visage, ONNX) reconstruit les détails haute fréquence (peau, yeux, sourcils) à partir de
cette image floue, à 512×512 px.
"""
from __future__ import annotations

import threading
from functools import lru_cache

import cv2
import numpy as np

from .config import load_config, providers
from .faces import ModelsMissing

SIZE = 512
_lock = threading.Lock()
_NP_DTYPES = {"tensor(float)": np.float32, "tensor(double)": np.float64, "tensor(float16)": np.float16}


@lru_cache(maxsize=1)
def _session():
    import onnxruntime as ort

    cfg = load_config()
    path = cfg.path("models") / cfg.models.restore
    if not path.is_file():
        raise ModelsMissing("Modèle de netteté absent : python scripts/download_models.py --level restore")
    on_gpu = cfg.get("render", {}).get("restore_device", "cpu") == "gpu"
    provs = providers(cfg, role="swap") if on_gpu else ["CPUExecutionProvider"]
    return ort.InferenceSession(str(path), providers=provs)


def to_input(crop: np.ndarray) -> np.ndarray:
    """`crop` (carré, n'importe quelle taille) redimensionné à 512 px, normalisé en [-1, 1], BCHW RGB."""
    x = cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC)[:, :, ::-1].astype(np.float32) / 255.0
    return ((x - 0.5) / 0.5).transpose(2, 0, 1)[None]


def from_output(y: np.ndarray) -> np.ndarray:
    img = ((np.clip(y[0], -1.0, 1.0) * 0.5 + 0.5) * 255.0).transpose(1, 2, 0).astype(np.uint8)
    return np.ascontiguousarray(img[:, :, ::-1])


def _feed(sess, x: np.ndarray, fidelity: float) -> dict:
    """Le modèle attend l'image (BCHW) et un poids de fidélité (scalaire) : reconnus par leur forme plutôt que
    par un nom d'entrée figé, les versions ONNX en circulation ne les nommant pas toutes pareil."""
    weight = np.array([fidelity])
    feed = {}
    for inp in sess.get_inputs():
        dtype = _NP_DTYPES.get(inp.type, np.float32)
        feed[inp.name] = x.astype(dtype) if len(inp.shape) == 4 else weight.astype(dtype)
    return feed


def restore(crop: np.ndarray, fidelity: float = 0.5) -> np.ndarray:
    """Visage `crop` restauré à 512 px. `fidelity` : 0 = très retouché (plus net, peut s'éloigner un peu de
    l'original), 1 = fidèle à l'entrée (plus sûr, moins de gain de netteté)."""
    sess = _session()
    x = to_input(crop)
    with _lock:
        y = sess.run(None, _feed(sess, x, fidelity))[0]
    return from_output(y)
