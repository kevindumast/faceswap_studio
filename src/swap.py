"""Swap d'un visage sur une frame avec inswapper_128."""
from __future__ import annotations

import threading
from functools import lru_cache

import numpy as np

from .config import load_config, providers
from .faces import ModelsMissing

_lock = threading.Lock()


@lru_cache(maxsize=1)
def swapper():
    from insightface.model_zoo import get_model

    cfg = load_config()
    path = cfg.path("models") / cfg.models.inswapper
    if not path.is_file():
        raise ModelsMissing(f"{path.name} absent. Lancez : python scripts/download_models.py")
    return get_model(str(path), providers=providers(cfg))


def swap_face(frame: np.ndarray, target_face, source_face) -> np.ndarray:
    """Remplace target_face (détecté dans frame) par l'identité source, recollé dans la frame."""
    with _lock:
        return swapper().get(frame, target_face, source_face, paste_back=True)
