"""Détection + embedding ArcFace (pack buffalo_l d'InsightFace), chargés une seule fois par process."""
from __future__ import annotations

import threading
from functools import lru_cache

import numpy as np

from .config import accelerator, load_config, providers


class ModelsMissing(RuntimeError):
    pass


_lock = threading.Lock()


@lru_cache(maxsize=1)
def analyzer():
    from insightface.app import FaceAnalysis

    cfg = load_config()
    pack = cfg.path("models") / cfg.models.detector_pack
    if not any(pack.glob("*.onnx")):
        raise ModelsMissing(f"Modèles absents dans {pack}. Lancez : python scripts/download_models.py")
    # Seuls détection + reconnaissance : on saute landmarks 3D / âge / genre (gros gain CPU).
    app = FaceAnalysis(name=str(pack), allowed_modules=["detection", "recognition"], providers=providers(cfg, role="analysis"))
    size = int(cfg.det_size)
    # ctx_id < 0 force insightface à repasser la détection sur CPU : on ne le fait que sans carte graphique.
    app.prepare(ctx_id=0 if accelerator(cfg) != "cpu" else -1, det_size=(size, size),
                det_thresh=float(cfg.get("det_thresh", 0.5)))
    return app


def detect(frame: np.ndarray) -> list:
    """Visages (avec embedding) triés du plus grand au plus petit. Thread-safe (l'API sert plusieurs requêtes)."""
    with _lock:
        faces = analyzer().get(frame)
    return sorted(faces, key=lambda f: area(f.bbox), reverse=True)


def detect_boxes(frame: np.ndarray, size: int | None = None, thresh: float | None = None) -> list:
    """Détection seule, sans embedding (~2x plus rapide) : le suivi n'a pas besoin d'ArcFace à chaque frame.

    size / thresh : taille d'analyse et confiance minimale pour cet appel (par défaut, ceux du rendu).
    """
    from insightface.app.common import Face

    with _lock:
        bboxes, kpss = analyzer().det_model.detect(
            frame, input_size=(size, size) if size else None, max_num=0, metric="default", det_thresh=thresh)
    faces = [Face(bbox=b[:4], kps=k, det_score=b[4]) for b, k in zip(bboxes, kpss)]
    return sorted(faces, key=lambda f: area(f.bbox), reverse=True)


def embed(frame: np.ndarray, face) -> None:
    """Calcule l'embedding ArcFace d'un visage détecté (remplit face.embedding / normed_embedding)."""
    with _lock:
        analyzer().models["recognition"].get(frame, face)


def area(bbox) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def face_crop(frame: np.ndarray, bbox, size: int = 160) -> np.ndarray:
    """Crop carré autour du visage avec marge, pour les vignettes de l'UI."""
    import cv2

    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1) * 0.8
    h, w = frame.shape[:2]
    l, t = int(max(0, cx - half)), int(max(0, cy - half))
    r, b = int(min(w, cx + half)), int(min(h, cy + half))
    crop = frame[t:b, l:r]
    if crop.size == 0:
        crop = frame
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
