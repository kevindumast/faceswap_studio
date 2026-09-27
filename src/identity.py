"""Identité source : embedding ArcFace moyen de 1 à 20 photos."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import load_config
from .faces import detect_boxes, embed


@dataclass
class SourceFace:
    """Ce qu'attend INSwapper côté source : juste un embedding normalisé."""
    normed_embedding: np.ndarray


@dataclass
class PhotoResult:
    path: Path
    ok: bool
    crop: np.ndarray | None = None
    embedding: np.ndarray | None = None


def read_image_full(path: Path) -> np.ndarray | None:
    # imdecode plutôt qu'imread : gère les chemins avec accents sous Windows. Applique l'orientation EXIF par défaut.
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def read_image(path: Path) -> np.ndarray | None:
    """Comme read_image_full, plafonné à 1600 px : suffisant pour la détection, plus rapide."""
    img = read_image_full(path)
    if img is not None and max(img.shape[:2]) > 1600:
        scale = 1600 / max(img.shape[:2])
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img


def _detect_tolerant(frame: np.ndarray) -> list:
    """Détection avec le profil « photo » (comme l'étape Visages) : plus tolérant que le détecteur du rendu,
    figé sur des réglages pensés pour la vitesse vidéo. Une photo importée est unique et précieuse : mieux
    vaut rater plus rarement un visage net que de réutiliser un seuil pensé pour des frames en rafale."""
    cfg = load_config()
    ph = cfg.get("photos", {})
    size, thresh = int(ph.get("det_size", 960)), float(ph.get("det_thresh", 0.3))
    faces = detect_boxes(frame, size=size, thresh=thresh)
    if faces:
        embed(frame, faces[0])
    return faces


def analyze_photo(path: Path) -> PhotoResult:
    from .faces import face_crop

    img = read_image(path)
    if img is None:
        return PhotoResult(path, ok=False)
    faces = _detect_tolerant(img)
    if not faces:
        return PhotoResult(path, ok=False)
    face = faces[0]  # le plus grand visage de la photo
    return PhotoResult(path, ok=True, crop=face_crop(img, face.bbox), embedding=face.normed_embedding)


def analyze_region(path: Path, box: tuple[float, float, float, float]) -> PhotoResult:
    """Comme analyze_photo, mais limité à une zone tracée à la main (box en fractions 0-1 de l'image).

    Utile quand la détection automatique échoue sur la photo entière (visage trop petit) : recadrer
    sur la zone indiquée « zoome » dessus et lui donne une meilleure chance d'être détecté.
    """
    from .faces import face_crop

    img = read_image(path)
    if img is None:
        return PhotoResult(path, ok=False)
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    l, t = int(max(0, min(x1, x2) * w)), int(max(0, min(y1, y2) * h))
    r, b = int(min(w, max(x1, x2) * w)), int(min(h, max(y1, y2) * h))
    if r - l < 20 or b - t < 20:
        return PhotoResult(path, ok=False)
    mx, my = int((r - l) * 0.3), int((b - t) * 0.3)
    cl, ct = max(0, l - mx), max(0, t - my)
    cr, cb = min(w, r + mx), min(h, b + my)
    region = img[ct:cb, cl:cr]
    faces = _detect_tolerant(region)
    if not faces:
        return PhotoResult(path, ok=False)
    face = faces[0]
    return PhotoResult(path, ok=True, crop=face_crop(region, face.bbox), embedding=face.normed_embedding)


def average_embedding(embeddings: list[np.ndarray]) -> np.ndarray:
    if not embeddings:
        raise ValueError("Aucun visage exploitable dans les photos source.")
    mean = np.mean(np.stack(embeddings), axis=0)
    return (mean / np.linalg.norm(mean)).astype(np.float32)


def source_from_dir(folder: Path) -> SourceFace:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    results = [analyze_photo(p) for p in sorted(folder.iterdir()) if p.suffix.lower() in exts]
    return SourceFace(average_embedding([r.embedding for r in results if r.ok]))
