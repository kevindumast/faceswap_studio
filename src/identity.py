"""Identité source : embedding ArcFace moyen de 1 à 20 photos."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .faces import detect


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


def read_image(path: Path) -> np.ndarray | None:
    # imdecode plutôt qu'imread : gère les chemins avec accents sous Windows.
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is not None and max(img.shape[:2]) > 1600:
        scale = 1600 / max(img.shape[:2])
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img


def analyze_photo(path: Path) -> PhotoResult:
    from .faces import face_crop

    img = read_image(path)
    if img is None:
        return PhotoResult(path, ok=False)
    faces = detect(img)
    if not faces:
        return PhotoResult(path, ok=False)
    face = faces[0]  # le plus grand visage de la photo
    return PhotoResult(path, ok=True, crop=face_crop(img, face.bbox), embedding=face.normed_embedding)


def average_embedding(embeddings: list[np.ndarray]) -> np.ndarray:
    if not embeddings:
        raise ValueError("Aucun visage exploitable dans les photos source.")
    mean = np.mean(np.stack(embeddings), axis=0)
    return (mean / np.linalg.norm(mean)).astype(np.float32)


def source_from_dir(folder: Path) -> SourceFace:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    results = [analyze_photo(p) for p in sorted(folder.iterdir()) if p.suffix.lower() in exts]
    return SourceFace(average_embedding([r.embedding for r in results if r.ok]))
