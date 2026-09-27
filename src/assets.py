"""Préparation des données d'une personne source à partir de ses photos (teint, etc.)."""
from __future__ import annotations

from pathlib import Path

from .faces import detect
from .identity import read_image
from .tone import ToneStats, source_tone


def person_tone(photos: list[Path]) -> ToneStats | None:
    """Teint de la personne : peau du cœur du visage, sur toutes ses photos où un visage est détecté."""
    images, faces = [], []
    for path in photos:
        img = read_image(path)
        if img is None:
            continue
        found = detect(img)
        if found:
            images.append(img)
            faces.append(found[0])
    return source_tone(images, faces)
