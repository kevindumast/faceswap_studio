"""Niveau 4 (personne entière), côté PC : photo de référence, estimation du GPU, recollage du résultat du Space.

Le Space génère la personne en 360p ou 480p, à 30 i/s, et renvoie le rectangle qu'il a régénéré autour d'elle.
Ici, on recolle ce rectangle (bords adoucis) sur l'extrait, à sa cadence et à sa taille : hors de la personne,
l'image garde sa netteté d'origine.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from src import media

PORTRAIT_RATIO = 0.22   # visage plus haut que 22 % de la photo : portrait, le corps et les habits seront inventés


@dataclass
class Reference:
    path: Path
    face_ratio: float     # hauteur du visage / hauteur de la photo (petit = photo en pied)

    @property
    def full_body(self) -> bool:
        return self.face_ratio <= PORTRAIT_RATIO


def face_ratio(path: Path) -> float | None:
    from src.faces import detect_boxes

    img = cv2.imread(str(path))
    if img is None:
        return None
    faces = detect_boxes(img)
    if not faces:
        return None
    tallest = max(faces, key=lambda f: f.bbox[3] - f.bbox[1])
    return float((tallest.bbox[3] - tallest.bbox[1]) / img.shape[0])


def choose_reference(photos: list[Path], ratio: Callable[[Path], float | None] = face_ratio) -> Reference:
    """La photo la plus « en pied » de la personne : celle où le visage occupe le moins de hauteur."""
    scored = [(r, p) for p in photos if (r := ratio(p)) is not None]
    if not scored:
        raise RuntimeError("Aucune photo exploitable pour cette personne (visage introuvable).")
    r, p = min(scored, key=lambda x: x[0])
    return Reference(p, r)


def gpu_seconds(seconds: float, resolution: str, steps: int, per_second: dict | None = None) -> float:
    """Temps de GPU attendu sur ZeroGPU (même formule que le Space, réglages dans config.yaml → levels.character)."""
    per_s = (per_second or {"360p": 12.0, "480p": 26.0}).get(resolution, 12.0)
    return 45 + seconds * per_s * steps / 6


class _Timeline:
    """Lit une vidéo à sa cadence et renvoie l'image affichée à un instant donné (lecture séquentielle)."""

    def __init__(self, path: Path):
        self.cap = cv2.VideoCapture(str(path))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.index, self.frame = -1, None

    def at(self, t: float) -> np.ndarray | None:
        target = max(0, int(t * self.fps + 1e-6))
        while self.index < target:
            ok, frame = self.cap.read()
            if not ok:
                break          # fin : on garde la dernière image
            self.frame, self.index = frame, self.index + 1
        return self.frame

    def close(self) -> None:
        self.cap.release()


def recompose(generated: Path, mask: Path | None, original: Path, out: Path, fps_str: str, crf: int, preset: str,
              feather: float = 0.015) -> int:
    """Recolle la personne générée sur l'extrait d'origine ; renvoie le nombre d'images écrites.

    Sans masque : l'image générée entière, agrandie (moins nette, mais cadence et durée d'origine).
    """
    info = media.probe(original)
    size = (info.width, info.height)
    k = max(3, int(info.height * feather) // 2 * 2 + 1)      # adoucissement proportionnel à la hauteur
    gen, msk = _Timeline(generated), _Timeline(mask) if mask else None
    cap = cv2.VideoCapture(str(original))
    fps = info.fps or 30.0
    count = 0
    try:
        with media.FrameWriter(out, size, fps_str, crf, preset) as writer:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                t = count / fps
                g = gen.at(t)
                if g is None:
                    raise media.MediaError("Vidéo générée vide.")
                g = cv2.resize(g, size, interpolation=cv2.INTER_CUBIC)
                m = msk.at(t) if msk else None
                if m is None:
                    frame = g
                else:
                    alpha = cv2.resize(m[:, :, 0], size, interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
                    alpha = cv2.GaussianBlur(alpha, (k, k), 0)[:, :, None]
                    frame = (frame * (1 - alpha) + g * alpha).astype(np.uint8)
                writer.write(frame)
                count += 1
    finally:
        cap.release()
        gen.close()
        if msk:
            msk.close()
    return count
