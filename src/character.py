"""Niveau 4 (personne entière), côté PC : photo de référence, estimation du GPU, recollage du résultat du Space.

Le Space génère la personne en 360p ou 480p, à 30 i/s, et renvoie le rectangle qu'il a régénéré autour d'elle.
Ici, on recolle ce rectangle (bords adoucis) sur l'extrait, à sa cadence et à sa taille : hors de la personne,
l'image garde sa netteté d'origine.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from src import media

# Cadrage d'une photo selon la part de sa hauteur occupée par le visage.
FULL_BODY_RATIO = 0.14   # jusqu'ici : de la tête aux pieds (idéal pour le niveau 4)
PORTRAIT_RATIO = 0.30    # au-delà : portrait, le corps et les habits seront inventés ; entre les deux : mi-corps


def framing_of(ratio: float | None) -> str:
    """full (tête aux pieds), half (mi-corps : le bas sera inventé), portrait, ou none (visage introuvable)."""
    if ratio is None:
        return "none"
    if ratio <= FULL_BODY_RATIO:
        return "full"
    return "half" if ratio <= PORTRAIT_RATIO else "portrait"


@dataclass
class Reference:
    path: Path
    face_ratio: float | None    # hauteur du visage / hauteur de la photo (petit = photo en pied ; None : introuvable)

    @property
    def framing(self) -> str:
        return framing_of(self.face_ratio)

    @property
    def full_body(self) -> bool:
        return self.framing == "full"


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


BLOCK_FRAMES = 76    # Wan-Animate génère par blocs de 77 images à 30 i/s, dont 1 reprise du bloc précédent
PREP_S = (10.0, 6.0)  # préparation sur le Space (squelette, silhouette) : fixe + par seconde d'extrait


def blocks(seconds: float) -> int:
    """Blocs de génération d'un extrait : 2,5 s → 1 ; 5 s → 2 ; 10 s → 4 (même calcul que le Space)."""
    return max(1, math.ceil((max(1, round(seconds * 30)) - 1) / BLOCK_FRAMES))


def gpu_seconds(seconds: float, resolution: str, steps: int, per_block_step: dict | None = None) -> float:
    """Temps de GPU attendu sur ZeroGPU pour un passage (même formule que le Space ; valeurs mesurées sur les rendus
    précédents, sinon config.yaml → levels.character.gpu_s_per_block_step)."""
    per = (per_block_step or {"360p": 16.0, "480p": 36.0}).get(resolution, 16.0)
    return PREP_S[0] + PREP_S[1] * seconds + blocks(seconds) * steps * per


def measured_block_step(gpu_s: float, seconds: float, steps: int) -> float:
    """Temps d'une étape d'un bloc, déduit d'un passage réel (pour recaler l'estimation des rendus suivants)."""
    return max(1.0, (gpu_s - PREP_S[0] - PREP_S[1] * seconds) / (blocks(seconds) * max(1, steps)))


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


def color_offset(original: np.ndarray, generated: np.ndarray, zone: np.ndarray, band: int) -> np.ndarray | None:
    """Écart de couleur (médiane par canal, original − généré) sur le bord intérieur de la zone refaite.

    Ce bord montre du décor des deux côtés (la personne est au milieu de la zone) : c'est là que le décor régénéré
    se voit s'il n'a pas exactement la teinte d'origine. Seuls comptent les pixels déjà proches des deux côtés (écart
    < 40) : pas ceux où la personne touche le bord, ni l'ancienne personne encore visible dans l'original.
    None : trop peu de décor commun pour mesurer (personne qui remplit la zone) ; on ne corrige pas.
    """
    inner = cv2.erode(zone, np.ones((2 * band + 1, 2 * band + 1), np.uint8))
    ring = (zone > 0) & (inner == 0)
    diff = original[ring].astype(np.int16) - generated[ring].astype(np.int16)
    decor = diff[(np.abs(diff) < 40).all(axis=1)]
    if len(decor) < max(500, 0.3 * len(diff)):
        return None
    return np.median(decor, axis=0).astype(np.float32)


def recompose(generated: Path, mask: Path | None, original: Path, out: Path, fps_str: str, crf: int, preset: str,
              feather: float = 0.015, match_color: bool = True) -> int:
    """Recolle la personne générée sur l'extrait d'origine ; renvoie le nombre d'images écrites.

    Sans masque : l'image générée entière, agrandie (moins nette, mais cadence et durée d'origine).
    match_color : la zone refaite reprend la teinte du décor d'origine (sinon un rectangle un peu plus clair ou plus
    foncé se voit autour de la personne) ; écart mesuré à chaque image, lissé dans le temps pour ne pas clignoter.
    """
    info = media.probe(original)
    size = (info.width, info.height)
    k = max(3, int(info.height * feather) // 2 * 2 + 1)      # adoucissement proportionnel à la hauteur
    band = max(4, int(info.height * 0.03))                     # bord intérieur où mesurer l'écart de couleur
    gen, msk = _Timeline(generated), _Timeline(mask) if mask else None
    cap = cv2.VideoCapture(str(original))
    fps = info.fps or 30.0
    count = 0
    offset: np.ndarray | None = None
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
                    zone = cv2.resize(m[:, :, 0], size, interpolation=cv2.INTER_LINEAR)
                    if match_color:
                        measured = color_offset(frame, g, (zone > 127).astype(np.uint8), band)
                        if measured is not None:
                            offset = measured if offset is None else 0.8 * offset + 0.2 * measured
                        if offset is not None:
                            g = np.clip(g.astype(np.float32) + offset, 0, 255)
                    alpha = cv2.GaussianBlur(zone.astype(np.float32) / 255, (k, k), 0)[:, :, None]
                    frame = (frame * (1 - alpha) + g * alpha).astype(np.uint8)
                writer.write(frame)
                count += 1
    finally:
        cap.release()
        gen.close()
        if msk:
            msk.close()
    return count
