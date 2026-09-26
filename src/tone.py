"""Niveau 2 : transfert du teint de la personne source sur la peau visible de la cible (visage, oreilles, cou).

La segmentation seule se trompe sur les profils (chemise prise pour de la peau…). On combine donc trois indices :
1. les classes « peau / nez / oreilles / cou » de BiSeNet ;
2. une zone géométrique autour du visage détecté (ellipse + bande du cou) ;
3. un filtre couleur : on ne garde que les pixels proches de la couleur du cœur du visage (toujours de la peau).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import parsing
from .parsing import EXCLUDE_TONE, TONE_CLASSES

MIN_PIXELS = 150


@dataclass
class ToneStats:
    """Couleur de peau en LAB (OpenCV 8 bits : L 0–255, a/b centrés sur 128) : médiane et dispersion robuste."""
    mean: np.ndarray
    std: np.ndarray

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "ToneStats":
        return cls(np.array(d["mean"], np.float32), np.array(d["std"], np.float32))


def _lab(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)


def robust_stats(pixels: np.ndarray) -> ToneStats:
    med = np.median(pixels, axis=0)
    mad = np.median(np.abs(pixels - med), axis=0) * 1.4826
    return ToneStats(med.astype(np.float32), np.maximum(mad, 2.0).astype(np.float32))


def _ellipse(shape, center, axes) -> np.ndarray:
    m = np.zeros(shape[:2], np.uint8)
    cv2.ellipse(m, (int(center[0]), int(center[1])), (max(1, int(axes[0])), max(1, int(axes[1]))), 0, 0, 360, 1, -1)
    return m.astype(bool)


def skin_regions(crop: np.ndarray, bbox, kps, labels: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(zone à recolorer, cœur du visage) dans le repère du crop. bbox/kps déjà exprimés dans ce repère."""
    if labels is None:
        labels = parsing.parse(crop)
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    fw, fh = x2 - x1, y2 - y1
    center = np.asarray(kps, np.float32).mean(axis=0)
    excluded = parsing.mask(labels, EXCLUDE_TONE)
    lab = _lab(crop)

    # Cœur : petite ellipse centrée sur les points clés (entre yeux et bouche), hors yeux / bouche / lunettes.
    core = _ellipse(crop.shape, center, (0.28 * fw, 0.36 * fh)) & ~excluded
    lum = lab[..., 0]
    core &= (lum > 25) & (lum < 245)  # ni reflets ni ombres profondes
    if core.sum() < MIN_PIXELS:
        return np.zeros(crop.shape[:2], bool), core
    ref = robust_stats(lab[core][:, 1:])  # a, b uniquement : L varie avec l'ombrage

    # Zone géométrique : visage élargi + cou sous le menton.
    geo = _ellipse(crop.shape, ((x1 + x2) / 2, (y1 + y2) / 2), (0.68 * fw, 0.62 * fh))
    neck = np.zeros(crop.shape[:2], bool)
    cx = (x1 + x2) / 2
    neck[int(max(0, y2 - 0.1 * fh)):int(min(crop.shape[0], y2 + 0.9 * fh)),
         int(max(0, cx - 0.38 * fw)):int(min(crop.shape[1], cx + 0.38 * fw))] = True
    geo |= neck

    # Filtre couleur (distance a/b normalisée au cœur du visage).
    d = np.sqrt(((lab[..., 1] - ref.mean[0]) / (ref.std[0] * 3.5)) ** 2 + ((lab[..., 2] - ref.mean[1]) / (ref.std[1] * 3.5)) ** 2)
    close = d < 1.0

    # Sous le menton, BiSeNet rate souvent le cou (barbe, ombre, chaîne) : on y accepte aussi le « fond »,
    # tant que la couleur est celle de la peau. Sans ça, on verrait une démarcation nette à la mâchoire.
    neck_extra = neck & parsing.mask(labels, (parsing.BACKGROUND,))
    candidates = parsing.mask(labels, TONE_CLASSES) | core | neck_extra
    region = candidates & geo & close & ~excluded
    return region, core


def source_tone(images: list[np.ndarray], faces: list) -> ToneStats | None:
    """Teint de référence d'une personne, à partir de ses photos (et du visage principal de chacune)."""
    pixels = []
    for img, face in zip(images, faces):
        bx1, by1, bx2, by2 = parsing.face_crop_box(face.bbox, img.shape)
        crop = img[by1:by2, bx1:bx2]
        if crop.size == 0:
            continue
        bbox = np.asarray(face.bbox[:4], np.float32) - [bx1, by1, bx1, by1]
        kps = np.asarray(face.kps, np.float32) - [bx1, by1]
        _, core = skin_regions(crop, bbox, kps)
        if core.sum() >= MIN_PIXELS:
            pixels.append(_lab(crop)[core])
    if not pixels:
        return None
    return robust_stats(np.concatenate(pixels))


def transfer(crop: np.ndarray, region: np.ndarray, target: ToneStats, source: ToneStats, strength_l: float) -> np.ndarray:
    """Recolore `region` : a/b ramenés sur la source (moyenne + dispersion), L seulement décalé en partie."""
    lab = _lab(crop)
    out = lab.copy()
    ratio = np.clip(source.std[1:] / target.std[1:], 0.5, 2.0)
    out[..., 1:] = (lab[..., 1:] - target.mean[1:]) * ratio + source.mean[1:]
    out[..., 0] = lab[..., 0] + strength_l * (source.mean[0] - target.mean[0])
    out_bgr = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    k = max(3, int(min(crop.shape[:2]) * 0.03) | 1)
    alpha = cv2.GaussianBlur(region.astype(np.float32), (k, k), 0)[..., None]
    return (crop.astype(np.float32) * (1 - alpha) + out_bgr.astype(np.float32) * alpha).astype(np.uint8)


class ToneMatcher:
    """Applique le teint source sur un visage suivi, image après image, avec lissage des stats de la cible."""

    def __init__(self, source: ToneStats, strength_l: float = 0.5, smoothing: float = 0.7):
        self.source = source
        self.strength_l = strength_l
        self.smoothing = smoothing
        self.target: ToneStats | None = None

    def apply(self, frame: np.ndarray, face) -> np.ndarray:
        x1, y1, x2, y2 = parsing.face_crop_box(face.bbox, frame.shape)
        if x2 - x1 < 24 or y2 - y1 < 24:
            return frame
        crop = frame[y1:y2, x1:x2]
        bbox = np.asarray(face.bbox[:4], np.float32) - [x1, y1, x1, y1]
        kps = np.asarray(face.kps, np.float32) - [x1, y1]
        region, core = skin_regions(crop, bbox, kps)
        if region.sum() < MIN_PIXELS or core.sum() < MIN_PIXELS:
            return frame
        current = robust_stats(_lab(crop)[core])
        if self.target is None:
            self.target = current
        else:  # EMA : évite que la couleur « respire » d'une image à l'autre
            s = self.smoothing
            self.target = ToneStats(s * self.target.mean + (1 - s) * current.mean, s * self.target.std + (1 - s) * current.std)
        out = frame.copy()
        out[y1:y2, x1:x2] = transfer(crop, region, self.target, self.source, self.strength_l)
        return out


def load_images(paths: list[Path]) -> list[np.ndarray]:
    imgs = []
    for p in paths:
        img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            imgs.append(img)
    return imgs
