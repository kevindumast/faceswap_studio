"""Niveau 2 : transfert du teint de la personne source sur la peau visible de la cible (visage, oreilles, cou).

La segmentation seule se trompe sur les profils (chemise prise pour de la peau…). On combine donc trois indices :
1. les classes « peau / nez / oreilles / cou » de BiSeNet ;
2. une zone géométrique autour du visage détecté (ellipse + bande du cou) ;
3. un filtre couleur : on ne garde que les pixels proches de la couleur du cœur du visage (toujours de la peau).

Stabilité (le teint « sautait », faisait des plaques et virait parfois au gris-bleu) :
- la correction est un décalage LAB constant sur tout l'extrait, calculé une fois sur des images réparties (pré-passe) :
  un changement de lumière dans la vidéo passe tel quel, et rien ne varie plus d'une image à l'autre ;
- pas d'étirement du chroma : il poussait les ombres, la barbe et le cou sous le neutre, vers le gris-bleu ;
- teints source et cible comparés chacun à la dominante de son éclairage (photo sous lumière chaude ≠ vidéo froide) ;
- zones sombres protégées, filtre couleur progressif, masque lissé dans le temps (déformé selon le mouvement du visage).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import parsing
from .parsing import EXCLUDE_TONE, TONE_CLASSES

MIN_PIXELS = 150
VERSION = 2                 # calcul du teint source : invalide les tone.json de la bibliothèque quand il change
NEUTRAL_MAX = 8.0           # dominante de l'éclairage retenue au plus (a/b, unités LAB 8 bits)
DARK = (20.0, 60.0)         # L : aucune correction en dessous du 1er seuil, pleine correction au-dessus du 2e


@dataclass
class ToneStats:
    """Couleur de peau en LAB (OpenCV 8 bits : L 0–255, a/b centrés sur 128) : médiane et dispersion robuste.

    neutral : dominante a/b de l'éclairage (écart au gris) des images où le teint a été mesuré.
    """
    mean: np.ndarray
    std: np.ndarray
    neutral: np.ndarray | None = None

    def to_dict(self) -> dict:
        d = {"mean": self.mean.tolist(), "std": self.std.tolist()}
        if self.neutral is not None:
            d["neutral"] = self.neutral.tolist()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ToneStats":
        neutral = d.get("neutral")
        return cls(np.array(d["mean"], np.float32), np.array(d["std"], np.float32),
                   None if neutral is None else np.array(neutral, np.float32))


@dataclass
class ToneSettings:
    strength_l: float = 0.5      # part de l'écart de luminosité transférée (0 = garde l'éclairage de la scène)
    smoothing: float = 0.8       # EMA de la couleur de référence du filtre (et du teint cible sans pré-passe)
    chroma_scale: float = 1.0    # étirement du chroma autour du teint cible (1 = simple décalage)
    wb: float = 0.5              # part de la compensation de la dominante de l'éclairage (photo ↔ vidéo)
    max_shift: float = 25.0      # décalage a/b maximal
    mask_smoothing: float = 0.6  # part du masque précédent (déformé) gardée d'une image à l'autre
    feather: float = 0.012       # flou du bord du masque (sigma, fraction du recadrage)
    hold: int = 6                # images où l'on garde le masque précédent si la peau n'est pas trouvée

    @classmethod
    def from_config(cls, lcfg) -> "ToneSettings":
        d = cls()
        return cls(strength_l=float(lcfg.get("tone_strength_l", d.strength_l)),
                   smoothing=float(lcfg.get("tone_smoothing", d.smoothing)),
                   chroma_scale=float(lcfg.get("tone_chroma_scale", d.chroma_scale)),
                   wb=float(lcfg.get("tone_wb_compensation", d.wb)),
                   max_shift=float(lcfg.get("tone_max_shift", d.max_shift)),
                   mask_smoothing=float(lcfg.get("mask_smoothing", d.mask_smoothing)),
                   feather=float(lcfg.get("tone_feather", d.feather)),
                   hold=int(lcfg.get("tone_hold", d.hold)))


def _lab(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)


def smoothstep(x, lo: float, hi: float):
    t = np.clip((np.asarray(x, np.float32) - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def robust_stats(pixels: np.ndarray) -> ToneStats:
    med = np.median(pixels, axis=0)
    mad = np.median(np.abs(pixels - med), axis=0) * 1.4826
    return ToneStats(med.astype(np.float32), np.maximum(mad, 2.0).astype(np.float32))


def _ema(prev: ToneStats | None, cur: ToneStats, s: float) -> ToneStats:
    if prev is None:
        return cur
    neutral = cur.neutral if prev.neutral is None or cur.neutral is None else s * prev.neutral + (1 - s) * cur.neutral
    return ToneStats(s * prev.mean + (1 - s) * cur.mean, s * prev.std + (1 - s) * cur.std, neutral)


def scene_neutral(img: np.ndarray) -> np.ndarray:
    """Dominante de couleur de l'éclairage (a, b centrés sur 0) : moyenne des pixels presque gris de l'image.

    Estimation grossière (« monde gris » restreint aux pixels peu colorés), bornée à ±NEUTRAL_MAX.
    """
    scale = 256 / max(img.shape[:2])
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else img
    lab = _lab(small).reshape(-1, 3)
    ab = lab[:, 1:] - 128
    mid = (lab[:, 0] > 40) & (lab[:, 0] < 230)
    grayish = mid & (np.linalg.norm(ab, axis=1) < 20)
    pick = ab[grayish] if grayish.sum() > 0.05 * len(ab) else ab[mid] if mid.any() else ab
    return np.clip(pick.mean(axis=0), -NEUTRAL_MAX, NEUTRAL_MAX).astype(np.float32)


def _ellipse(shape, center, axes) -> np.ndarray:
    m = np.zeros(shape[:2], np.uint8)
    cv2.ellipse(m, (int(center[0]), int(center[1])), (max(1, int(axes[0])), max(1, int(axes[1]))), 0, 0, 360, 1, -1)
    return m.astype(bool)


def face_core(lab: np.ndarray, bbox, kps, excluded: np.ndarray) -> np.ndarray:
    """Cœur du visage : petite ellipse centrée sur les points clés (entre yeux et bouche), hors yeux / bouche / lunettes,
    ni reflets ni ombres profondes."""
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    center = np.asarray(kps, np.float32).mean(axis=0)
    core = _ellipse(lab.shape, center, (0.28 * (x2 - x1), 0.36 * (y2 - y1))) & ~excluded
    lum = lab[..., 0]
    return core & (lum > 25) & (lum < 245)


def skin_regions(crop: np.ndarray, bbox, kps, labels: np.ndarray | None = None,
                 ref: ToneStats | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(poids 0–1 de la zone à recolorer, cœur du visage) dans le repère du crop. bbox/kps déjà exprimés dans ce repère.

    ref : couleur (a, b) de référence du filtre ; par défaut celle du cœur du visage sur ce crop.
    """
    if labels is None:
        labels = parsing.parse(crop)
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    fw, fh = x2 - x1, y2 - y1
    excluded = parsing.mask(labels, EXCLUDE_TONE)
    lab = _lab(crop)
    core = face_core(lab, bbox, kps, excluded)
    if core.sum() < MIN_PIXELS:
        return np.zeros(crop.shape[:2], np.float32), core
    if ref is None:
        ref = robust_stats(lab[core][:, 1:])  # a, b uniquement : L varie avec l'ombrage

    # Zone géométrique : visage élargi + cou sous le menton.
    geo = _ellipse(crop.shape, ((x1 + x2) / 2, (y1 + y2) / 2), (0.68 * fw, 0.62 * fh))
    neck = np.zeros(crop.shape[:2], bool)
    cx = (x1 + x2) / 2
    neck[int(max(0, y2 - 0.1 * fh)):int(min(crop.shape[0], y2 + 0.9 * fh)),
         int(max(0, cx - 0.38 * fw)):int(min(crop.shape[1], cx + 0.38 * fw))] = True
    geo |= neck

    # Filtre couleur progressif (distance a/b normalisée au cœur du visage) : pas de pixels qui clignotent au seuil.
    d = np.sqrt(((lab[..., 1] - ref.mean[0]) / (ref.std[0] * 3.5)) ** 2 + ((lab[..., 2] - ref.mean[1]) / (ref.std[1] * 3.5)) ** 2)
    close = 1.0 - smoothstep(d, 0.8, 1.3)

    # Sous le menton, BiSeNet rate souvent le cou (barbe, ombre, chaîne) : on y accepte aussi le « fond »,
    # tant que la couleur est celle de la peau. Sans ça, on verrait une démarcation nette à la mâchoire.
    neck_extra = neck & parsing.mask(labels, (parsing.BACKGROUND,))
    candidates = parsing.mask(labels, TONE_CLASSES) | core | neck_extra
    weights = (candidates & geo & ~excluded).astype(np.float32) * close * smoothstep(lab[..., 0], *DARK)
    return weights, core


def source_tone(images: list[np.ndarray], faces: list) -> ToneStats | None:
    """Teint de référence d'une personne : médiane des teints de ses photos (et du visage principal de chacune).

    Mesuré photo par photo puis combiné : un gros tas de pixels de photos aux éclairages différents gonflait la
    dispersion. Une photo au teint très éloigné des autres (lumière colorée, filtre) est écartée.
    """
    measures = []
    for img, face in zip(images, faces):
        bx1, by1, bx2, by2 = parsing.face_crop_box(face.bbox, img.shape)
        crop = img[by1:by2, bx1:bx2]
        if crop.size == 0:
            continue
        bbox = np.asarray(face.bbox[:4], np.float32) - [bx1, by1, bx1, by1]
        kps = np.asarray(face.kps, np.float32) - [bx1, by1]
        _, core = skin_regions(crop, bbox, kps)
        if core.sum() >= MIN_PIXELS:
            measures.append((robust_stats(_lab(crop)[core]), scene_neutral(img)))
    if not measures:
        return None
    ab = np.stack([s.mean[1:] for s, _ in measures])
    dist = np.linalg.norm(ab - np.median(ab, axis=0), axis=1)
    kept = [m for m, dd in zip(measures, dist) if dd <= max(6.0, 2.5 * float(np.median(dist)))]
    return ToneStats(np.median(np.stack([s.mean for s, _ in kept]), axis=0).astype(np.float32),
                     np.median(np.stack([s.std for s, _ in kept]), axis=0).astype(np.float32),
                     np.median(np.stack([n for _, n in kept]), axis=0).astype(np.float32))


def tone_shift(target: ToneStats, source: ToneStats, strength_l: float, wb: float = 0.0,
               max_shift: float = 25.0) -> np.ndarray:
    """Décalage LAB (L, a, b) à ajouter à la peau de la cible pour lui donner le teint de la source.

    a/b : écart des deux teints, chacun compté par rapport à la dominante de son éclairage (wb = part de cette
    compensation). L : seulement une part (strength_l) de l'écart, l'éclairage de la scène reste maître.
    """
    src_ab, tgt_ab = source.mean[1:].astype(np.float32), target.mean[1:].astype(np.float32)
    if wb > 0 and source.neutral is not None and target.neutral is not None:
        src_ab = src_ab - wb * source.neutral
        tgt_ab = tgt_ab - wb * target.neutral
    ab = np.clip(src_ab - tgt_ab, -max_shift, max_shift)
    return np.array([strength_l * (source.mean[0] - target.mean[0]), ab[0], ab[1]], np.float32)


def transfer(crop: np.ndarray, weights: np.ndarray, shift: np.ndarray, chroma_scale: float = 1.0,
             target_ab: np.ndarray | None = None) -> np.ndarray:
    """Recolore le crop selon les poids (0–1) : décalage LAB `shift`, et étirement du chroma en option."""
    lab = _lab(crop)
    out = lab + shift
    if chroma_scale != 1.0 and target_ab is not None:
        out[..., 1:] = (lab[..., 1:] - target_ab) * chroma_scale + target_ab + shift[1:]
    out_bgr = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    alpha = weights[..., None]
    return (crop.astype(np.float32) * (1 - alpha) + out_bgr.astype(np.float32) * alpha).astype(np.uint8)


class ToneMatcher:
    """Applique le teint source sur un visage suivi, image après image.

    Sans pré-passe (prepare), le teint de la cible est suivi lentement ; avec, il est figé pour tout l'extrait.
    """

    def __init__(self, source: ToneStats, settings: ToneSettings | None = None):
        self.source = source
        self.cfg = settings or ToneSettings()
        self.target: ToneStats | None = None      # teint de la cible (L, a, b) + dominante de la scène
        self.frozen = False
        self.ref: ToneStats | None = None         # couleur (a, b) du cœur, lissée : référence du filtre couleur
        self.prev: tuple | None = None            # (poids, x1, y1, points clés) de l'image précédente
        self.missed = 0

    @staticmethod
    def _local(frame: np.ndarray, face):
        x1, y1, x2, y2 = parsing.face_crop_box(face.bbox, frame.shape)
        bbox = np.asarray(face.bbox[:4], np.float32) - [x1, y1, x1, y1]
        kps = np.asarray(face.kps, np.float32) - [x1, y1]
        return (x1, y1, x2, y2), bbox, kps

    def prepare(self, samples: list[tuple[np.ndarray, object]]) -> None:
        """Teint de la cible figé pour tout l'extrait : médiane sur des images réparties (images originales)."""
        means, stds, neutrals = [], [], []
        for frame, face in samples:
            (x1, y1, x2, y2), bbox, kps = self._local(frame, face)
            if x2 - x1 < 24 or y2 - y1 < 24:
                continue
            crop = frame[y1:y2, x1:x2]
            _, core = skin_regions(crop, bbox, kps)
            if core.sum() < MIN_PIXELS:
                continue
            s = robust_stats(_lab(crop)[core])
            means.append(s.mean)
            stds.append(s.std)
            neutrals.append(scene_neutral(frame))
        if means:
            self.target = ToneStats(np.median(means, axis=0).astype(np.float32), np.median(stds, axis=0).astype(np.float32),
                                    np.median(neutrals, axis=0).astype(np.float32))
            self.frozen = True

    def shift(self) -> np.ndarray:
        """Décalage LAB (L, a, b) vers le teint de la personne ; nul tant que le teint de la cible est inconnu."""
        if self.target is None:
            return np.zeros(3, np.float32)
        return tone_shift(self.target, self.source, self.cfg.strength_l, self.cfg.wb, self.cfg.max_shift)

    def _warp_prev(self, x1: int, y1: int, kps: np.ndarray, shape) -> np.ndarray | None:
        """Masque de l'image précédente, déplacé comme le visage (similitude sur les points clés), dans le crop courant."""
        if self.prev is None:
            return None
        weights, px, py, pkps = self.prev
        A, _ = cv2.estimateAffinePartial2D(pkps, kps, method=cv2.LMEDS)
        if A is None:
            return None
        T = A.copy()
        T[:, 2] += A[:, :2] @ np.array([px, py], np.float64) - np.array([x1, y1], np.float64)
        return cv2.warpAffine(weights, T, (shape[1], shape[0]), flags=cv2.INTER_LINEAR, borderValue=0.0)

    def apply(self, frame: np.ndarray, face, original: np.ndarray | None = None) -> np.ndarray:
        """Recolore la peau de `frame`. Zone et mesures prises sur `original` (l'image avant swap) si fourni."""
        cfg = self.cfg
        src = frame if original is None else original
        (x1, y1, x2, y2), bbox, kps = self._local(frame, face)
        if x2 - x1 < 24 or y2 - y1 < 24:
            self.prev = None
            return frame
        crop = src[y1:y2, x1:x2]
        labels = parsing.parse(crop)
        lab = _lab(crop)
        core = face_core(lab, bbox, kps, parsing.mask(labels, EXCLUDE_TONE))
        weights = None
        if core.sum() >= MIN_PIXELS:
            current = robust_stats(lab[core])
            self.ref = _ema(self.ref, ToneStats(current.mean[1:], current.std[1:]), cfg.smoothing)
            if not self.frozen:   # sans pré-passe : suivi lent du teint cible
                current.neutral = scene_neutral(src)
                self.target = _ema(self.target, current, max(cfg.smoothing, 0.95))
            weights, _ = skin_regions(crop, bbox, kps, labels, ref=self.ref)
            if weights.sum() < MIN_PIXELS:
                weights = None

        kps_frame = np.asarray(face.kps, np.float32)
        prev = self._warp_prev(x1, y1, kps_frame, crop.shape)
        if weights is not None:
            if cfg.feather > 0:
                weights = cv2.GaussianBlur(weights, (0, 0), cfg.feather * min(crop.shape[:2]))
            if prev is not None and cfg.mask_smoothing > 0:
                size = max(np.sqrt(max(bbox[2] - bbox[0], 1) * max(bbox[3] - bbox[1], 1)), 1.0)
                motion = float(np.mean(np.linalg.norm(kps_frame - self.prev[3], axis=1))) / size
                m = cfg.mask_smoothing * max(0.0, 1.0 - motion / 0.1)
                weights = m * prev + (1 - m) * weights
            self.missed = 0
        elif prev is not None and self.missed < cfg.hold:
            # Peau introuvable sur cette image (profil, flou) : masque précédent, qui s'efface en quelques images.
            self.missed += 1
            weights = prev * (0.85 ** self.missed)
        else:
            self.prev = None
            return frame
        self.prev = (weights, x1, y1, kps_frame.copy())
        if self.target is None:
            return frame
        shift = self.shift()
        out = frame.copy()
        out[y1:y2, x1:x2] = transfer(frame[y1:y2, x1:x2], weights, shift, cfg.chroma_scale, self.target.mean[1:])
        return out


def load_images(paths: list[Path]) -> list[np.ndarray]:
    imgs = []
    for p in paths:
        img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            imgs.append(img)
    return imgs
