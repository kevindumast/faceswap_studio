"""Niveau 3 : tête entière de la photo de la personne (visage, cheveux, forme du crâne), animée par la tête du clip.

- LivePortrait (4 modèles ONNX, carte graphique) anime la photo avec la pose de la cible (absolue : la tête suit le
  corps) et son expression, sur un recadrage 512 aligné sur les 5 points clés et élargi aux cheveux ;
- BiSeNet découpe la nouvelle tête, et l'ancienne avec ses accessoires (casquette, casque…) ;
- le décor que cachait l'ancienne tête vient d'un fond propre reconstruit une fois pour l'extrait à partir des autres
  images (plan fixe : quand la tête bouge, le décor derrière se découvre). LaMa (CPU, ~4 s, DirectML le refuse) ne
  comble que ce qui n'est jamais visible. Caméra qui bouge : effaceur rapide d'OpenCV, image par image ;
- la nouvelle tête prend la lumière de la scène et le teint de la personne (calcul du niveau 2) ; le cou de la cible
  reçoit ce même teint.

Masques et couleur sont lissés dans le repère du recadrage aligné : il suit le visage, donc d'une image à l'autre le
masque de la tête y reste presque immobile.
"""
from __future__ import annotations

import threading
import warnings
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from . import parsing
from .config import load_config, providers
from .faces import ModelsMissing

P = parsing
# Nouvelle tête collée : tout sauf le cou, les vêtements et le fond.
HEAD = (P.SKIN, P.L_BROW, P.R_BROW, P.L_EYE, P.R_EYE, P.GLASSES, P.L_EAR, P.R_EAR, P.EARRING, P.NOSE, P.MOUTH,
        P.U_LIP, P.L_LIP, P.HAIR, P.HAT)
BODY = (P.NECK, P.NECKLACE, P.CLOTH)           # jamais « ancienne tête » : reste de la cible
# Ancienne tête effacée là où la nouvelle ne la couvre pas : cheveux, chapeau, bijoux partout ; visage (peau, oreilles,
# traits) seulement au-dessus du menton. Sous le menton, la « peau » est le cou (BiSeNet le classe souvent ainsi) :
# l'effacer le creusait. De 3/4, sans ça, joue et mâchoire de la cible dépassaient derrière la nouvelle tête.
ERASE = (P.HAIR, P.HAT, P.EARRING)
FACE_PARTS = (P.SKIN, P.NOSE, P.L_EAR, P.R_EAR, P.L_BROW, P.R_BROW, P.L_EYE, P.R_EYE, P.GLASSES, P.MOUTH,
              P.U_LIP, P.L_LIP)
KEEP = BODY + FACE_PARTS                        # jamais pris pour un accessoire (casque…) par l'écart au fond
SKIN_CORE = (P.SKIN, P.NOSE)
# Gabarit FFHQ 512 (5 points clés normalisés), celui de LivePortrait dans FaceFusion.
TEMPLATE = np.array([[0.37691676, 0.46864664], [0.62285697, 0.46912813], [0.50123859, 0.61331904],
                     [0.39308822, 0.72541100], [0.61150205, 0.72490465]], np.float32)
CROP = 512
LP_MODELS = ("feature_extractor", "motion_extractor", "generator", "stitcher")
FILES = {name: f"live_portrait_{name}.onnx" for name in LP_MODELS} | {"lama": "lama_fp32.onnx"}
MAX_COLOR = np.array([30.0, 15.0, 15.0], np.float32)   # correction de lumière / teint de la tête générée, au plus
_lock = threading.Lock()


def model_path(name: str) -> Path:
    cfg = load_config()
    return cfg.path("models") / "head" / cfg.get("models", {}).get("head", {}).get(name, FILES[name])


@lru_cache(maxsize=None)
def _session(name: str):
    import onnxruntime as ort

    path = model_path(name)
    if not path.is_file():
        raise ModelsMissing("Modèles du niveau 3 absents : python scripts/download_models.py --level head")
    cfg = load_config()
    hcfg = cfg.get("levels", {}).get("head", {})
    # Processeur par défaut : sur une MX450, le générateur (~1,1 s par appel) fait décrocher le pilote NVIDIA
    # (délai de 2 s de Windows dépassé, écran noir, carte inutilisable ensuite pour le process), même découpé.
    # LaMa toujours sur le CPU : DirectML refuse sa transformée de Fourier (et il ne sert qu'une fois par extrait).
    on_gpu = name != "lama" and hcfg.get("device", "cpu") == "gpu"
    provs = providers(cfg, role="swap") if on_gpu else ["CPUExecutionProvider"]
    so = ort.SessionOptions()
    if on_gpu and name == "generator" and hcfg.get("split_gpu_work", True):
        so.add_session_config_entry("ep.dml.disable_graph_fusion", "1")   # petites commandes plutôt qu'un gros bloc
    return ort.InferenceSession(str(path), so, providers=provs)


def run(name: str, feed: dict) -> list:
    sess = _session(name)
    with _lock:
        return sess.run(None, feed)


def align(frame: np.ndarray, kps, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Recadrage 512 aligné sur les 5 points clés, visage réduit de `scale` autour du nez (cheveux compris)."""
    kps = np.asarray(kps, np.float32)
    kps = (kps - kps[2]) * scale + kps[2]
    M = cv2.estimateAffinePartial2D(kps, TEMPLATE * CROP, method=cv2.RANSAC, ransacReprojThreshold=100)[0]
    crop = cv2.warpAffine(frame, M, (CROP, CROP), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_AREA)
    return crop, M


def to_input(crop: np.ndarray) -> np.ndarray:
    x = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_AREA)[:, :, ::-1] / 255.0
    return x.transpose(2, 0, 1)[None].astype(np.float32)


def from_output(y: np.ndarray) -> np.ndarray:
    img = np.ascontiguousarray((y.transpose(1, 2, 0).clip(0, 1) * 255).astype(np.uint8)[:, :, ::-1])
    return img if img.shape[0] == CROP else cv2.resize(img, (CROP, CROP))


def rotation(pitch: float, yaw: float, roll: float) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_euler("xyz", [pitch, yaw, roll], degrees=True).as_matrix().astype(np.float32)


@dataclass
class Motion:
    """Sortie du modèle de mouvement : pose (degrés), échelle, position, expression et points canoniques (21×3)."""
    pitch: float
    yaw: float
    roll: float
    scale: np.ndarray
    translation: np.ndarray
    expression: np.ndarray
    points: np.ndarray

    @classmethod
    def extract(cls, crop: np.ndarray) -> "Motion":
        pitch, yaw, roll, scale, translation, expression, points = run("motion_extractor", {"input": to_input(crop)})
        return cls(float(np.ravel(pitch)[0]), float(np.ravel(yaw)[0]), float(np.ravel(roll)[0]),
                   np.asarray(scale, np.float32), np.asarray(translation, np.float32),
                   np.asarray(expression, np.float32), np.asarray(points, np.float32))

    def keypoints(self) -> np.ndarray:
        rot = rotation(self.pitch, self.yaw, self.roll)
        return (self.scale * (self.points @ rot.T + self.expression) + self.translation).astype(np.float32)


class MotionSmoother:
    """EMA adaptative de la pose (forte au repos, nulle quand la tête tourne vite) ; expression lissée plus légèrement
    pour garder une bouche réactive. LivePortrait tremble un peu d'une image à l'autre sans ça."""

    def __init__(self, smoothing: float):
        self.smoothing = smoothing
        self.prev: Motion | None = None
        self.change = 0.0          # rotation depuis l'image précédente (degrés)

    def __call__(self, m: Motion) -> Motion:
        p = self.prev
        if p is None or self.smoothing <= 0:
            self.prev, self.change = m, 0.0
            return m
        self.change = max(abs(m.pitch - p.pitch), abs(m.yaw - p.yaw), abs(m.roll - p.roll))
        a = self.smoothing * max(0.0, 1.0 - self.change / 8.0)
        e = self.smoothing * 0.5
        out = replace(m, pitch=a * p.pitch + (1 - a) * m.pitch, yaw=a * p.yaw + (1 - a) * m.yaw,
                      roll=a * p.roll + (1 - a) * m.roll, scale=a * p.scale + (1 - a) * m.scale,
                      translation=a * p.translation + (1 - a) * m.translation,
                      expression=e * p.expression + (1 - e) * m.expression)
        self.prev = out
        return out


@dataclass
class SourceHead:
    """La tête de la photo, calculée une fois : volume d'apparence + mouvement et points de la photo."""
    feature: np.ndarray
    motion: Motion
    keypoints: np.ndarray

    @classmethod
    def from_photo(cls, img: np.ndarray, kps, scale: float) -> "SourceHead":
        crop, _ = align(img, kps, scale)
        x = to_input(crop)
        feature = run("feature_extractor", {"input": x})[0]
        pitch, yaw, roll, s, t, exp, pts = run("motion_extractor", {"input": x})
        motion = Motion(float(np.ravel(pitch)[0]), float(np.ravel(yaw)[0]), float(np.ravel(roll)[0]),
                        np.asarray(s, np.float32), np.asarray(t, np.float32), np.asarray(exp, np.float32),
                        np.asarray(pts, np.float32))
        return cls(feature, motion, motion.keypoints())

    @classmethod
    def from_path(cls, path: Path, scale: float) -> "SourceHead":
        from .faces import detect
        from .identity import read_image_full

        img = read_image_full(path)
        faces = detect(img) if img is not None else []
        if not faces:
            raise RuntimeError(f"Aucun visage sur la photo choisie pour la tête ({path.name}).")
        return cls.from_photo(img, faces[0].kps, scale)


def generate(source: SourceHead, target: Motion) -> np.ndarray:
    """La tête de la photo avec la pose, la taille, la position et l'expression de la cible (image 512 BGR)."""
    driving = replace(target, points=source.motion.points).keypoints()
    driving = run("stitcher", {"source": driving, "target": source.keypoints})[0]
    out = run("generator", {"feature_volume": source.feature, "source": driving, "target": source.keypoints})[0][0]
    return from_output(out)


def frontal_score(kps, bbox) -> float:
    """Photo de tête idéale : visage grand (pixels) et de face (nez centré entre les yeux, yeux à l'horizontale)."""
    kps = np.asarray(kps, np.float32)
    eyes = kps[1] - kps[0]
    dist = max(float(np.linalg.norm(eyes)), 1.0)
    yaw = abs(float(kps[2, 0] - (kps[0, 0] + kps[1, 0]) / 2)) / dist
    roll = abs(float(np.arctan2(eyes[1], eyes[0])))
    size = min(float(bbox[3] - bbox[1]), 400.0) / 400.0
    return size * max(0.0, 1.0 - 2.0 * yaw) * max(0.0, 1.0 - roll)


def head_score(path: Path) -> float | None:
    """Note d'une photo pour la tête du niveau 3 (None : aucun visage)."""
    from .faces import detect
    from .identity import read_image

    img = read_image(path) if path.is_file() else None
    faces = detect(img) if img is not None else []
    return round(frontal_score(faces[0].kps, faces[0].bbox), 4) if faces else None


def choose_photo(paths: list[Path]) -> Path | None:
    """La photo la plus de face et la plus grande (choix automatique de la photo de tête)."""
    scores = [(s, p) for p in paths if (s := head_score(p)) is not None]
    return max(scores, key=lambda sp: sp[0])[1] if scores else None


# ─── Fond propre ───

@dataclass
class Plate:
    """Décor sans la personne autour de la tête (repère de l'image entière, à partir de (x0, y0))."""
    image: np.ndarray
    x0: int
    y0: int

    def crop(self, M: np.ndarray) -> np.ndarray:
        """Le fond dans le repère du recadrage aligné (M : image entière → recadrage)."""
        T = M.astype(np.float64).copy()
        T[:, 2] += M[:, :2] @ np.array([self.x0, self.y0], np.float64)
        return cv2.warpAffine(self.image, T, (CROP, CROP), borderMode=cv2.BORDER_REPLICATE)


def inpaint(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Comble `mask` : LaMa (CPU) si installé, sinon l'effaceur rapide d'OpenCV."""
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return img
    h, w = img.shape[:2]
    side = int(max(ys.max() - ys.min(), xs.max() - xs.min()) * 1.5) + 64
    cx, cy = (xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2
    x0, y0 = max(0, cx - side // 2), max(0, cy - side // 2)
    x1, y1 = min(w, x0 + side), min(h, y0 + side)
    region, hole = img[y0:y1, x0:x1], mask[y0:y1, x0:x1].astype(np.uint8)
    try:
        small = cv2.resize(region, (512, 512), interpolation=cv2.INTER_AREA)
        m = (cv2.resize(hole, (512, 512), interpolation=cv2.INTER_NEAREST) > 0).astype(np.float32)
        x = (small[:, :, ::-1] / 255.0).transpose(2, 0, 1)[None].astype(np.float32)
        y = run("lama", {"image": x, "mask": m[None, None]})[0][0].transpose(1, 2, 0)
        y = (y if y.max() > 2 else y * 255).clip(0, 255).astype(np.uint8)[:, :, ::-1]
        filled = cv2.resize(y, (x1 - x0, y1 - y0), interpolation=cv2.INTER_CUBIC)
    except ModelsMissing:
        filled = cv2.inpaint(region, hole, 5, cv2.INPAINT_TELEA)
    out = img.copy()
    out[y0:y1, x0:x1][hole > 0] = filled[hole > 0]
    return out


def build_plate(samples: list[tuple[np.ndarray, object]], scale: float, max_dev: float = 10.0) -> Plate | None:
    """Fond propre autour de la tête : médiane des images où il est visible (hors personne). None si la caméra bouge
    ou si trop peu d'images : on se rabat alors sur l'effaceur image par image."""
    if len(samples) < 4:
        return None
    h, w = samples[0][0].shape[:2]
    zones, valids = [], []
    for frame, face in samples:
        crop, M = align(frame, face.kps, scale)
        free = (parsing.parse(crop) == P.BACKGROUND).astype(np.uint8)
        free = cv2.erode(free, np.ones((15, 15), np.uint8))    # marge : mèches, flou de mouvement
        IM = cv2.invertAffineTransform(M)
        valids.append(cv2.warpAffine(free, IM, (w, h), flags=cv2.INTER_NEAREST, borderValue=0).astype(bool))
        zones.append(cv2.warpAffine(np.ones((CROP, CROP), np.uint8), IM, (w, h), flags=cv2.INTER_NEAREST).astype(bool))
    zone = np.any(zones, axis=0)
    ys, xs = np.nonzero(zone)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    plate = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
    seen = np.zeros((y1 - y0, x1 - x0), np.int32)
    devs = []
    for r in range(y0, y1, 48):                                 # par bandes : mémoire bornée même en 1080p
        r1 = min(y1, r + 48)
        valid = np.stack([v[r:r1, x0:x1] for v in valids])
        data = np.where(valid[..., None], np.stack([f[r:r1, x0:x1] for f, _ in samples]).astype(np.float32), np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)     # pixels jamais visibles : NaN attendu
            med = np.nanmedian(data, axis=0)
            # Écart au fond : 75e centile sur les images (la médiane s'annule dès que la moitié tombe juste par hasard).
            dev = np.nanpercentile(np.nanmean(np.abs(data - med[None]), axis=-1), 75, axis=0)
        count = valid.sum(axis=0)
        plate[r - y0:r1 - y0] = np.nan_to_num(med).clip(0, 255).astype(np.uint8)
        seen[r - y0:r1 - y0] = count
        multi = count >= 3
        if multi.any():
            devs.append(dev[multi])
    if not devs:
        return None
    devs = np.concatenate(devs)
    if len(devs) < 500 or float(np.median(devs)) > max_dev:    # la caméra bouge : pas de fond fixe
        return None
    unseen = (seen == 0) & zone[y0:y1, x0:x1]
    if unseen.any():
        plate = inpaint(plate, cv2.dilate(unseen.astype(np.uint8), np.ones((5, 5), np.uint8)))
    return Plate(plate, x0, y0)


# ─── Composition image par image ───

def chin_row(scale: float) -> int:
    """Ligne du menton dans le recadrage aligné : yeux et bouche y sont toujours à la même place (gabarit)."""
    nose = TEMPLATE[2, 1]

    def row(v: float) -> float:
        return CROP * (nose + (v - nose) / scale)

    eyes, mouth = row(TEMPLATE[:2, 1].mean()), row(TEMPLATE[3:, 1].mean())
    return int(mouth + 0.8 * (mouth - eyes))


@lru_cache(maxsize=1)
def accessory_zone() -> np.ndarray:
    """Où chercher les accessoires de l'ancienne tête (casque, casquette) dans le recadrage : autour et au-dessus du
    visage, jamais sous le menton (épaules, micro…)."""
    m = np.zeros((CROP, CROP), np.uint8)
    cv2.ellipse(m, (256, 236), (230, 205), 0, 0, 360, 1, -1)
    m[380:] = 0
    return m.astype(bool)


def _median_lab(img: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    if mask.sum() < 300:
        return None
    return np.median(cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)[mask], axis=0)


class HeadEngine:
    """Remplace la tête d'un visage suivi, image après image."""

    def __init__(self, source: SourceHead, scale: float = 1.5, smoothing: float = 0.5, tone=None):
        self.source = source
        self.scale = scale
        self.chin = chin_row(scale)
        self.motion = MotionSmoother(smoothing)
        self.smoothing = smoothing
        self.tone = tone                      # tone.ToneMatcher du niveau 2 (teint de la personne), ou None
        self.plate: Plate | None = None
        self.alpha: np.ndarray | None = None  # masque de la tête de l'image précédente (repère du recadrage)
        self.color: np.ndarray | None = None  # correction LAB de la tête générée, lissée

    def prepare(self, samples: list[tuple[np.ndarray, object]]) -> None:
        self.plate = build_plate(samples, self.scale)

    def _color_shift(self, head: np.ndarray, head_labels: np.ndarray, crop: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Décalage LAB : lumière de la scène (visage de la cible) + teint de la personne (écart du niveau 2)."""
        generated = _median_lab(head, parsing.mask(head_labels, SKIN_CORE))
        target = _median_lab(crop, parsing.mask(labels, SKIN_CORE))
        shift = self.color
        if generated is not None and target is not None:
            wanted = target + (self.tone.shift() if self.tone is not None else 0.0)
            now = np.clip(wanted - generated, -MAX_COLOR, MAX_COLOR)
            shift = now if shift is None else 0.8 * shift + 0.2 * now
        self.color = shift
        return np.zeros(3, np.float32) if shift is None else shift

    def apply(self, frame: np.ndarray, face, original: np.ndarray | None = None) -> np.ndarray:
        """frame : image à composer (cou déjà au teint de la personne) ; original : image d'origine pour les analyses."""
        from .swap import paste

        src = frame if original is None else original
        crop, M = align(src, face.kps, self.scale)
        labels = parsing.parse(crop)
        head = generate(self.source, self.motion(Motion.extract(crop)))
        head_labels = parsing.parse(head)
        new = parsing.mask(head_labels, HEAD)
        if new.sum() < 500:
            return frame
        # Haut du cou de la photo compris dans la nouvelle tête : il fait la jonction avec le cou de la cible.
        rows = np.arange(CROP)[:, None]
        new |= parsing.mask(head_labels, (P.NECK,)) & (rows < self.chin + 30)
        shift = self._color_shift(head, head_labels, crop, labels)
        lab = cv2.cvtColor(head, cv2.COLOR_BGR2LAB).astype(np.float32) + shift
        head = cv2.cvtColor(lab.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

        # Ancienne tête à effacer : cheveux, chapeau, bijoux ; + accessoires vus comme « fond » par BiSeNet (casque…) :
        # ce qui s'écarte du fond propre autour de la tête, hors peau et hors corps.
        old = parsing.mask(labels, ERASE) | (parsing.mask(labels, FACE_PARTS) & (rows < self.chin))
        plate = self.plate.crop(M) if self.plate is not None else None
        if plate is not None:
            differs = np.abs(crop.astype(np.int16) - plate.astype(np.int16)).max(axis=2) > 30
            old |= differs & accessory_zone() & ~parsing.mask(labels, KEEP)
        old = cv2.morphologyEx(old.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)).astype(bool)

        # Bord de la nouvelle tête resserré : sinon un liseré du fond de la photo reste autour des cheveux.
        new = cv2.erode(new.astype(np.uint8), np.ones((5, 5), np.uint8))
        alpha = cv2.GaussianBlur(new.astype(np.float32), (0, 0), 2.5)
        if self.alpha is not None and self.smoothing > 0:         # repère aligné : le masque bouge peu
            m = self.smoothing * max(0.0, 1.0 - self.motion.change / 8.0)
            alpha = m * self.alpha + (1 - m) * alpha
        self.alpha = alpha

        dst, _ = align(frame, face.kps, self.scale)                # image à composer (cou recoloré) dans le recadrage
        hole = old & (alpha < 0.5)
        hole_alpha = np.zeros((CROP, CROP), np.float32)
        filled = dst
        if hole.any():
            hole = cv2.dilate(hole.astype(np.uint8), np.ones((7, 7), np.uint8))
            background = plate if plate is not None else cv2.inpaint(
                dst, cv2.dilate(old.astype(np.uint8), np.ones((15, 15), np.uint8)), 5, cv2.INPAINT_TELEA)
            hole_alpha = cv2.GaussianBlur(hole.astype(np.float32), (0, 0), 2.0) * (1 - alpha)
            filled = (background * hole_alpha[..., None] + dst * (1 - hole_alpha[..., None])).astype(np.uint8)
        a = alpha[..., None]
        composed = (head * a + filled * (1 - a)).astype(np.uint8)
        return paste(frame, composed, M, np.maximum(alpha, hole_alpha))
