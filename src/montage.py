"""Montage : plusieurs rendus enchaînés en une vidéo à partager (vertical 9:16 pour TikTok / Reels / Shorts).

- mise en page : vidéo entière sur fond flouté, ou recadrage qui suit les visages ;
- écran titre, légende par séquence, écran de fin, transitions (fondu au noir, flash blanc) ;
- effets : filtre couleur, neige, vignettage, zoom lent ;
- masques « réalité augmentée » qui suivent les visages : bonnet de Père Noël, bois et nez de renne, nez rouge ;
- son : celui de chaque séquence, plus une musique de fond qui baisse toute seule quand quelqu'un parle.

Tout est dessiné ici (aucune image à télécharger) ; l'image passe par OpenCV puis ffmpeg, le son par ffmpeg.
"""
from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from . import media

FPS = 30
FORMATS = {"9:16": (1080, 1920), "1:1": (1080, 1080), "16:9": (1920, 1080)}
FITS = ("blur", "crop")
GRADES = ("none", "warm", "cinema", "vintage", "bw")
MASKS = ("none", "santa", "reindeer", "nose")
TRANSITIONS = ("fade", "flash", "cut")
TITLE_S, OUTRO_S = 3.0, 2.5
Progress = Callable[[int, int], None]


class MontageError(RuntimeError):
    pass


@dataclass
class Clip:
    path: Path
    caption: str = ""


@dataclass
class MontageOptions:
    format: str = "9:16"
    fit: str = "blur"              # blur : vidéo entière sur fond flouté ; crop : plein cadre, centré sur les visages
    title: str = ""
    subtitle: str = ""
    outro: str = ""
    grade: str = "none"
    snow: bool = False
    vignette: bool = False
    zoom: bool = False             # zoom lent sur chaque séquence (effet « vivant »)
    mask: str = "none"
    transition: str = "fade"
    music: Path | None = None
    music_volume: float = 0.35
    clip_audio: bool = True        # garder le son des séquences
    ai_label: bool = True
    crf: int = 18
    preset: str = "veryfast"

    def validate(self) -> None:
        checks = ((self.format, FORMATS), (self.fit, FITS), (self.grade, GRADES), (self.mask, MASKS),
                  (self.transition, TRANSITIONS))
        for value, allowed in checks:
            if value not in allowed:
                raise MontageError(f"Option inconnue : {value}.")


@dataclass
class Segment:
    kind: str                      # title | clip | outro
    frames: int
    clip: Clip | None = None
    info: media.VideoInfo | None = None

    @property
    def seconds(self) -> float:
        return self.frames / FPS


# ─── Texte ───

FONT_CANDIDATES = {
    "title": ("C:/Windows/Fonts/georgiab.ttf", "C:/Windows/Fonts/segoeuib.ttf", "DejaVuSerif-Bold.ttf", "DejaVuSans-Bold.ttf"),
    "text": ("C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arialbd.ttf", "DejaVuSans-Bold.ttf"),
}


@lru_cache(maxsize=32)
def font(kind: str, size: int):
    from PIL import ImageFont

    for candidate in FONT_CANDIDATES[kind]:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _wrap(text: str, fnt, max_width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in text.splitlines() or [""]:
        words, line = paragraph.split(), ""
        for word in words:
            trial = f"{line} {word}".strip()
            if fnt.getlength(trial) <= max_width or not line:
                line = trial
            else:
                lines.append(line)
                line = word
        lines.append(line)
    return [ln for ln in lines if ln] or [""]


def text_sprite(text: str, size: int, max_width: int, kind: str = "text", color=(255, 255, 255),
                pill: bool = False) -> np.ndarray:
    """Texte centré (plusieurs lignes si besoin) en RGBA, avec ombre portée ; `pill` : fond arrondi sombre."""
    from PIL import Image, ImageDraw, ImageFilter

    fnt = font(kind, size)
    lines = _wrap(text, fnt, max_width)
    gap = int(size * 0.25)
    boxes = [fnt.getbbox(ln) for ln in lines]
    heights = [b[3] - b[1] for b in boxes]
    width = int(max(fnt.getlength(ln) for ln in lines))
    line_h = max(heights + [size])
    pad = int(size * 0.6) if pill else int(size * 0.3)
    w, h = width + 2 * pad, line_h * len(lines) + gap * (len(lines) - 1) + 2 * pad
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    if pill:
        draw.rounded_rectangle([0, 0, w - 1, h - 1], radius=min(h // 2, size), fill=(0, 0, 0, 150))
    shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(shadow)
    y = pad
    for ln, box in zip(lines, boxes):
        x = (w - fnt.getlength(ln)) / 2
        if not pill:
            sdraw.text((x, y - box[1] + size * 0.06), ln, font=fnt, fill=(0, 0, 0, 170))
        draw.text((x, y - box[1]), ln, font=fnt, fill=(*color, 255))
        y += line_h + gap
    if not pill:
        shadow = shadow.filter(ImageFilter.GaussianBlur(size * 0.08))
        img = Image.alpha_composite(shadow, img)
    return np.array(img)[:, :, [2, 1, 0, 3]]        # RGBA → BGRA


def blend(dst: np.ndarray, sprite: np.ndarray, x: int, y: int, opacity: float = 1.0) -> None:
    """Colle un sprite BGRA dans dst (BGR) à (x, y), en place ; les bords hors de l'image sont ignorés."""
    h, w = sprite.shape[:2]
    H, W = dst.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0 or opacity <= 0:
        return
    part = sprite[y0 - y:y1 - y, x0 - x:x1 - x]
    a = part[:, :, 3:4].astype(np.float32) * (opacity / 255.0)
    roi = dst[y0:y1, x0:x1].astype(np.float32)
    dst[y0:y1, x0:x1] = (part[:, :, :3] * a + roi * (1 - a)).astype(np.uint8)


# ─── Mise en page ───

def _cover(frame: np.ndarray, W: int, H: int, cx: float = 0.5, cy: float = 0.5, zoom: float = 1.0) -> np.ndarray:
    """Remplit W×H (recadrage), centré sur (cx, cy) en fractions de l'image source."""
    h, w = frame.shape[:2]
    k = max(W / w, H / h) * zoom
    cw, ch = W / k, H / k
    x0 = min(max(cx * w - cw / 2, 0), w - cw)
    y0 = min(max(cy * h - ch / 2, 0), h - ch)
    M = np.array([[k, 0, -x0 * k], [0, k, -y0 * k]], np.float32)
    return cv2.warpAffine(frame, M, (W, H), flags=cv2.INTER_LINEAR if k > 1 else cv2.INTER_AREA,
                          borderMode=cv2.BORDER_REPLICATE)


def blurred_background(frame: np.ndarray, W: int, H: int, darken: float = 0.55) -> np.ndarray:
    small = _cover(frame, max(8, W // 10), max(8, H // 10))
    small = cv2.GaussianBlur(small, (0, 0), 3)
    return (cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR) * darken).astype(np.uint8)


def layout(frame: np.ndarray, W: int, H: int, fit: str, zoom: float = 1.0, focus: tuple[float, float] | None = None
           ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Image source → toile W×H. Renvoie la toile et le rectangle (x0, y0, x1, y1) occupé par la vidéo."""
    if fit == "crop":
        cx, cy = focus or (0.5, 0.5)
        return _cover(frame, W, H, cx, cy, zoom), (0, 0, W, H)
    h, w = frame.shape[:2]
    k = min(W / w, H / h)
    fw, fh = int(round(w * k)), int(round(h * k))
    canvas = blurred_background(frame, W, H)
    inner = _cover(frame, fw, fh, zoom=zoom) if zoom > 1.0 else cv2.resize(frame, (fw, fh), interpolation=cv2.INTER_AREA)
    x0, y0 = (W - fw) // 2, (H - fh) // 2
    canvas[y0:y0 + fh, x0:x0 + fw] = inner
    return canvas, (x0, y0, x0 + fw, y0 + fh)


# ─── Couleur et effets ───

@lru_cache(maxsize=8)
def _luts(grade: str) -> np.ndarray | None:
    """Tables par canal (B, G, R) : 3 × 256."""
    x = np.arange(256, dtype=np.float32) / 255.0
    if grade == "warm":
        b, g, r = x ** 1.08 * 0.94, x ** 0.98, np.clip(x ** 0.92 * 1.04, 0, 1)
    elif grade == "cinema":     # ombres bleu-vert, hautes lumières orangées, contraste en S
        s = 0.5 - 0.5 * np.cos(np.pi * x)
        base = 0.65 * x + 0.35 * s
        b = np.clip(base + 0.06 * (1 - x) - 0.05 * x, 0, 1)
        g = np.clip(base + 0.02 * (1 - x), 0, 1)
        r = np.clip(base - 0.03 * (1 - x) + 0.05 * x, 0, 1)
    elif grade == "vintage":    # noirs délavés, teinte crème
        b = 0.10 + 0.78 * x ** 1.05
        g = 0.07 + 0.86 * x
        r = 0.08 + 0.90 * x ** 0.95
    else:
        return None
    return (np.stack([b, g, r]) * 255).clip(0, 255).astype(np.uint8)


def apply_grade(img: np.ndarray, grade: str) -> np.ndarray:
    if grade == "none":
        return img
    if grade == "bw":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8)).apply(gray)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    lut = _luts(grade)
    out = cv2.merge([cv2.LUT(c, lut[i]) for i, c in enumerate(cv2.split(img))])
    if grade == "warm":         # couleurs un peu plus vives
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
        hsv[..., 1] = cv2.LUT(hsv[..., 1], np.clip(np.arange(256) * 1.12, 0, 255).astype(np.uint8))
        out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return out


@lru_cache(maxsize=4)
def vignette_map(W: int, H: int, strength: float = 0.45) -> np.ndarray:
    """Assombrissement des bords, en uint8 3 canaux (255 = intact) : appliqué par cv2.multiply, sans passer en flottant."""
    y, x = np.ogrid[-1:1:complex(0, H), -1:1:complex(0, W)]
    d = np.sqrt(x * x + y * y) / math.sqrt(2)
    k = 1 - strength * np.clip((d - 0.35) / 0.65, 0, 1) ** 1.6
    return np.repeat((k * 255).round().astype(np.uint8)[..., None], 3, axis=2)


def apply_vignette(img: np.ndarray) -> np.ndarray:
    return cv2.multiply(img, vignette_map(img.shape[1], img.shape[0]), scale=1 / 255)


class Snow:
    """Flocons qui tombent en se balançant, trois tailles (les gros, proches, plus flous et plus rapides)."""

    def __init__(self, W: int, H: int, count: int | None = None, seed: int = 7):
        self.W, self.H = W, H
        rng = np.random.default_rng(seed)
        n = count or int(W * H / 9000)
        self.x = rng.uniform(0, W, n)
        self.y = rng.uniform(0, H, n)
        self.depth = rng.choice([0, 1, 2], n, p=[0.55, 0.3, 0.15])
        self.r = np.array([1.6, 2.8, 4.6])[self.depth] * (W / 1080) * rng.uniform(0.8, 1.25, n)
        self.speed = np.array([1.4, 2.4, 3.8])[self.depth] * (H / 1920) * rng.uniform(0.8, 1.2, n)
        self.phase = rng.uniform(0, 2 * np.pi, n)
        self.t = 0

    def step(self) -> None:
        self.t += 1
        self.y += self.speed
        self.x += np.sin(self.t / 28 + self.phase) * 0.6 * (self.depth + 1)
        wrapped = self.y > self.H + 10
        self.y[wrapped] = -10
        self.x %= self.W

    def draw(self, img: np.ndarray) -> np.ndarray:
        layer = np.zeros(img.shape[:2], np.uint8)
        for x, y, r in zip(self.x.astype(int), self.y.astype(int), self.r):
            cv2.circle(layer, (x, y), max(1, int(round(r))), 255, -1, lineType=cv2.LINE_AA)
        layer = cv2.GaussianBlur(layer, (0, 0), 1.1)
        self.step()
        # img + (255 - img) × alpha, en uint8 (4× plus rapide qu'en flottant sur une image 1080×1920).
        alpha = cv2.cvtColor(cv2.convertScaleAbs(layer, alpha=0.9), cv2.COLOR_GRAY2BGR)
        return cv2.add(img, cv2.multiply(cv2.subtract(np.full_like(img, 255), img), alpha, scale=1 / 255))


# ─── Masques qui suivent les visages ───

@dataclass
class Sprite:
    image: np.ndarray        # BGRA
    anchor: tuple[float, float]   # point du sprite posé sur le point d'ancrage du visage (px)
    ref_width: float         # largeur (px du sprite) qui correspond à `face_width` × écart des yeux


def _pil_to_bgra(img) -> np.ndarray:
    return np.array(img)[:, :, [2, 1, 0, 3]]


@lru_cache(maxsize=1)
def santa_hat() -> Sprite:
    from PIL import Image, ImageDraw, ImageFilter

    W, H = 760, 700
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    # Cône rouge qui retombe sur le côté, avec ombrage.
    cone = [(110, 520), (230, 230), (420, 90), (600, 150), (640, 270), (560, 230), (470, 260), (650, 520)]
    d.polygon(cone, fill=(196, 22, 38, 255))
    shade = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(shade).polygon([(470, 260), (560, 230), (640, 270), (650, 520), (520, 520)], fill=(110, 8, 22, 120))
    img = Image.alpha_composite(img, shade.filter(ImageFilter.GaussianBlur(18)))
    hl = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(hl).polygon([(200, 480), (280, 260), (400, 150), (330, 300), (280, 480)], fill=(255, 120, 120, 70))
    img = Image.alpha_composite(img, hl.filter(ImageFilter.GaussianBlur(20)))
    # Bord de fourrure blanc et pompon, duveteux.
    fur = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    f = ImageDraw.Draw(fur)
    f.rounded_rectangle([70, 470, 690, 600], radius=64, fill=(250, 248, 244, 255))
    rng = np.random.default_rng(3)
    for _ in range(140):
        x, y = rng.uniform(80, 680), rng.uniform(470, 600)
        r = rng.uniform(14, 30)
        f.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 255, 255))
    for _ in range(40):
        a, rr = rng.uniform(0, 2 * np.pi), rng.uniform(0, 60)
        x, y, r = 640 + rr * np.cos(a), 290 + rr * np.sin(a), rng.uniform(18, 30)
        f.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 255, 255))
    fur_shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(fur_shadow).rounded_rectangle([80, 560, 680, 610], radius=30, fill=(150, 150, 160, 110))
    img = Image.alpha_composite(img, fur.filter(ImageFilter.GaussianBlur(1.2)))
    img = Image.alpha_composite(img, fur_shadow.filter(ImageFilter.GaussianBlur(14)))
    return Sprite(_pil_to_bgra(img), anchor=(380, 600), ref_width=620)


@lru_cache(maxsize=1)
def reindeer_antlers() -> Sprite:
    from PIL import Image, ImageDraw, ImageFilter

    W, H = 1000, 620
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    brown, dark = (122, 78, 44, 255), (84, 52, 28, 255)

    def branch(points, width, color):
        d.line(points, fill=color, width=width, joint="curve")
        for p in (points[0], points[-1]):
            d.ellipse([p[0] - width / 2, p[1] - width / 2, p[0] + width / 2, p[1] + width / 2], fill=color)

    for side in (-1, 1):
        def X(x, side=side):
            return 500 + side * x
        branch([(X(150), 600), (X(200), 430), (X(260), 280), (X(330), 120), (X(360), 30)], 46, dark)
        branch([(X(150), 600), (X(200), 430), (X(260), 280), (X(330), 120), (X(360), 30)], 34, brown)
        for start, end in (((X(215), 400), (X(330), 330)), ((X(250), 300), (X(400), 250)), ((X(300), 190), (X(450), 150)),
                           ((X(240), 330), (X(170), 220))):
            branch([start, end], 26, brown)
    # Petit bandeau qui tient les bois.
    d.rounded_rectangle([300, 570, 700, 615], radius=22, fill=(150, 20, 30, 255))
    img = img.filter(ImageFilter.GaussianBlur(0.8))
    return Sprite(_pil_to_bgra(img), anchor=(500, 600), ref_width=480)


@lru_cache(maxsize=1)
def red_nose() -> Sprite:
    from PIL import Image, ImageDraw, ImageFilter

    S = 300
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse([20, 20, 280, 280], fill=(214, 18, 30, 255))
    shade = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ImageDraw.Draw(shade).ellipse([90, 110, 300, 300], fill=(110, 0, 10, 140))
    img = Image.alpha_composite(img, shade.filter(ImageFilter.GaussianBlur(30)))
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).ellipse([20, 20, 280, 280], fill=255)
    img.putalpha(mask)
    glow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ImageDraw.Draw(glow).ellipse([70, 60, 150, 130], fill=(255, 255, 255, 190))
    img = Image.alpha_composite(img, glow.filter(ImageFilter.GaussianBlur(10)))
    return Sprite(_pil_to_bgra(img), anchor=(150, 150), ref_width=260)


def place_sprite(dst: np.ndarray, sprite: Sprite, point: np.ndarray, width: float, angle: float) -> None:
    """Pose le sprite : son ancre sur `point`, sa largeur de référence à `width` px, tourné de `angle` (radians)."""
    k = width / sprite.ref_width
    if k <= 0.01:
        return
    ca, sa = math.cos(angle) * k, math.sin(angle) * k
    ax, ay = sprite.anchor
    M = np.array([[ca, -sa, 0], [sa, ca, 0]], np.float32)
    M[:, 2] = point - M[:, :2] @ np.array([ax, ay], np.float32)
    h, w = sprite.image.shape[:2]
    corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], np.float32) @ M.T
    H, Wd = dst.shape[:2]
    x0, y0 = (int(max(0, math.floor(v))) for v in corners.min(axis=0))
    x1, y1 = int(min(Wd, math.ceil(corners[:, 0].max()))), int(min(H, math.ceil(corners[:, 1].max())))
    if x1 <= x0 or y1 <= y0:
        return
    M[:, 2] -= (x0, y0)
    warped = cv2.warpAffine(sprite.image, M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
    blend(dst, warped, x0, y0)


@dataclass
class _Track:
    kps: np.ndarray
    missed: int = 0


class FaceMasks:
    """Masques posés sur chaque visage, avec un suivi lissé (pas de tremblement d'une image à l'autre)."""

    def __init__(self, kind: str, detect: Callable[[np.ndarray], list] | None = None, smoothing: float = 0.55,
                 every: int = 2):
        self.kind = kind
        self.detect = detect or _default_detect
        self.smoothing = smoothing
        self.every = every
        self.tracks: list[_Track] = []
        self.n = 0

    def reset(self) -> None:
        self.tracks, self.n = [], 0

    def _update(self, frame: np.ndarray) -> None:
        faces = self.detect(frame)
        found = [np.asarray(f.kps, np.float32) for f in faces]
        used = set()
        for tr in self.tracks:
            center = tr.kps.mean(axis=0)
            scale = float(np.linalg.norm(tr.kps[1] - tr.kps[0])) + 1
            best, dist = None, 1e9
            for i, k in enumerate(found):
                dd = float(np.linalg.norm(k.mean(axis=0) - center))
                if i not in used and dd < dist and dd < 2.5 * scale:
                    best, dist = i, dd
            if best is None:
                tr.missed += 1
                continue
            used.add(best)
            tr.kps = self.smoothing * tr.kps + (1 - self.smoothing) * found[best]
            tr.missed = 0
        self.tracks = [t for t in self.tracks if t.missed <= 4]
        self.tracks += [_Track(k) for i, k in enumerate(found) if i not in used]

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Suit les visages (une détection toutes les `every` images) et pose le masque ; kind « none » : suivi seul."""
        if self.n % self.every == 0:
            self._update(frame)
        self.n += 1
        if self.kind == "none":
            return frame
        out = frame.copy()
        for tr in self.tracks:
            if tr.missed:
                continue
            draw_mask(out, self.kind, tr.kps)
        return out

    def faces_center(self, shape) -> tuple[float, float] | None:
        live = [t.kps.mean(axis=0) for t in self.tracks if not t.missed]
        if not live:
            return None
        c = np.mean(live, axis=0)
        return float(c[0] / shape[1]), float(c[1] / shape[0])


def draw_mask(img: np.ndarray, kind: str, kps: np.ndarray) -> None:
    """Dessine le masque `kind` sur un visage (5 points clés InsightFace : yeux, nez, coins de la bouche)."""
    left, right, nose = kps[0], kps[1], kps[2]
    eyes = right - left
    d = float(np.linalg.norm(eyes))
    if d < 6:
        return
    angle = math.atan2(eyes[1], eyes[0])
    up = np.array([eyes[1], -eyes[0]], np.float32) / d          # vers le haut du visage
    mid = (left + right) / 2
    if kind == "santa":
        place_sprite(img, santa_hat(), mid + up * d * 1.05, d * 2.75, angle)
    elif kind == "reindeer":
        place_sprite(img, reindeer_antlers(), mid + up * d * 1.35, d * 2.4, angle)
        place_sprite(img, red_nose(), nose + up * d * 0.05, d * 0.62, angle)
    elif kind == "nose":
        place_sprite(img, red_nose(), nose + up * d * 0.05, d * 0.62, angle)


def _default_detect(frame: np.ndarray) -> list:
    from .faces import detect_boxes

    h, w = frame.shape[:2]
    k = min(1.0, 960 / max(h, w))
    small = cv2.resize(frame, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else frame
    faces = detect_boxes(small, size=480, thresh=0.45)   # 480 : visages des rendus bien assez grands, ~35 % plus rapide
    for f in faces:
        f.kps = np.asarray(f.kps, np.float32) / k
        f.bbox = np.asarray(f.bbox, np.float32) / k
    return faces


# ─── Plan du montage ───

def plan(clips: list[Clip], opts: MontageOptions) -> list[Segment]:
    if not clips:
        raise MontageError("Ajoute au moins une séquence au montage.")
    segments = []
    if opts.title.strip():
        segments.append(Segment("title", round(TITLE_S * FPS)))
    for c in clips:
        if not c.path.is_file():
            raise MontageError(f"Séquence introuvable : {c.path.name}.")
        info = media.probe(c.path)
        # Durée de l'image (nombre d'images), pas celle du conteneur : le son y dépasse souvent de quelques ms.
        cap = cv2.VideoCapture(str(c.path))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        seconds = count / info.fps if count > 0 else info.duration
        segments.append(Segment("clip", max(1, round(seconds * FPS)), c, info))
    if opts.outro.strip():
        segments.append(Segment("outro", round(OUTRO_S * FPS)))
    return segments


def _fade_factor(i: int, n: int, transition: str, first: bool, last: bool) -> tuple[float, tuple[int, int, int]]:
    """Fondu en début et fin de segment : (part de l'image gardée, couleur vers laquelle on fond)."""
    if transition == "cut":
        return 1.0, (0, 0, 0)
    span = round((0.12 if transition == "flash" else 0.3) * FPS)
    color = (255, 255, 255) if transition == "flash" else (0, 0, 0)
    k = 1.0
    if not first and i < span:
        k = min(k, (i + 1) / (span + 1))
    if not last and n - 1 - i < span:
        k = min(k, (n - i) / (span + 1))
    return k, color


def _read_frames(path: Path, info: media.VideoInfo, count: int):
    """Images à 30 i/s (une image source répétée ou sautée selon sa cadence)."""
    cap = cv2.VideoCapture(str(path))
    try:
        idx, frame, ok = -1, None, True
        for i in range(count):
            want = int(i * info.fps / FPS + 1e-6)
            while idx < want and ok:
                ok, nxt = cap.read()
                if ok:
                    frame, idx = nxt, idx + 1
            if frame is None:
                raise MontageError(f"Impossible de lire {path.name}.")
            yield frame
    finally:
        cap.release()


def _first_frame(path: Path, last: bool = False) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    try:
        if last:
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n - 2))
        ok, frame = cap.read()
        if not ok:
            raise MontageError(f"Impossible de lire {path.name}.")
        return frame
    finally:
        cap.release()


class Renderer:
    """Fabrique les images du montage, segment par segment."""

    def __init__(self, segments: list[Segment], opts: MontageOptions, detect: Callable | None = None):
        self.segments = segments
        self.opts = opts
        self.W, self.H = FORMATS[opts.format]
        self.snow = Snow(self.W, self.H) if opts.snow else None
        self.masks = FaceMasks(opts.mask, detect) if opts.mask != "none" or opts.fit == "crop" else None
        self.focus: tuple[float, float] | None = None
        u = min(self.W, self.H)
        self.label = None
        if opts.ai_label:
            import tempfile

            from PIL import Image

            with tempfile.TemporaryDirectory() as tmp:
                path = media.render_label(Path(tmp) / "label.png", int(min(self.W, self.H) * 0.75))
                self.label = np.array(Image.open(path).convert("RGBA"))[:, :, [2, 1, 0, 3]]
        self.caption_size = int(u * 0.05)

    def _finish(self, canvas: np.ndarray, k: float, color) -> np.ndarray:
        canvas = apply_grade(canvas, self.opts.grade)
        if self.opts.vignette:
            canvas = apply_vignette(canvas)
        if self.snow is not None:
            canvas = self.snow.draw(canvas)
        if self.label is not None:
            m = int(min(self.W, self.H) * 0.03)
            blend(canvas, self.label, self.W - self.label.shape[1] - m, self.H - self.label.shape[0] - m, 0.9)
        if k < 1.0:
            canvas = cv2.addWeighted(canvas, k, np.full_like(canvas, color), 1 - k, 0)
        return canvas

    def _card(self, seg: Segment, text: str, sub: str, bg: np.ndarray, first: bool, last: bool):
        W, H, u = self.W, self.H, min(self.W, self.H)
        title = text_sprite(text, int(u * 0.11), int(W * 0.84), "title")
        subtitle = text_sprite(sub, int(u * 0.05), int(W * 0.8), "text", (235, 235, 235)) if sub.strip() else None
        base = blurred_background(bg, W, H, darken=0.45)
        for i in range(seg.frames):
            t = i / FPS
            zoom = 1.0 + 0.04 * i / max(1, seg.frames)
            canvas = _cover(base, W, H, zoom=zoom)
            appear = min(1.0, t / 0.6)
            ease = 1 - (1 - appear) ** 3
            y = int(H * 0.42 - title.shape[0] / 2 + (1 - ease) * u * 0.04)
            blend(canvas, title, (W - title.shape[1]) // 2, y, ease)
            if subtitle is not None:
                s_appear = min(1.0, max(0.0, (t - 0.35) / 0.6))
                blend(canvas, subtitle, (W - subtitle.shape[1]) // 2, y + title.shape[0] + int(u * 0.03), s_appear)
            k, color = _fade_factor(i, seg.frames, self.opts.transition, first, last)
            yield self._finish(canvas, k, color)

    def _clip(self, seg: Segment, first: bool, last: bool):
        W, H = self.W, self.H
        caption = text_sprite(seg.clip.caption, self.caption_size, int(W * 0.86), "text", pill=True) \
            if seg.clip.caption.strip() else None
        if self.masks is not None:
            self.masks.reset()
        self.focus = None
        for i, frame in enumerate(_read_frames(seg.clip.path, seg.info, seg.frames)):
            if self.masks is not None:
                frame = self.masks.apply(frame)
                center = self.masks.faces_center(frame.shape)
                if center is not None:   # recadrage : suit doucement le centre des visages
                    self.focus = center if self.focus is None else (0.9 * self.focus[0] + 0.1 * center[0],
                                                                     0.9 * self.focus[1] + 0.1 * center[1])
            zoom = 1.0 + 0.06 * i / max(1, seg.frames) if self.opts.zoom else 1.0
            canvas, (x0, y0, x1, y1) = layout(frame, W, H, self.opts.fit, zoom, self.focus)
            if caption is not None:
                appear = min(1.0, max(0.0, (i / FPS - 0.25) / 0.4))
                space = H - y1
                if self.opts.fit == "blur" and space > caption.shape[0] * 1.6:
                    cy = y1 + (space - caption.shape[0]) // 2
                else:
                    cy = int(H * 0.8) - caption.shape[0] // 2
                blend(canvas, caption, (W - caption.shape[1]) // 2, cy, appear)
            k, color = _fade_factor(i, seg.frames, self.opts.transition, first, last)
            yield self._finish(canvas, k, color)

    def frames(self):
        clips = [s for s in self.segments if s.kind == "clip"]
        for n, seg in enumerate(self.segments):
            first, last = n == 0, n == len(self.segments) - 1
            if seg.kind == "title":
                yield from self._card(seg, self.opts.title, self.opts.subtitle, _first_frame(clips[0].clip.path), first, last)
            elif seg.kind == "outro":
                yield from self._card(seg, self.opts.outro, "", _first_frame(clips[-1].clip.path, last=True), first, last)
            else:
                yield from self._clip(seg, first, last)


# ─── Son ───

def audio_graph(segments: list[Segment], opts: MontageOptions, music_input: int | None) -> tuple[list[str], str]:
    """Inputs ffmpeg supplémentaires (après la vidéo, input 0) et filtre audio → [aout]."""
    inputs: list[str] = []
    chains, labels = [], []
    k = 1
    for n, seg in enumerate(segments):
        d = seg.seconds
        lab = f"a{n}"
        if seg.kind == "clip" and seg.info.has_audio and opts.clip_audio:
            inputs += ["-i", str(seg.clip.path)]
            chains.append(f"[{k}:a]aformat=sample_rates=48000:channel_layouts=stereo,atrim=0:{d:.4f},"
                          f"apad=whole_dur={d:.4f},afade=t=in:d=0.08,afade=t=out:st={max(0.0, d - 0.15):.4f}:d=0.15[{lab}]")
            k += 1
        else:
            chains.append(f"anullsrc=r=48000:cl=stereo,atrim=0:{d:.4f}[{lab}]")
        labels.append(f"[{lab}]")
    chains.append(f"{''.join(labels)}concat=n={len(labels)}:v=0:a=1[voice]")
    total = sum(s.seconds for s in segments)
    if opts.music is not None:
        m = k
        inputs += ["-stream_loop", "-1", "-i", str(opts.music)]
        vol = max(0.0, min(1.5, opts.music_volume))
        chains.append(f"[{m}:a]aformat=sample_rates=48000:channel_layouts=stereo,atrim=0:{total:.4f},volume={vol:.3f},"
                      f"afade=t=in:d=1,afade=t=out:st={max(0.0, total - 2):.4f}:d=2[music]")
        if opts.clip_audio:
            # La musique baisse toute seule quand quelqu'un parle (compression déclenchée par les voix).
            chains.append("[voice]asplit=2[v1][v2]")
            chains.append("[music][v1]sidechaincompress=threshold=0.03:ratio=6:attack=30:release=400[duck]")
            chains.append("[v2][duck]amix=inputs=2:duration=first:normalize=0[aout]")
        else:
            chains.append("[voice][music]amix=inputs=2:duration=first:normalize=0[aout]")
    else:
        chains.append("[voice]anull[aout]")
    return inputs, ";".join(chains)


# ─── Assemblage ───

def render(clips: list[Clip], opts: MontageOptions, out: Path, progress: Progress | None = None,
           detect: Callable | None = None) -> dict:
    """Fabrique le montage dans `out` (MP4 H.264 + AAC). Renvoie durée, images et taille."""
    opts.validate()
    report = progress or (lambda *_: None)
    segments = plan(clips, opts)
    total = sum(s.frames for s in segments)
    W, H = FORMATS[opts.format]
    work = out.parent / f".{out.stem}_work"
    work.mkdir(parents=True, exist_ok=True)
    silent = work / "video.mp4"
    try:
        renderer = Renderer(segments, opts, detect)
        with media.FrameWriter(silent, (W, H), str(FPS), opts.crf, opts.preset) as writer:
            for i, frame in enumerate(renderer.frames()):
                writer.write(frame)
                if i % 10 == 0:
                    report(i, total)
        report(total, total)
        extra, graph = audio_graph(segments, opts, None)
        tmp = out.with_name(f"{out.stem}.tmp{out.suffix}")
        media.ffmpeg("-i", str(silent), *extra, "-filter_complex", graph, "-map", "0:v", "-map", "[aout]",
                     "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-t", f"{total / FPS:.3f}",
                     "-movflags", "+faststart", str(tmp))
        tmp.replace(out)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return {"frames": total, "duration": total / FPS, "width": W, "height": H}


def preview(clips: list[Clip], opts: MontageOptions, detect: Callable | None = None) -> np.ndarray:
    """Une image représentative (milieu de la 1re séquence) pour juger la mise en page sans tout calculer."""
    opts.validate()
    segments = [s for s in plan(clips, opts) if s.kind == "clip"][:1]
    seg = segments[0]
    mid = seg.frames // 2
    renderer = Renderer(segments, opts, detect)
    for i, frame in enumerate(renderer._clip(seg, True, True)):
        if i >= mid:
            return frame
    raise MontageError("Séquence vide.")
