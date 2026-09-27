"""Qui apparaît dans ce passage ? Plusieurs images analysées, visages regroupés par personne.

Regarder une seule image (début / milieu / fin) rate les personnes absentes à cet instant, de dos ou masquées.
On échantillonne donc tout le passage, avec une détection plus fine et plus tolérante que pendant le rendu,
puis on regroupe les visages par identité (ArcFace) : deux visages d'une même image sont forcément deux personnes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import media
from .config import load_config
from .faces import area, detect_boxes, embed, face_crop


@dataclass
class Person:
    """Une personne vue dans le passage, représentée par sa meilleure apparition."""
    t: float
    box: list[float]                  # normalisée 0..1 dans l'image de t
    score: float
    height_px: int
    crop: np.ndarray
    embedding_sum: np.ndarray
    times: set = field(default_factory=set)
    quality: float = 0.0
    maybe_same: int | None = None     # index d'une personne principale dont ce visage est sans doute un profil flou

    @property
    def mean(self) -> np.ndarray:
        return self.embedding_sum / np.linalg.norm(self.embedding_sum)


def sample_times(start: float, end: float, samples: int) -> list[float]:
    """Instants répartis sur le passage, en évitant les toutes premières et dernières images (fondus, coupes)."""
    margin = min(0.2, (end - start) / (samples * 4))
    return [round(float(t), 3) for t in np.linspace(start + margin, end - margin, samples)]


def scan_passage(source: Path, start: float, end: float) -> tuple[list[Person], dict[float, np.ndarray]]:
    """Personnes du passage (les plus présentes d'abord) + les images où elles apparaissent le mieux."""
    cfg = load_config()
    sc = cfg.get("scan", {})
    samples = int(sc.get("samples", 12))
    size, thresh = int(sc.get("det_size", 960)), float(sc.get("det_thresh", 0.3))
    min_px, same = int(sc.get("min_face_px", 24)), float(sc.get("same_person", 0.4))
    max_height = int(cfg.render.max_height)

    frames: dict[float, np.ndarray] = {}
    candidates = []
    for t in sample_times(start, end, samples):
        try:
            frame = media.extract_frame(source, t, max_height)
        except media.MediaError:
            continue
        frames[t] = frame
        for face in detect_boxes(frame, size=size, thresh=thresh):
            height = int(face.bbox[3] - face.bbox[1])
            if height < min_px:
                continue
            embed(frame, face)
            candidates.append((t, face, frame, area(face.bbox) * float(face.det_score)))

    # Regroupement glouton, du visage le plus net au moins net.
    people: list[Person] = []
    for t, face, frame, quality in sorted(candidates, key=lambda c: c[3], reverse=True):
        emb = face.normed_embedding
        best, best_sim = None, same
        for p in people:
            if t in p.times:          # déjà vue dans cette image : ce visage est quelqu'un d'autre
                continue
            sim = float(np.dot(p.mean, emb))
            if sim >= best_sim:
                best, best_sim = p, sim
        if best is None:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = [float(v) for v in face.bbox[:4]]
            people.append(Person(
                t=t, box=[x1 / w, y1 / h, x2 / w, y2 / h], score=float(face.det_score), height_px=int(y2 - y1),
                crop=face_crop(frame, face.bbox, 112), embedding_sum=emb.astype(np.float64).copy(), times={t},
                quality=quality,
            ))
        else:
            best.embedding_sum += emb
            best.times.add(t)

    people.sort(key=lambda p: (len(p.times), p.quality), reverse=True)
    _flag_probable_duplicates(people)
    used = {p.t for p in people}
    return people, {t: f for t, f in frames.items() if t in used}


# Un visage vu une seule fois, peu net, qui ressemble un peu à une personne bien présente, est presque toujours cette
# personne de profil ou floue (mesuré : ~0,17 contre ~0 entre deux personnes différentes). On le signale plutôt que de
# le proposer comme une personne à part : l'associer à quelqu'un d'autre ferait alterner deux visages sur le même acteur.
DUPLICATE_SIM = 0.12
DUPLICATE_MAX_SCORE = 0.6


def _flag_probable_duplicates(people: list[Person]) -> None:
    main = [(i, p) for i, p in enumerate(people) if len(p.times) >= 2]
    for p in people:
        if len(p.times) != 1 or p.score >= DUPLICATE_MAX_SCORE or not main:
            continue
        i, sim = max(((i, float(np.dot(p.mean, q.mean))) for i, q in main if q is not p), key=lambda x: x[1],
                     default=(None, 0.0))
        if i is not None and sim >= DUPLICATE_SIM:
            p.maybe_same = i
