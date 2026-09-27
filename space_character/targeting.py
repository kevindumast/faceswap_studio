"""Choix de LA personne à remplacer dans chaque image, quand plusieurs personnes sont à l'écran.

Le prétraitement officiel de Wan-Animate prend la plus grande silhouette de chaque image. Ici, on part du visage
choisi dans l'appli (instant + cadre), on retrouve la silhouette qui le porte, puis on la suit d'image en image par
recouvrement. Numpy seulement : testé sur le PC, utilisé sur le Space.
"""
from __future__ import annotations

import math

import numpy as np

MIN_SCORE = 0.25      # silhouettes moins sûres ignorées (le détecteur garde tout à partir de 0,05)
MIN_IOU = 0.05        # en dessous : ce n'est plus la même personne (sauf si le centre est tout proche)


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _center(box) -> tuple[float, float]:
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def face_fit(person, face) -> float:
    """Plus c'est bas, plus le visage est plausible pour cette silhouette : centré horizontalement, en haut."""
    w, h = person[2] - person[0], person[3] - person[1]
    if w <= 0 or h <= 0:
        return math.inf
    fx, fy = _center(face)
    score = abs(fx - (person[0] + person[2]) / 2) / w + abs(fy - person[1]) / h
    if not (person[0] <= fx <= person[2] and person[1] <= fy <= person[3]):
        score += 10
    return score


def usable(boxes) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 5) if len(boxes) else np.zeros((0, 5))
    return boxes[boxes[:, 4] >= MIN_SCORE]


def pick_track(candidates: list, anchor_idx: int, face_box) -> list[np.ndarray | None]:
    """Une silhouette [x1, y1, x2, y2, score] par image (None si aucune personne détectée nulle part).

    candidates[i] : silhouettes détectées dans l'image i ; face_box : visage choisi, en pixels, vers l'image anchor_idx.
    Image sans détection (occultation, flou) : on garde la dernière position connue.
    """
    frames = [usable(c) for c in candidates]
    n = len(frames)
    out: list[np.ndarray | None] = [None] * n
    order = sorted(range(n), key=lambda i: abs(i - anchor_idx))
    start = next((i for i in order if len(frames[i])), None)
    if start is None:
        return out
    boxes = frames[start]
    out[start] = boxes[min(range(len(boxes)), key=lambda k: face_fit(boxes[k], face_box))]
    for step in (1, -1):
        prev = out[start]
        i = start + step
        while 0 <= i < n:
            boxes = frames[i]
            if len(boxes):
                k = max(range(len(boxes)), key=lambda j: iou(boxes[j], prev))
                if iou(boxes[k], prev) >= MIN_IOU:
                    prev = boxes[k]
                else:  # mouvement rapide : même personne si son centre est resté proche
                    k = min(range(len(boxes)), key=lambda j: math.dist(_center(boxes[j]), _center(prev)))
                    diag = math.hypot(prev[2] - prev[0], prev[3] - prev[1])
                    if math.dist(_center(boxes[k]), _center(prev)) < 0.5 * diag:
                        prev = boxes[k]
            out[i] = prev
            i += step
    return out
