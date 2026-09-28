"""Suivi du visage cible et lissage des landmarks entre frames (anti-jitter)."""
from __future__ import annotations

from typing import Callable

import numpy as np

from .faces import area, iou


class TargetTracker:
    """Choisit, frame après frame, le visage qui correspond à l'identité cible.

    - Vérification d'identité (ArcFace) toutes les `verify_every` frames ou dès que le suivi est perdu.
    - Entre deux vérifications, on suit le visage par recouvrement de boîte (pas d'embedding : plus rapide).
    - En secours, un visage qui recouvre la position précédente est accepté avec un seuil d'identité plus bas
      (profil, flou de mouvement).
    """

    def __init__(self, reference: np.ndarray, threshold: float, smoothing: float,
                 hold_frames: int = 2, verify_every: int = 12):
        self.reference = reference
        self.threshold = threshold
        self.smoothing = smoothing
        self.hold_frames = hold_frames
        self.verify_every = verify_every
        self.prev_bbox = None
        self.prev_kps = None
        self.missed = 0
        self.since_verify = verify_every

    def _by_identity(self, faces: list):
        best, best_score = None, -1.0
        for face in faces:
            sim = float(np.dot(face.normed_embedding, self.reference))
            overlap = iou(face.bbox, self.prev_bbox) if self.prev_bbox is not None else 0.0
            ok = sim >= self.threshold or (overlap > 0.5 and sim >= self.threshold * 0.5)
            score = sim + 0.3 * overlap
            if ok and score > best_score:
                best, best_score = face, score
        return best

    def _by_position(self, faces: list):
        if self.prev_bbox is None or not faces:
            return None
        best = max(faces, key=lambda f: iou(f.bbox, self.prev_bbox))
        return best if iou(best.bbox, self.prev_bbox) > 0.5 else None

    def smooth(self, face) -> None:
        """EMA adaptative : forte au repos, quasi nulle quand la tête bouge vite (pas de retard).

        La boîte est lissée comme les points clés : le recadrage du teint, son ellipse et la bande du cou en dépendent.
        """
        kps = np.asarray(face.kps, dtype=np.float32)
        bbox = np.asarray(face.bbox[:4], dtype=np.float32)
        if self.prev_kps is not None and self.smoothing > 0:
            size = max(np.sqrt(area(bbox)), 1.0)
            motion = float(np.mean(np.linalg.norm(kps - self.prev_kps, axis=1))) / size
            alpha = self.smoothing * max(0.0, 1.0 - motion / 0.05)
            kps = alpha * self.prev_kps + (1 - alpha) * kps
            bbox = alpha * np.asarray(self.prev_bbox[:4], np.float32) + (1 - alpha) * bbox
            face.kps = kps
            face.bbox = bbox
        self.prev_kps = kps
        self.prev_bbox = bbox

    def update(self, faces: list, embed: Callable[[object], None]):
        """Renvoie le visage à swapper pour cette frame (ou None). `embed(face)` calcule l'embedding à la demande."""
        face = None
        if self.since_verify < self.verify_every:
            face = self._by_position(faces)
            if face is not None:
                self.since_verify += 1
        if face is None:
            for f in faces:
                embed(f)
            face = self._by_identity(faces)
            self.since_verify = 0
        if face is not None:
            self.missed = 0
            self.smooth(face)
            return face
        self.missed += 1
        if self.missed > self.hold_frames:
            self.prev_kps = None
            self.prev_bbox = None
            self.since_verify = self.verify_every
        return None
