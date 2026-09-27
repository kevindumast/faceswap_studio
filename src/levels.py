"""Niveaux de transformation et stratégie de remplacement associée à chacun.

Une stratégie est instanciée par association (visage du clip → personne) : elle peut garder un état entre images
(lissage du teint, etc.). Les niveaux « image par image » passent par swap_segment ; le niveau 4 (personnage entier)
est traité d'un bloc sur le GPU distant.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import load_config
from .identity import SourceFace

FACE, FACE_TONE, HEAD, CHARACTER = "face", "face_tone", "head", "character"
ALL_LEVELS = (FACE, FACE_TONE, HEAD, CHARACTER)
FRAME_LEVELS = (FACE, FACE_TONE, HEAD)
# Modèles nécessaires en local, par niveau (groupes de src/models.py).
MODEL_GROUPS = {FACE: ("base",), FACE_TONE: ("base", "tone")}


@dataclass
class PersonAssets:
    """Tout ce qu'on sait d'une personne source : identité, teint, photos."""
    source: SourceFace
    tone: object | None = None                     # tone.ToneStats
    photos: list[Path] = field(default_factory=list)


class Strategy:
    def __init__(self, person: PersonAssets):
        self.person = person

    def apply(self, frame: np.ndarray, face) -> np.ndarray:
        raise NotImplementedError


class FaceStrategy(Strategy):
    """Niveau 1 : traits du visage (inswapper)."""

    def apply(self, frame, face):
        from .swap import swap_face

        return swap_face(frame, face, self.person.source)


class FaceToneStrategy(FaceStrategy):
    """Niveau 2 : visage puis teint de la personne sur la peau visible (visage, oreilles, cou)."""

    def __init__(self, person: PersonAssets):
        super().__init__(person)
        from .tone import ToneMatcher

        lcfg = load_config().levels.face_tone
        self.matcher = ToneMatcher(person.tone, float(lcfg.tone_strength_l), float(lcfg.tone_smoothing)) if person.tone else None

    def apply(self, frame, face):
        frame = super().apply(frame, face)
        return self.matcher.apply(frame, face) if self.matcher else frame


STRATEGIES = {FACE: FaceStrategy, FACE_TONE: FaceToneStrategy}


def make_strategy(level: str, person: PersonAssets) -> Strategy:
    if level not in STRATEGIES:
        raise ValueError(f"Niveau non disponible : {level}")
    return STRATEGIES[level](person)


def models_ready(level: str) -> bool:
    from . import models

    return level in MODEL_GROUPS and all(models.is_ready(g) for g in MODEL_GROUPS[level])
