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
MODEL_GROUPS = {FACE: ("base",), FACE_TONE: ("base", "tone"), HEAD: ("base", "tone", "head")}


@dataclass
class PersonAssets:
    """Tout ce qu'on sait d'une personne source : identité, teint, photos."""
    source: SourceFace
    tone: object | None = None                     # tone.ToneStats
    photos: list[Path] = field(default_factory=list)
    reference: Path | None = None                  # photo choisie pour le niveau 4 (sinon : la plus en pied)
    head_reference: Path | None = None             # photo de la tête au niveau 3 (sinon : la plus de face)


class Strategy:
    samples = 0   # images réparties sur l'extrait à fournir à prepare() avant le rendu (0 = pas de pré-passe)

    def __init__(self, person: PersonAssets):
        self.person = person

    def prepare(self, samples: list[tuple[np.ndarray, object]]) -> None:
        """Pré-passe : (image originale, visage de la cible) sur des images réparties sur tout l'extrait."""

    def apply(self, frame: np.ndarray, face) -> np.ndarray:
        raise NotImplementedError


class FaceStrategy(Strategy):
    """Niveau 1 : traits du visage (inswapper)."""

    def __init__(self, person: PersonAssets, restore: bool = False):
        super().__init__(person)
        from .swap import SwapState

        self.swap_state = SwapState()
        self.restore = restore    # option « netteté » : restauration du visage généré avant recollage

    def apply(self, frame, face):
        from .swap import swap_face

        return swap_face(frame, face, self.person.source, self.swap_state, restore=self.restore)


class FaceToneStrategy(FaceStrategy):
    """Niveau 2 : visage puis teint de la personne sur la peau visible (visage, oreilles, cou)."""

    def __init__(self, person: PersonAssets, restore: bool = False):
        super().__init__(person, restore)
        from .tone import ToneMatcher, ToneSettings

        lcfg = load_config().levels.face_tone
        self.matcher = ToneMatcher(person.tone, ToneSettings.from_config(lcfg)) if person.tone else None
        self.samples = int(lcfg.get("tone_samples", 16)) if self.matcher else 0

    def prepare(self, samples):
        if self.matcher:
            self.matcher.prepare(samples)

    def apply(self, frame, face):
        swapped = super().apply(frame, face)
        # Zone de peau et mesures sur l'image d'origine : BiSeNet n'est pas trompé par le visage généré.
        return self.matcher.apply(swapped, face, original=frame) if self.matcher else swapped


class HeadStrategy(Strategy):
    """Niveau 3 : tête entière de la photo de la personne (cheveux, forme, teint), animée par la tête du clip."""

    def __init__(self, person: PersonAssets):
        super().__init__(person)
        from .head import HeadEngine, SourceHead, choose_photo
        from .tone import ToneMatcher, ToneSettings

        levels = load_config().levels
        hcfg = levels.get("head", {})
        scale = float(hcfg.get("crop_scale", 1.5))
        photo = person.head_reference or choose_photo(person.photos)
        if photo is None:
            raise RuntimeError("Aucune photo de face exploitable pour la tête.")
        self.tone = ToneMatcher(person.tone, ToneSettings.from_config(levels.face_tone)) if person.tone else None
        self.engine = HeadEngine(SourceHead.from_path(photo, scale), scale, float(hcfg.get("motion_smoothing", 0.5)),
                                 self.tone)
        self.samples = int(hcfg.get("plate_samples", 24))

    def prepare(self, samples):
        if self.tone:
            self.tone.prepare(samples)
        self.engine.prepare(samples)

    def apply(self, frame, face):
        # Cou et oreilles restantes au teint de la personne, puis la nouvelle tête par-dessus.
        toned = self.tone.apply(frame, face, original=frame) if self.tone else frame
        return self.engine.apply(toned, face, original=frame)


STRATEGIES = {FACE: FaceStrategy, FACE_TONE: FaceToneStrategy, HEAD: HeadStrategy}


def make_strategy(level: str, person: PersonAssets, restore: bool = False) -> Strategy:
    """`restore` : option « netteté » (niveaux 1 et 2 seulement, ignorée ailleurs)."""
    if level not in STRATEGIES:
        raise ValueError(f"Niveau non disponible : {level}")
    cls = STRATEGIES[level]
    return cls(person, restore) if level in (FACE, FACE_TONE) else cls(person)


def models_ready(level: str) -> bool:
    from . import models

    return level in MODEL_GROUPS and all(models.is_ready(g) for g in MODEL_GROUPS[level])
