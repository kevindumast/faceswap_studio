"""Orchestration : découpe → remplacement image par image (selon le niveau) → assemblage.

Les trois temps sont séparés pour qu'on puisse envoyer seulement l'extrait coupé à un GPU distant :
cut() et assemble() restent toujours en local (son, étiquette IA, réinsertion dans la vidéo complète).

CLI : python -m src.pipeline --video clip.mp4 --start 12 --end 20 --faces data/source_faces --out out.mp4 [--level face_tone]
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from . import media
from .config import load_config
from .faces import detect, detect_boxes, embed, iou
from .identity import source_from_dir
from .levels import FACE, FACE_TONE, PersonAssets, make_strategy
from .temporal import TargetTracker

Progress = Callable[[str, int, int], None]


class Cancelled(Exception):
    pass


class RenderError(RuntimeError):
    pass


@dataclass
class RenderOptions:
    start: float
    end: float
    output: str = "segment"            # segment | full
    stabilize: bool = True
    ai_label: bool = True
    level: str = FACE


@dataclass
class FaceMapping:
    """Qui remplace qui : une personne source → un visage du clip."""
    person: PersonAssets
    target: dict | None = None         # {"t": s, "box": [x1, y1, x2, y2] normalisés} ; None = plus grand visage
    label: str = "Visage"


@dataclass
class CutSegment:
    path: Path
    info: media.VideoInfo              # infos de la vidéo source
    size: tuple[int, int]              # taille de travail (source réduite à render.max_height)


@dataclass
class RenderStats:
    frames: int = 0
    swapped: int = 0
    seconds: float = 0.0
    sec_per_frame: float = 0.0
    warnings: list[str] = field(default_factory=list)


def validate_range(start: float, end: float, duration: float) -> None:
    seg = load_config().segment
    if start < 0 or end > duration + 0.05:
        raise RenderError("Le passage dépasse la durée de la vidéo.")
    length = end - start
    if length < seg.min_s - 0.01 or length > seg.max_s + 0.01:
        raise RenderError(f"Le passage doit durer entre {seg.min_s} et {seg.max_s} s (actuellement {length:.1f} s).")


def cut(source: Path, opts: RenderOptions, work_dir: Path) -> CutSegment:
    """Découpe précise du passage, à la taille de travail, avec le son d'origine."""
    info = media.probe(source)
    validate_range(opts.start, opts.end, info.duration)
    size = media.scaled_size(info, int(load_config().render.max_height))
    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / "cut.mp4"
    media.cut_segment(source, path, opts.start, opts.end, size, info.has_audio)
    return CutSegment(path, info, size)


def relative_targets(mappings: list[FaceMapping], start: float, length: float) -> list[FaceMapping]:
    """Instants absolus (dans la vidéo source) → instants dans l'extrait coupé."""
    out = []
    for m in mappings:
        target = None
        if m.target is not None:
            target = {**m.target, "t": min(max(float(m.target["t"]) - start, 0.0), max(length - 0.05, 0.0))}
        out.append(FaceMapping(m.person, target, m.label))
    return out


def _reference_from_target(clip: Path, target: dict, size: tuple[int, int], cache: dict) -> np.ndarray | None:
    t = round(float(target["t"]), 3)
    if t not in cache:
        frame = cv2.resize(media.extract_frame(clip, t), size)
        cache[t] = detect(frame)
    faces = cache[t]
    if not faces:
        return None
    w, h = size
    x1, y1, x2, y2 = target["box"]
    box = [x1 * w, y1 * h, x2 * w, y2 * h]
    best = max(faces, key=lambda f: iou(f.bbox, box))
    return best.normed_embedding if iou(best.bbox, box) > 0.2 else None


def swap_segment(clip: Path, mappings: list[FaceMapping], level: str, out: Path, *, stabilize: bool = True,
                 progress: Progress | None = None, preview: Path | None = None,
                 stats: RenderStats | None = None) -> RenderStats:
    """Remplace les visages d'un extrait (instants des cibles relatifs à l'extrait). Écrit une vidéo sans son."""
    if not mappings:
        raise RenderError("Aucun visage à remplacer.")
    rcfg = load_config().render
    report = progress or (lambda *_: None)
    stats = stats or RenderStats()
    info = media.probe(clip)
    size = (info.width, info.height)

    # Identités cibles : une référence ArcFace par visage du clip à remplacer, une stratégie par association.
    smoothing = float(rcfg.smoothing) if stabilize else 0.0
    threshold = float(rcfg.similarity_threshold)
    ref_cache: dict = {}
    active: list[list] = []   # [mapping, tracker | None (créé à la 1re frame utile), stratégie, nb d'images remplacées]
    for m in mappings:
        strategy = make_strategy(level, m.person)
        if m.target is None:
            active.append([m, None, strategy, 0])
            continue
        reference = _reference_from_target(clip, m.target, size, ref_cache)
        if reference is not None:
            active.append([m, TargetTracker(reference, threshold, smoothing), strategy, 0])
        elif len(mappings) == 1:
            stats.warnings.append(f"{m.label} introuvable à l'instant choisi : on prend le plus grand visage.")
            active.append([m, None, strategy, 0])
        else:
            stats.warnings.append(f"{m.label} introuvable à l'instant choisi : ignoré.")
    if not active:
        raise RenderError("Aucun des visages choisis n'a été retrouvé dans la vidéo.")

    cap = cv2.VideoCapture(str(clip))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or round(info.duration * info.fps)
    every = int(rcfg.preview_every)
    try:
        with media.FrameWriter(out, size, info.fps_str, int(rcfg.crf), str(rcfg.preset)) as writer:
            i = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                faces = detect_boxes(frame)

                def embed_once(f, frame=frame):
                    if f.embedding is None:  # un visage peut être vérifié par plusieurs suivis
                        embed(frame, f)

                # Chaque suivi choisit son visage ; un visage pris n'est plus disponible pour les suivants.
                claimed, picks = [], []
                for entry in active:
                    tracker = entry[1]
                    free = [f for f in faces if all(f is not c for c in claimed)]
                    if tracker is None:
                        if not free:
                            continue
                        # Pas de cible choisie : identité du plus grand visage de la première frame utile.
                        embed_once(free[0])
                        tracker = entry[1] = TargetTracker(free[0].normed_embedding, threshold, smoothing)
                    face = tracker.update(free, embed_once)
                    if face is not None:
                        claimed.append(face)
                        picks.append((face, entry[2]))
                        entry[3] += 1
                for face, strategy in picks:
                    frame = strategy.apply(frame, face)
                if picks:
                    stats.swapped += 1
                writer.write(frame)
                i += 1
                if preview is not None and (i == 1 or i % every == 0):
                    small = cv2.resize(frame, (int(frame.shape[1] * 480 / frame.shape[0]) // 2 * 2, 480))
                    tmp = preview.with_suffix(".tmp.jpg")
                    cv2.imwrite(str(tmp), small, [cv2.IMWRITE_JPEG_QUALITY, 82])
                    tmp.replace(preview)
                report("swap", i, max(total, i))
            stats.frames = i
    finally:
        cap.release()

    if stats.frames == 0:
        raise RenderError("Aucune frame lue dans l'extrait.")
    if stats.swapped == 0:
        raise RenderError("Aucun visage cible détecté dans ce passage.")
    for mapping, _, _, count in active:
        if count < stats.frames * 0.5:
            stats.warnings.append(f"{mapping.label} remplacé sur {count}/{stats.frames} images seulement (profils, occlusions).")
    return stats


def assemble(source: Path, segment: CutSegment, swapped: Path, opts: RenderOptions, out: Path, work_dir: Path) -> None:
    """Son d'origine, étiquette IA, et réinsertion éventuelle dans la vidéo complète."""
    rcfg = load_config().render
    label = media.render_label(work_dir / "label.png", segment.size[1]) if opts.ai_label else None
    crf, preset = int(rcfg.crf), str(rcfg.preset)
    if opts.output == "full":
        media.assemble_full(source, swapped, out, segment.info, segment.size, opts.start, opts.end, label, crf, preset)
    else:
        media.assemble_segment(swapped, segment.path, out, segment.info.has_audio, label, crf, preset)


def render(source: Path, mappings: list[FaceMapping], opts: RenderOptions, work_dir: Path, out: Path,
           progress: Progress | None = None, preview: Path | None = None) -> RenderStats:
    """Rendu complet en local. Les instants des cibles sont absolus (dans la vidéo source)."""
    report = progress or (lambda *_: None)
    stats = RenderStats()
    t0 = time.perf_counter()

    report("cut", 0, 1)
    segment = cut(source, opts, work_dir)
    report("cut", 1, 1)

    swapped = work_dir / "swapped.mp4"
    local = relative_targets(mappings, opts.start, opts.end - opts.start)
    swap_segment(segment.path, local, opts.level, swapped, stabilize=opts.stabilize, progress=report,
                 preview=preview, stats=stats)
    stats.sec_per_frame = (time.perf_counter() - t0) / stats.frames

    report("assemble", 0, 1)
    assemble(source, segment, swapped, opts, out, work_dir)
    report("assemble", 1, 1)
    stats.seconds = time.perf_counter() - t0
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Face swap sur un passage de vidéo")
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--start", required=True, type=float)
    ap.add_argument("--end", required=True, type=float)
    ap.add_argument("--faces", required=True, type=Path, help="dossier de photos du visage source")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--level", choices=[FACE, FACE_TONE], default=FACE)
    ap.add_argument("--full", action="store_true", help="réinsérer le passage dans la vidéo complète")
    ap.add_argument("--no-label", action="store_true")
    ap.add_argument("--no-stabilize", action="store_true")
    args = ap.parse_args()

    person = PersonAssets(source=source_from_dir(args.faces))
    if args.level == FACE_TONE:
        from .assets import person_tone

        person.tone = person_tone(sorted(p for p in args.faces.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}))
    opts = RenderOptions(args.start, args.end, "full" if args.full else "segment",
                         stabilize=not args.no_stabilize, ai_label=not args.no_label, level=args.level)

    def show(stage: str, done: int, total: int) -> None:
        print(f"\r{stage:<9} {done}/{total}", end="", flush=True)

    stats = render(args.video, [FaceMapping(person)], opts, args.out.parent / f".{args.out.stem}_work", args.out, show)
    print(f"\nOK : {args.out}  ({stats.swapped}/{stats.frames} frames, {stats.sec_per_frame:.2f} s/frame)")
    for w in stats.warnings:
        print("! " + w)


if __name__ == "__main__":
    main()
