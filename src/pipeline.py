"""Orchestration bout en bout : découpe → swap frame par frame → assemblage.

CLI : python -m src.pipeline --video clip.mp4 --start 12 --end 20 --faces data/source_faces --out out.mp4
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
from .identity import SourceFace, source_from_dir
from .swap import swap_face
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


@dataclass
class FaceMapping:
    """Qui remplace qui : une identité source → un visage du clip."""
    source: SourceFace
    target: dict | None = None         # {"t": s, "box": [x1, y1, x2, y2] normalisés} ; None = plus grand visage
    label: str = "Visage"


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


def _reference_from_target(source: Path, target: dict, size: tuple[int, int], cache: dict) -> np.ndarray | None:
    t = round(float(target["t"]), 3)
    if t not in cache:
        frame = cv2.resize(media.extract_frame(source, t), size)
        cache[t] = detect(frame)
    faces = cache[t]
    if not faces:
        return None
    w, h = size
    x1, y1, x2, y2 = target["box"]
    box = [x1 * w, y1 * h, x2 * w, y2 * h]
    best = max(faces, key=lambda f: iou(f.bbox, box))
    return best.normed_embedding if iou(best.bbox, box) > 0.2 else None


def render(source: Path, mappings: list[FaceMapping], opts: RenderOptions, work_dir: Path, out: Path,
           progress: Progress | None = None, preview: Path | None = None) -> RenderStats:
    if not mappings:
        raise RenderError("Aucun visage à remplacer.")
    cfg = load_config()
    rcfg = cfg.render
    report = progress or (lambda *_: None)
    work_dir.mkdir(parents=True, exist_ok=True)
    stats = RenderStats()
    t0 = time.perf_counter()

    info = media.probe(source)
    validate_range(opts.start, opts.end, info.duration)
    size = media.scaled_size(info, int(rcfg.max_height))

    # 1. Découpe précise
    report("cut", 0, 1)
    cut = work_dir / "cut.mp4"
    media.cut_segment(source, cut, opts.start, opts.end, size, info.has_audio)
    report("cut", 1, 1)

    # 2. Identités cibles : une référence ArcFace par visage du clip à remplacer
    smoothing = float(rcfg.smoothing) if opts.stabilize else 0.0
    threshold = float(rcfg.similarity_threshold)
    ref_cache: dict = {}
    active: list[list] = []   # [mapping, tracker | None (créé à la 1re frame utile), nb de frames remplacées]
    for m in mappings:
        if m.target is None:
            active.append([m, None, 0])
            continue
        reference = _reference_from_target(source, m.target, size, ref_cache)
        if reference is not None:
            active.append([m, TargetTracker(reference, threshold, smoothing), 0])
        elif len(mappings) == 1:
            stats.warnings.append(f"{m.label} introuvable à l'instant choisi : on prend le plus grand visage.")
            active.append([m, None, 0])
        else:
            stats.warnings.append(f"{m.label} introuvable à l'instant choisi : ignoré.")
    if not active:
        raise RenderError("Aucun des visages choisis n'a été retrouvé dans la vidéo.")

    # 3. Swap frame par frame, encodé à la volée
    swapped = work_dir / "swapped.mp4"
    cap = cv2.VideoCapture(str(cut))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or round((opts.end - opts.start) * info.fps)
    every = int(rcfg.preview_every)
    try:
        with media.FrameWriter(swapped, size, info.fps_str, int(rcfg.crf), str(rcfg.preset)) as writer:
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
                    mapping, tracker = entry[0], entry[1]
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
                        picks.append((face, mapping.source))
                        entry[2] += 1
                for face, src in picks:
                    frame = swap_face(frame, face, src)
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
    for mapping, _, count in active:
        if count < stats.frames * 0.5:
            stats.warnings.append(f"{mapping.label} remplacé sur {count}/{stats.frames} images seulement (profils, occlusions).")
    stats.sec_per_frame = (time.perf_counter() - t0) / stats.frames

    # 4. Assemblage + audio d'origine
    report("assemble", 0, 1)
    label = media.render_label(work_dir / "label.png", size[1]) if opts.ai_label else None
    crf, preset = int(rcfg.crf), str(rcfg.preset)
    if opts.output == "full":
        media.assemble_full(source, swapped, out, info, size, opts.start, opts.end, label, crf, preset)
    else:
        media.assemble_segment(swapped, cut, out, info.has_audio, label, crf, preset)
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
    ap.add_argument("--full", action="store_true", help="réinsérer le passage dans la vidéo complète")
    ap.add_argument("--no-label", action="store_true")
    ap.add_argument("--no-stabilize", action="store_true")
    args = ap.parse_args()

    mappings = [FaceMapping(source=source_from_dir(args.faces))]
    opts = RenderOptions(args.start, args.end, "full" if args.full else "segment",
                         stabilize=not args.no_stabilize, ai_label=not args.no_label)

    def show(stage: str, done: int, total: int) -> None:
        print(f"\r{stage:<9} {done}/{total}", end="", flush=True)

    stats = render(args.video, mappings, opts, args.out.parent / f".{args.out.stem}_work", args.out, show)
    print(f"\nOK : {args.out}  ({stats.swapped}/{stats.frames} frames, {stats.sec_per_frame:.2f} s/frame)")
    for w in stats.warnings:
        print("! " + w)


if __name__ == "__main__":
    main()
