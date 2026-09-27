"""Orchestration : découpe → remplacement image par image (selon le niveau) → assemblage.

Les trois temps sont séparés pour qu'on puisse envoyer seulement l'extrait coupé à un GPU distant :
cut() et assemble() restent toujours en local (son, étiquette IA, réinsertion dans la vidéo complète).

CLI : python -m src.pipeline --video clip.mp4 --start 12 --end 20 --faces data/source_faces --out out.mp4 [--level face_tone]
"""
from __future__ import annotations

import argparse
import contextlib
import json
import time
from dataclasses import asdict, dataclass, field
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


class Paused(Exception):
    """Pause demandée : le morceau en cours est refermé proprement et le point de reprise enregistré."""


@dataclass
class Checkpoint:
    """Point de reprise d'un rendu local : morceaux déjà écrits + compteurs (work_dir/checkpoint.json).

    Permet de mettre un rendu en pause, de fermer l'appli ou d'éteindre le PC, puis de reprendre à la même image.
    """
    path: Path
    frames_done: int = 0
    chunks: list[str] = field(default_factory=list)
    swapped: int = 0
    reused: int = 0
    counts: dict[str, int] = field(default_factory=dict)
    elapsed: float = 0.0                # temps de calcul cumulé sur toutes les sessions

    @classmethod
    def load(cls, work_dir: Path) -> "Checkpoint":
        path = work_dir / "checkpoint.json"
        if path.is_file():
            return cls(path=path, **json.loads(path.read_text(encoding="utf-8")))
        return cls(path=path)

    def save(self) -> None:
        data = {k: v for k, v in asdict(self).items() if k != "path"}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        tmp.replace(self.path)          # jamais de JSON à moitié écrit si le PC s'éteint à ce moment-là

    def clear(self) -> None:
        for name in self.chunks:
            (self.path.parent / name).unlink(missing_ok=True)
        self.path.unlink(missing_ok=True)
        _drop_partials(self.path.parent)


def partial_preview(work_dir: Path) -> Path | None:
    """Ce qui est déjà rendu (rendu en pause) : morceaux recollés + son d'origine de l'extrait, sans réencodage.

    Un fichier par point de reprise (partial_<images>.mp4) : fabriqué une seule fois, jamais réécrit pendant qu'il est lu.
    """
    checkpoint = Checkpoint.load(work_dir)
    parts = [work_dir / name for name in checkpoint.chunks]
    if not parts or not all(p.is_file() for p in parts):
        return None
    out = work_dir / f"partial_{checkpoint.frames_done}.mp4"
    if not out.is_file():
        _drop_partials(work_dir)
        media.concat_with_audio(parts, work_dir / "cut.mp4", out)
    return out


def _drop_partials(work_dir: Path) -> None:
    for old in work_dir.glob("partial_*.mp4"):
        with contextlib.suppress(OSError):   # encore ouvert par le navigateur (Windows) : il partira la fois suivante
            old.unlink()


def _tick(report: Progress, i: int, total: int) -> bool:
    """Progression ; True si une pause a été demandée (on referme alors proprement le morceau en cours)."""
    try:
        report("swap", i, max(total, i))
    except Paused:
        return True
    return False


@dataclass
class RenderOptions:
    start: float
    end: float
    output: str = "segment"            # segment | full
    stabilize: bool = True
    ai_label: bool = True
    level: str = FACE
    limit_fps: bool = False            # vidéos > render.fps_cap : une image sur N (60 → 30 i/s)


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
    fps_str: str                       # cadence de sortie (celle de la source, ou plafonnée)


@dataclass
class RenderStats:
    frames: int = 0
    swapped: int = 0
    seconds: float = 0.0
    sec_per_frame: float = 0.0
    reused: int = 0                    # images identiques à la précédente : résultat réutilisé, pas recalculé
    sec_per_computed: float = 0.0      # temps par image réellement calculée (sert à l'estimation des prochains rendus)
    warnings: list[str] = field(default_factory=list)


def validate_range(start: float, end: float, duration: float) -> None:
    seg = load_config().segment
    if start < 0 or end > duration + 0.05:
        raise RenderError("Le passage dépasse la durée de la vidéo.")
    length = end - start
    if length < seg.min_s - 0.01 or length > seg.max_s + 0.01:
        raise RenderError(f"Le passage doit durer entre {seg.min_s} et {seg.max_s} s (actuellement {length:.1f} s).")


def cut(source: Path, opts: RenderOptions, work_dir: Path, reuse: bool = False) -> CutSegment:
    """Découpe précise du passage, à la taille de travail, avec le son d'origine."""
    rcfg = load_config().render
    info = media.probe(source)
    validate_range(opts.start, opts.end, info.duration)
    size = media.scaled_size(info, int(rcfg.max_height))
    fps_str = media.capped_fps(info.fps_str, float(rcfg.get("fps_cap", 30))) if opts.limit_fps else info.fps_str
    work_dir.mkdir(parents=True, exist_ok=True)
    path = work_dir / "cut.mp4"
    if not (reuse and path.is_file()):  # reprise après une pause : l'extrait déjà découpé est réutilisé tel quel
        media.cut_segment(source, path, opts.start, opts.end, size, info.has_audio,
                          fps_str if fps_str != info.fps_str else None)
    return CutSegment(path, info, size, fps_str)


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


class DuplicateDetector:
    """Repère les images identiques à la précédente (vidéos converties en 50/60 i/s en dupliquant des images).

    Comparaison zone par zone (blocs de 10×10 px sur une vignette 320×180) : une moyenne sur toute l'image masquerait
    un petit mouvement de lèvres dans un plan fixe. Mesuré : copies ≤ 2,4 d'écart max par bloc, vraies images ≥ 7,5.
    """

    def __init__(self, threshold: float):
        self.threshold = threshold
        self.prev: np.ndarray | None = None

    def is_duplicate(self, frame: np.ndarray) -> bool:
        small = cv2.cvtColor(cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
        prev, self.prev = self.prev, small
        if prev is None or self.threshold <= 0:
            return False
        blocks = np.abs(small - prev).reshape(18, 10, 32, 10).mean(axis=(1, 3))
        return float(blocks.max()) < self.threshold


def swap_segment(clip: Path, mappings: list[FaceMapping], level: str, out: Path, *, stabilize: bool = True,
                 progress: Progress | None = None, preview: Path | None = None,
                 stats: RenderStats | None = None, checkpoint: Checkpoint | None = None) -> RenderStats:
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

    # Avec un point de reprise : écriture par morceaux, et on repart après les images déjà rendues.
    # Un morceau est refermé toutes les ~2 min : un arrêt brutal (PC éteint, plantage) ne perd que ce qui suit.
    start_frame = checkpoint.frames_done if checkpoint else 0
    if checkpoint:
        stats.swapped, stats.reused = checkpoint.swapped, checkpoint.reused
        for entry in active:
            entry[3] = checkpoint.counts.get(entry[0].label, 0)
    roll_every = float(rcfg.get("checkpoint_every_s", 120))
    mark = [time.perf_counter()]
    paused = False

    def new_writer() -> media.FrameWriter:
        target = checkpoint.path.parent / f"chunk_{len(checkpoint.chunks):03d}.mp4" if checkpoint else out
        return media.FrameWriter(target, size, info.fps_str, int(rcfg.crf), str(rcfg.preset))

    def save_chunk(writer: media.FrameWriter, frames_done: int) -> None:
        """Referme le morceau en cours et enregistre le point de reprise."""
        writer.close()
        if writer.dst.is_file():
            checkpoint.chunks.append(writer.dst.name)
        checkpoint.frames_done = frames_done
        checkpoint.swapped, checkpoint.reused = stats.swapped, stats.reused
        checkpoint.counts = {entry[0].label: entry[3] for entry in active}
        now = time.perf_counter()
        checkpoint.elapsed += now - mark[0]
        mark[0] = now
        checkpoint.save()

    cap = cv2.VideoCapture(str(clip))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or round(info.duration * info.fps)
    every = int(rcfg.preview_every)
    duplicates = DuplicateDetector(float(rcfg.get("duplicate_block_diff", 4.0)))
    last_out: np.ndarray | None = None
    writer = new_writer()
    try:
        i = start_frame
        for _ in range(start_frame):  # reprise : images déjà dans les morceaux précédents
            if not cap.grab():
                break
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if last_out is not None and duplicates.is_duplicate(frame):
                # Copie de l'image précédente : même résultat, sans refaire détection ni swap.
                writer.write(last_out)
                stats.reused += 1
                i += 1
            else:
                if last_out is None:
                    duplicates.is_duplicate(frame)  # mémorise la première image
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
                last_out = frame
                i += 1
                if preview is not None and (i == 1 or i % every == 0):
                    small = cv2.resize(frame, (int(frame.shape[1] * 480 / frame.shape[0]) // 2 * 2, 480))
                    tmp = preview.with_suffix(".tmp.jpg")
                    cv2.imwrite(str(tmp), small, [cv2.IMWRITE_JPEG_QUALITY, 82])
                    tmp.replace(preview)
            if _tick(report, i, total):
                paused = True
                break
            if checkpoint is not None and time.perf_counter() - mark[0] >= roll_every:
                save_chunk(writer, i)
                writer = new_writer()
        stats.frames = i
    except BaseException:
        writer.abort()
        raise
    finally:
        cap.release()

    if checkpoint is not None:
        save_chunk(writer, stats.frames)
        if paused:
            raise Paused()
        if checkpoint.chunks:
            media.concat_videos([checkpoint.path.parent / c for c in checkpoint.chunks], out)
    else:
        writer.close()
        if paused:
            raise Paused()

    if stats.frames == 0:
        raise RenderError("Aucune frame lue dans l'extrait.")
    if stats.swapped == 0:
        raise RenderError("Aucun visage cible détecté dans ce passage.")
    computed = stats.frames - stats.reused
    for mapping, _, _, count in active:
        if count < computed * 0.5:
            stats.warnings.append(f"{mapping.label} remplacé sur {count}/{computed} images seulement (profils, occlusions).")
    return stats


def assemble(source: Path, segment: CutSegment, swapped: Path, opts: RenderOptions, out: Path, work_dir: Path) -> None:
    """Son d'origine, étiquette IA, et réinsertion éventuelle dans la vidéo complète."""
    rcfg = load_config().render
    label = media.render_label(work_dir / "label.png", segment.size[1]) if opts.ai_label else None
    crf, preset = int(rcfg.crf), str(rcfg.preset)
    if opts.output == "full":
        media.assemble_full(source, swapped, out, segment.info, segment.size, opts.start, opts.end, label, crf, preset,
                            segment.fps_str)
    else:
        media.assemble_segment(swapped, segment.path, out, segment.info.has_audio, label, crf, preset)


def render(source: Path, mappings: list[FaceMapping], opts: RenderOptions, work_dir: Path, out: Path,
           progress: Progress | None = None, preview: Path | None = None) -> RenderStats:
    """Rendu complet en local. Les instants des cibles sont absolus (dans la vidéo source)."""
    report = progress or (lambda *_: None)
    stats = RenderStats()
    t0 = time.perf_counter()
    work_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Checkpoint.load(work_dir)       # reprise après une pause, s'il y en a une

    report("cut", 0, 1)
    segment = cut(source, opts, work_dir, reuse=checkpoint.frames_done > 0)
    report("cut", 1, 1)

    swapped = work_dir / "swapped.mp4"
    local = relative_targets(mappings, opts.start, opts.end - opts.start)
    swap_segment(segment.path, local, opts.level, swapped, stabilize=opts.stabilize, progress=report,
                 preview=preview, stats=stats, checkpoint=checkpoint)   # peut lever Paused
    elapsed = checkpoint.elapsed                 # temps de calcul cumulé, toutes sessions confondues
    stats.sec_per_frame = elapsed / stats.frames
    stats.sec_per_computed = elapsed / max(1, stats.frames - stats.reused)

    report("assemble", 0, 1)
    assemble(source, segment, swapped, opts, out, work_dir)
    report("assemble", 1, 1)
    checkpoint.clear()
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
    ap.add_argument("--limit-fps", action="store_true", help="plafonner à render.fps_cap i/s (une image sur N)")
    args = ap.parse_args()

    person = PersonAssets(source=source_from_dir(args.faces))
    if args.level == FACE_TONE:
        from .assets import person_tone

        person.tone = person_tone(sorted(p for p in args.faces.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}))
    opts = RenderOptions(args.start, args.end, "full" if args.full else "segment",
                         stabilize=not args.no_stabilize, ai_label=not args.no_label, level=args.level,
                         limit_fps=args.limit_fps)

    def show(stage: str, done: int, total: int) -> None:
        print(f"\r{stage:<9} {done}/{total}", end="", flush=True)

    stats = render(args.video, [FaceMapping(person)], opts, args.out.parent / f".{args.out.stem}_work", args.out, show)
    print(f"\nOK : {args.out}  ({stats.swapped}/{stats.frames} frames, {stats.reused} copies réutilisées, "
          f"{stats.sec_per_frame:.2f} s/frame, {stats.seconds:.0f} s au total)")
    for w in stats.warnings:
        print("! " + w)


if __name__ == "__main__":
    main()
