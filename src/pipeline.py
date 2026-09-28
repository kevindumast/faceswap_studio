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
from fractions import Fraction
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from . import media
from .config import load_config
from .faces import detect, detect_boxes, detect_in_region, embed, iou
from .identity import source_from_dir
from .levels import FACE, FACE_TONE, HEAD, PersonAssets, make_strategy
from .review import FramePlan, save_stats
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
    restore: bool = False              # option « netteté » (niveaux 1 et 2) : voir src/restore.py


@dataclass
class FaceMapping:
    """Qui remplace qui : une personne source → un visage du clip."""
    person: PersonAssets
    target: dict | None = None         # {"t": s, "box": [x1, y1, x2, y2] normalisés} ; None = plus grand visage
    label: str = "Visage"
    person_id: str | None = None       # personne de la bibliothèque (vérification avant l'assemblage)


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
        out.append(FaceMapping(m.person, target, m.label, m.person_id))
    return out


def _reference_from_target(clip: Path, target: dict, size: tuple[int, int], cache: dict) -> np.ndarray | None:
    t = round(float(target["t"]), 3)
    if t not in cache:
        frame = cv2.resize(media.extract_frame(clip, t), size)
        cache[t] = frame, detect(frame)
    frame, faces = cache[t]
    w, h = size
    x1, y1, x2, y2 = target["box"]
    box = [x1 * w, y1 * h, x2 * w, y2 * h]
    if faces:
        best = max(faces, key=lambda f: iou(f.bbox, box))
        if iou(best.bbox, box) > 0.2:
            return best.normed_embedding
    sc = load_config().get("scan", {})
    det_size, thresh = int(sc.get("det_size", 640)), float(sc.get("det_thresh", 0.3))
    # Visage trouvé à l'étape Visages avec un seuil plus tolérant que celui du rendu : même seuil ici.
    loose = [f for f in detect_boxes(frame, det_size, thresh) if iou(f.bbox, box) > 0.2]
    if loose:
        face = max(loose, key=lambda f: iou(f.bbox, box))
        embed(frame, face)
        return face.normed_embedding
    # Visage ajouté à la main à l'étape Visages, trop petit pour la détection sur l'image entière : on zoome dessus.
    face = detect_in_region(frame, target["box"], det_size, thresh)
    return face.normed_embedding if face is not None else None


class DuplicateDetector:
    """Repère les images identiques à la précédente (vidéos converties en 50/60 i/s en dupliquant des images).

    Comparaison zone par zone (blocs de 10×10 px sur une vignette 320×180) : une moyenne sur toute l'image masquerait
    un petit mouvement de lèvres dans un plan fixe. Mesuré : copies ≤ 2,4 d'écart max par bloc, vraies images ≥ 7,5.
    """

    def __init__(self, threshold: float):
        self.threshold = threshold
        self.prev: np.ndarray | None = None
        self.diff: float | None = None     # écart moyen avec l'image précédente (changements de plan, vérification)

    def is_duplicate(self, frame: np.ndarray) -> bool:
        small = cv2.cvtColor(cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
        prev, self.prev = self.prev, small
        if prev is None:
            self.diff = None
            return False
        delta = np.abs(small - prev)
        self.diff = float(delta.mean())
        if self.threshold <= 0:
            return False
        blocks = delta.reshape(18, 10, 32, 10).mean(axis=(1, 3))
        return float(blocks.max()) < self.threshold


def prepare_strategies(clip: Path, active: list, threshold: float) -> None:
    """Pré-passe des stratégies qui en demandent une (teint du niveau 2) : visage de chaque cible sur des images
    réparties sur tout l'extrait. Lecture séquentielle (pas de saut dans la vidéo) : même résultat à chaque reprise."""
    todo = [entry for entry in active if entry[2].samples > 0]
    if not todo:
        return
    cap = cv2.VideoCapture(str(clip))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    count = max(entry[2].samples for entry in todo)
    wanted = set(np.linspace(0, max(total - 1, 0), num=min(count, max(total, 1))).round().astype(int).tolist())
    samples: dict[int, list] = {id(entry): [] for entry in todo}
    try:
        for i in range(max(wanted, default=-1) + 1):
            if not cap.grab():
                break
            if i not in wanted:
                continue
            ok, frame = cap.retrieve()
            faces = detect_boxes(frame) if ok else []
            for entry in todo:
                tracker = entry[1]
                if not faces:
                    break
                if tracker is None:        # pas de cible choisie : le plus grand visage
                    samples[id(entry)].append((frame, faces[0]))
                    continue
                for f in faces:
                    if f.embedding is None:
                        embed(frame, f)
                best = max(faces, key=lambda f: float(np.dot(f.normed_embedding, tracker.reference)))
                if float(np.dot(best.normed_embedding, tracker.reference)) >= threshold:
                    samples[id(entry)].append((frame, best))
    finally:
        cap.release()
    for entry in todo:
        entry[2].prepare(samples[id(entry)])


def _compute(frame: np.ndarray, active: list, threshold: float, smoothing: float) -> tuple[np.ndarray, list, list]:
    """Détection, suivi et remplacement sur une image. Renvoie l'image, les (visage, association) remplacés et tous
    les visages détectés."""
    faces = detect_boxes(frame)

    def embed_once(f):
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
            picks.append((face, entry))
    out = frame
    for face, entry in picks:
        out = entry[2].apply(out, face)
    return out, picks, faces


def _engine_failure(exc: BaseException) -> bool:
    """Erreur levée par onnxruntime (carte graphique bloquée puis réinitialisée par Windows, par exemple), ou session
    recréée sur le CPU faute de carte revenue à temps (insightface le refuse)."""
    return type(exc).__module__.startswith("onnxruntime") or "activated providers" in str(exc)


def _after_gpu_reset(fn: Callable, attempts: int = 3, wait_s: float = 3.0):
    """Relance `fn` sur des sessions neuves, en laissant à la carte le temps de revenir (quelques secondes)."""
    for attempt in range(1, attempts + 1):
        time.sleep(wait_s * attempt)
        reset_sessions()
        try:
            return fn()
        except Exception as exc:
            if not _engine_failure(exc) or attempt == attempts:
                raise


def reset_sessions() -> None:
    """Oublie toutes les sessions ONNX : elles sont recréées sur la carte réinitialisée au prochain appel."""
    from . import faces, head, parsing, restore, swap

    for factory in (faces.analyzer, swap.swapper, parsing._session, head._session, restore._session):
        factory.cache_clear()


def swap_segment(clip: Path, mappings: list[FaceMapping], level: str, out: Path, *, stabilize: bool = True,
                 restore: bool = False, progress: Progress | None = None, preview: Path | None = None,
                 stats: RenderStats | None = None, checkpoint: Checkpoint | None = None,
                 plan: FramePlan | None = None) -> RenderStats:
    """Remplace les visages d'un extrait (instants des cibles relatifs à l'extrait). Écrit une vidéo sans son.

    plan : journal image par image (visages, qui les a remplacés) pour la vérification avant l'assemblage.
    """
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
    # [mapping, tracker | None (créé à la 1re frame utile), stratégie, nb d'images remplacées, index de l'association]
    active: list[list] = []
    for k, m in enumerate(mappings):
        strategy = make_strategy(level, m.person, restore)
        if m.target is None:
            active.append([m, None, strategy, 0, k])
            continue
        reference = _reference_from_target(clip, m.target, size, ref_cache)
        if reference is not None:
            active.append([m, TargetTracker(reference, threshold, smoothing), strategy, 0, k])
        elif len(mappings) == 1:
            stats.warnings.append(f"{m.label} introuvable à l'instant choisi : on prend le plus grand visage.")
            active.append([m, None, strategy, 0, k])
        else:
            stats.warnings.append(f"{m.label} introuvable à l'instant choisi : ignoré.")
            if plan is not None:
                plan.mappings[k]["active"] = False
    if not active:
        raise RenderError("Aucun des visages choisis n'a été retrouvé dans la vidéo.")
    prepare_strategies(clip, active, threshold)

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
        if plan is not None:
            plan.save()                 # avant le point de reprise : jamais un journal plus court que les morceaux
        checkpoint.save()

    cap = cv2.VideoCapture(str(clip))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or round(info.duration * info.fps)
    every = int(rcfg.preview_every)
    duplicates = DuplicateDetector(float(rcfg.get("duplicate_block_diff", 4.0)))
    last_out: np.ndarray | None = None
    gpu_resets = 0
    if plan is not None:
        plan.truncate(start_frame)
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
                if plan is not None:
                    plan.add_duplicate()
                i += 1
            else:
                if last_out is None:
                    duplicates.is_duplicate(frame)  # mémorise la première image
                try:
                    frame, picks, faces = _compute(frame, active, threshold, smoothing)
                except Exception as exc:
                    if not _engine_failure(exc):
                        raise
                    # Carte graphique réinitialisée par Windows (calcul trop long, carte trop chaude) : sessions
                    # recréées, image refaite (3 essais). Au-delà, le rendu s'arrête (reprise possible ensuite).
                    gpu_resets += 1
                    frame, picks, faces = _after_gpu_reset(lambda f=frame: _compute(f, active, threshold, smoothing))
                if plan is not None:
                    references = {e[4]: e[1].reference for e in active if e[1] is not None}
                    plan.add(faces, [(face, e[4], e[2]) for face, e in picks], references, duplicates.diff)
                for _, entry in picks:
                    entry[3] += 1
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
    if gpu_resets:
        stats.warnings.append(f"Carte graphique réinitialisée {gpu_resets} fois par Windows (calcul trop long) : "
                              "images concernées recalculées.")
    if plan is not None:
        plan.save()
    else:   # sans journal (Space ZeroGPU) : simple compte ; avec, la vérification donne le détail par visage
        computed = stats.frames - stats.reused
        for mapping, _, _, count, _ in active:
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


def render_frames(source: Path, mappings: list[FaceMapping], opts: RenderOptions, work_dir: Path,
                  progress: Progress | None = None, preview: Path | None = None) -> tuple[CutSegment, RenderStats]:
    """Découpe et remplacement image par image, sans assemblage : work_dir/swapped.mp4 (sans son), plan.json (journal
    pour la vérification) et stats.json. Les instants des cibles sont absolus (dans la vidéo source)."""
    report = progress or (lambda *_: None)
    stats = RenderStats()
    work_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Checkpoint.load(work_dir)       # reprise après une pause, s'il y en a une

    report("cut", 0, 1)
    segment = cut(source, opts, work_dir, reuse=checkpoint.frames_done > 0)
    report("cut", 1, 1)

    swapped = work_dir / "swapped.mp4"
    local = relative_targets(mappings, opts.start, opts.end - opts.start)
    # Journal repris avec le rendu ; un rendu commencé avant la vérification n'en a pas (il s'assemble directement).
    plan = FramePlan.load(work_dir)
    if checkpoint.frames_done == 0:
        plan = FramePlan(work_dir / "plan.json", segment.size, float(Fraction(segment.fps_str)),
                         [{"label": m.label, "person": m.person_id, "active": True} for m in local])
    swap_segment(segment.path, local, opts.level, swapped, stabilize=opts.stabilize, restore=opts.restore,
                 progress=report, preview=preview, stats=stats, checkpoint=checkpoint, plan=plan)   # peut lever Paused
    elapsed = checkpoint.elapsed                 # temps de calcul cumulé, toutes sessions confondues
    stats.sec_per_frame = elapsed / stats.frames
    stats.sec_per_computed = elapsed / max(1, stats.frames - stats.reused)
    stats.seconds = elapsed
    save_stats(work_dir, stats)
    return segment, stats


def finish(source: Path, segment: CutSegment, opts: RenderOptions, work_dir: Path, out: Path,
           progress: Progress | None = None) -> None:
    """Assemblage (son, étiquette, vidéo complète), puis ménage des morceaux du rendu."""
    report = progress or (lambda *_: None)
    report("assemble", 0, 1)
    assemble(source, segment, work_dir / "swapped.mp4", opts, out, work_dir)
    report("assemble", 1, 1)
    Checkpoint.load(work_dir).clear()


def render(source: Path, mappings: list[FaceMapping], opts: RenderOptions, work_dir: Path, out: Path,
           progress: Progress | None = None, preview: Path | None = None) -> RenderStats:
    """Rendu complet en local, sans vérification. Les instants des cibles sont absolus (dans la vidéo source)."""
    t0 = time.perf_counter()
    segment, stats = render_frames(source, mappings, opts, work_dir, progress, preview)
    finish(source, segment, opts, work_dir, out, progress)
    stats.seconds = time.perf_counter() - t0
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Face swap sur un passage de vidéo")
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--start", required=True, type=float)
    ap.add_argument("--end", required=True, type=float)
    ap.add_argument("--faces", required=True, type=Path, help="dossier de photos du visage source")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--level", choices=[FACE, FACE_TONE, HEAD], default=FACE)
    ap.add_argument("--full", action="store_true", help="réinsérer le passage dans la vidéo complète")
    ap.add_argument("--no-label", action="store_true")
    ap.add_argument("--no-stabilize", action="store_true")
    ap.add_argument("--limit-fps", action="store_true", help="plafonner à render.fps_cap i/s (une image sur N)")
    ap.add_argument("--restore", action="store_true", help="option « netteté » (niveaux 1 et 2, CodeFormer)")
    args = ap.parse_args()

    photos = sorted(p for p in args.faces.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    person = PersonAssets(source=source_from_dir(args.faces), photos=photos)
    if args.level in (FACE_TONE, HEAD):
        from .assets import person_tone

        person.tone = person_tone(photos)
    opts = RenderOptions(args.start, args.end, "full" if args.full else "segment",
                         stabilize=not args.no_stabilize, ai_label=not args.no_label, level=args.level,
                         limit_fps=args.limit_fps, restore=args.restore)

    def show(stage: str, done: int, total: int) -> None:
        print(f"\r{stage:<9} {done}/{total}", end="", flush=True)

    stats = render(args.video, [FaceMapping(person)], opts, args.out.parent / f".{args.out.stem}_work", args.out, show)
    print(f"\nOK : {args.out}  ({stats.swapped}/{stats.frames} frames, {stats.reused} copies réutilisées, "
          f"{stats.sec_per_frame:.2f} s/frame, {stats.seconds:.0f} s au total)")
    for w in stats.warnings:
        print("! " + w)


if __name__ == "__main__":
    main()
