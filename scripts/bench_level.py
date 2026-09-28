"""Banc de test d'un niveau image par image : mêmes images, plusieurs réglages, mesures de stabilité du teint.

Exemple, sur l'extrait d'un rendu déjà fait (comparé au résultat de l'ancienne version) :
  python scripts/bench_level.py --video data/jobs/<job>/cut.mp4 --start 40 --end 46 --person <personne> ^
      --compare avant=data/jobs/<job>/result.mp4 --variant apres --variant "sans_compensation:levels.face_tone.tone_wb_compensation=0"

--variant nom[:clé=valeur,clé=valeur] : un rendu avec la config actuelle, modifiée par ces clés (notation pointée).
--compare nom=vidéo : une vidéo déjà rendue du même passage (même début que --video), découpée sur la même plage.

Sorties (dossier --out) : cote_a_cote.mp4, planche.jpg (visages zoomés, mêmes images pour tous), mesures.json.
Mesures, sur la peau détectée (BiSeNet) du visage principal de chaque image :
- saut de teint : écart a/b (LAB) de la médiane de la peau d'une image à la suivante (moyenne et 95e centile) ;
- gris-bleu : part des pixels de peau dont b ≤ 130 (neutre = 128 ; une peau est nettement au-dessus) ;
- sans visage : images où aucun visage n'est trouvé ; s/image : temps de rendu.
L'original sert de référence : ses valeurs sont le « bruit » naturel de la mesure.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import media, parsing  # noqa: E402
from src.config import load_config  # noqa: E402
from src.faces import detect_boxes, iou  # noqa: E402
from src.levels import FACE_TONE, HEAD, PersonAssets  # noqa: E402
from src.pipeline import FaceMapping, RenderStats, swap_segment  # noqa: E402

SKIN = (parsing.SKIN, parsing.NOSE, parsing.NECK)
GREY_BLUE_B = 130


def parse_variant(spec: str) -> tuple[str, dict]:
    name, _, rest = spec.partition(":")
    overrides = {}
    for item in filter(None, rest.split(",")):
        key, _, value = item.partition("=")
        overrides[key.strip()] = yaml.safe_load(value)
    return name.strip(), overrides


def apply_overrides(base: dict, overrides: dict) -> None:
    """Remet la config chargée à `base`, puis applique les clés pointées (modifie la config en mémoire)."""
    cfg = load_config()
    cfg.clear()
    cfg.update(copy.deepcopy(base))
    for key, value in overrides.items():
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = value


def cut_range(src: Path, start: float, end: float, dst: Path) -> Path:
    info = media.probe(src)
    size = media.scaled_size(info, int(load_config().render.max_height))
    media.cut_segment(src, dst, start, end, size, info.has_audio, None)
    return dst


def load_person(args) -> PersonAssets:
    if args.person:
        from app import library

        return library.assets(args.person, args.level)
    from src.assets import person_tone
    from src.identity import source_from_dir

    exts = {".jpg", ".jpeg", ".png", ".webp"}
    photos = sorted(p for p in args.faces.iterdir() if p.suffix.lower() in exts)
    person = PersonAssets(source=source_from_dir(args.faces), photos=photos)
    if args.level in (FACE_TONE, HEAD):
        person.tone = person_tone(photos)
    return person


def read_frames(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def face_boxes(frames: list[np.ndarray]) -> list[np.ndarray | None]:
    """Boîte du visage suivi sur chaque image (le plus grand au départ, puis le plus proche du précédent)."""
    boxes, prev = [], None
    for frame in frames:
        faces = detect_boxes(frame)
        if not faces:
            boxes.append(None)
            continue
        face = faces[0] if prev is None else max(faces, key=lambda f: iou(f.bbox, prev))
        if prev is not None and iou(face.bbox, prev) < 0.2:
            face = faces[0]
        prev = np.asarray(face.bbox[:4], np.float32)
        boxes.append(prev)
    return boxes


def skin_measures(frames: list[np.ndarray], boxes: list) -> dict:
    medians, grey = [], []
    for frame, box in zip(frames, boxes):
        if box is None:
            medians.append(None)
            continue
        x1, y1, x2, y2 = parsing.face_crop_box(box, frame.shape)
        crop = frame[y1:y2, x1:x2]
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
        skin = parsing.mask(parsing.parse(crop), SKIN) & (lab[..., 0] > 25) & (lab[..., 0] < 245)
        if skin.sum() < 150:
            medians.append(None)
            continue
        medians.append(np.median(lab[skin], axis=0))
        grey.append(float((lab[..., 2][skin] <= GREY_BLUE_B).mean()))
    jumps = [float(np.linalg.norm(b[1:] - a[1:])) for a, b in zip(medians, medians[1:]) if a is not None and b is not None]
    found = [m for m in medians if m is not None]
    return {
        "saut_teint_moyen": round(float(np.mean(jumps)), 2) if jumps else None,
        "saut_teint_p95": round(float(np.percentile(jumps, 95)), 2) if jumps else None,
        "gris_bleu_pct": round(100 * float(np.mean(grey)), 2) if grey else None,
        "gris_bleu_pire_image_pct": round(100 * float(np.max(grey)), 2) if grey else None,
        "teint_median_lab": [round(float(v), 1) for v in np.median(found, axis=0)] if found else None,
        "sans_visage": sum(1 for m in medians if m is None),
    }


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (min(out.shape[1], 12 + 11 * len(text)), 28), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def side_by_side(videos: dict[str, list[np.ndarray]], dst: Path, fps_str: str, height: int) -> None:
    n = min(len(v) for v in videos.values())
    first = next(iter(videos.values()))[0]
    width = int(first.shape[1] * height / first.shape[0]) // 2 * 2
    writer = media.FrameWriter(dst, (width * len(videos), height), fps_str, 20, "veryfast")
    for i in range(n):
        tiles = [label(cv2.resize(v[i], (width, height), interpolation=cv2.INTER_AREA), name) for name, v in videos.items()]
        writer.write(np.hstack(tiles))
    writer.close()


def contact_sheet(videos: dict[str, list[np.ndarray]], boxes: list, dst: Path, count: int, tile: int = 220) -> list[int]:
    """Visages zoomés : une ligne par vidéo, les mêmes images (celles où l'original a un visage) en colonnes."""
    n = min(len(v) for v in videos.values())
    usable = [i for i in range(n) if boxes[i] is not None]
    if not usable:
        return []
    picks = [usable[int(round(k))] for k in np.linspace(0, len(usable) - 1, num=min(count, len(usable)))]
    rows = []
    for name, frames in videos.items():
        cells = []
        for i in picks:
            x1, y1, x2, y2 = parsing.face_crop_box(boxes[i], frames[i].shape, scale=1.8, down=0.08)
            cells.append(cv2.resize(frames[i][y1:y2, x1:x2], (tile, tile), interpolation=cv2.INTER_AREA))
        rows.append(np.hstack([label(cells[0], name)] + cells[1:]))
    header = np.hstack([label(np.zeros((28, tile, 3), np.uint8), f"image {i}") for i in picks])
    cv2.imwrite(str(dst), np.vstack([header] + rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
    return picks


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # console Windows redirigée : cp1252 sinon
    ap = argparse.ArgumentParser(description="Banc de test d'un niveau image par image (mêmes images, plusieurs réglages)")
    ap.add_argument("--video", required=True, type=Path, help="vidéo d'origine (ex. data/jobs/<job>/cut.mp4)")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, required=True)
    who = ap.add_mutually_exclusive_group(required=True)
    who.add_argument("--person", help="id d'une personne de la bibliothèque (data/people/<id>)")
    who.add_argument("--faces", type=Path, help="dossier de photos de la personne source")
    ap.add_argument("--level", default=FACE_TONE)
    ap.add_argument("--variant", action="append", default=[], help="nom[:clé=valeur,…] ; répétable")
    ap.add_argument("--compare", action="append", default=[], help="nom=vidéo déjà rendue du même passage ; répétable")
    ap.add_argument("--sheet", type=int, default=6, help="images sur la planche")
    ap.add_argument("--height", type=int, default=360, help="hauteur de chaque vignette de la vidéo côte à côte")
    ap.add_argument("--sans-mesures", action="store_true", help="seulement la vidéo côte à côte et la planche (rapide)")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "bench" / time.strftime("%Y%m%d-%H%M%S"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    clip = cut_range(args.video, args.start, args.end, args.out / "original.mp4")
    info = media.probe(clip)
    videos = {"original": clip}
    for spec in args.compare:
        name, _, path = spec.partition("=")
        videos[name] = cut_range(Path(path), args.start, args.end, args.out / f"{name}.mp4")

    report: dict[str, dict] = {}
    person = load_person(args)
    base = copy.deepcopy(dict(load_config()))
    for name, overrides in [parse_variant(v) for v in args.variant] or [("actuel", {})]:
        apply_overrides(base, overrides)
        out = args.out / f"{name}.mp4"
        t0 = time.perf_counter()
        stats = swap_segment(clip, [FaceMapping(person)], args.level, out, stats=RenderStats())
        elapsed = time.perf_counter() - t0
        videos[name] = out
        report[name] = {"reglages": overrides, "s_par_image": round(elapsed / max(1, stats.frames), 3),
                        "images_remplacees": stats.swapped, "images": stats.frames}
        print(f"{name} : {stats.swapped}/{stats.frames} images, {elapsed:.0f} s")
    apply_overrides(base, {})

    frames = {name: read_frames(path) for name, path in videos.items()}
    boxes = face_boxes(frames["original"])
    for name, seq in frames.items():
        if args.sans_mesures:
            break
        report.setdefault(name, {}).update(skin_measures(seq, face_boxes(seq) if name != "original" else boxes))
        print(f"mesuré : {name}")

    side_by_side(frames, args.out / "cote_a_cote.mp4", info.fps_str, args.height)
    picks = contact_sheet(frames, boxes, args.out / "planche.jpg", args.sheet)
    (args.out / "mesures.json").write_text(json.dumps({"images_planche": picks, "videos": report}, ensure_ascii=False,
                                                      indent=2), encoding="utf-8")

    cols = ["saut_teint_moyen", "saut_teint_p95", "gris_bleu_pct", "gris_bleu_pire_image_pct", "sans_visage", "s_par_image"]
    print("\n" + f"{'vidéo':<22}" + "".join(f"{c:>26}" for c in cols))
    for name, r in report.items():
        print(f"{name:<22}" + "".join(f"{str(r.get(c, '')):>26}" for c in cols))
    print(f"\nRésultats : {args.out}")


if __name__ == "__main__":
    main()
