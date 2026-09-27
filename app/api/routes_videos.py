"""Vidéos : import de fichier, récupération par URL, préparation (proxy + filmstrip), visages d'une frame."""
from __future__ import annotations

import json
import shutil
import threading
import time
from collections import OrderedDict
from contextlib import suppress

from pathlib import Path

import cv2
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app import db
from src import fetch, media
from src.config import load_config
from src.faces import ModelsMissing, detect, face_crop, iou
from src.scan import Person, face_in_region, scan_passage

from .common import background, get_or_404, jpeg_data_url, not_found, save_upload

router = APIRouter(prefix="/api/videos", tags=["videos"])


class UrlIn(BaseModel):
    url: str


class RegionIn(BaseModel):
    """Zone tracée à la main sur une image analysée du passage [start, end]."""
    start: float
    end: float
    t: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 en fractions 0-1 de l'image


def public(video: dict) -> dict:
    base = f"/api/videos/{video['id']}"
    ready = video["status"] == "ready"
    return {
        "id": video["id"],
        "kind": video["kind"],
        "title": video["title"],
        "url": video["url"],
        "status": video["status"],
        "progress": video["progress"],
        "error": video["error"],
        "info": video["info"],
        "filmstrip": video["filmstrip"],
        "proxy_url": f"{base}/proxy.mp4" if ready else None,
        "filmstrip_url": f"{base}/filmstrip.jpg" if ready else None,
        "poster_url": f"{base}/poster.jpg" if ready else None,
        "created_at": video["created_at"],
    }


def prepare(video_id: str, source: Path) -> None:
    """Probe → proxy H.264 → filmstrip → poster. Tourne dans le pool `background`."""
    cfg = load_config()
    out = db.folder("videos", video_id)
    try:
        db.update("videos", video_id, status="preparing", progress=0, source=str(source))
        info = media.probe(source)
        if info.duration < cfg.segment.min_s:
            raise media.MediaError(f"Vidéo trop courte : il faut au moins {cfg.segment.min_s} s.")
        db.update("videos", video_id, info=info.to_dict())

        last = [0.0]

        def on_progress(frac: float) -> None:
            if frac - last[0] >= 0.02:
                last[0] = frac
                db.update("videos", video_id, progress=round(frac * 0.9, 3))

        media.make_proxy(source, out / "proxy.mp4", info, int(cfg.proxy.max_height), on_progress)
        film = media.make_filmstrip(out / "proxy.mp4", out / "filmstrip.jpg", out / "filmstrip.json", info,
                                    int(cfg.proxy.filmstrip_count), int(cfg.proxy.filmstrip_height))
        media.make_poster(out / "proxy.mp4", out / "poster.jpg", t=min(1.0, info.duration / 2))
        db.update("videos", video_id, status="ready", progress=1, filmstrip=film)
    except Exception as exc:  # toute erreur remonte à l'UI
        db.update("videos", video_id, status="error", error=str(exc))


def download_then_prepare(video_id: str, url: str) -> None:
    cfg = load_config()
    out = db.folder("videos", video_id)
    try:
        last = [0.0]

        def on_progress(frac: float) -> None:
            if frac - last[0] >= 0.01:
                last[0] = frac
                db.update("videos", video_id, progress=round(frac, 3))

        source = fetch.download(url, out, int(cfg.youtube.max_height), on_progress)
    except Exception as exc:
        db.update("videos", video_id, status="error", error=str(exc))
        return
    prepare(video_id, source)


@router.get("")
def list_videos() -> list[dict]:
    return [public(v) for v in db.all_rows("videos", 20)]


@router.post("/upload")
async def upload_video(file: UploadFile = File(...)) -> dict:
    cfg = load_config()
    ext = Path(file.filename or "").suffix.lower()
    if ext not in cfg.upload.video_ext:
        raise HTTPException(415, f"Format non supporté ({ext or '?'}). Formats : {', '.join(cfg.upload.video_ext)}.")
    video_id = db.new_id()
    source = db.folder("videos", video_id) / f"source{ext}"
    await save_upload(file, source, float(cfg.upload.max_mb))
    db.insert("videos", id=video_id, kind="upload", title=Path(file.filename).stem, status="preparing",
              progress=0, source=str(source), created_at=time.time())
    background.submit(prepare, video_id, source)
    return public(db.get("videos", video_id))


@router.post("/url/info")
def url_info(body: UrlIn) -> dict:
    """Aperçu rapide (titre, miniature, durée) avant de lancer le téléchargement."""
    cfg = load_config()
    try:
        info = fetch.video_info(body.url)
    except fetch.FetchError as exc:
        raise HTTPException(422, str(exc)) from exc
    info["too_long"] = info["duration"] > cfg.youtube.max_duration_s
    info["max_duration_s"] = cfg.youtube.max_duration_s
    return info


@router.post("/url")
def from_url(body: UrlIn) -> dict:
    cfg = load_config()
    try:
        info = fetch.video_info(body.url)
    except fetch.FetchError as exc:
        raise HTTPException(422, str(exc)) from exc
    if info["duration"] > cfg.youtube.max_duration_s:
        raise HTTPException(422, f"Vidéo trop longue (max {cfg.youtube.max_duration_s // 60} min).")
    video_id = db.new_id()
    db.folder("videos", video_id).mkdir(parents=True, exist_ok=True)
    db.insert("videos", id=video_id, kind="url", title=info["title"], url=info["webpage_url"],
              status="downloading", progress=0, created_at=time.time())
    background.submit(download_then_prepare, video_id, info["webpage_url"])
    return public(db.get("videos", video_id))


@router.get("/{video_id}")
def get_video(video_id: str) -> dict:
    return public(get_or_404("videos", video_id, "Vidéo"))


@router.delete("/{video_id}")
def delete_video(video_id: str) -> dict:
    get_or_404("videos", video_id, "Vidéo")
    db.delete("videos", video_id)
    shutil.rmtree(db.folder("videos", video_id), ignore_errors=True)
    return {"ok": True}


def _asset(video_id: str, name: str, media_type: str) -> FileResponse:
    get_or_404("videos", video_id, "Vidéo")
    path = db.folder("videos", video_id) / name
    if not path.is_file():
        raise not_found("Fichier")
    return FileResponse(path, media_type=media_type)


@router.get("/{video_id}/proxy.mp4")
def proxy(video_id: str) -> FileResponse:
    return _asset(video_id, "proxy.mp4", "video/mp4")


@router.get("/{video_id}/filmstrip.jpg")
def filmstrip(video_id: str) -> FileResponse:
    return _asset(video_id, "filmstrip.jpg", "image/jpeg")


@router.get("/{video_id}/poster.jpg")
def poster(video_id: str) -> FileResponse:
    return _asset(video_id, "poster.jpg", "image/jpeg")


@router.get("/{video_id}/faces")
def faces_at(video_id: str, t: float) -> dict:
    """Visages détectés à l'instant t, pour choisir celui à remplacer. Boîtes normalisées 0..1."""
    video = get_or_404("videos", video_id, "Vidéo")
    if video["status"] != "ready":
        raise HTTPException(409, "Vidéo pas encore prête.")
    t = min(max(t, 0.0), video["info"]["duration"] - 0.05)
    try:
        frame = media.extract_frame(db.folder("videos", video_id) / "proxy.mp4", t)
        faces = detect(frame)
    except ModelsMissing as exc:
        raise HTTPException(503, str(exc)) from exc
    except media.MediaError as exc:
        raise HTTPException(422, str(exc)) from exc
    h, w = frame.shape[:2]
    return {
        "t": t,
        "width": w,
        "height": h,
        "frame": jpeg_data_url(frame, 82),
        "faces": [
            {
                "box": [float(f.bbox[0] / w), float(f.bbox[1] / h), float(f.bbox[2] / w), float(f.bbox[3] / h)],
                "score": float(f.det_score),
                "crop": jpeg_data_url(face_crop(frame, f.bbox, 112)),
            }
            for f in faces
        ],
    }


# Résultats de recherche par passage : l'analyse prend quelques secondes, on la garde en mémoire.
_scan_cache: OrderedDict[tuple, dict] = OrderedDict()
_scan_lock = threading.Lock()


def _scan_source(video: dict) -> Path:
    return Path(video["source"]) if video.get("source") and Path(video["source"]).is_file() \
        else db.folder("videos", video["id"]) / "proxy.mp4"


def _preview(img) -> str:
    scale = min(1.0, 720 / img.shape[0])
    return jpeg_data_url(cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), 82)


def _scan_face(p: Person) -> dict:
    return {"t": p.t, "box": p.box, "score": p.score, "seen": len(p.times), "height_px": p.height_px,
            "maybe_same": p.maybe_same, "seen_at": [{"t": t, "box": box} for t, box in p.seen_at], "crop": jpeg_data_url(p.crop)}


# Visages ajoutés à la main (étape Visages), gardés avec la vidéo : ils survivent à un redémarrage de l'API.
def _manual_path(video_id: str) -> Path:
    return db.folder("videos", video_id) / "manual_faces.json"


def _load_manual(video_id: str) -> list[dict]:
    path = _manual_path(video_id)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []


def _save_manual(video_id: str, faces: list[dict]) -> None:
    path = _manual_path(video_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(faces), encoding="utf-8")


@router.get("/{video_id}/scan")
def scan(video_id: str, start: float, end: float) -> dict:
    """Personnes présentes dans le passage [start, end] : une entrée par personne, avec sa meilleure apparition.

    Les visages ajoutés à la main sur ce passage viennent en fin de liste.
    """
    video = get_or_404("videos", video_id, "Vidéo")
    if video["status"] != "ready":
        raise HTTPException(409, "Vidéo pas encore prête.")
    duration = video["info"]["duration"]
    start, end = max(0.0, start), min(end, duration)
    if end - start < 0.5:
        raise HTTPException(422, "Passage trop court.")
    key = (video_id, round(start, 2), round(end, 2))
    with _scan_lock:
        result = _scan_cache.get(key)
        if result is not None:
            _scan_cache.move_to_end(key)
    if result is None:
        try:
            people, frames = scan_passage(_scan_source(video), start, end)
        except ModelsMissing as exc:
            raise HTTPException(503, str(exc)) from exc
        result = {
            "start": start,
            "end": end,
            "samples": int(load_config().get("scan", {}).get("samples", 12)),
            "faces": [_scan_face(p) for p in people],
            "frames": {f"{t:.3f}": _preview(img) for t, img in frames.items()},
            "width": video["info"]["width"],
            "height": video["info"]["height"],
        }
        with _scan_lock:
            _scan_cache[key] = result
            while len(_scan_cache) > 12:
                _scan_cache.popitem(last=False)
    manual = [f for f in _load_manual(video_id) if start <= f["t"] <= end]
    if not manual:
        return result
    frames = dict(result["frames"])
    for f in manual:   # passage modifié depuis l'ajout : l'image où le visage a été indiqué n'est plus analysée
        k = f"{f['t']:.3f}"
        if k not in frames:
            with suppress(media.MediaError):
                frames[k] = _preview(media.extract_frame(_scan_source(video), f["t"], int(load_config().render.max_height)))
    return {**result, "faces": result["faces"] + manual, "frames": frames}


@router.post("/{video_id}/scan/faces")
def add_scan_face(video_id: str, body: RegionIn) -> dict:
    """Ajoute le visage d'une zone tracée à la main. S'il est déjà dans la liste, renvoie seulement sa place."""
    video = get_or_404("videos", video_id, "Vidéo")
    faces = scan(video_id, body.start, body.end)["faces"]
    try:
        person = face_in_region(_scan_source(video), body.t, body.box)
    except ModelsMissing as exc:
        raise HTTPException(503, str(exc)) from exc
    except media.MediaError as exc:
        raise HTTPException(422, str(exc)) from exc
    if person is None:
        raise HTTPException(422, "Aucun visage trouvé dans ce cadre, même en zoomant. De dos ou de profil très marqué, "
                                 "il ne pourra pas être remplacé : le rendu doit retrouver le visage sur chaque image.")
    for i, f in enumerate(faces):
        if any(abs(a["t"] - body.t) < 1e-3 and iou(a["box"], person.box) > 0.4 for a in f["seen_at"]):
            return {"index": i, "face": None}
    face = {**_scan_face(person), "id": db.new_id(), "manual": True}
    with _scan_lock:
        _save_manual(video_id, _load_manual(video_id) + [face])
    return {"index": len(faces), "face": face}


@router.delete("/{video_id}/scan/faces/{face_id}")
def delete_scan_face(video_id: str, face_id: str) -> dict:
    get_or_404("videos", video_id, "Vidéo")
    with _scan_lock:
        _save_manual(video_id, [f for f in _load_manual(video_id) if f["id"] != face_id])
    return {"ok": True}
