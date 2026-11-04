"""Montages : plusieurs rendus terminés enchaînés en une vidéo à partager (vertical, effets, musique). Voir src/montage.py."""
from __future__ import annotations

import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import cv2
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app import db
from src import montage

from .common import get_or_404, jpeg_data_url, not_found, save_upload

router = APIRouter(prefix="/api/montages", tags=["montages"])

# Un montage à la fois : il occupe le processeur (masques qui suivent les visages, encodage 1080×1920).
runner = ThreadPoolExecutor(max_workers=1, thread_name_prefix="montage")
_preview_lock = threading.Lock()
MUSIC_EXT = (".mp3", ".m4a", ".aac", ".wav", ".ogg", ".flac")
MAX_CLIPS = 40


class ClipIn(BaseModel):
    job_id: str
    caption: str = Field(default="", max_length=120)


class MontageIn(BaseModel):
    clips: list[ClipIn] = Field(min_length=1, max_length=MAX_CLIPS)
    format: Literal["9:16", "1:1", "16:9"] = "9:16"
    fit: Literal["blur", "crop"] = "blur"
    title: str = Field(default="", max_length=80)
    subtitle: str = Field(default="", max_length=120)
    outro: str = Field(default="", max_length=80)
    grade: Literal["none", "warm", "cinema", "vintage", "bw"] = "none"
    snow: bool = False
    vignette: bool = False
    zoom: bool = False
    mask: Literal["none", "santa", "reindeer", "nose"] = "none"
    transition: Literal["fade", "flash", "cut"] = "fade"
    music_id: str | None = None
    music_volume: float = Field(default=0.35, ge=0, le=1.5)
    clip_audio: bool = True
    ai_label: bool = True


def _music_dir() -> Path:
    return db.data_dir() / "music"


def _music_path(music_id: str) -> Path:
    if not music_id.isalnum():
        raise not_found("Musique")
    found = [p for p in _music_dir().glob(f"{music_id}.*") if p.suffix.lower() in MUSIC_EXT]
    if not found:
        raise not_found("Musique")
    return found[0]


def _clip_path(job_id: str) -> Path:
    job = get_or_404("jobs", job_id, "Rendu")
    path = db.folder("jobs", job_id) / "result.mp4"
    if job["status"] not in ("done", "review") or not path.is_file():
        raise HTTPException(422, "Seuls les rendus terminés peuvent entrer dans un montage.")
    return path


def _options(body: MontageIn) -> tuple[list[montage.Clip], montage.MontageOptions]:
    clips = [montage.Clip(_clip_path(c.job_id), c.caption.strip()) for c in body.clips]
    opts = montage.MontageOptions(
        format=body.format, fit=body.fit, title=body.title.strip(), subtitle=body.subtitle.strip(),
        outro=body.outro.strip(), grade=body.grade, snow=body.snow, vignette=body.vignette, zoom=body.zoom,
        mask=body.mask, transition=body.transition, music=_music_path(body.music_id) if body.music_id else None,
        music_volume=body.music_volume, clip_audio=body.clip_audio, ai_label=body.ai_label)
    return clips, opts


def public(row: dict) -> dict:
    base = f"/api/montages/{row['id']}"
    done = row["status"] == "done"
    return {
        "id": row["id"],
        "title": row["title"],
        "status": row["status"],
        "progress": row["progress"] or 0,
        "error": row["error"],
        "info": row["info"],
        "params": row["params"],
        "created_at": row["created_at"],
        "finished_at": row["finished_at"],
        "result_url": f"{base}/result.mp4" if done else None,
        "download_url": f"{base}/result.mp4?download=1" if done else None,
    }


class _Cancelled(Exception):
    pass


def _run(montage_id: str) -> None:
    row = db.get("montages", montage_id)
    if row is None or row["status"] != "queued":
        return
    db.update("montages", montage_id, status="running", progress=0)
    last = [0.0]

    def progress(done: int, total: int) -> None:
        if time.time() - last[0] > 0.7 or done >= total:
            last[0] = time.time()
            if db.get("montages", montage_id)["status"] == "cancelling":
                raise _Cancelled()
            db.update("montages", montage_id, progress=round(done / max(1, total), 4))

    try:
        clips, opts = _options(MontageIn(**row["params"]))
        out = db.folder("montages", montage_id)
        out.mkdir(parents=True, exist_ok=True)
        info = montage.render(clips, opts, out / "result.mp4", progress)
        db.update("montages", montage_id, status="done", progress=1, info=info, finished_at=time.time())
    except _Cancelled:
        db.update("montages", montage_id, status="cancelled", finished_at=time.time())
    except HTTPException as exc:
        db.update("montages", montage_id, status="error", error=str(exc.detail), finished_at=time.time())
    except Exception as exc:   # message lisible dans l'appli plutôt qu'un montage bloqué « en cours »
        db.update("montages", montage_id, status="error", error=str(exc)[:500] or type(exc).__name__,
                  finished_at=time.time())


def resume_interrupted() -> None:
    """Au démarrage de l'API : un montage coupé par un redémarrage est marqué, ceux en attente repartent."""
    for row in db.all_rows("montages", 200):
        if row["status"] in ("running", "cancelling"):
            db.update("montages", row["id"], status="error", error="Interrompu par un redémarrage du serveur : relance-le.")
        elif row["status"] == "queued":
            runner.submit(_run, row["id"])


@router.get("")
def list_montages() -> list[dict]:
    return [public(r) for r in db.all_rows("montages", 50)]


@router.post("")
def create_montage(body: MontageIn) -> dict:
    _options(body)                     # refuse tout de suite un rendu manquant ou une musique inconnue
    montage_id = db.new_id()
    title = body.title.strip() or f"Montage de {len(body.clips)} séquence{'s' if len(body.clips) > 1 else ''}"
    db.insert("montages", id=montage_id, title=title, params=body.model_dump(), status="queued", progress=0,
              created_at=time.time())
    runner.submit(_run, montage_id)
    return public(db.get("montages", montage_id))


@router.post("/preview")
def preview(body: MontageIn) -> dict:
    """Une image du montage avec ces réglages (milieu de la 1re séquence), sans rien calculer d'autre."""
    clips, opts = _options(body)
    try:
        with _preview_lock:
            img = montage.preview(clips[:1], opts)
    except montage.MontageError as exc:
        raise HTTPException(422, str(exc)) from exc
    h, w = img.shape[:2]
    k = 960 / max(h, w)
    return {"image": jpeg_data_url(cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA), 85)}


@router.post("/music")
async def upload_music(file: UploadFile = File(...)) -> dict:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in MUSIC_EXT:
        raise HTTPException(415, f"Format de musique non pris en charge ({', '.join(MUSIC_EXT)}).")
    music_id = db.new_id()
    _music_dir().mkdir(parents=True, exist_ok=True)
    await save_upload(file, _music_dir() / f"{music_id}{ext}", 50)
    return {"id": music_id, "name": Path(file.filename).stem}


@router.get("/{montage_id}")
def get_montage(montage_id: str) -> dict:
    return public(get_or_404("montages", montage_id, "Montage"))


@router.post("/{montage_id}/cancel")
def cancel_montage(montage_id: str) -> dict:
    row = get_or_404("montages", montage_id, "Montage")
    if row["status"] == "queued":
        db.update("montages", montage_id, status="cancelled", finished_at=time.time())
    elif row["status"] == "running":
        db.update("montages", montage_id, status="cancelling")
    return public(db.get("montages", montage_id))


@router.delete("/{montage_id}")
def delete_montage(montage_id: str) -> dict:
    row = get_or_404("montages", montage_id, "Montage")
    if row["status"] in ("running", "cancelling"):
        raise HTTPException(409, "Annule le montage avant de le supprimer.")
    db.delete("montages", montage_id)
    shutil.rmtree(db.folder("montages", montage_id), ignore_errors=True)
    return {"ok": True}


@router.get("/{montage_id}/result.mp4")
def result(montage_id: str, download: bool = False) -> FileResponse:
    row = get_or_404("montages", montage_id, "Montage")
    path = db.folder("montages", montage_id) / "result.mp4"
    if not path.is_file():
        raise not_found("Fichier")

    name = None
    if download:
        name = (re.sub(r"[^\w\- ]+", "", row["title"] or "montage").strip()[:60] or "montage") + ".mp4"
    return FileResponse(path, media_type="video/mp4", filename=name,
                        content_disposition_type="attachment" if download else "inline")
