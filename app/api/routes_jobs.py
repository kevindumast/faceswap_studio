"""Jobs de rendu : création (validée), suivi, annulation, résultat."""
from __future__ import annotations

import re
import shutil
from fractions import Fraction
import time
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app import db
from src import media
from src.config import load_config

from src.levels import CHARACTER, FACE, FACE_TONE, models_ready

from .common import get_or_404, not_found
from .routes_faces import person_ids

# Niveaux livrés à ce stade (phase A) ; le niveau 3 et le niveau 4 arrivent dans les phases suivantes.
AVAILABLE_LEVELS = (FACE, FACE_TONE)


def output_fps(info: dict, limit: bool) -> float:
    """Cadence du rendu : celle de la vidéo, ou plafonnée (une image sur N) si l'option est cochée."""
    fps_str = info.get("fps_str") or str(info["fps"])
    if limit:
        fps_str = media.capped_fps(fps_str, float(load_config().render.get("fps_cap", 30)))
    return float(Fraction(fps_str))


def gpu_configured() -> bool:
    """ZeroGPU branché (Space + clé) ? Sinon l'option GPU est refusée proprement."""
    from app.worker.zerogpu_client import configured

    return configured()

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


class Target(BaseModel):
    t: float
    box: list[float] = Field(min_length=4, max_length=4)


class Mapping(Target):
    """Un visage du clip (repéré à l'instant t par sa boîte normalisée) → une personne du set source."""
    person: str


class JobIn(BaseModel):
    video_id: str
    face_set_id: str
    start: float
    end: float
    output: Literal["segment", "full"] = "segment"
    stabilize: bool = True
    ai_label: bool = True
    mappings: list[Mapping] = Field(default_factory=list, max_length=8)
    target: Target | None = None  # ancien format : un seul visage, toutes les photos
    level: Literal["face", "face_tone", "head", "character"] = "face"
    use_gpu: bool = False         # jamais implicite : CPU sauf case cochée par l'utilisateur
    resolution: Literal["360p", "480p"] = "360p"  # niveau 4 uniquement
    limit_fps: bool = False       # vidéos > render.fps_cap i/s : une image sur N (60 → 30)
    consent: bool = False


def check_level(level: str, use_gpu: bool, active_mappings: int) -> None:
    """Refuse ce qui ne peut pas tourner, sans jamais basculer d'un moteur à l'autre à la place de l'utilisateur."""
    if level == CHARACTER and not use_gpu:
        raise HTTPException(422, "Le niveau 4 (personne entière) nécessite l'option GPU.")
    if use_gpu and not gpu_configured():
        raise HTTPException(422, "Aucun GPU ZeroGPU n'est branché : décoche l'option GPU ou configure-le dans « Moteur ».")
    if level not in AVAILABLE_LEVELS:
        raise HTTPException(422, "Ce niveau n'est pas encore disponible.")
    if level == CHARACTER and active_mappings != 1:
        raise HTTPException(422, "Le niveau 4 remplace une seule personne à la fois.")
    if not use_gpu and not models_ready(level):
        raise HTTPException(422, "Les modèles de ce niveau ne sont pas installés (bouton « Installer » à l'étape Visages).")


def public(job: dict) -> dict:
    base = f"/api/jobs/{job['id']}"
    done = job["status"] == "done"
    params = job["params"]
    elapsed = None
    if job["started_at"]:
        elapsed = (job["finished_at"] or time.time()) - job["started_at"]
    eta = None
    if job["status"] == "running" and job["stage"] == "swap" and job["done"] and elapsed:
        rate = elapsed / job["done"]
        eta = rate * max(job["total"] - job["done"], 0) + 5
    params.setdefault("mappings", [])
    params.setdefault("level", FACE)
    params.setdefault("use_gpu", False)
    video = db.get("videos", job["video_id"])
    if params["output"] == "full" and video:
        before = f"/api/videos/{job['video_id']}/proxy.mp4"
    else:
        before = f"{base}/before.mp4"
    return {
        "id": job["id"],
        "video_id": job["video_id"],
        "video_title": video["title"] if video else None,
        "face_set_id": job["face_set_id"],
        "params": params,
        "status": job["status"],
        "stage": job["stage"],
        "done": job["done"],
        "total": job["total"],
        "elapsed": elapsed,
        "eta": eta,
        "sec_per_frame": job["sec_per_frame"],
        "queue_ahead": db.jobs_ahead(job["created_at"]) if job["status"] == "queued" else 0,
        "pausable": not params.get("use_gpu"),     # ZeroGPU calcule l'extrait d'un coup : rien à reprendre
        "error": job["error"],
        "warnings": job["warnings"] or [],
        "preview_url": f"{base}/preview.jpg",
        "result_url": f"{base}/result.mp4" if done else None,
        "before_url": before if done else None,
        "created_at": job["created_at"],
    }


@router.get("")
def list_jobs() -> list[dict]:
    return [public(j) for j in db.all_rows("jobs", 30)]


@router.post("")
def create_job(body: JobIn) -> dict:
    cfg = load_config()
    if not body.consent:
        raise HTTPException(422, "Confirmez que le visage source est le vôtre ou celui d'une personne consentante.")
    video = get_or_404("videos", body.video_id, "Vidéo")
    if video["status"] != "ready":
        raise HTTPException(409, "La vidéo n'est pas prête.")
    if not body.face_set_id.isalnum() or not person_ids(body.face_set_id):
        raise HTTPException(422, "Aucune personne source pour cette vidéo : ajoute des photos ou choisis quelqu'un dans ta bibliothèque.")
    length = body.end - body.start
    seg = cfg.segment
    if body.start < 0 or body.end > video["info"]["duration"] + 0.05:
        raise HTTPException(422, "Le passage dépasse la durée de la vidéo.")
    if length < seg.min_s - 0.01 or length > seg.max_s + 0.01:
        raise HTTPException(422, f"Le passage doit durer entre {seg.min_s} et {seg.max_s} s.")
    known = person_ids(body.face_set_id)
    unknown = {m.person for m in body.mappings} - known
    if unknown:
        raise HTTPException(422, f"Personne(s) inconnue(s) : {', '.join(sorted(unknown))}.")
    check_level(body.level, body.use_gpu, len(body.mappings))

    job_id = db.new_id()
    params = body.model_dump(exclude={"video_id", "face_set_id", "consent"})
    db.insert("jobs", id=job_id, video_id=body.video_id, face_set_id=body.face_set_id, params=params,
              status="queued", total=round(length * output_fps(video["info"], body.limit_fps)), created_at=time.time())
    return public(db.get("jobs", job_id))


@router.get("/{job_id}")
def get_job(job_id: str) -> dict:
    return public(get_or_404("jobs", job_id, "Rendu"))


@router.post("/{job_id}/pause")
def pause_job(job_id: str) -> dict:
    """Met un rendu local en pause : le morceau en cours est refermé, la reprise repartira de la même image."""
    job = get_or_404("jobs", job_id, "Rendu")
    if job["params"].get("use_gpu"):
        raise HTTPException(409, "Pause impossible sur ZeroGPU : le Space calcule tout l'extrait d'un coup.")
    if job["status"] == "running":
        db.update("jobs", job_id, status="pausing")      # le worker s'arrête proprement à l'image suivante
    elif job["status"] == "queued":
        db.update("jobs", job_id, status="paused")
    elif job["status"] not in ("pausing", "paused"):
        raise HTTPException(409, "Ce rendu ne peut pas être mis en pause.")
    return public(db.get("jobs", job_id))


@router.post("/{job_id}/resume")
def resume_job(job_id: str) -> dict:
    """Remet un rendu en pause dans la file : il reprendra à l'image où il s'était arrêté."""
    job = get_or_404("jobs", job_id, "Rendu")
    if job["status"] != "paused":
        raise HTTPException(409, "Ce rendu n'est pas en pause.")
    db.update("jobs", job_id, status="queued", error=None)
    return public(db.get("jobs", job_id))


@router.post("/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    job = get_or_404("jobs", job_id, "Rendu")
    if job["status"] in ("queued", "paused"):
        db.update("jobs", job_id, status="cancelled", finished_at=time.time())
    elif job["status"] in ("running", "pausing"):
        db.update("jobs", job_id, status="cancelling")  # le worker s'arrête à la frame suivante
    return public(db.get("jobs", job_id))


@router.delete("/{job_id}")
def delete_job(job_id: str) -> dict:
    job = get_or_404("jobs", job_id, "Rendu")
    if job["status"] in ("running", "cancelling", "pausing"):
        raise HTTPException(409, "Annulez ou mettez le rendu en pause avant de le supprimer.")
    db.delete("jobs", job_id)
    shutil.rmtree(db.folder("jobs", job_id), ignore_errors=True)
    return {"ok": True}


def _file(job_id: str, name: str, media_type: str, download_name: str | None = None) -> FileResponse:
    get_or_404("jobs", job_id, "Rendu")
    path = db.folder("jobs", job_id) / name
    if not path.is_file():
        raise not_found("Fichier")
    headers = {"Cache-Control": "no-store"} if name == "preview.jpg" else None
    return FileResponse(path, media_type=media_type, filename=download_name, headers=headers,
                        content_disposition_type="attachment" if download_name else "inline")


@router.get("/{job_id}/preview.jpg")
def preview(job_id: str) -> FileResponse:
    return _file(job_id, "preview.jpg", "image/jpeg")


@router.get("/{job_id}/before.mp4")
def before(job_id: str) -> FileResponse:
    return _file(job_id, "cut.mp4", "video/mp4")


@router.get("/{job_id}/result.mp4")
def result(job_id: str, download: bool = False) -> FileResponse:
    name = None
    if download:
        job = db.get("jobs", job_id)
        video = db.get("videos", job["video_id"]) if job else None
        title = re.sub(r"[^\w\- ]+", "", (video or {}).get("title") or "faceswap").strip()[:60] or "faceswap"
        name = f"{title} - swap.mp4"
    return _file(job_id, "result.mp4", "video/mp4", name)
