"""Jobs de rendu : création (validée), suivi, annulation, résultat."""
from __future__ import annotations

import re
import shutil
import time
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app import db
from src.config import load_config

from .common import get_or_404, not_found
from .routes_faces import person_ids

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
    consent: bool = False


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
    face_dir = db.folder("faces", body.face_set_id) if body.face_set_id.isalnum() else None
    if face_dir is None or not any(face_dir.glob("*.npy")):
        raise HTTPException(422, "Aucune photo source avec un visage détecté.")
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

    job_id = db.new_id()
    params = body.model_dump(exclude={"video_id", "face_set_id", "consent"})
    db.insert("jobs", id=job_id, video_id=body.video_id, face_set_id=body.face_set_id, params=params,
              status="queued", total=round(length * video["info"]["fps"]), created_at=time.time())
    return public(db.get("jobs", job_id))


@router.get("/{job_id}")
def get_job(job_id: str) -> dict:
    return public(get_or_404("jobs", job_id, "Rendu"))


@router.post("/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    job = get_or_404("jobs", job_id, "Rendu")
    if job["status"] == "queued":
        db.update("jobs", job_id, status="cancelled", finished_at=time.time())
    elif job["status"] == "running":
        db.update("jobs", job_id, status="cancelling")  # le worker s'arrête à la frame suivante
    return public(db.get("jobs", job_id))


@router.delete("/{job_id}")
def delete_job(job_id: str) -> dict:
    job = get_or_404("jobs", job_id, "Rendu")
    if job["status"] in ("running", "cancelling"):
        raise HTTPException(409, "Annulez le rendu avant de le supprimer.")
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
