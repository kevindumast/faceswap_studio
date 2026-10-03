"""Jobs de rendu : création (validée), suivi, annulation, résultat."""
from __future__ import annotations

import re
import shutil
import threading
from fractions import Fraction
import time
from typing import Literal

import cv2
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app import db
from src import media, models, review
from src.config import load_config
from src.faces import face_crop

from src.levels import CHARACTER, FACE, FACE_TONE, HEAD, models_ready
from src.pipeline import Checkpoint, partial_preview

from .common import get_or_404, not_found
from .routes_faces import freeze, person_ids

AVAILABLE_LEVELS = (FACE, FACE_TONE, HEAD, CHARACTER)


def output_fps(info: dict, limit: bool) -> float:
    """Cadence du rendu : celle de la vidéo, ou plafonnée (une image sur N) si l'option est cochée."""
    fps_str = info.get("fps_str") or str(info["fps"])
    if limit:
        fps_str = media.capped_fps(fps_str, float(load_config().render.get("fps_cap", 30)))
    return float(Fraction(fps_str))


def gpu_configured(level: str = FACE) -> bool:
    """ZeroGPU branché (Space + clé) pour ce niveau ? Sinon l'option GPU est refusée proprement."""
    from app.worker.zerogpu_client import character_configured, configured

    return character_configured() if level == CHARACTER else configured()

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


class Target(BaseModel):
    t: float
    box: list[float] = Field(min_length=4, max_length=4)


class Mapping(Target):
    """Un visage du clip (repéré à l'instant t par sa boîte normalisée) → une personne du set source."""
    person: str


# Garde-fou : une même personne peut remplacer plusieurs visages (acteur vu sous plusieurs angles).
MAX_MAPPINGS = 24


class JobIn(BaseModel):
    video_id: str
    face_set_id: str
    start: float
    end: float
    output: Literal["segment", "full"] = "segment"
    stabilize: bool = True
    ai_label: bool = True
    mappings: list[Mapping] = Field(default_factory=list)
    target: Target | None = None  # ancien format : un seul visage, toutes les photos
    level: Literal["face", "face_tone", "head", "character"] = "face"
    use_gpu: bool = False         # jamais implicite : CPU sauf case cochée par l'utilisateur
    resolution: Literal["360p", "480p"] = "360p"  # niveau 4 uniquement
    steps: int | None = Field(default=None, ge=2, le=20)  # niveau 4 : étapes de génération (4 = modèle distillé)
    relight: bool = False         # niveau 4 : la personne prend la lumière du décor (sinon : les couleurs de sa photo)
    face_pass: bool = True        # niveau 4 : visage refait net sur ce PC après la génération
    limit_fps: bool = False       # vidéos > render.fps_cap i/s : une image sur N (60 → 30)
    restore: bool = False         # option « netteté » (niveaux 1 et 2 uniquement) : voir src/restore.py
    review: bool = False          # rendu sur ce PC : arrêt avant l'assemblage si des visages sont mal suivis
    consent: bool = False


def check_level(level: str, use_gpu: bool, active_mappings: int, length: float = 0.0, restore: bool = False) -> None:
    """Refuse ce qui ne peut pas tourner, sans jamais basculer d'un moteur à l'autre à la place de l'utilisateur."""
    if restore and level not in (FACE, FACE_TONE):
        raise HTTPException(422, "L'option « netteté » ne s'applique qu'aux niveaux 1 et 2.")
    if restore and use_gpu:
        raise HTTPException(422, "L'option « netteté » tourne sur ce PC, pas encore sur ZeroGPU : décoche l'option GPU ou l'option « netteté ».")
    if restore and not models.is_ready("restore"):
        raise HTTPException(422, "Le modèle de l'option « netteté » n'est pas installé (bouton « Installer » à l'étape Rendu).")
    if level == CHARACTER and not use_gpu:
        raise HTTPException(422, "Le niveau 4 (personne entière) nécessite l'option GPU.")
    if level == CHARACTER and not gpu_configured(CHARACTER):
        raise HTTPException(422, "Le Space du niveau 4 n'est pas branché : Moteur → Niveau 4 (deploy_space.py --kind character).")
    if level == HEAD and use_gpu:
        raise HTTPException(422, "Le niveau 3 (tête complète) tourne sur ta carte graphique, pas sur ZeroGPU : décoche l'option GPU.")
    if use_gpu and level != CHARACTER and not gpu_configured():
        raise HTTPException(422, "Aucun GPU ZeroGPU n'est branché : décoche l'option GPU ou configure-le dans « Moteur ».")
    max_s = float(load_config().get("levels", {}).get(CHARACTER, {}).get("max_s", 10))
    if level == CHARACTER and length > max_s + 0.01:
        raise HTTPException(422, f"Le niveau 4 est limité à {max_s:g} s d'extrait (quota GPU) : raccourcis le passage.")
    if level not in AVAILABLE_LEVELS:
        raise HTTPException(422, "Ce niveau n'est pas encore disponible.")
    max_people = int(load_config().get("levels", {}).get(CHARACTER, {}).get("max_people", 2))
    if level == CHARACTER and not 1 <= active_mappings <= max_people:
        raise HTTPException(422, f"Le niveau 4 remplace 1 à {max_people} personnes par rendu (un passage GPU chacune).")
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
    if job["status"] == "running" and job["stage"] in ("swap", "fix") and job["done"] and elapsed:
        rate = elapsed / job["done"]
        eta = rate * max(job["total"] - job["done"], 0) + 5
    params.setdefault("mappings", [])
    params.setdefault("level", FACE)
    params.setdefault("use_gpu", False)
    video = db.get("videos", job["video_id"])
    # En pause : images déjà rendues (point de reprise), qu'on peut revoir avant de reprendre.
    partial = Checkpoint.load(db.folder("jobs", job["id"])).frames_done if job["status"] == "paused" else 0
    if done and params["output"] == "full" and video:
        before = f"/api/videos/{job['video_id']}/proxy.mp4"
    else:
        before = f"{base}/before.mp4"   # aperçu en pause : seul le passage est comparé, même en « vidéo complète »
    in_review = job["status"] == "review" or bool(params.get("review_step"))
    return {
        "id": job["id"],
        "video_id": job["video_id"],
        # Titre aussi gardé dans le rendu : il reste lisible si la vidéo source a été supprimée.
        "video_title": video["title"] if video else params.get("video_title"),
        "video_missing": video is None,
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
        # ZeroGPU calcule l'extrait d'un coup : rien à reprendre ; corrections et assemblage : courts, pas de pause.
        "pausable": not params.get("use_gpu") and not params.get("review_step"),
        "error": job["error"],
        "warnings": job["warnings"] or [],
        "preview_url": f"{base}/preview.jpg",
        "result_url": f"{base}/result.mp4" if done else None,
        "partial_url": f"{base}/partial.mp4?v={partial}" if partial else None,
        "before_url": before if done or partial or in_review else None,
        # Rendu terminé qui garde son journal et sa vidéo sans son : on peut encore le revoir et le corriger.
        "reviewable": done and _reviewable(job["id"]),
        "created_at": job["created_at"],
    }


def _reviewable(job_id: str) -> bool:
    d = db.folder("jobs", job_id)
    return all((d / name).is_file() for name in (review.PLAN, "swapped.mp4", "cut.mp4"))


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
    if len(body.mappings) > MAX_MAPPINGS:
        raise HTTPException(422, f"{len(body.mappings)} visages à remplacer : {MAX_MAPPINGS} au maximum par rendu. "
                                 "Passe les moins présents sur « Ne pas remplacer ».")
    known = person_ids(body.face_set_id)
    unknown = {m.person for m in body.mappings} - known
    if unknown:
        raise HTTPException(422, f"Personne(s) inconnue(s) : {', '.join(sorted(unknown))}.")
    check_level(body.level, body.use_gpu, len(body.mappings), length, body.restore)

    freeze(body.face_set_id)   # les personnes de ce rendu ne changeront plus (voir routes_faces)
    job_id = db.new_id()
    params = body.model_dump(exclude={"video_id", "face_set_id", "consent"}) | {"video_title": video["title"]}
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
    if job["params"].get("review_step"):
        raise HTTPException(409, "Pas de pause pendant les corrections ou l'assemblage : annule pour revenir à la vérification.")
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
    if job["status"] == "queued" and job["params"].get("review_step"):
        # Correction ou assemblage pas encore commencé : retour à la vérification, rien n'est perdu.
        params = {k: v for k, v in job["params"].items() if k != "review_step"}
        db.update("jobs", job_id, status="review", stage="review", params=params)
    elif job["status"] in ("queued", "paused", "review"):
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


_partial_lock = threading.Lock()     # deux requêtes du lecteur vidéo ne fabriquent pas l'aperçu en même temps


@router.get("/{job_id}/partial.mp4")
def partial(job_id: str) -> FileResponse:
    """Ce qui est déjà rendu d'un rendu en pause, pour juger le résultat avant de reprendre (ou d'annuler)."""
    job = get_or_404("jobs", job_id, "Rendu")
    if job["status"] != "paused":
        raise HTTPException(409, "Aperçu disponible seulement quand le rendu est en pause.")
    with _partial_lock:
        path = partial_preview(db.folder("jobs", job_id))
    if path is None:
        raise not_found("Aperçu")
    return FileResponse(path, media_type="video/mp4")


@router.get("/{job_id}/before.mp4")
def before(job_id: str) -> FileResponse:
    return _file(job_id, "cut.mp4", "video/mp4")


# ---------------------------------------------------------------------------------------------------------------
# Vérification avant l'assemblage (rendus sur ce PC) : voir src/review.py


class Decision(BaseModel):
    issue: int
    action: Literal["assign", "remove", "keep"]
    person: str | None = None


class ReviewIn(BaseModel):
    version: str
    decisions: list[Decision] = Field(default_factory=list, max_length=1000)


def _plan(job_id: str) -> review.FramePlan:
    plan = review.FramePlan.load(db.folder("jobs", job_id))
    if plan is None:
        raise HTTPException(409, "Ce rendu n'a pas de journal image par image : relance-le pour pouvoir le vérifier.")
    return plan


def _in_review(job: dict) -> None:
    if job["status"] != "review":
        raise HTTPException(409, "Ce rendu n'est pas en vérification (recharge la page).")


@router.get("/{job_id}/review")
def get_review(job_id: str) -> dict:
    """Visages mal suivis (pistes à corriger) et visages de chaque image, pour l'écran de vérification."""
    job = get_or_404("jobs", job_id, "Rendu")
    plan = _plan(job_id)
    tracks, issues = review.analyze(plan)
    base = f"/api/jobs/{job_id}"
    return {
        "version": plan.version,
        "fps": plan.fps,
        "frames": len(plan.frames),
        "width": plan.size[0],
        "height": plan.size[1],
        "mappings": plan.mappings,
        "issues": issues,
        "boxes": review.frame_boxes(plan, tracks),
        # Choix déjà enregistrés mais pas encore recalculés (recalcul interrompu) : à relancer.
        "pending": sum(1 for e in plan.frames if e and e.get("dirty")),
        "after_url": f"{base}/swapped.mp4?v={plan.version}",
        "before_url": f"{base}/before.mp4",
    }


@router.post("/{job_id}/review")
def post_review(job_id: str, body: ReviewIn) -> dict:
    """Enregistre les choix ; s'il y a des images à recalculer, le worker s'en charge (étape « fix »)."""
    job = get_or_404("jobs", job_id, "Rendu")
    _in_review(job)
    plan = _plan(job_id)
    if body.version != plan.version:
        raise HTTPException(409, "La vérification a changé entre-temps : recharge la page.")
    rcfg = load_config().render
    smoothing = float(rcfg.smoothing) if job["params"].get("stabilize", True) else 0.0
    try:
        review.apply_decisions(plan, [d.model_dump() for d in body.decisions], smoothing)
    except review.DecisionError as exc:
        raise HTTPException(422, str(exc)) from exc
    plan.save()
    if any(e and e.get("dirty") for e in plan.frames):
        db.update("jobs", job_id, status="queued", stage="fix", done=0, total=len(plan.frames), error=None,
                  params={**job["params"], "review_step": "fix"})
    return public(db.get("jobs", job_id))


@router.post("/{job_id}/assemble")
def assemble_reviewed(job_id: str) -> dict:
    """Vérification terminée : son, étiquette et (vidéo complète) réinsertion, comme un rendu sans vérification."""
    job = get_or_404("jobs", job_id, "Rendu")
    _in_review(job)
    if any(e and e.get("dirty") for e in _plan(job_id).frames):
        raise HTTPException(409, "Des corrections ne sont pas encore calculées : relance le recalcul avant d'assembler.")
    db.update("jobs", job_id, status="queued", stage="assemble", done=0, total=1, error=None,
              params={**job["params"], "review_step": "assemble"})
    return public(db.get("jobs", job_id))


@router.post("/{job_id}/reopen")
def reopen(job_id: str) -> dict:
    """Rendu terminé : retour à la vérification pour corriger des images, puis nouvel assemblage."""
    job = get_or_404("jobs", job_id, "Rendu")
    if job["status"] != "done":
        raise HTTPException(409, "Seul un rendu terminé peut être revu.")
    if not _reviewable(job_id):
        raise HTTPException(409, "Ce rendu ne garde pas de quoi être revu (rendu sur GPU, ou lancé avant cette fonction).")
    db.update("jobs", job_id, status="review", stage="review", error=None)
    return public(db.get("jobs", job_id))


@router.get("/{job_id}/swapped.mp4")
def swapped(job_id: str) -> FileResponse:
    return _file(job_id, "swapped.mp4", "video/mp4")


@router.get("/{job_id}/review/crop.jpg")
def review_crop(job_id: str, frame: int, x1: float, y1: float, x2: float, y2: float) -> FileResponse:
    """Visage d'origine d'une image de l'extrait (vignette d'une piste), gardé en cache."""
    get_or_404("jobs", job_id, "Rendu")
    d = db.folder("jobs", job_id)
    plan = _plan(job_id)
    if not 0 <= frame < len(plan.frames):
        raise not_found("Image")
    cache = d / "review_cache"
    cache.mkdir(exist_ok=True)
    path = cache / f"{frame}_{round(x1)}_{round(y1)}_{round(x2)}_{round(y2)}.jpg"
    if not path.is_file():
        # Première image dont l'instant est ≥ t : on vise un peu avant le début de l'image voulue.
        img = media.extract_frame(d / "cut.mp4", max(0.0, (frame - 0.4) / plan.fps))
        tmp = path.with_suffix(".tmp.jpg")
        cv2.imwrite(str(tmp), face_crop(img, (x1, y1, x2, y2), 192), [cv2.IMWRITE_JPEG_QUALITY, 86])
        tmp.replace(path)
    return FileResponse(path, media_type="image/jpeg")


@router.get("/{job_id}/result.mp4")
def result(job_id: str, download: bool = False) -> FileResponse:
    name = None
    if download:
        job = db.get("jobs", job_id)
        video = db.get("videos", job["video_id"]) if job else None
        title = re.sub(r"[^\w\- ]+", "", (video or {}).get("title") or "faceswap").strip()[:60] or "faceswap"
        name = f"{title} - swap.mp4"
    return _file(job_id, "result.mp4", "video/mp4", name)
