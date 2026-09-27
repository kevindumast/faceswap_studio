"""Bibliothèque de personnes : lister, chercher, renommer, gérer les photos, supprimer."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app import db, library
from src.config import load_config
from src.faces import ModelsMissing

from .common import not_found, save_upload

router = APIRouter(prefix="/api/people", tags=["people"])


class RenameIn(BaseModel):
    name: str


class MoveIn(BaseModel):
    person: str  # id d'une autre personne, ou "new"


class AssignPendingIn(BaseModel):
    box: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)  # x1, y1, x2, y2 en fractions 0-1 de l'image
    target: str  # id d'une personne, ou "new"
    name: str | None = None  # nom de la nouvelle personne (si target == "new")


def _load(pid: str) -> dict:
    try:
        return library.load(pid)
    except library.PersonNotFound:
        raise not_found("Personne")


@router.get("")
def list_people(q: str = "") -> list[dict]:
    return [library.public(p) for p in library.search(q)]


@router.post("/photos")
async def import_photos(files: list[UploadFile] = File(...), consent: bool = Form(False)) -> dict:
    """Ajoute des photos directement dans la bibliothèque (hors d'une vidéo)."""
    if not consent:
        raise HTTPException(422, "Confirmez que les photos sont les vôtres ou celles d'une personne consentante.")
    cfg = load_config()
    if len(files) > cfg.photos.max:
        raise HTTPException(422, f"{cfg.photos.max} photos maximum par envoi.")
    tmp = library.root() / "_imports"
    tmp.mkdir(parents=True, exist_ok=True)
    results = []
    for upload in files:
        ext = Path(upload.filename or "").suffix.lower()
        if ext not in cfg.upload.photo_ext:
            raise HTTPException(415, f"{upload.filename} : format non supporté (jpg, png, webp).")
        raw = tmp / f"{db.new_id()}{ext}"
        await save_upload(upload, raw, float(cfg.upload.photo_max_mb))
        try:
            res = library.import_photo(raw, upload.filename or raw.name)
        except ModelsMissing as exc:
            raise HTTPException(503, str(exc)) from exc
        finally:
            raw.unlink(missing_ok=True)
        img = res.pop("image", None)
        if not res["ok"] and img is not None:
            pending = library.add_pending(img, upload.filename or raw.name)
            res["pending_id"] = pending["id"]
        results.append(res)
    return {"imported": results, "people": [library.public(p) for p in library.all_people()]}


@router.get("/photos/pending")
def list_pending() -> list[dict]:
    """Photos importées sans visage détecté, en attente d'un rattachement manuel."""
    return library.public_pending()


@router.get("/photos/pending/{rid}/{kind}.jpg")
def pending_file(rid: str, kind: str) -> FileResponse:
    path = library.pending_file(rid, kind)
    if path is None:
        raise not_found("Photo")
    return FileResponse(path, media_type="image/jpeg")


@router.delete("/photos/pending/{rid}")
def delete_pending(rid: str) -> dict:
    library.delete_pending(rid)
    return {"ok": True}


@router.post("/photos/pending/{rid}/assign")
def assign_pending(rid: str, body: AssignPendingIn) -> dict:
    """Rattache une photo sans visage auto-détecté à une personne, via une zone tracée à la main."""
    if body.target != "new" and not library.exists(body.target):
        raise HTTPException(422, "Personne inconnue.")
    try:
        person = library.assign_pending(rid, body.box, body.target, body.name)
    except library.PersonNotFound:
        raise not_found("Photo en attente")
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return library.public(person)


@router.get("/{pid}")
def get_person(pid: str) -> dict:
    return library.public(_load(pid))


@router.patch("/{pid}")
def rename(pid: str, body: RenameIn) -> dict:
    _load(pid)
    try:
        return library.public(library.rename(pid, body.name))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.delete("/{pid}")
def delete(pid: str) -> dict:
    _load(pid)
    library.delete(pid)
    return {"ok": True}


@router.patch("/{pid}/photos/{photo_id}")
def move_photo(pid: str, photo_id: str, body: MoveIn) -> dict:
    """Corrige le rangement : cette photo appartient à une autre personne (ou à une nouvelle)."""
    _load(pid)
    if body.person != "new" and not library.exists(body.person):
        raise HTTPException(422, "Personne inconnue.")
    try:
        dest = library.move_photo(pid, photo_id, body.person)
    except library.PersonNotFound:
        raise not_found("Photo")
    return {"moved_to": dest, "source_exists": library.exists(pid)}


@router.delete("/{pid}/photos/{photo_id}")
def delete_photo(pid: str, photo_id: str) -> dict:
    _load(pid)
    still = library.delete_photo(pid, photo_id)
    return {"ok": True, "person_exists": still}


@router.get("/{pid}/framing")
def framing(pid: str) -> dict:
    """Photos en pied ou portraits, et celle que le niveau 4 utilisera (calculé une fois par photo)."""
    _load(pid)
    try:
        return library.framing(pid)
    except ModelsMissing as exc:
        raise HTTPException(503, str(exc)) from exc


@router.get("/{pid}/photos/{photo_id}/{kind}.jpg")
def photo_file(pid: str, photo_id: str, kind: str) -> FileResponse:
    _load(pid)
    path = library.photo_file(pid, photo_id, kind)
    if path is None:
        raise not_found("Photo")
    return FileResponse(path, media_type="image/jpeg")
