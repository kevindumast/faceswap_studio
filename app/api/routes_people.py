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
        res.pop("image", None)
        results.append(res)
    return {"imported": results, "people": [library.public(p) for p in library.all_people()]}


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


@router.get("/{pid}/photos/{photo_id}/{kind}.jpg")
def photo_file(pid: str, photo_id: str, kind: str) -> FileResponse:
    _load(pid)
    path = library.photo_file(pid, photo_id, kind)
    if path is None:
        raise not_found("Photo")
    return FileResponse(path, media_type="image/jpeg")
