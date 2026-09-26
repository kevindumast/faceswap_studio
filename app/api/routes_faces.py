"""Visages source : un « set » de 1 à 20 photos, rangées automatiquement par personne.

Chaque photo est analysée dès l'ajout ; son embedding ArcFace est comparé à la moyenne de chaque personne
déjà connue : assez proche → même personne, sinon → nouvelle personne (A, B, C…). On peut corriger à la main.

Pas de table : data/faces/<set_id>/set.json + par photo <id>.jpg, <id>_crop.jpg, <id>.npy (embedding).
"""
from __future__ import annotations

import json
import shutil
import string
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app import db
from src.config import load_config
from src.faces import ModelsMissing
from src.identity import SourceFace, analyze_photo, average_embedding

from .common import not_found, save_upload

router = APIRouter(prefix="/api/faces", tags=["faces"])

# Similarité cosinus ArcFace : même personne ≈ 0,4–0,8 ; personnes différentes < 0,25.
SAME_PERSON = 0.35


class MoveIn(BaseModel):
    person: str  # id d'une personne existante, ou "new"


def _dir(set_id: str) -> Path:
    try:
        d = db.folder("faces", set_id)
    except ValueError:
        raise not_found("Set de visages")
    if not (d / "set.json").is_file():
        raise not_found("Set de visages")
    return d


def _load(d: Path) -> dict:
    data = json.loads((d / "set.json").read_text(encoding="utf-8"))
    if "persons" not in data:  # ancien format : une seule identité
        data["persons"] = [{"id": "A", "name": "Personne A"}] if any(p["ok"] for p in data["photos"]) else []
        for p in data["photos"]:
            p["person"] = "A" if p["ok"] else None
    return data


def _save(d: Path, data: dict) -> None:
    _prune(data)
    (d / "set.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _prune(data: dict) -> None:
    """Retire les personnes qui n'ont plus aucune photo valide."""
    used = {p["person"] for p in data["photos"] if p["ok"]}
    data["persons"] = [p for p in data["persons"] if p["id"] in used]


def _new_person(data: dict) -> str:
    taken = {p["id"] for p in data["persons"]}
    letter = next((c for c in string.ascii_uppercase if c not in taken), None)
    if letter is None:
        raise HTTPException(422, "Trop de personnes différentes.")
    data["persons"].append({"id": letter, "name": f"Personne {letter}"})
    data["persons"].sort(key=lambda p: p["id"])
    return letter


def _person_embeddings(d: Path, data: dict, person: str) -> list[np.ndarray]:
    return [np.load(d / f"{p['id']}.npy") for p in data["photos"] if p["ok"] and p["person"] == person]


def _closest_person(d: Path, data: dict, emb: np.ndarray) -> str | None:
    best, best_sim = None, SAME_PERSON
    for person in data["persons"]:
        embs = _person_embeddings(d, data, person["id"])
        if not embs:
            continue
        sim = float(np.dot(average_embedding(embs), emb))
        if sim >= best_sim:
            best, best_sim = person["id"], sim
    return best


def public(set_id: str, data: dict) -> dict:
    photos = [
        {**p, "crop_url": f"/api/faces/{set_id}/photos/{p['id']}/crop.jpg" if p["ok"] else None,
         "photo_url": f"/api/faces/{set_id}/photos/{p['id']}/photo.jpg"}
        for p in data["photos"]
    ]
    persons = [
        {**person, "count": sum(1 for p in photos if p["ok"] and p["person"] == person["id"]),
         "cover_url": next((p["crop_url"] for p in photos if p["ok"] and p["person"] == person["id"]), None)}
        for person in data["persons"]
    ]
    return {"id": set_id, "photos": photos, "persons": persons, "ok_count": sum(p["ok"] for p in photos)}


def person_ids(set_id: str) -> set[str]:
    try:
        d = _dir(set_id)
    except HTTPException:
        return set()
    return {p["id"] for p in _load(d)["persons"]}


def load_source_face(set_id: str, person: str | None = None) -> SourceFace:
    """Identité d'une personne = moyenne des embeddings de ses photos (toutes les photos si person=None)."""
    d = db.folder("faces", set_id)
    data = _load(d)
    if person is None:
        embeddings = [np.load(d / f"{p['id']}.npy") for p in data["photos"] if p["ok"]]
    else:
        embeddings = _person_embeddings(d, data, person)
    return SourceFace(average_embedding(embeddings))


async def _add(set_id: str, d: Path, files: list[UploadFile]) -> dict:
    cfg = load_config()
    data = _load(d)
    if len(data["photos"]) + len(files) > cfg.photos.max:
        raise HTTPException(422, f"{cfg.photos.max} photos maximum.")
    for upload in files:
        ext = Path(upload.filename or "").suffix.lower()
        if ext not in cfg.upload.photo_ext:
            raise HTTPException(415, f"{upload.filename} : format non supporté (jpg, png, webp).")
        photo_id = db.new_id()
        raw = d / f"{photo_id}{ext}"
        await save_upload(upload, raw, float(cfg.upload.photo_max_mb))
        try:
            result = analyze_photo(raw)
        except ModelsMissing as exc:
            raise HTTPException(503, str(exc)) from exc
        entry = {"id": photo_id, "name": upload.filename, "ok": result.ok, "person": None}
        img = cv2.imdecode(np.fromfile(str(raw), np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            scale = min(1.0, 480 / max(img.shape[:2]))
            thumb = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(d / f"{photo_id}.jpg"), thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if raw.suffix != ".jpg" or img is None:
            raw.unlink(missing_ok=True)
        if result.ok:
            cv2.imwrite(str(d / f"{photo_id}_crop.jpg"), result.crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
            np.save(d / f"{photo_id}.npy", result.embedding)
            # Rangement automatique : personne la plus proche, sinon nouvelle personne.
            entry["person"] = _closest_person(d, data, result.embedding) or _new_person(data)
        data["photos"].append(entry)
    _save(d, data)
    return public(set_id, data)


@router.post("")
async def create_set(files: list[UploadFile] = File(...), consent: bool = Form(False)) -> dict:
    if not consent:
        raise HTTPException(422, "Confirmez que les photos sont les vôtres ou celles d'une personne consentante.")
    set_id = db.new_id()
    d = db.folder("faces", set_id)
    d.mkdir(parents=True)
    _save(d, {"photos": [], "persons": []})
    try:
        return await _add(set_id, d, files)
    except BaseException:
        shutil.rmtree(d, ignore_errors=True)
        raise


@router.post("/{set_id}/photos")
async def add_photos(set_id: str, files: list[UploadFile] = File(...)) -> dict:
    return await _add(set_id, _dir(set_id), files)


@router.get("/{set_id}")
def get_set(set_id: str) -> dict:
    d = _dir(set_id)
    return public(set_id, _load(d))


@router.patch("/{set_id}/photos/{photo_id}")
def move_photo(set_id: str, photo_id: str, body: MoveIn) -> dict:
    """Corrige le rangement : déplace une photo vers une autre personne (ou une nouvelle)."""
    d = _dir(set_id)
    data = _load(d)
    photo = next((p for p in data["photos"] if p["id"] == photo_id), None)
    if photo is None or not photo["ok"]:
        raise not_found("Photo")
    if body.person == "new":
        photo["person"] = _new_person(data)
    elif body.person in {p["id"] for p in data["persons"]}:
        photo["person"] = body.person
    else:
        raise HTTPException(422, "Personne inconnue.")
    _save(d, data)
    return public(set_id, data)


@router.delete("/{set_id}/photos/{photo_id}")
def delete_photo(set_id: str, photo_id: str) -> dict:
    d = _dir(set_id)
    if not photo_id.isalnum():
        raise not_found("Photo")
    data = _load(d)
    data["photos"] = [p for p in data["photos"] if p["id"] != photo_id]
    for f in d.glob(f"{photo_id}*"):
        f.unlink(missing_ok=True)
    _save(d, data)
    return public(set_id, data)


@router.get("/{set_id}/photos/{photo_id}/{kind}.jpg")
def photo_file(set_id: str, photo_id: str, kind: str) -> FileResponse:
    d = _dir(set_id)
    if not photo_id.isalnum() or kind not in ("crop", "photo"):
        raise not_found("Photo")
    path = d / (f"{photo_id}_crop.jpg" if kind == "crop" else f"{photo_id}.jpg")
    if not path.is_file():
        raise not_found("Photo")
    return FileResponse(path, media_type="image/jpeg")
