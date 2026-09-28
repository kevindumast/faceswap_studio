"""Session de visages d'une vidéo : les personnes de la bibliothèque choisies pour ce rendu.

data/faces/<set_id>/set.json  {people: [id de personne…], rejected: [{id, name}], frozen?: true}
Les photos importées ici entrent dans la bibliothèque (app/library.py), reconnues ou comme nouvelle personne ;
seules les photos sans visage restent dans la session (nettoyées avec elle au bout de retention_hours).
Chaque création a sa propre session. Celle d'un rendu lancé est figée : la modifier ensuite (rendu rouvert depuis
l'historique, autre essai) crée une copie, et les rendus déjà faits gardent leurs personnes.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app import db, library
from src.config import load_config
from src.faces import ModelsMissing
from src.levels import PersonAssets

from .common import not_found, save_upload

router = APIRouter(prefix="/api/faces", tags=["faces"])


class PersonRef(BaseModel):
    person_id: str


def _dir(set_id: str) -> Path:
    try:
        d = db.folder("faces", set_id)
    except ValueError:
        raise not_found("Session de visages")
    if not (d / "set.json").is_file():
        raise not_found("Session de visages")
    return d


def _load(d: Path) -> dict:
    data = json.loads((d / "set.json").read_text(encoding="utf-8"))
    if "people" not in data:  # ancienne session (personnes A, B… internes) : non reprise automatiquement
        return {"people": [], "rejected": [], "legacy": True}
    return data


def _save(d: Path, data: dict) -> None:
    data = {k: v for k, v in data.items() if k in ("people", "rejected", "frozen")}
    (d / "set.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def public(set_id: str, data: dict) -> dict:
    persons = [library.public(library.load(pid)) for pid in data["people"] if library.exists(pid)]
    rejected = [{**r, "photo_url": f"/api/faces/{set_id}/rejected/{r['id']}.jpg"} for r in data.get("rejected", [])]
    return {"id": set_id, "persons": persons, "rejected": rejected, "ok_count": sum(p["count"] for p in persons),
            "legacy": bool(data.get("legacy"))}


def _writable(set_id: str) -> tuple[str, Path, dict]:
    """Session à modifier : celle d'un rendu déjà lancé n'est pas touchée, la modification part sur une copie."""
    d = _dir(set_id)
    data = _load(d)
    if not data.get("frozen"):
        return set_id, d, data
    new_id = db.new_id()
    nd = db.folder("faces", new_id)
    nd.mkdir(parents=True)
    for r in data.get("rejected", []):
        if (d / f"{r['id']}.jpg").is_file():
            shutil.copy2(d / f"{r['id']}.jpg", nd / f"{r['id']}.jpg")
    data = {"people": list(data["people"]), "rejected": list(data.get("rejected", []))}
    _save(nd, data)
    return new_id, nd, data


def freeze(set_id: str) -> None:
    """Fige la session d'un rendu : ses personnes ne bougeront plus, même si on continue à éditer la création."""
    d = _dir(set_id)
    data = _load(d)
    if not data.get("frozen") and not data.get("legacy"):
        _save(d, {**data, "frozen": True})


def person_ids(set_id: str) -> set[str]:
    try:
        d = _dir(set_id)
    except HTTPException:
        return set()
    return {pid for pid in _load(d)["people"] if library.exists(pid)}


def load_person_assets(set_id: str, person: str | None, level: str) -> PersonAssets:
    """Données d'une personne de la session, pour le worker."""
    if person is None or person not in person_ids(set_id):
        raise RuntimeError("Personne absente de cette session (ancienne session ou personne retirée).")
    return library.assets(person, level)


async def _import(d: Path, data: dict, files: list[UploadFile]) -> list[dict]:
    """Importe les photos dans la bibliothèque et ajoute les personnes concernées à la session."""
    cfg = load_config()
    if len(files) > cfg.photos.max:
        raise HTTPException(422, f"{cfg.photos.max} photos maximum par envoi.")
    results = []
    for upload in files:
        ext = Path(upload.filename or "").suffix.lower()
        if ext not in cfg.upload.photo_ext:
            raise HTTPException(415, f"{upload.filename} : format non supporté (jpg, png, webp).")
        raw = d / f"upload_{db.new_id()}{ext}"
        await save_upload(upload, raw, float(cfg.upload.photo_max_mb))
        try:
            res = library.import_photo(raw, upload.filename or raw.name)
        except ModelsMissing as exc:
            raw.unlink(missing_ok=True)
            raise HTTPException(503, str(exc)) from exc
        if res["ok"]:
            if res["person_id"] not in data["people"]:
                data["people"].append(res["person_id"])
        else:  # pas de visage : on garde une vignette pour l'afficher, dans la session seulement
            rid = db.new_id()
            img = res.pop("image")
            if img is not None:
                scale = min(1.0, 480 / max(img.shape[:2]))
                cv2.imwrite(str(d / f"{rid}.jpg"), cv2.resize(img, None, fx=scale, fy=scale), [cv2.IMWRITE_JPEG_QUALITY, 85])
            data.setdefault("rejected", []).append({"id": rid, "name": res["name"]})
            raw.unlink(missing_ok=True)
        res.pop("image", None)
        results.append(res)
    _save(d, data)
    return results


def _require_consent(consent: bool) -> None:
    if not consent:
        raise HTTPException(422, "Confirmez que les photos sont les vôtres ou celles d'une personne consentante.")


@router.post("")
async def create_set(files: list[UploadFile] = File(default=[]), consent: bool = Form(False)) -> dict:
    """Nouvelle session ; avec des photos, elles sont importées tout de suite."""
    if files:
        _require_consent(consent)
    set_id = db.new_id()
    d = db.folder("faces", set_id)
    d.mkdir(parents=True)
    data = {"people": [], "rejected": []}
    _save(d, data)
    try:
        results = await _import(d, data, files) if files else []
    except BaseException:
        shutil.rmtree(d, ignore_errors=True)
        raise
    return {**public(set_id, data), "imported": results}


@router.post("/{set_id}/photos")
async def add_photos(set_id: str, files: list[UploadFile] = File(...), consent: bool = Form(True)) -> dict:
    _require_consent(consent)
    set_id, d, data = _writable(set_id)
    results = await _import(d, data, files)
    return {**public(set_id, data), "imported": results}


@router.get("/{set_id}")
def get_set(set_id: str) -> dict:
    d = _dir(set_id)
    return public(set_id, _load(d))


@router.post("/{set_id}/people")
def add_person(set_id: str, body: PersonRef) -> dict:
    """Ajoute une personne de la bibliothèque à cette vidéo."""
    _dir(set_id)
    if not library.exists(body.person_id):
        raise not_found("Personne")
    set_id, d, data = _writable(set_id)
    if body.person_id not in data["people"]:
        data["people"].append(body.person_id)
        _save(d, data)
    return public(set_id, data)


@router.delete("/{set_id}/people/{person_id}")
def remove_person(set_id: str, person_id: str) -> dict:
    """Retire une personne de cette vidéo (elle reste dans la bibliothèque)."""
    set_id, d, data = _writable(set_id)
    data["people"] = [p for p in data["people"] if p != person_id]
    _save(d, data)
    return public(set_id, data)


@router.delete("/{set_id}/rejected/{rid}")
def delete_rejected(set_id: str, rid: str) -> dict:
    if not rid.isalnum():
        raise not_found("Photo")
    set_id, d, data = _writable(set_id)
    data["rejected"] = [r for r in data.get("rejected", []) if r["id"] != rid]
    (d / f"{rid}.jpg").unlink(missing_ok=True)
    _save(d, data)
    return public(set_id, data)


@router.get("/{set_id}/rejected/{rid}.jpg")
def rejected_file(set_id: str, rid: str) -> FileResponse:
    d = _dir(set_id)
    path = d / f"{rid}.jpg"
    if not rid.isalnum() or not path.is_file():
        raise not_found("Photo")
    return FileResponse(path, media_type="image/jpeg")
