"""Bibliothèque de personnes : permanente et partagée entre toutes les vidéos.

data/people/<id>/person.json  {id, name, created_at, updated_at, photos: [{id, name, added_at}]}
                  <photo>.jpg (480 px, UI), <photo>_full.jpg (1024 px, niveaux 3-4), <photo>_crop.jpg, <photo>.npy (ArcFace)
                  tone.json   (cache du teint, recalculé si les photos changent)

Jamais touchée par le nettoyage automatique (qui ne vise que data/faces, les sessions de vidéo).
À l'import, chaque photo est comparée à toutes les personnes connues : assez proche → rangée dans cette personne
(« Reconnu : Kevin »), sinon → nouvelle personne.
"""
from __future__ import annotations

import json
import shutil
import threading
import time
import unicodedata
from pathlib import Path

import cv2
import numpy as np

from app import db
from src.identity import SourceFace, analyze_photo, average_embedding
from src.levels import FACE_TONE, PersonAssets

# Similarité cosinus ArcFace : même personne ≈ 0,4–0,8 ; personnes différentes < 0,25.
SAME_PERSON = 0.35
NAME_MAX = 40

_lock = threading.RLock()  # l'API sert plusieurs requêtes en parallèle


class PersonNotFound(KeyError):
    pass


def root() -> Path:
    return db.data_dir() / "people"


def _dir(pid: str) -> Path:
    if not pid or not pid.isalnum():
        raise PersonNotFound(pid)
    return root() / pid


def exists(pid: str) -> bool:
    try:
        return (_dir(pid) / "person.json").is_file()
    except PersonNotFound:
        return False


def load(pid: str) -> dict:
    path = _dir(pid) / "person.json"
    if not path.is_file():
        raise PersonNotFound(pid)
    return json.loads(path.read_text(encoding="utf-8"))


def _save(person: dict) -> None:
    person["updated_at"] = time.time()
    d = _dir(person["id"])
    d.mkdir(parents=True, exist_ok=True)
    (d / "person.json").write_text(json.dumps(person, ensure_ascii=False), encoding="utf-8")


def all_people() -> list[dict]:
    if not root().is_dir():
        return []
    people = []
    for d in root().iterdir():
        if (d / "person.json").is_file():
            people.append(json.loads((d / "person.json").read_text(encoding="utf-8")))
    return sorted(people, key=lambda p: p.get("updated_at", 0), reverse=True)


def _fold(text: str) -> str:
    """Minuscules sans accents, pour la recherche (« élodie » trouve « Elodie »)."""
    return "".join(c for c in unicodedata.normalize("NFD", text.lower()) if unicodedata.category(c) != "Mn")


def search(query: str = "") -> list[dict]:
    q = _fold(query.strip())
    return [p for p in all_people() if not q or q in _fold(p["name"])]


def _default_name() -> str:
    taken = {p["name"] for p in all_people()}
    n = 1
    while f"Personne {n}" in taken:
        n += 1
    return f"Personne {n}"


def create(name: str | None = None) -> dict:
    with _lock:
        now = time.time()
        person = {"id": db.new_id(), "name": clean_name(name) if name else _default_name(),
                  "created_at": now, "updated_at": now, "photos": []}
        _save(person)
        return person


def clean_name(name: str) -> str:
    name = " ".join(str(name).split())[:NAME_MAX]
    if not name:
        raise ValueError("Le nom ne peut pas être vide.")
    return name


def rename(pid: str, name: str) -> dict:
    with _lock:
        person = load(pid)
        person["name"] = clean_name(name)
        _save(person)
        return person


def delete(pid: str) -> None:
    with _lock:
        shutil.rmtree(_dir(pid), ignore_errors=True)


def embeddings(person: dict) -> list[np.ndarray]:
    d = _dir(person["id"])
    return [np.load(d / f"{ph['id']}.npy") for ph in person["photos"] if (d / f"{ph['id']}.npy").is_file()]


def closest(embedding: np.ndarray) -> str | None:
    """Personne de la bibliothèque la plus ressemblante, si elle dépasse le seuil « même personne »."""
    best, best_sim = None, SAME_PERSON
    for person in all_people():
        embs = embeddings(person)
        if not embs:
            continue
        sim = float(np.dot(average_embedding(embs), embedding))
        if sim >= best_sim:
            best, best_sim = person["id"], sim
    return best


def _write_images(d: Path, photo_id: str, img: np.ndarray, crop: np.ndarray, embedding: np.ndarray) -> None:
    # 1024 px pour les niveaux qui ont besoin de détails (tête, personne entière), 480 px pour l'UI.
    for suffix, side, quality in (("_full", 1024, 92), ("", 480, 85)):
        scale = min(1.0, side / max(img.shape[:2]))
        resized = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(d / f"{photo_id}{suffix}.jpg"), resized, [cv2.IMWRITE_JPEG_QUALITY, quality])
    cv2.imwrite(str(d / f"{photo_id}_crop.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
    np.save(d / f"{photo_id}.npy", embedding)


def import_photo(raw: Path, original_name: str) -> dict:
    """Analyse une photo et la range dans la bonne personne (reconnue ou nouvelle). Le fichier brut est consommé."""
    result = analyze_photo(raw)
    img = cv2.imdecode(np.fromfile(str(raw), np.uint8), cv2.IMREAD_COLOR)
    if not result.ok or img is None:
        return {"ok": False, "name": original_name, "image": img}
    with _lock:
        pid = closest(result.embedding)
        created = pid is None
        person = create() if created else load(pid)
        photo_id = db.new_id()
        _write_images(_dir(person["id"]), photo_id, img, result.crop, result.embedding)
        person["photos"].append({"id": photo_id, "name": original_name, "added_at": time.time()})
        _save(person)
    raw.unlink(missing_ok=True)
    return {"ok": True, "name": original_name, "person_id": person["id"], "person_name": person["name"], "created": created}


def move_photo(pid: str, photo_id: str, target: str) -> str:
    """Déplace une photo vers une autre personne (ou une nouvelle). Une personne vidée de ses photos est supprimée."""
    with _lock:
        source = load(pid)
        photo = next((ph for ph in source["photos"] if ph["id"] == photo_id), None)
        if photo is None:
            raise PersonNotFound(photo_id)
        dest = create() if target == "new" else load(target)
        if dest["id"] == source["id"]:
            return dest["id"]
        src_dir, dst_dir = _dir(source["id"]), _dir(dest["id"])
        for f in src_dir.glob(f"{photo_id}*"):
            shutil.move(str(f), dst_dir / f.name)
        source["photos"] = [ph for ph in source["photos"] if ph["id"] != photo_id]
        dest["photos"].append(photo)
        _save(dest)
        if source["photos"]:
            _save(source)
        else:
            delete(source["id"])
        return dest["id"]


def delete_photo(pid: str, photo_id: str) -> bool:
    """Supprime une photo. Renvoie False si la personne, vidée, a été supprimée avec."""
    with _lock:
        person = load(pid)
        person["photos"] = [ph for ph in person["photos"] if ph["id"] != photo_id]
        for f in _dir(pid).glob(f"{photo_id}*"):
            f.unlink(missing_ok=True)
        if not person["photos"]:
            delete(pid)
            return False
        _save(person)
        return True


def photo_file(pid: str, photo_id: str, kind: str) -> Path | None:
    if not photo_id.isalnum() or kind not in ("crop", "photo"):
        return None
    path = _dir(pid) / (f"{photo_id}_crop.jpg" if kind == "crop" else f"{photo_id}.jpg")
    return path if path.is_file() else None


def public(person: dict) -> dict:
    base = f"/api/people/{person['id']}/photos"
    photos = [{"id": ph["id"], "name": ph["name"], "crop_url": f"{base}/{ph['id']}/crop.jpg",
               "photo_url": f"{base}/{ph['id']}/photo.jpg"} for ph in person["photos"]]
    return {"id": person["id"], "name": person["name"], "count": len(photos),
            "cover_url": photos[0]["crop_url"] if photos else None, "photos": photos,
            "created_at": person.get("created_at"), "updated_at": person.get("updated_at")}


def photo_paths(pid: str) -> list[Path]:
    """Photos de la personne, en meilleure qualité disponible (1024 px, sinon vignette)."""
    d = _dir(pid)
    out = []
    for ph in load(pid)["photos"]:
        full = d / f"{ph['id']}_full.jpg"
        out.append(full if full.is_file() else d / f"{ph['id']}.jpg")
    return out


def assets(pid: str, level: str) -> PersonAssets:
    """Tout ce dont un niveau a besoin pour cette personne. Le teint est mis en cache (recalculé si les photos changent)."""
    if not exists(pid):
        raise RuntimeError("Cette personne a été supprimée de la bibliothèque.")
    person = load(pid)
    embs = embeddings(person)
    if not embs:
        raise RuntimeError(f"{person['name']} n'a plus de photo utilisable.")
    out = PersonAssets(source=SourceFace(average_embedding(embs)), photos=photo_paths(pid))
    if level in (FACE_TONE,):
        from src.assets import person_tone
        from src.tone import ToneStats

        key = ",".join(sorted(ph["id"] for ph in person["photos"]))
        cache = _dir(pid) / "tone.json"
        cached = json.loads(cache.read_text(encoding="utf-8")) if cache.is_file() else None
        if cached and cached.get("key") == key:
            out.tone = ToneStats.from_dict(cached["tone"])
        else:
            out.tone = person_tone(out.photos)
            if out.tone is None:
                raise RuntimeError(f"Teint introuvable sur les photos de {person['name']} (visage trop petit ou masqué).")
            cache.write_text(json.dumps({"key": key, "tone": out.tone.to_dict()}), encoding="utf-8")
    return out


def import_legacy_set(set_dir: Path) -> list[str]:
    """Reprend une ancienne session (personnes A, B… stockées dans la session) dans la bibliothèque.

    Renvoie les ids des personnes créées ; la session est réécrite au nouveau format.
    """
    data = json.loads((set_dir / "set.json").read_text(encoding="utf-8"))
    if "people" in data:
        return []
    created = []
    with _lock:
        for letter in sorted({ph.get("person") for ph in data.get("photos", []) if ph.get("ok") and ph.get("person")}):
            person = create()
            dst = _dir(person["id"])
            for ph in data["photos"]:
                if ph.get("person") != letter or not ph.get("ok"):
                    continue
                for f in set_dir.glob(f"{ph['id']}*"):
                    shutil.copy2(f, dst / f.name)
                person["photos"].append({"id": ph["id"], "name": ph["name"], "added_at": time.time()})
            _save(person)
            created.append(person["id"])
        (set_dir / "set.json").write_text(json.dumps({"people": created, "rejected": []}, ensure_ascii=False), encoding="utf-8")
    return created
