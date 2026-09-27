"""API : validation des jobs et rangement des photos par personne (sans lancer de rendu)."""
import json
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from app.api.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def ready_video(client, sample_video):
    from app import db
    from src import media

    info = media.probe(sample_video)
    vid = db.new_id()
    db.insert("videos", id=vid, kind="upload", title="mire", status="ready", progress=1,
              source=str(sample_video), info=info.to_dict(), created_at=time.time())
    return vid


def _face_set(embeddings: dict[str, list[np.ndarray]]) -> str:
    """Crée un set de visages directement sur disque, avec des embeddings connus."""
    from app import db

    set_id = db.new_id()
    d = db.folder("faces", set_id)
    d.mkdir(parents=True)
    photos = []
    for person, embs in embeddings.items():
        for e in embs:
            pid = db.new_id()
            np.save(d / f"{pid}.npy", e / np.linalg.norm(e))
            photos.append({"id": pid, "name": f"{pid}.jpg", "ok": True, "person": person})
    persons = [{"id": p, "name": f"Personne {p}"} for p in embeddings]
    (d / "set.json").write_text(json.dumps({"photos": photos, "persons": persons}), encoding="utf-8")
    return set_id


def _job(client, video, face_set, start, end, mappings):
    return client.post("/api/jobs", json={
        "video_id": video, "face_set_id": face_set, "start": start, "end": end,
        "consent": True, "mappings": mappings,
    })


def test_status(client):
    s = client.get("/api/status").json()
    assert s["segment"] == {"min_s": 5, "max_s": 60}


def test_segment_bounds(client, ready_video):
    rng = np.random.default_rng(0)
    fs = _face_set({"A": [rng.normal(size=512)]})
    box = {"t": 1.0, "box": [0.1, 0.1, 0.3, 0.4], "person": "A"}
    assert _job(client, ready_video, fs, 0, 4, [box]).status_code == 422      # trop court
    assert _job(client, ready_video, fs, 0, 9, [box]).status_code == 422      # dépasse la vidéo
    ok = _job(client, ready_video, fs, 1, 7, [box])
    assert ok.status_code == 200 and ok.json()["total"] == 150                # 6 s × 25 i/s
    client.post(f"/api/jobs/{ok.json()['id']}/cancel")


def test_unknown_person_rejected(client, ready_video):
    fs = _face_set({"A": [np.ones(512)]})
    r = _job(client, ready_video, fs, 1, 7, [{"t": 1.0, "box": [0, 0, 0.2, 0.2], "person": "Z"}])
    assert r.status_code == 422 and "Z" in r.json()["detail"]


def test_consent_required(client, ready_video):
    fs = _face_set({"A": [np.ones(512)]})
    r = client.post("/api/jobs", json={"video_id": ready_video, "face_set_id": fs, "start": 1, "end": 7,
                                       "mappings": [{"t": 1, "box": [0, 0, 0.2, 0.2], "person": "A"}]})
    assert r.status_code == 422


def test_grouping_by_similarity():
    """Deux identités bien distinctes → deux personnes ; une variante proche → même personne."""
    from app.api import routes_faces
    from app import db

    rng = np.random.default_rng(1)
    a, b = rng.normal(size=512), rng.normal(size=512)
    set_id = _face_set({"A": [a]})
    d = db.folder("faces", set_id)
    data = routes_faces._load(d)
    near_a = (a / np.linalg.norm(a)) + 0.3 * rng.normal(size=512) / np.sqrt(512)
    assert routes_faces._closest_person(d, data, near_a / np.linalg.norm(near_a)) == "A"
    assert routes_faces._closest_person(d, data, b / np.linalg.norm(b)) is None


def test_move_photo_and_prune(client):
    rng = np.random.default_rng(2)
    set_id = _face_set({"A": [rng.normal(size=512), rng.normal(size=512)]})
    data = client.get(f"/api/faces/{set_id}").json()
    photo = data["photos"][0]["id"]
    moved = client.patch(f"/api/faces/{set_id}/photos/{photo}", json={"person": "new"}).json()
    assert [(p["id"], p["count"]) for p in moved["persons"]] == [("A", 1), ("B", 1)]
    back = client.patch(f"/api/faces/{set_id}/photos/{photo}", json={"person": "A"}).json()
    assert [(p["id"], p["count"]) for p in back["persons"]] == [("A", 2)]  # B vide → retirée
