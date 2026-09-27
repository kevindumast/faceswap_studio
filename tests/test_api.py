"""API : validation des jobs, bibliothèque de personnes et sessions de vidéo (sans lancer de rendu)."""
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


def _job(client, video, face_set, start, end, mappings):
    return client.post("/api/jobs", json={
        "video_id": video, "face_set_id": face_set, "start": start, "end": end,
        "consent": True, "mappings": mappings,
    })


def _unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


def test_status(client):
    s = client.get("/api/status").json()
    assert s["segment"] == {"min_s": 5, "max_s": 60}


def test_segment_bounds(client, ready_video, make_session):
    rng = np.random.default_rng(0)
    fs, ids = make_session({"Kevin": [rng.normal(size=512)]})
    box = {"t": 1.0, "box": [0.1, 0.1, 0.3, 0.4], "person": ids["Kevin"]}
    assert _job(client, ready_video, fs, 0, 4, [box]).status_code == 422      # trop court
    assert _job(client, ready_video, fs, 0, 9, [box]).status_code == 422      # dépasse la vidéo (8 s)
    assert _job(client, ready_video, fs, 0, 61, [box]).status_code == 422     # dépasse la limite de 60 s
    ok = _job(client, ready_video, fs, 1, 7, [box])
    assert ok.status_code == 200 and ok.json()["total"] == 150                # 6 s × 25 i/s
    client.post(f"/api/jobs/{ok.json()['id']}/cancel")


def test_unknown_person_rejected(client, ready_video, make_session):
    fs, _ = make_session({"Kevin": [np.ones(512)]})
    r = _job(client, ready_video, fs, 1, 7, [{"t": 1.0, "box": [0, 0, 0.2, 0.2], "person": "Z"}])
    assert r.status_code == 422 and "Z" in r.json()["detail"]


def test_person_removed_from_video_is_refused(client, ready_video, make_session):
    fs, ids = make_session({"Kevin": [np.ones(512)]})
    client.delete(f"/api/faces/{fs}/people/{ids['Kevin']}")
    r = _job(client, ready_video, fs, 1, 7, [{"t": 1.0, "box": [0, 0, 0.2, 0.2], "person": ids["Kevin"]}])
    assert r.status_code == 422


def test_consent_required(client, ready_video, make_session):
    fs, ids = make_session({"Kevin": [np.ones(512)]})
    r = client.post("/api/jobs", json={"video_id": ready_video, "face_set_id": fs, "start": 1, "end": 7,
                                       "mappings": [{"t": 1, "box": [0, 0, 0.2, 0.2], "person": ids["Kevin"]}]})
    assert r.status_code == 422


def test_library_recognizes_known_person(make_session):
    """Une nouvelle photo proche d'une personne connue est reconnue ; un visage inconnu ne l'est pas."""
    from app import library

    rng = np.random.default_rng(1)
    a, b = rng.normal(size=512), rng.normal(size=512)
    _, ids = make_session({"Kevin": [a]})
    near_a = _unit(_unit(a) + 0.3 * rng.normal(size=512) / np.sqrt(512))
    assert library.closest(near_a) == ids["Kevin"]
    assert library.closest(_unit(b)) is None


def test_rename_and_accent_insensitive_search(client, make_session):
    _, ids = make_session({"Personne test": [np.random.default_rng(3).normal(size=512)]})
    r = client.patch(f"/api/people/{ids['Personne test']}", json={"name": "  Élodie   Martin "})
    assert r.status_code == 200 and r.json()["name"] == "Élodie Martin"
    found = client.get("/api/people", params={"q": "elodie"}).json()
    assert [p["id"] for p in found] == [ids["Personne test"]]
    assert client.patch(f"/api/people/{ids['Personne test']}", json={"name": "   "}).status_code == 422


def test_move_photo_to_new_person_and_back(client, make_session):
    rng = np.random.default_rng(2)
    _, ids = make_session({"Kevin": [rng.normal(size=512), rng.normal(size=512)]})
    kevin = client.get(f"/api/people/{ids['Kevin']}").json()
    photo = kevin["photos"][0]["id"]
    moved = client.patch(f"/api/people/{ids['Kevin']}/photos/{photo}", json={"person": "new"}).json()
    new_pid = moved["moved_to"]
    assert new_pid != ids["Kevin"] and moved["source_exists"]
    assert client.get(f"/api/people/{ids['Kevin']}").json()["count"] == 1
    back = client.patch(f"/api/people/{new_pid}/photos/{photo}", json={"person": ids["Kevin"]}).json()
    assert back["moved_to"] == ids["Kevin"] and not back["source_exists"]   # la personne vidée disparaît
    assert client.get(f"/api/people/{ids['Kevin']}").json()["count"] == 2


def test_session_add_and_remove_person(client, make_session):
    _, ids = make_session({"Pote": [np.random.default_rng(4).normal(size=512)]})
    session = client.post("/api/faces").json()
    assert session["persons"] == []
    added = client.post(f"/api/faces/{session['id']}/people", json={"person_id": ids["Pote"]}).json()
    assert [p["id"] for p in added["persons"]] == [ids["Pote"]]
    removed = client.delete(f"/api/faces/{session['id']}/people/{ids['Pote']}").json()
    assert removed["persons"] == []
    assert client.get(f"/api/people/{ids['Pote']}").status_code == 200    # toujours dans la bibliothèque


def test_library_survives_cleanup(make_session):
    from app import db, library

    set_id, ids = make_session({"Kevin": [np.random.default_rng(5).normal(size=512)]})
    db.cleanup(-1)                      # tout ce qui est « plus vieux que dans 1 h » : toutes les sessions
    assert not db.folder("faces", set_id).exists()
    assert library.exists(ids["Kevin"])


def test_legacy_session_is_imported(make_session):
    """Une ancienne session (personnes A, B internes) est reprise dans la bibliothèque, photos comprises."""
    from app import db, library

    rng = np.random.default_rng(6)
    set_id = db.new_id()
    d = db.folder("faces", set_id)
    d.mkdir(parents=True)
    photos = []
    for letter in ("A", "A", "B"):
        ph = db.new_id()
        np.save(d / f"{ph}.npy", _unit(rng.normal(size=512)))
        (d / f"{ph}_crop.jpg").write_bytes(b"jpg")
        photos.append({"id": ph, "name": f"{ph}.jpg", "ok": True, "person": letter})
    (d / "set.json").write_text(json.dumps({"photos": photos, "persons": [{"id": "A"}, {"id": "B"}]}), encoding="utf-8")

    created = library.import_legacy_set(d)
    assert [len(library.load(pid)["photos"]) for pid in created] == [2, 1]
    assert json.loads((d / "set.json").read_text(encoding="utf-8"))["people"] == created
    assert library.import_legacy_set(d) == []                  # déjà reprise : ne refait rien
