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
    assert s["segment"] == {"min_s": 0.5, "max_s": 60}


def test_segment_bounds(client, ready_video, make_session):
    rng = np.random.default_rng(0)
    fs, ids = make_session({"Kevin": [rng.normal(size=512)]})
    box = {"t": 1.0, "box": [0.1, 0.1, 0.3, 0.4], "person": ids["Kevin"]}
    assert _job(client, ready_video, fs, 0, 0.3, [box]).status_code == 422    # trop court (moins de 0,5 s)
    assert _job(client, ready_video, fs, 0, 9, [box]).status_code == 422      # dépasse la vidéo (8 s)
    assert _job(client, ready_video, fs, 0, 61, [box]).status_code == 422     # dépasse la limite de 60 s
    ok = _job(client, ready_video, fs, 1, 7, [box])
    assert ok.status_code == 200 and ok.json()["total"] == 150                # 6 s × 25 i/s
    client.post(f"/api/jobs/{ok.json()['id']}/cancel")


def test_unknown_person_rejected(client, ready_video, make_session):
    fs, _ = make_session({"Kevin": [np.ones(512)]})
    r = _job(client, ready_video, fs, 1, 7, [{"t": 1.0, "box": [0, 0, 0.2, 0.2], "person": "Z"}])
    assert r.status_code == 422 and "Z" in r.json()["detail"]


def test_same_person_on_many_faces(client, ready_video, make_session):
    """Une personne peut remplacer plusieurs visages ; au-delà du garde-fou, message clair."""
    from app.api.routes_jobs import MAX_MAPPINGS

    fs, ids = make_session({"Kevin": [np.ones(512)]})
    faces = [{"t": 1.0, "box": [0.04 * k, 0, 0.04 * k + 0.03, 0.1], "person": ids["Kevin"]} for k in range(MAX_MAPPINGS + 1)]
    ok = _job(client, ready_video, fs, 1, 7, faces[:9])
    assert ok.status_code == 200 and len(ok.json()["params"]["mappings"]) == 9
    client.post(f"/api/jobs/{ok.json()['id']}/cancel")
    r = _job(client, ready_video, fs, 1, 7, faces)
    assert r.status_code == 422 and "au maximum" in r.json()["detail"]


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


def test_rendered_session_is_frozen(client, ready_video, make_session):
    """Les personnes d'un rendu lancé ne bougent plus : modifier la session ensuite part sur une copie."""
    rng = np.random.default_rng(5)
    fs, ids = make_session({"Kevin": [rng.normal(size=512)], "Pote": [rng.normal(size=512)]})
    job = _job(client, ready_video, fs, 1, 7, [{"t": 1.0, "box": [0, 0, 0.2, 0.2], "person": ids["Kevin"]}]).json()
    client.post(f"/api/jobs/{job['id']}/cancel")
    edited = client.delete(f"/api/faces/{fs}/people/{ids['Pote']}").json()
    assert edited["id"] != fs and [p["id"] for p in edited["persons"]] == [ids["Kevin"]]
    assert {p["id"] for p in client.get(f"/api/faces/{fs}").json()["persons"]} == set(ids.values())
    again = client.post(f"/api/faces/{edited['id']}/people", json={"person_id": ids["Pote"]}).json()
    assert again["id"] == edited["id"]                                      # la copie, elle, reste modifiable


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


def test_manual_face_added_merged_and_removed(client, sample_video, monkeypatch):
    """Visage indiqué à la main à l'étape Visages : ajouté en fin de liste, gardé avec la vidéo, supprimable."""
    from app import db
    from app.api import routes_videos
    from src import media
    from src.scan import Person

    video = db.new_id()   # vidéo à part : le test de nettoyage vide la table des vidéos
    db.insert("videos", id=video, kind="upload", title="mire", status="ready", progress=1,
              source=str(sample_video), info=media.probe(sample_video).to_dict(), created_at=time.time())

    def person(t, box):
        return Person(t=t, box=box, score=0.8, height_px=50, crop=np.zeros((112, 112, 3), np.uint8),
                      embedding_sum=np.ones(512), times={t}, seen_at=[(t, box)])

    found = [0.1, 0.1, 0.2, 0.3]
    monkeypatch.setattr(routes_videos, "scan_passage",
                        lambda src, s, e, density=0: ([person(2.0, found)], {2.0: np.zeros((360, 640, 3), np.uint8)}))
    routes_videos._scan_cache.clear()
    base = f"/api/videos/{video}/scan"
    passage = {"start": 1, "end": 7}
    assert client.get(base, params=passage).json()["faces"][0]["seen_at"] == [{"t": 2.0, "box": found}]

    region = {**passage, "t": 2.0, "box": [0.5, 0.5, 0.6, 0.7]}
    monkeypatch.setattr(routes_videos, "face_in_region", lambda src, t, box: None)
    assert client.post(f"{base}/faces", json=region).status_code == 422               # rien dans le cadre
    monkeypatch.setattr(routes_videos, "face_in_region", lambda src, t, box: person(t, found))
    assert client.post(f"{base}/faces", json=region).json() == {"index": 0, "face": None}   # déjà dans la liste

    monkeypatch.setattr(routes_videos, "face_in_region", lambda src, t, box: person(t, [0.5, 0.5, 0.6, 0.7]))
    added = client.post(f"{base}/faces", json=region).json()
    assert added["index"] == 1 and added["face"]["manual"] and added["face"]["id"]
    assert [f.get("manual", False) for f in client.get(base, params=passage).json()["faces"]] == [False, True]
    assert len(client.get(base, params={"start": 3, "end": 8}).json()["faces"]) == 1    # hors de ce passage-là

    assert client.delete(f"{base}/faces/{added['face']['id']}").json() == {"ok": True}
    assert len(client.get(base, params=passage).json()["faces"]) == 1
    routes_videos._scan_cache.clear()


def test_more_frames_are_remembered_for_the_passage(client, sample_video, monkeypatch):
    """« Plus d'images » : la densité est gardée avec la vidéo (le rendu revoit la même analyse), jusqu'au maximum."""
    from app import db
    from app.api import routes_videos
    from src import media

    video = db.new_id()
    db.insert("videos", id=video, kind="upload", title="mire", status="ready", progress=1,
              source=str(sample_video), info=media.probe(sample_video).to_dict(), created_at=time.time())
    asked = []
    monkeypatch.setattr(routes_videos, "scan_passage", lambda src, s, e, density=0: (asked.append(density), ([], {}))[1])
    routes_videos._scan_cache.clear()
    base = f"/api/videos/{video}/scan"
    passage = {"start": 0, "end": 8}

    first = client.get(base, params=passage).json()
    assert (first["samples"], first["next_samples"]) == (12, 23)
    more = client.post(f"{base}/more", json=passage).json()
    assert (more["samples"], more["next_samples"]) == (23, 45)
    assert client.get(base, params=passage).json()["samples"] == 23                   # gardé, sans nouvelle analyse
    assert client.get(base, params={"start": 0, "end": 7}).json()["samples"] == 12   # autre passage : inchangé
    last = client.post(f"{base}/more", json=passage).json()
    assert (last["samples"], last["next_samples"]) == (45, None)       # 8 s : plus serré, moins de 0,1 s entre deux
    assert client.post(f"{base}/more", json=passage).status_code == 409
    assert asked == [0, 1, 0, 2]
    routes_videos._scan_cache.clear()
