"""Niveau 2 (teint) sur image de synthèse, validations des niveaux / de l'option GPU, instants relatifs."""
import cv2
import numpy as np
import pytest

from src import parsing, tone


def _synthetic_face():
    """Crop 240×240 : peau (ellipse) + chemise en bas, avec la carte de segmentation correspondante."""
    rng = np.random.default_rng(0)
    img = np.full((240, 240, 3), (40, 140, 230), np.uint8)             # fond orange
    labels = np.zeros((240, 240), np.uint8)
    skin = np.zeros((240, 240), np.uint8)
    cv2.ellipse(skin, (120, 115), (55, 70), 0, 0, 360, 1, -1)
    img[skin > 0] = (60, 90, 140)                                      # peau brune
    labels[skin > 0] = parsing.SKIN
    img[205:] = (200, 120, 40)                                         # chemise bleue
    labels[205:] = parsing.CLOTH
    img = np.clip(img.astype(np.int16) + rng.integers(-6, 7, img.shape), 0, 255).astype(np.uint8)
    bbox = np.array([70, 50, 170, 175], np.float32)
    kps = np.array([[100, 95], [140, 95], [120, 118], [104, 140], [136, 140]], np.float32)
    return img, labels, bbox, kps


def _lab_mean(img, mask):
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)[mask].mean(axis=0)


def test_skin_region_excludes_shirt_and_background():
    img, labels, bbox, kps = _synthetic_face()
    region, core = tone.skin_regions(img, bbox, kps, labels)
    assert core.sum() > 500
    assert region[labels == parsing.SKIN].mean() > 0.9     # la peau est prise
    assert not region[210:].any()                           # la chemise jamais
    assert not region[:, :20].any()                         # le fond non plus


def test_transfer_moves_skin_toward_source_only():
    img, labels, bbox, kps = _synthetic_face()
    region, core = tone.skin_regions(img, bbox, kps, labels)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    target = tone.robust_stats(lab[core])
    light = np.full((20, 20, 3), (150, 180, 225), np.uint8)             # peau claire de référence
    source = tone.robust_stats(cv2.cvtColor(light, cv2.COLOR_BGR2LAB).astype(np.float32).reshape(-1, 3))
    out = tone.transfer(img, region, target, source, strength_l=0.5)

    inner = np.zeros_like(region)
    inner[90:150, 90:150] = True
    inner &= region
    before, after = _lab_mean(img, inner), _lab_mean(out, inner)
    assert np.linalg.norm(after[1:] - source.mean[1:]) < np.linalg.norm(before[1:] - source.mean[1:]) * 0.3
    assert after[0] > before[0] + 10                                    # plus clair, mais pas tout (strength 0.5)
    assert np.abs(out[215:].astype(int) - img[215:].astype(int)).max() == 0   # chemise intacte


def test_tone_stats_roundtrip():
    s = tone.ToneStats(np.array([120, 140, 150], np.float32), np.array([8, 3, 4], np.float32))
    back = tone.ToneStats.from_dict(s.to_dict())
    assert np.allclose(back.mean, s.mean) and np.allclose(back.std, s.std)


def test_relative_targets():
    from src.identity import SourceFace
    from src.levels import PersonAssets
    from src.pipeline import FaceMapping, relative_targets

    p = PersonAssets(source=SourceFace(np.ones(512, np.float32)))
    maps = [FaceMapping(p, {"t": 22.0, "box": [0, 0, 1, 1]}), FaceMapping(p, {"t": 40.0, "box": [0, 0, 1, 1]}), FaceMapping(p)]
    rel = relative_targets(maps, start=20.0, length=6.0)
    assert rel[0].target["t"] == pytest.approx(2.0)
    assert rel[1].target["t"] == pytest.approx(5.95)       # borné à la fin de l'extrait
    assert rel[2].target is None
    assert maps[0].target["t"] == 22.0                      # l'original n'est pas modifié


# ─── Validations API ───

@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from app.api.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def setup(client, sample_video):
    import json
    import time

    from app import db
    from src import media

    info = media.probe(sample_video)
    vid = db.new_id()
    db.insert("videos", id=vid, kind="upload", title="mire", status="ready", progress=1,
              source=str(sample_video), info=info.to_dict(), created_at=time.time())
    from app import library

    person = library.create("Kevin")
    np.save(library.root() / person["id"] / "p1.npy", np.ones(512, np.float32) / np.sqrt(512))
    person["photos"].append({"id": "p1", "name": "a.jpg", "added_at": time.time()})
    library._save(person)
    set_id = db.new_id()
    d = db.folder("faces", set_id)
    d.mkdir(parents=True)
    (d / "set.json").write_text(json.dumps({"people": [person["id"]], "rejected": []}), encoding="utf-8")
    return vid, set_id, person["id"]


def _job(client, setup, **extra):
    vid, set_id, pid = setup
    body = {"video_id": vid, "face_set_id": set_id, "start": 1, "end": 7, "consent": True,
            "mappings": [{"t": 1, "box": [0.1, 0.1, 0.3, 0.4], "person": pid}], **extra}
    return client.post("/api/jobs", json=body)


def test_default_is_level1_on_cpu(client, setup):
    r = _job(client, setup)
    assert r.status_code == 200
    assert r.json()["params"]["level"] == "face" and r.json()["params"]["use_gpu"] is False
    client.post(f"/api/jobs/{r.json()['id']}/cancel")


def test_level2_accepted_when_models_installed(client, setup):
    from src.levels import models_ready

    r = _job(client, setup, level="face_tone")
    assert r.status_code == (200 if models_ready("face_tone") else 422)
    if r.status_code == 200:
        client.post(f"/api/jobs/{r.json()['id']}/cancel")


def test_level4_requires_gpu_option(client, setup):
    r = _job(client, setup, level="character")
    assert r.status_code == 422 and "option GPU" in r.json()["detail"]


def test_gpu_option_refused_when_not_configured(client, setup):
    r = _job(client, setup, use_gpu=True)
    assert r.status_code == 422 and "ZeroGPU" in r.json()["detail"]


def test_status_lists_levels(client):
    s = client.get("/api/status").json()
    assert set(s["levels"]) == {"face", "face_tone", "head", "character"}
    assert s["levels"]["face"]["available"] and s["levels"]["character"]["gpu_only"]
    assert s["gpu"]["configured"] is False
