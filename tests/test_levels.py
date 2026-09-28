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
    out = tone.transfer(img, region, tone.tone_shift(target, source, strength_l=0.5))

    inner = np.zeros(region.shape, bool)
    inner[90:150, 90:150] = True
    inner &= region > 0.99
    before, after = _lab_mean(img, inner), _lab_mean(out, inner)
    assert np.linalg.norm(after[1:] - source.mean[1:]) < np.linalg.norm(before[1:] - source.mean[1:]) * 0.3
    assert after[0] > before[0] + 10                                    # plus clair, mais pas tout (strength 0.5)
    assert np.abs(out[215:].astype(int) - img[215:].astype(int)).max() == 0   # chemise intacte


def _from_lab(lab: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def test_shadows_and_beard_never_turn_grey_blue():
    """Peau chaude + ombre / barbe peu colorées, source plus claire et moins chaude : rien ne passe sous le neutre."""
    rng = np.random.default_rng(1)
    lab = np.zeros((240, 240, 3), np.float32)
    lab[:] = (150, 150, 165)                                            # fond
    skin = np.zeros((240, 240), np.uint8)
    cv2.ellipse(skin, (120, 115), (55, 70), 0, 0, 360, 1, -1)
    lab[skin > 0] = (130, 146, 152)                                     # peau chaude
    beard_zone = np.zeros_like(skin)
    beard_zone[160:185, 80:160] = 1
    lab[(skin > 0) & (beard_zone > 0)] = (38, 134, 135)                 # barbe / ombre sombre, peu colorée
    lab[..., 1:] += rng.normal(0, 2.5, (240, 240, 2))
    img = _from_lab(lab)
    labels = np.where(skin > 0, parsing.SKIN, parsing.BACKGROUND).astype(np.uint8)
    bbox = np.array([70, 50, 170, 175], np.float32)
    kps = np.array([[100, 95], [140, 95], [120, 118], [104, 140], [136, 140]], np.float32)
    region, core = tone.skin_regions(img, bbox, kps, labels)
    before = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    target = tone.robust_stats(before[core])
    source = tone.ToneStats(np.array([175, 137, 140], np.float32), np.array([12, 9, 9], np.float32))
    out = tone.transfer(img, region, tone.tone_shift(target, source, strength_l=0.5, max_shift=25))
    after = cv2.cvtColor(out, cv2.COLOR_BGR2LAB).astype(np.float32)
    assert after[..., 2][region > 0.05].min() >= 128                    # jamais bleu
    beard = np.zeros(skin.shape, bool)
    beard[165:180, 90:150] = True
    beard &= skin > 0
    assert np.abs(after[beard] - before[beard])[:, 1:].mean() < 4       # les ombres gardent leur couleur


def test_default_tone_shift_is_a_plain_offset():
    assert tone.ToneSettings().chroma_scale == 1.0
    target = tone.ToneStats(np.array([120, 150, 160], np.float32), np.array([5, 2, 2], np.float32))
    source = tone.ToneStats(np.array([160, 140, 150], np.float32), np.array([20, 9, 9], np.float32))
    assert np.allclose(tone.tone_shift(target, source, strength_l=0.5, max_shift=25), [20, -10, -10])
    far = tone.ToneStats(np.array([160, 100, 110], np.float32), np.array([5, 2, 2], np.float32))
    assert np.allclose(tone.tone_shift(target, far, 0.5, max_shift=25)[1:], [-25, -25])   # borné


def test_tone_shift_compensates_lighting_cast():
    """Photos prises sous lumière chaude : sans compensation, la vidéo serait réchauffée à tort."""
    target = tone.ToneStats(np.array([140, 145, 150], np.float32), np.array([5, 2, 2], np.float32),
                            neutral=np.array([0, 0], np.float32))
    source = tone.ToneStats(np.array([140, 150, 160], np.float32), np.array([5, 2, 2], np.float32),
                            neutral=np.array([5, 8], np.float32))
    assert np.allclose(tone.tone_shift(target, source, 0.0, wb=0.0)[1:], [5, 10])
    assert np.allclose(tone.tone_shift(target, source, 0.0, wb=1.0)[1:], [0, 2])


def _fake_face(dx=0.0, dy=0.0):
    from types import SimpleNamespace

    bbox = np.array([150 + dx, 110 + dy, 250 + dx, 235 + dy], np.float32)
    kps = np.array([[180, 155], [220, 155], [200, 178], [184, 200], [216, 200]], np.float32) + [dx, dy]
    return SimpleNamespace(bbox=bbox, kps=kps)


def _sequence(n=30):
    """Visage fixe, bruit de compression différent à chaque image, détection qui tremble de ±3 px."""
    rng = np.random.default_rng(2)
    frames, faces = [], []
    for _ in range(n):
        img = np.full((360, 400, 3), (40, 140, 230), np.uint8)
        cv2.ellipse(img, (200, 175), (55, 70), 0, 0, 360, (60, 90, 140), -1)
        img[265:] = (200, 120, 40)
        img = np.clip(img.astype(np.int16) + rng.integers(-8, 9, img.shape), 0, 255).astype(np.uint8)
        frames.append(img)
        faces.append(_fake_face(*rng.uniform(-3, 3, 2)))
    return frames, faces


def _parse_by_color(crop):
    """BiSeNet simulé : peau = rouge modéré, chemise = bleu dominant, reste = fond."""
    b, r = crop[..., 0].astype(int), crop[..., 2].astype(int)
    labels = np.full(crop.shape[:2], parsing.BACKGROUND, np.uint8)
    labels[(r < 185) & (b < 110)] = parsing.SKIN
    labels[b > 160] = parsing.CLOTH
    return labels


def _light_source():
    return tone.ToneStats(np.array([175, 140, 150], np.float32), np.array([8, 4, 4], np.float32))


def test_tone_matcher_is_stable_over_time(monkeypatch):
    monkeypatch.setattr(tone.parsing, "parse", _parse_by_color)
    frames, faces = _sequence()
    matcher = tone.ToneMatcher(_light_source())
    matcher.prepare(list(zip(frames[::2], faces[::2])))
    assert matcher.frozen
    inner = np.zeros((360, 400), bool)
    inner[150:200, 175:225] = True
    medians = []
    for frame, face in zip(frames, faces):
        out = matcher.apply(frame, face, original=frame)
        medians.append(np.median(cv2.cvtColor(out, cv2.COLOR_BGR2LAB).astype(np.float32)[inner], axis=0))
    medians = np.array(medians)
    assert np.abs(np.diff(medians[:, 1:], axis=0)).max() < 1.5          # le teint ne saute pas
    assert np.abs(medians[:, 1:] - _light_source().mean[1:]).max() < 4   # et il est appliqué sur toutes les images


def test_tone_matcher_holds_mask_when_skin_is_lost(monkeypatch):
    monkeypatch.setattr(tone.parsing, "parse", _parse_by_color)
    frames, faces = _sequence(4)
    matcher = tone.ToneMatcher(_light_source())
    matcher.prepare(list(zip(frames, faces)))
    matcher.apply(frames[0], faces[0])
    monkeypatch.setattr(tone.parsing, "parse", lambda crop: np.full(crop.shape[:2], parsing.HAIR, np.uint8))
    out = matcher.apply(frames[1], faces[1])                             # peau introuvable sur cette image
    assert np.abs(out.astype(int) - frames[1].astype(int)).max() > 5     # le teint ne disparaît pas d'un coup


def test_source_tone_ignores_an_outlier_photo(monkeypatch):
    monkeypatch.setattr(tone.parsing, "parse", lambda crop: np.full(crop.shape[:2], parsing.SKIN, np.uint8))
    rng = np.random.default_rng(3)
    images, faces = [], []
    for color in [(60, 90, 140)] * 4 + [(40, 60, 200)]:                 # 4 photos normales + 1 sous lumière rouge
        img = np.full((360, 400, 3), color, np.uint8)
        images.append(np.clip(img.astype(np.int16) + rng.integers(-5, 6, img.shape), 0, 255).astype(np.uint8))
        faces.append(_fake_face())
    result = tone.source_tone(images, faces)
    normal = tone.source_tone(images[:4], faces[:4])
    assert np.abs(result.mean[1:] - normal.mean[1:]).max() < 1.0
    assert result.neutral is not None


def test_tone_stats_roundtrip():
    s = tone.ToneStats(np.array([120, 140, 150], np.float32), np.array([8, 3, 4], np.float32))
    back = tone.ToneStats.from_dict(s.to_dict())
    assert np.allclose(back.mean, s.mean) and np.allclose(back.std, s.std) and back.neutral is None
    s.neutral = np.array([2, -3], np.float32)
    assert np.allclose(tone.ToneStats.from_dict(s.to_dict()).neutral, s.neutral)


def test_prepare_samples_are_spread_and_deterministic(monkeypatch, sample_video):
    """Pré-passe : images réparties sur tout l'extrait, les mêmes à chaque appel (reprise après pause)."""
    from types import SimpleNamespace

    from src import pipeline

    class Recorder:
        samples = 16

        def __init__(self):
            self.got = []

        def prepare(self, samples):
            self.got = [float(frame.mean()) for frame, _ in samples]

    face = SimpleNamespace(bbox=np.array([10, 10, 60, 70], np.float32), kps=np.zeros((5, 2), np.float32), embedding=None)
    monkeypatch.setattr(pipeline, "detect_boxes", lambda frame: [face])
    runs = []
    for _ in range(2):
        rec = Recorder()
        pipeline.prepare_strategies(sample_video, [[None, None, rec, 0]], threshold=0.3)
        runs.append(rec.got)
    assert len(runs[0]) == 16 and runs[0] == runs[1]
    assert len(set(runs[0])) > 8                                         # des images différentes, pas 16 fois la même


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
