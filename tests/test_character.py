"""Niveau 4 (personne entière) : suivi de la personne choisie, photo en pied, recollage, API, worker avec un faux Space."""
import importlib.util
import json
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent


def _targeting():
    """Module du Space importé par son chemin (le dossier du Space a son propre app.py, à ne pas confondre)."""
    spec = importlib.util.spec_from_file_location("targeting", ROOT / "space_character" / "targeting.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- Suivi de la personne choisie (space_character/targeting.py) ---------------------------------------------------

def test_track_starts_on_the_chosen_face_and_ignores_the_person_crossing():
    pick_track = _targeting().pick_track

    frames = []
    for i in range(20):
        passer = [10 + i * 8, 10, 60 + i * 8, 200, 0.9]     # traverse l'image et passe devant la personne choisie
        chosen = [150, 12, 200, 210, 0.9]
        frames.append(np.array([passer, chosen]))
    track = pick_track(frames, 0, (165, 20, 185, 45))          # visage en haut de la 2e silhouette
    assert all(box[0] == 150 for box in track)


def test_track_keeps_last_position_when_detection_misses_and_goes_backward_from_anchor():
    pick_track = _targeting().pick_track

    frames = [np.array([[100 + i, 0, 200 + i, 300, 0.9], [400, 0, 500, 300, 0.9]]) for i in range(10)]
    frames[3] = np.zeros((0, 5))                                  # personne non détectée sur une image
    frames[6] = np.array([[100, 0, 200, 300, 0.1]])               # détection trop peu sûre : ignorée
    track = pick_track(frames, 8, (140, 10, 170, 50))           # ancre vers la fin : suivi dans les deux sens
    # En remontant depuis l'ancre, une image sans détection garde la position de l'image suivante.
    assert track[0][0] == 100 and track[3][0] == track[4][0] == 104 and track[6][0] == track[7][0] == 107
    assert track[9][0] == 109


def test_track_empty_when_nobody_is_detected():
    pick_track = _targeting().pick_track

    assert pick_track([np.zeros((0, 5))] * 4, 1, (0, 0, 10, 10)) == [None] * 4


# --- Photo de référence, estimation GPU ---------------------------------------------------------------------------

def test_reference_is_the_most_full_body_photo(tmp_path):
    from src.character import choose_reference

    ratios = {"portrait": 0.45, "en_pied": 0.11, "sans_visage": None}
    ref = choose_reference([tmp_path / n for n in ratios], ratio=lambda p: ratios[p.name])
    assert ref.path.name == "en_pied" and ref.full_body
    assert not choose_reference([tmp_path / "portrait"], ratio=lambda p: 0.45).full_body
    with pytest.raises(RuntimeError):
        choose_reference([tmp_path / "x"], ratio=lambda p: None)


def test_gpu_estimate_matches_the_space():
    from src.character import gpu_seconds

    assert gpu_seconds(5, "360p", 6) == pytest.approx(105)
    assert gpu_seconds(5, "480p", 6) > gpu_seconds(5, "360p", 6) * 1.5
    space = (ROOT / "space_character" / "app.py").read_text(encoding="utf-8")
    assert 'per_s = {"360p": 12.0, "480p": 26.0}' in space and "return 45 + seconds * per_s * steps / 6" in space


# --- Recollage sur l'extrait d'origine ----------------------------------------------------------------------------

def _solid(path: Path, color, size, fps: str, seconds: float, right_half: bool = False) -> None:
    from src import media

    w, h = size
    frame = np.zeros((h, w, 3), np.uint8)
    if right_half:
        frame[:, w // 2:] = color
    else:
        frame[:] = color
    with media.FrameWriter(path, size, fps, 18, "veryfast") as writer:
        for _ in range(round(seconds * float(fps))):
            writer.write(frame)


def test_recompose_keeps_original_outside_the_mask_and_its_frame_rate(sample_video, tmp_path):
    from src import media
    from src.character import recompose

    clip = tmp_path / "clip.mp4"
    media.cut_segment(sample_video, clip, 1.0, 3.0, (640, 360), False)           # 50 images à 25 i/s
    generated, mask = tmp_path / "gen.mp4", tmp_path / "mask.mp4"
    _solid(generated, (0, 0, 255), (320, 176), "30", 2.0)                          # personne « générée » : rouge, 30 i/s
    _solid(mask, (255, 255, 255), (320, 176), "30", 2.0, right_half=True)          # zone refaite : moitié droite
    out = tmp_path / "out.mp4"
    assert recompose(generated, mask, clip, out, "25", 18, "veryfast") == 50
    info = media.probe(out)
    assert (info.width, info.height, info.fps) == (640, 360, 25)
    cap_out, cap_ref = cv2.VideoCapture(str(out)), cv2.VideoCapture(str(clip))
    _, got = cap_out.read()
    _, ref = cap_ref.read()
    assert np.abs(got[:, :250].astype(int) - ref[:, :250].astype(int)).mean() < 6   # gauche : image d'origine
    assert got[:, 400:, 2].mean() > 200 and got[:, 400:, :2].mean() < 50            # droite : personne générée


# --- API -----------------------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    from app.api.main import app

    with TestClient(app) as c:
        yield c
        c.delete("/api/settings/zerogpu")


@pytest.fixture(scope="module")
def ready_video(client, sample_video):
    from app import db
    from src import media

    vid = db.new_id()
    db.insert("videos", id=vid, kind="upload", title="mire", status="ready", progress=1,
              source=str(sample_video), info=media.probe(sample_video).to_dict(), created_at=time.time())
    return vid


def _body(video, set_id, people, **extra):
    mappings = [{"t": 1, "box": [0.1 + 0.4 * i, 0.1, 0.3 + 0.4 * i, 0.4], "person": p} for i, p in enumerate(people)]
    return {"video_id": video, "face_set_id": set_id, "start": 1, "end": 7, "consent": True, "level": "character",
            "mappings": mappings, **extra}


def test_settings_store_the_level4_space(client):
    z = client.put("/api/settings/zerogpu", json={"space": "kevin/faceswap-gpu", "token": "hf_" + "x" * 30, "key": "k",
                                                   "character_space": "kevin/faceswap-character"}).json()["zerogpu"]
    assert z["character_space"] == "kevin/faceswap-character" and z["character_configured"] and not z["character_tested"]
    z = client.put("/api/settings/zerogpu", json={"space": "kevin/faceswap-gpu"}).json()["zerogpu"]
    assert z["character_space"] == "kevin/faceswap-character"                      # absent = inchangé
    assert client.get("/api/status").json()["gpu"]["character"]["configured"] is True
    assert client.put("/api/settings/zerogpu", json={"space": "kevin/faceswap-gpu", "character_space": "pas bon"}).status_code == 422
    z = client.put("/api/settings/zerogpu", json={"space": "kevin/faceswap-gpu", "character_space": ""}).json()["zerogpu"]
    assert z["character_space"] is None and not z["character_configured"]
    client.delete("/api/settings/zerogpu")


def test_level4_rules(client, ready_video, make_session, monkeypatch):
    from src.config import load_config

    set_id, ids = make_session({"Kevin": [np.ones(512)], "Pote": [-np.ones(512)]})
    one = _body(ready_video, set_id, [ids["Kevin"]], use_gpu=True, resolution="480p")
    r = client.post("/api/jobs", json={**one, "use_gpu": False})
    assert r.status_code == 422 and "option GPU" in r.json()["detail"]
    r = client.post("/api/jobs", json=one)
    assert r.status_code == 422 and "niveau 4 n'est pas branché" in r.json()["detail"]

    client.put("/api/settings/zerogpu", json={"space": "kevin/faceswap-gpu", "token": "hf_x", "key": "k",
                                              "character_space": "kevin/faceswap-character"})
    r = client.post("/api/jobs", json=_body(ready_video, set_id, [ids["Kevin"], ids["Pote"]], use_gpu=True))
    assert r.status_code == 422 and "une seule personne" in r.json()["detail"]
    monkeypatch.setitem(load_config()["levels"]["character"], "max_s", 5.5)
    r = client.post("/api/jobs", json=one)
    assert r.status_code == 422 and "limité à 5.5 s" in r.json()["detail"]
    monkeypatch.undo()
    r = client.post("/api/jobs", json=one)
    assert r.status_code == 200
    job = r.json()
    assert job["params"]["level"] == "character" and job["params"]["resolution"] == "480p" and job["pausable"] is False
    client.post(f"/api/jobs/{job['id']}/cancel")
    client.delete("/api/settings/zerogpu")


# --- Client et worker avec un faux Space ---------------------------------------------------------------------------

def test_client_replace_copies_video_and_mask(monkeypatch, tmp_path):
    import gradio_client

    from app.worker.zerogpu_client import ZeroGPUClient

    video, mask = tmp_path / "v.mp4", tmp_path / "m.mp4"
    video.write_bytes(b"video")
    mask.write_bytes(b"mask")
    calls = []

    class Job:
        def done(self):
            return True

        def result(self):
            return str(video), {"path": str(mask)}, json.dumps({"frames": 150, "gpu_seconds": 98.5})

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def submit(self, *args, api_name):
            calls.append((api_name, args))
            return Job()

    monkeypatch.setattr(gradio_client, "Client", FakeClient)
    monkeypatch.setattr(gradio_client, "handle_file", lambda p: p)
    stats = ZeroGPUClient("kevin/c", "tok", "key").replace(tmp_path / "clip.mp4", tmp_path / "ref.jpg", {"t": 1.0},
                                                           tmp_path / "out.mp4", tmp_path / "mask_out.mp4", poll=0)
    assert calls[0][0] == "/replace" and calls[0][1][1].endswith("ref.jpg") and json.loads(calls[0][1][2]) == {"t": 1.0}
    assert (tmp_path / "out.mp4").read_bytes() == b"video" and (tmp_path / "mask_out.mp4").read_bytes() == b"mask"
    assert stats["gpu_seconds"] == 98.5


def test_worker_level4_end_to_end_with_fake_space(sample_video, tmp_path, monkeypatch):
    from app.worker import main as worker
    from src import media
    from src.character import Reference
    from src.identity import SourceFace
    from src.levels import PersonAssets
    from src.pipeline import FaceMapping, RenderOptions

    sent = {}

    class FakeSpace:
        def replace(self, clip, reference, payload, out, mask_out, on_progress=None, should_cancel=None, on_stage=None):
            sent.update(payload=payload, clip=media.probe(clip), reference=reference)
            on_stage("queue", 1, 2, None)                     # 2e dans la file du Space
            on_stage("gpu", 0, 0, None)                       # calcul lancé, en attente d'un GPU
            on_stage("progress", 500, 1000, "génération")     # étape annoncée par le Space
            _solid(out, (0, 0, 255), (320, 176), "30", 6.0)
            _solid(mask_out, (255, 255, 255), (320, 176), "30", 6.0, right_half=True)
            return {"frames": 180, "gpu_seconds": 120.0, "warnings": ["test"]}

    photo = tmp_path / "moi.jpg"
    photo.write_bytes(b"jpg")
    monkeypatch.setattr(worker.ZeroGPUClient, "from_settings", classmethod(lambda cls, kind="faces": FakeSpace()))
    woken = []

    def fake_wait(kind, on_wait, should_cancel=None, poll=10.0):
        woken.append(kind)
        on_wait(12.4, 1800)                                   # Space endormi : 12 s de réveil sur ~30 min

    monkeypatch.setattr(worker, "wait_until_ready", fake_wait)
    monkeypatch.setattr(worker, "choose_reference", lambda photos: Reference(photos[0], 0.4))   # portrait seulement
    person = PersonAssets(source=SourceFace(np.ones(512, np.float32) / np.sqrt(512)), photos=[photo])
    mapping = FaceMapping(person, target={"t": 3.0, "box": [0.4, 0.1, 0.6, 0.4]}, label="Visage 1")
    opts = RenderOptions(start=1.0, end=7.0, output="segment", stabilize=True, ai_label=True, level="character")
    job = {"id": "test", "params": {"resolution": "360p"}}
    out = tmp_path / "job"
    progress = []
    stats = worker.run_character(job, sample_video, opts, mapping, out, lambda *a: progress.append(a))

    assert sent["payload"]["t"] == pytest.approx(2.0) and sent["payload"]["resolution"] == "360p"
    assert not sent["clip"].has_audio and sent["reference"] == photo
    result = media.probe(out / "result.mp4")
    assert result.duration == pytest.approx(6.0, abs=0.1) and result.has_audio and (result.width, result.fps) == (640, 25)
    assert stats.frames == 150 and "test" in stats.warnings and any("photo en pied" in w for w in stats.warnings)
    assert woken == ["character"]
    stages = [a for a in progress if a[0] in ("wake", "queue", "gpu", "generate")]
    assert stages == [("wake", 12, 1800), ("queue", 1, 2), ("gpu", 0, 0), ("generate", 500, 1000)]
    assert not (out / "generated.mp4").exists()


# --- Réveil et état du Space (faux Hugging Face) -------------------------------------------------------------------

class _FakeHf:
    """Suite d'états renvoyés par get_space_runtime ; restart_space enregistré."""

    def __init__(self, stages):
        self.stages, self.restarts = list(stages), []

    def __call__(self, token=None):
        return self

    def get_space_runtime(self, space):
        stage = self.stages.pop(0) if len(self.stages) > 1 else self.stages[0]
        return SimpleNamespace(stage=stage, hardware=None, raw={"errorMessage": "OOM" if "ERROR" in stage else None})

    def restart_space(self, space):
        self.restarts.append(space)


@pytest.fixture
def spaces_configured():
    from app import db
    from app.worker import zerogpu_client as zg

    for k, v in (("zerogpu_space", "kevin/gpu"), ("zerogpu_key", "k"), ("zerogpu_character_space", "kevin/character")):
        db.set_meta(k, v)
    zg._states.clear()
    yield zg
    for k in ("zerogpu_space", "zerogpu_key", "zerogpu_character_space", "space_waking:faces", "space_waking:character"):
        db.set_meta(k, "")
    zg._states.clear()


def test_state_keeps_the_wake_chrono_until_ready(spaces_configured, monkeypatch):
    import huggingface_hub

    zg = spaces_configured
    fake = _FakeHf(["SLEEPING", "SLEEPING", "APP_STARTING", "APP_STARTING", "RUNNING"])
    monkeypatch.setattr(huggingface_hub, "HfApi", fake)
    assert zg.state("character", max_age=0)["phase"] == "asleep"
    woke = zg.wake("character")                                    # endormi → redémarré, chrono lancé
    assert fake.restarts == ["kevin/character"] and woke["waking_since"] is not None
    starting = zg.state("character", max_age=0)
    assert starting["phase"] == "starting" and starting["waking_since"] == woke["waking_since"]
    assert starting["expected_s"] == 1800
    ready = zg.state("character", max_age=0)
    assert ready["phase"] == "ready" and ready["waking_since"] is None
    assert zg.wake("character")["phase"] == "ready" and len(fake.restarts) == 1   # déjà prêt : rien à faire


def test_wait_until_ready_wakes_once_and_reports_the_chrono(spaces_configured, monkeypatch):
    import huggingface_hub

    zg = spaces_configured
    fake = _FakeHf(["SLEEPING", "SLEEPING", "SLEEPING", "APP_STARTING", "APP_STARTING", "RUNNING"])
    monkeypatch.setattr(huggingface_hub, "HfApi", fake)
    seen = []
    zg.wait_until_ready("faces", on_wait=lambda elapsed, expected: seen.append(expected), poll=0)
    assert fake.restarts == ["kevin/gpu"] and seen and set(seen) == {180}


def test_wait_until_ready_stops_on_a_broken_space_or_cancel(spaces_configured, monkeypatch):
    import huggingface_hub

    zg = spaces_configured
    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeHf(["RUNTIME_ERROR"]))
    with pytest.raises(zg.ZeroGPUError, match="en erreur.*OOM"):          # une relance tentée, puis erreur claire
        zg.wait_until_ready("faces", on_wait=lambda *a: None, poll=0)
    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeHf(["APP_STARTING"]))
    with pytest.raises(zg.RemoteCancelled):
        zg.wait_until_ready("faces", on_wait=lambda *a: None, should_cancel=lambda: True, poll=0)


def test_state_api(client, spaces_configured, monkeypatch):
    import huggingface_hub

    fake = _FakeHf(["SLEEPING", "SLEEPING", "APP_STARTING"])
    monkeypatch.setattr(huggingface_hub, "HfApi", fake)
    assert client.get("/api/settings/zerogpu/state?kind=character").json()["phase"] == "asleep"
    r = client.post("/api/settings/zerogpu/wake?kind=character").json()
    assert fake.restarts == ["kevin/character"] and r["waking_since"]
    from app import db

    db.set_meta("zerogpu_character_space", "")
    assert client.get("/api/settings/zerogpu/state?kind=character").json()["phase"] == "unconfigured"
