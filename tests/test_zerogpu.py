"""Branchement ZeroGPU : réglages, validation de l'option GPU, messages d'erreur, contenu envoyé, client simulé."""
import json
import time
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient


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


def test_settings_never_expose_the_token(client):
    assert client.get("/api/settings").json()["zerogpu"]["configured"] is False
    token = "hf_" + "s3cr3t" * 5 + "AB12"
    r = client.put("/api/settings/zerogpu", json={"space": "kevin/faceswap-gpu", "token": token, "key": "k123"})
    z = r.json()["zerogpu"]
    assert z == {"space": "kevin/faceswap-gpu", "token_set": True, "key_set": True, "token_hint": "hf_…AB12",
                 "configured": True, "tested": False, "character_space": None, "character_configured": False,
                 "character_tested": False}
    exposed = json.dumps(client.get("/api/settings").json())
    assert "s3cr3t" not in exposed and "k123" not in exposed
    assert client.put("/api/settings/zerogpu", json={"space": "pas un space"}).status_code == 422
    assert client.delete("/api/settings/zerogpu").json()["zerogpu"]["configured"] is False


def test_gpu_option_accepted_only_when_configured(client, ready_video, make_session):
    fs, ids = make_session({"Kevin": [np.ones(512)]})
    body = {"video_id": ready_video, "face_set_id": fs, "start": 1, "end": 7, "consent": True, "use_gpu": True,
            "mappings": [{"t": 1, "box": [0.1, 0.1, 0.3, 0.4], "person": ids["Kevin"]}]}
    assert client.post("/api/jobs", json=body).status_code == 422
    client.put("/api/settings/zerogpu", json={"space": "kevin/faceswap-gpu", "token": "hf_x", "key": "k"})
    r = client.post("/api/jobs", json=body)
    assert r.status_code == 200 and r.json()["params"]["use_gpu"] is True
    client.post(f"/api/jobs/{r.json()['id']}/cancel")
    assert client.get("/api/status").json()["gpu"]["configured"] is True
    client.delete("/api/settings/zerogpu")


@pytest.mark.parametrize("raw, expected", [
    ("You have exceeded your GPU quota (60s requested vs. 12s left)", "Quota ZeroGPU épuisé"),
    ("Clé APP_KEY invalide.", "Clé APP_KEY refusée"),
    ("Could not fetch config for https://kevin-x.hf.space: 404 Not Found", "Space introuvable"),
    ("GPU task aborted", "coupé avant la fin"),
    ("The requested GPU duration (340s) is larger than the maximum allowed", "refuse de réserver 340 s"),
    ("You have exceeded your free ZeroGPU quota (204s requested vs. 95s left). Try again in 2:14:05.",
     "ce rendu réserve 204 s de GPU, il en reste 95 s (5 min/jour en gratuit, 40 en PRO). Réessaie dans 2 h 14"),
    ("You have exceeded your free ZeroGPU quota (204s requested vs. 95s left). Try again in 0:07:30.", "Réessaie dans 7 min"),
])
def test_friendly_errors(raw, expected):
    from app.worker.zerogpu_client import friendly

    assert expected in friendly(RuntimeError(raw))


def test_retry_time_is_given_as_a_clock_time():
    from datetime import datetime

    from app.worker.zerogpu_client import clock_text, retry_text

    now = datetime(2026, 9, 29, 22, 51).timestamp()
    assert retry_text(19 * 3600 + 29 * 60 + 10, now) == "dans 19 h 29 (demain vers 18 h 20)"
    assert retry_text(450, now) == "dans 7 min (vers 22 h 58)"
    assert clock_text(datetime(2026, 10, 2, 9, 5).timestamp(), now) == "le 02/10 vers 9 h 05"


def test_quota_refusal_is_remembered_until_the_retry_time(client):
    from app import db
    from app.worker import zerogpu_client as zg

    db.set_meta("zerogpu_quota", "")
    assert zg.quota_status() is None and client.get("/api/status").json()["gpu"]["quota"] is None
    err = zg.space_error(RuntimeError("You have exceeded your free ZeroGPU quota (88s requested vs. 21s left). "
                                      "Try again in 19:29:10. Subscribe to Hugging Face PRO…"))
    assert "il en reste 21 s" in str(err) and "Réessaie dans 19 h 29" in str(err)
    q = client.get("/api/status").json()["gpu"]["quota"]
    assert q["left_s"] == 21 and q["retry_at"] == pytest.approx(time.time() + 70150, abs=5)
    zg.consume_quota(15.4)                                               # calcul réussi entre-temps
    assert zg.quota_status()["left_s"] == 6
    zg.space_error(RuntimeError("Clé APP_KEY invalide."))                 # autre erreur : rien ne change
    assert zg.quota_status()["left_s"] == 6
    q["retry_at"] = time.time() - 1                                       # heure passée : quota revenu
    db.set_meta("zerogpu_quota", json.dumps(q))
    assert zg.quota_status() is None
    db.set_meta("zerogpu_quota", "")


def test_friendly_error_without_text_names_its_type():
    from app.worker.zerogpu_client import friendly

    class ReadError(Exception):
        pass

    assert friendly(ReadError()).endswith("ReadError")


def test_friendly_error_for_dropped_stream():
    import concurrent.futures

    from app.worker.zerogpu_client import friendly

    assert "Connexion avec le Space coupée" in friendly(concurrent.futures.CancelledError())


def test_remote_payload_sends_embeddings_not_photos():
    from app.worker.main import remote_payload
    from src.identity import SourceFace
    from src.levels import PersonAssets
    from src.pipeline import FaceMapping

    emb = np.ones(512, np.float32) / np.sqrt(512)
    person = PersonAssets(source=SourceFace(emb))
    params = {"level": "face", "stabilize": True, "mappings": [{"t": 22.0, "box": [0.1, 0.1, 0.3, 0.4], "person": "p1"}]}
    payload = remote_payload(params, [FaceMapping(person, {"t": 22.0, "box": [0.1, 0.1, 0.3, 0.4]}, "Visage 1")], 20.0, 6.0)
    assert payload["mappings"][0]["t"] == pytest.approx(2.0)          # relatif à l'extrait
    assert set(payload["people"]) == {"p1"}
    assert len(payload["people"]["p1"]["embedding"]) == 512 and payload["people"]["p1"]["tone"] is None
    assert "photos" not in json.dumps(payload)


class _FakeJob:
    def __init__(self, result, steps=3):
        self._result, self._steps, self.cancelled = result, steps, False

    def done(self):
        self._steps -= 1
        return self._steps < 0

    def status(self):
        return SimpleNamespace(progress_data=[SimpleNamespace(index=2 - max(self._steps, 0), length=10)])

    def cancel(self):
        self.cancelled = True
        return True

    def result(self):
        return self._result


def _fake_client(monkeypatch, job):
    import gradio_client

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def submit(self, *args, api_name):
            assert api_name == "/swap"
            return job

        def predict(self, key, api_name):
            return json.dumps({"version": "1", "levels": ["face"]})

    monkeypatch.setattr(gradio_client, "Client", FakeClient)
    monkeypatch.setattr(gradio_client, "handle_file", lambda p: p)


def test_client_swap_reports_progress_and_copies_result(monkeypatch, tmp_path):
    from app.worker.zerogpu_client import ZeroGPUClient

    produced = tmp_path / "remote.mp4"
    produced.write_bytes(b"video")
    _fake_client(monkeypatch, _FakeJob((str(produced), json.dumps({"frames": 10, "swapped": 9, "reused": 1, "gpu_seconds": 3.5}))))
    seen = []
    stats = ZeroGPUClient("kevin/x", "tok", "key").swap(tmp_path / "clip.mp4", {}, tmp_path / "out.mp4",
                                                        on_progress=lambda d, t: seen.append((d, t)), poll=0)
    assert (tmp_path / "out.mp4").read_bytes() == b"video"
    assert stats["gpu_seconds"] == 3.5 and seen and seen[-1][1] == 10


def test_client_swap_cancel(monkeypatch, tmp_path):
    from app.worker.zerogpu_client import RemoteCancelled, ZeroGPUClient

    job = _FakeJob(("x", "{}"), steps=10)
    _fake_client(monkeypatch, job)
    with pytest.raises(RemoteCancelled):
        ZeroGPUClient("kevin/x", "tok", "key").swap(tmp_path / "c.mp4", {}, tmp_path / "o.mp4", should_cancel=lambda: True, poll=0)
    assert job.cancelled


def test_pro_account_is_detected_and_raises_the_daily_quota(client, monkeypatch):
    import huggingface_hub

    from app import db
    from app.worker import zerogpu_client as zg

    calls = []

    class FakeApi:
        def __init__(self, token=None):
            pass

        def whoami(self):
            calls.append(1)
            return {"name": "kevin", "isPro": True}

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    db.set_meta("zerogpu_pro", "")
    client.put("/api/settings/zerogpu", json={"space": "kevin/faceswap-gpu", "token": "hf_x", "key": "k"})
    assert zg.account_pro() is None and client.get("/api/status").json()["gpu"]["free_quota_s"] == 300
    zg.refresh_pro()
    zg.refresh_pro()                                                      # relu au plus une fois par heure
    gpu = client.get("/api/status").json()["gpu"]
    assert calls == [1] and gpu["pro"] is True and gpu["free_quota_s"] == 40 * 60
    db.set_meta("zerogpu_pro", "")
    client.delete("/api/settings/zerogpu")

