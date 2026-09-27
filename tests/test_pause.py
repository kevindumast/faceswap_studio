"""Pause / reprise : le rendu repart à la même image et la vidéo finale est complète."""
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient


def count_frames(path) -> int:
    """Images réellement présentes (et pas la durée de l'en-tête)."""
    import subprocess

    from src.media import binary

    out = subprocess.run([binary("ffprobe"), "-v", "error", "-count_frames", "-select_streams", "v:0",
                          "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return int(out.strip())


def test_pause_then_resume_gives_complete_video(sample_video, tmp_path):
    from src import media
    from src.identity import SourceFace
    from src.levels import PersonAssets
    from src.pipeline import Checkpoint, FaceMapping, Paused, RenderError, swap_segment

    clip = tmp_path / "clip.mp4"
    media.cut_segment(sample_video, clip, 1.0, 4.0, (640, 360), False)          # 3 s à 25 i/s = 75 images
    person = PersonAssets(source=SourceFace(np.ones(512, np.float32) / np.sqrt(512)))
    out = tmp_path / "swapped.mp4"

    def pause_at_20(stage, done, total):
        if done >= 20:
            raise Paused()

    with pytest.raises(Paused):
        swap_segment(clip, [FaceMapping(person)], "face", out, progress=pause_at_20, checkpoint=Checkpoint.load(tmp_path))
    saved = Checkpoint.load(tmp_path)
    assert saved.frames_done == 20 and saved.chunks == ["chunk_000.mp4"]
    assert not out.exists()

    # Reprise : la mire n'a pas de visage, donc fin « aucun visage », mais la vidéo recollée doit être complète.
    with pytest.raises(RenderError):
        swap_segment(clip, [FaceMapping(person)], "face", out, checkpoint=saved)
    assert Checkpoint.load(tmp_path).chunks == ["chunk_000.mp4", "chunk_001.mp4"]
    info = media.probe(out)
    assert count_frames(out) == 75
    assert info.duration == pytest.approx(3.0, abs=0.05) and info.fps == pytest.approx(25)


def test_partial_preview_while_paused(sample_video, tmp_path):
    """En pause : ce qui est déjà rendu se revoit, avec le son d'origine, et disparaît une fois le rendu fini."""
    from src import media
    from src.identity import SourceFace
    from src.levels import PersonAssets
    from src.pipeline import Checkpoint, FaceMapping, Paused, partial_preview, swap_segment

    assert partial_preview(tmp_path) is None                                    # rien de rendu : pas d'aperçu
    clip = tmp_path / "cut.mp4"
    media.cut_segment(sample_video, clip, 1.0, 4.0, (640, 360), True)           # 75 images, avec son
    person = PersonAssets(source=SourceFace(np.ones(512, np.float32) / np.sqrt(512)))

    def pause_at_20(stage, done, total):
        if done >= 20:
            raise Paused()

    with pytest.raises(Paused):
        swap_segment(clip, [FaceMapping(person)], "face", tmp_path / "swapped.mp4", progress=pause_at_20,
                     checkpoint=Checkpoint.load(tmp_path))
    preview = partial_preview(tmp_path)
    assert preview.name == "partial_20.mp4" and count_frames(preview) == 20
    info = media.probe(preview)
    assert info.has_audio and info.duration == pytest.approx(0.8, abs=0.1)
    built = preview.stat().st_mtime_ns
    assert partial_preview(tmp_path).stat().st_mtime_ns == built                # pas refait à chaque requête du lecteur

    Checkpoint.load(tmp_path).clear()
    assert not list(tmp_path.glob("partial_*"))


def test_crash_resumes_from_last_saved_chunk(sample_video, tmp_path, monkeypatch):
    """Arrêt brutal (PC éteint) : on repart du dernier morceau refermé, sans image en double ni manquante."""
    from src import media
    from src.config import load_config
    from src.identity import SourceFace
    from src.levels import PersonAssets
    from src.pipeline import Checkpoint, FaceMapping, RenderError, swap_segment

    monkeypatch.setitem(load_config()["render"], "checkpoint_every_s", 0)     # un morceau refermé à chaque image
    clip = tmp_path / "clip.mp4"
    media.cut_segment(sample_video, clip, 1.0, 2.0, (640, 360), False)          # 25 images
    person = PersonAssets(source=SourceFace(np.ones(512, np.float32) / np.sqrt(512)))
    out = tmp_path / "swapped.mp4"

    def crash_at_10(stage, done, total):
        if done >= 10:
            raise KeyboardInterrupt()                                            # ni pause ni annulation

    with pytest.raises(KeyboardInterrupt):
        swap_segment(clip, [FaceMapping(person)], "face", out, progress=crash_at_10, checkpoint=Checkpoint.load(tmp_path))
    saved = Checkpoint.load(tmp_path)
    assert saved.frames_done == 9 and len(saved.chunks) == 9

    monkeypatch.setitem(load_config()["render"], "checkpoint_every_s", 3600)
    with pytest.raises(RenderError):                                             # la mire n'a pas de visage
        swap_segment(clip, [FaceMapping(person)], "face", out, checkpoint=saved)
    assert Checkpoint.load(tmp_path).frames_done == 25
    assert count_frames(out) == 25                                               # ni perdue ni en double


@pytest.fixture(scope="module")
def client():
    from app.api.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def ready_video(client, sample_video):
    from app import db
    from src import media

    vid = db.new_id()
    db.insert("videos", id=vid, kind="upload", title="mire", status="ready", progress=1,
              source=str(sample_video), info=media.probe(sample_video).to_dict(), created_at=time.time())
    return vid


def _job(client, video, set_id, pid, **extra):
    return client.post("/api/jobs", json={"video_id": video, "face_set_id": set_id, "start": 1, "end": 7, "consent": True,
                                          "mappings": [{"t": 1, "box": [0.1, 0.1, 0.3, 0.4], "person": pid}], **extra}).json()


def test_pause_resume_api(client, ready_video, make_session):
    set_id, ids = make_session({"Kevin": [np.ones(512)]})
    job = _job(client, ready_video, set_id, ids["Kevin"])
    assert job["pausable"] is True
    paused = client.post(f"/api/jobs/{job['id']}/pause").json()
    assert paused["status"] == "paused"
    assert paused["partial_url"] is None                                         # mis en pause avant la 1re image
    assert client.get(f"/api/jobs/{job['id']}/partial.mp4").status_code == 404
    assert client.post(f"/api/jobs/{job['id']}/resume").json()["status"] == "queued"
    assert client.get(f"/api/jobs/{job['id']}/partial.mp4").status_code == 409    # seulement en pause
    assert client.post(f"/api/jobs/{job['id']}/resume").status_code == 409            # plus en pause
    client.post(f"/api/jobs/{job['id']}/pause")
    assert client.post(f"/api/jobs/{job['id']}/cancel").json()["status"] == "cancelled"


def test_gpu_job_cannot_be_paused(client, ready_video, make_session):
    set_id, ids = make_session({"Kevin": [np.ones(512)]})
    client.put("/api/settings/zerogpu", json={"space": "kevin/faceswap-gpu", "token": "t", "key": "k"})
    job = _job(client, ready_video, set_id, ids["Kevin"], use_gpu=True)
    assert job["pausable"] is False
    r = client.post(f"/api/jobs/{job['id']}/pause")
    assert r.status_code == 409 and "ZeroGPU" in r.json()["detail"]
    client.post(f"/api/jobs/{job['id']}/cancel")
    client.delete("/api/settings/zerogpu")
