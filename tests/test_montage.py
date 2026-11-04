"""Montage : enchaînement de rendus en vertical, effets, masques qui suivent les visages, son."""
import time
from types import SimpleNamespace

import numpy as np
import pytest

from src import media, montage


def _count(path) -> int:
    out = media.run([media.binary("ffprobe"), "-v", "error", "-count_frames", "-select_streams", "v",
                     "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)]).stdout
    return int(out.strip())


@pytest.fixture(scope="module")
def clips(sample_video, tmp_path_factory):
    d = tmp_path_factory.mktemp("montage")
    a, b = d / "a.mp4", d / "b.mp4"
    info = media.probe(sample_video)
    media.cut_segment(sample_video, a, 1.0, 2.0, (640, 360), info.has_audio)       # 1 s à 25 i/s
    media.cut_segment(sample_video, b, 3.0, 4.6, (640, 360), False)                # 1,6 s (40 images) sans son
    return [montage.Clip(a, "Papa"), montage.Clip(b, "")]


def _fake_face(x=320.0, y=180.0, d=40.0):
    kps = np.array([[x - d / 2, y], [x + d / 2, y], [x, y + d * 0.5], [x - d * 0.4, y + d], [x + d * 0.4, y + d]], np.float32)
    return SimpleNamespace(kps=kps, bbox=np.array([x - d, y - d, x + d, y + d * 1.5], np.float32))


def test_vertical_montage_with_title_outro_and_music(clips, tmp_path):
    music = tmp_path / "music.wav"
    media.ffmpeg("-f", "lavfi", "-i", "sine=frequency=440:duration=1.5", str(music))     # plus courte : bouclée
    opts = montage.MontageOptions(title="Joyeux Noël", subtitle="2026", outro="Fin", snow=True, grade="warm",
                                  vignette=True, zoom=True, mask="santa", music=music, transition="flash")
    seen = []
    info = montage.render(clips, opts, tmp_path / "out.mp4", lambda d, t: seen.append((d, t)),
                          detect=lambda f: [_fake_face()])
    expected = round(montage.TITLE_S * 30) + 30 + 48 + round(montage.OUTRO_S * 30)
    assert info["frames"] == expected and (info["width"], info["height"]) == (1080, 1920)
    out = media.probe(tmp_path / "out.mp4")
    assert (out.width, out.height) == (1080, 1920) and out.has_audio
    assert _count(tmp_path / "out.mp4") == expected
    assert out.duration == pytest.approx(expected / 30, abs=0.05)
    assert seen[-1] == (expected, expected)
    assert not (tmp_path / ".out_work").exists()


def test_square_crop_without_extras(clips, tmp_path):
    opts = montage.MontageOptions(format="1:1", fit="crop", ai_label=False, transition="cut", clip_audio=False)
    info = montage.render(clips, opts, tmp_path / "sq.mp4", detect=lambda f: [])
    assert info["frames"] == 78 and media.probe(tmp_path / "sq.mp4").width == 1080


def test_unknown_option_is_refused(clips, tmp_path):
    with pytest.raises(montage.MontageError):
        montage.render(clips, montage.MontageOptions(mask="licorne"), tmp_path / "x.mp4")


def test_masks_are_drawn_on_the_face():
    for kind in ("santa", "reindeer", "nose"):
        img = np.zeros((360, 640, 3), np.uint8)
        montage.draw_mask(img, kind, _fake_face().kps)
        assert img.any(), kind
        ys, xs = np.nonzero(img.any(axis=2))
        assert abs(xs.mean() - 320) < 40                         # centré sur le visage
    hat = np.zeros((360, 640, 3), np.uint8)
    montage.draw_mask(hat, "santa", _fake_face().kps)
    assert np.nonzero(hat.any(axis=2))[0].mean() < 180         # le bonnet est au-dessus des yeux
    red = hat[..., 2].astype(int) - hat[..., 1]
    assert (red > 100).sum() > 500                             # bien rouge


def test_face_tracking_is_smoothed():
    faces = [[_fake_face(300)], [_fake_face(340)]]
    fm = montage.FaceMasks("nose", detect=lambda f: faces.pop(0), smoothing=0.5, every=1)
    frame = np.zeros((360, 640, 3), np.uint8)
    fm.apply(frame)
    fm.apply(frame)
    assert len(fm.tracks) == 1 and fm.tracks[0].kps[:, 0].mean() == pytest.approx(320, abs=1)


def test_layout_blur_keeps_whole_video():
    frame = np.full((360, 640, 3), 200, np.uint8)
    canvas, (x0, y0, x1, y1) = montage.layout(frame, 1080, 1920, "blur")
    assert canvas.shape == (1920, 1080, 3) and (x1 - x0, y1 - y0) == (1080, 608)


def test_api_montage_flow(sample_video, tmp_path):
    from fastapi.testclient import TestClient

    from app import db
    from app.api.main import app

    with TestClient(app) as client:
        job_id = db.new_id()
        d = db.folder("jobs", job_id)
        d.mkdir(parents=True)
        media.cut_segment(sample_video, d / "result.mp4", 1.0, 2.0, (640, 360), True)
        db.insert("jobs", id=job_id, video_id="x", face_set_id="y", params={}, status="done", created_at=time.time())
        bad = client.post("/api/montages", json={"clips": [{"job_id": "inconnu"}]})
        assert bad.status_code == 404
        music = client.post("/api/montages/music", files={"file": ("x.txt", b"no", "text/plain")})
        assert music.status_code == 415
        body = {"clips": [{"job_id": job_id, "caption": "Test"}], "title": "Noël", "ai_label": False}
        prev = client.post("/api/montages/preview", json=body)
        assert prev.status_code == 200 and prev.json()["image"].startswith("data:image/jpeg")
        m = client.post("/api/montages", json=body).json()
        for _ in range(120):
            m = client.get(f"/api/montages/{m['id']}").json()
            if m["status"] in ("done", "error"):
                break
            time.sleep(0.25)
        assert m["status"] == "done", m["error"]
        assert m["info"]["frames"] == 90 + 30 and m["result_url"]
        assert client.get(m["download_url"]).headers["content-disposition"].startswith("attachment")
        assert client.delete(f"/api/montages/{m['id']}").json()["ok"]
