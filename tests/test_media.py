import pytest

from src import media
from src.fetch import FetchError, normalize_url


def test_probe(sample_video):
    info = media.probe(sample_video)
    assert (info.width, info.height) == (640, 360)
    assert info.fps == pytest.approx(25)
    assert info.duration == pytest.approx(8, abs=0.1)
    assert info.has_audio


def test_cut_is_frame_accurate(sample_video, tmp_path):
    info = media.probe(sample_video)
    out = tmp_path / "cut.mp4"
    media.cut_segment(sample_video, out, 2.0, 7.0, (640, 360), info.has_audio)
    cut = media.probe(out)
    assert cut.duration == pytest.approx(5.0, abs=1 / 25 + 0.03)  # ±1 image (+ marge conteneur audio)


def _frames(path) -> int:
    out = media.run([media.binary("ffprobe"), "-v", "error", "-count_frames", "-select_streams", "v",
                     "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)]).stdout
    return int(out.strip())


@pytest.mark.parametrize("label", [False, True])
def test_assemble_segment_keeps_every_frame(sample_video, tmp_path, label):
    """Son remis sur l'extrait swappé : aucune image perdue à la fin (-shortest en perdait 3 sur 25)."""
    info = media.probe(sample_video)
    cut = tmp_path / "cut.mp4"
    media.cut_segment(sample_video, cut, 1.0, 2.0, (640, 360), info.has_audio)
    swapped = tmp_path / "swapped.mp4"
    media.ffmpeg("-i", str(cut), "-an", "-c:v", "libx264", "-preset", "veryfast", str(swapped))
    png = media.render_label(tmp_path / "label.png", 360) if label else None
    out = tmp_path / "out.mp4"
    media.assemble_segment(swapped, cut, out, True, png, 18, "veryfast")
    assert _frames(out) == _frames(swapped) == 25
    assert media.probe(out).has_audio


def test_proxy_and_filmstrip(sample_video, tmp_path):
    info = media.probe(sample_video)
    seen = []
    media.make_proxy(sample_video, tmp_path / "proxy.mp4", info, 240, seen.append)
    proxy = media.probe(tmp_path / "proxy.mp4")
    assert proxy.height == 240 and proxy.codec == "h264"
    assert seen and seen[-1] > 0.9
    meta = media.make_filmstrip(tmp_path / "proxy.mp4", tmp_path / "f.jpg", tmp_path / "f.json", info, 20, 60)
    assert (tmp_path / "f.jpg").stat().st_size > 0
    assert meta["cols"] * meta["rows"] >= 20 and meta["tile_h"] == 60


def test_extract_frame(sample_video):
    frame = media.extract_frame(sample_video, 3.0)
    assert frame.shape == (360, 640, 3)
    assert media.extract_frame(sample_video, 3.0, 240).shape == (240, 426, 3)  # comme scaled_size


@pytest.mark.parametrize("size, max_res, expected", [
    ((1920, 1080), 720, (1280, 720)),
    ((1080, 1920), 720, (720, 1280)),    # vertical : 720p = 720 de large, pas 720 de haut
    ((1080, 1920), 1080, (1080, 1920)),
    ((360, 640), 1080, (360, 640)),
])
def test_scaled_size_uses_short_side(size, max_res, expected):
    info = media.VideoInfo(10, *size, 25, "25/1", True, "h264")
    assert media.scaled_size(info, max_res) == expected


@pytest.mark.parametrize("url, expected", [
    ("https://www.youtube.com/watch?v=x9yop0nYR9g&list=RDx9yop0nYR9g&start_radio=1",
     "https://www.youtube.com/watch?v=x9yop0nYR9g"),
    ("https://youtu.be/x9yop0nYR9g?si=abc", "https://www.youtube.com/watch?v=x9yop0nYR9g"),
    ("https://www.youtube.com/shorts/abcDEF123", "https://www.youtube.com/watch?v=abcDEF123"),
    ("https://vimeo.com/123", "https://vimeo.com/123"),
])
def test_normalize_url(url, expected):
    assert normalize_url(url) == expected


def test_normalize_url_rejects_garbage():
    with pytest.raises(FetchError):
        normalize_url("pas une url")
