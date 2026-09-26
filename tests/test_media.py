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
