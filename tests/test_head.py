"""Niveau 3 (tête complète) sans les modèles : choix de la photo, lissage du mouvement, fond propre, composition."""
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from src import head, parsing

KPS = np.array([[270, 270], [330, 270], [300, 305], [275, 335], [325, 335]], np.float32)


def test_frontal_photo_wins_over_profile_and_small_face():
    bbox = np.array([0, 0, 200, 260], np.float32)
    frontal = head.frontal_score(KPS, bbox)
    turned = KPS.copy()
    turned[2, 0] += 25                                   # nez décalé : tête tournée
    assert head.frontal_score(turned, bbox) < frontal * 0.5
    assert head.frontal_score(KPS, bbox * 0.3) < frontal


def _motion(yaw: float, mouth: float = 0.0) -> head.Motion:
    exp = np.zeros((1, 21, 3), np.float32)
    exp[0, 0, 1] = mouth
    return head.Motion(0.0, yaw, 0.0, np.ones((1, 1), np.float32), np.zeros((1, 3), np.float32), exp,
                       np.zeros((1, 21, 3), np.float32))


def test_motion_smoother_steadies_at_rest_and_follows_fast_turns():
    smooth = head.MotionSmoother(0.5)
    rng = np.random.default_rng(0)
    raw = [float(v) for v in rng.normal(0, 1.0, 40)]    # tremblement de ±1°
    out = [smooth(_motion(v)).yaw for v in raw]
    assert np.abs(np.diff(out)).mean() < np.abs(np.diff(raw)).mean() * 0.8
    smooth = head.MotionSmoother(0.5)
    for i in range(6):
        assert smooth(_motion(15.0 * i)).yaw == pytest.approx(15.0 * i)   # 15°/image : pas de retard


def test_plate_crop_maps_the_background_into_the_aligned_crop():
    rng = np.random.default_rng(1)
    image = rng.integers(0, 255, (300, 300, 3), dtype=np.uint8)
    plate = head.Plate(image, x0=100, y0=50)
    M = np.array([[1.0, 0, -100], [0, 1.0, -50]], np.float64)        # recadrage = image entière décalée de (100, 50)
    crop = plate.crop(M)
    assert np.array_equal(crop[:300, :300], image)


def _scene(center_x: int, cam_dx: int = 0):
    """Décor rayé fixe (ou décalé de cam_dx : caméra qui bouge) + tête rouge qui se déplace."""
    frame = np.zeros((600, 800, 3), np.uint8)
    for x in range(0, 800, 40):
        frame[:, x:x + 20] = (60, 160, 60)
    frame[:, :] = np.roll(frame, cam_dx, axis=1)
    cv2.circle(frame, (center_x, 300), 90, (40, 40, 220), -1)
    face = SimpleNamespace(kps=KPS + [center_x - 300, 0], bbox=np.array([center_x - 60, 230, center_x + 60, 380]))
    return frame, face


NECK = (150, 170, 210)


def _parse_red(crop):
    labels = np.full(crop.shape[:2], parsing.BACKGROUND, np.uint8)
    labels[(crop[..., 2] > 180) & (crop[..., 1] < 100)] = parsing.HAIR     # rouge : l'ancienne tête
    labels[(crop[..., 0] > 180) & (crop[..., 1] < 100)] = parsing.HAIR     # bleu : la nouvelle
    labels[np.abs(crop.astype(int) - NECK).sum(axis=2) < 40] = parsing.SKIN   # cou, que BiSeNet classe « peau »
    return labels


@pytest.fixture
def no_lama(monkeypatch):
    """Pixels jamais visibles : effaceur d'OpenCV (LaMa sur CPU prend ~4 s)."""
    def run(name, feed):
        raise head.ModelsMissing(name)

    monkeypatch.setattr(head, "run", run)


def test_plate_rebuilds_the_background_hidden_by_the_moving_head(monkeypatch, no_lama):
    monkeypatch.setattr(head.parsing, "parse", _parse_red)
    samples = [_scene(x) for x in range(200, 601, 50)]
    plate = head.build_plate(samples, scale=1.5)
    assert plate is not None
    clean, _ = _scene(-500)                                              # le décor seul
    y, x = 300 - plate.y0, 400 - plate.x0                                # là où la tête est passée
    assert np.abs(plate.image[y, x].astype(int) - clean[300, 400].astype(int)).max() < 10


def test_no_plate_when_the_camera_moves(monkeypatch, no_lama):
    monkeypatch.setattr(head.parsing, "parse", _parse_red)
    samples = [_scene(300 + 20 * i, cam_dx=17 * i) for i in range(10)]
    assert head.build_plate(samples, scale=1.5) is None


def _fake_models(monkeypatch):
    """LivePortrait simulé : la « nouvelle tête » est un disque bleu plus petit que l'ancienne, au centre."""
    generated = np.zeros((512, 512, 3), np.uint8)
    generated[:] = (60, 160, 60)
    cv2.circle(generated, (256, 256), 50, (220, 40, 40), -1)
    out = (generated[:, :, ::-1].astype(np.float32) / 255).transpose(2, 0, 1)[None]

    def run(name, feed):
        if name == "motion_extractor":
            m = _motion(0.0)
            return [np.float32(0), np.float32(0), np.float32(0), m.scale, m.translation, m.expression, m.points]
        if name == "stitcher":
            return [feed["source"]]
        if name == "generator":
            return [out]
        return [np.zeros((1, 32, 16, 64, 64), np.float32)]

    monkeypatch.setattr(head, "run", run)


def test_engine_replaces_the_head_and_fills_the_rest_from_the_plate(monkeypatch):
    _fake_models(monkeypatch)
    monkeypatch.setattr(head.parsing, "parse", _parse_red)
    frame, face = _scene(400)
    clean, _ = _scene(-500)
    source = head.SourceHead.from_photo(frame, face.kps, 1.5)
    engine = head.HeadEngine(source, scale=1.5, smoothing=0.0)
    engine.plate = head.Plate(clean, 0, 0)
    out = engine.apply(frame, face)
    red = (out[..., 2] > 180) & (out[..., 1] < 100)
    blue = (out[..., 0] > 180) & (out[..., 1] < 100)
    assert red.sum() < 0.02 * (np.pi * 90 ** 2)                          # l'ancienne tête a disparu
    ys, xs = np.nonzero(blue)
    assert blue.sum() > 2000 and abs(xs.mean() - 400) < 10 and 230 < ys.mean() < 300   # la nouvelle, à sa place
    ring = np.zeros(red.shape, np.uint8)
    cv2.circle(ring, (400, 300), 85, 1, 6)
    ring = ring.astype(bool)
    assert np.median(np.abs(out[ring].astype(int) - clean[ring].astype(int))) < 8   # décor reconstruit autour
    assert np.array_equal(out[:, :150], frame[:, :150])                  # loin de la tête : intact


def test_engine_keeps_the_neck_but_erases_the_old_cheek(monkeypatch):
    _fake_models(monkeypatch)
    monkeypatch.setattr(head.parsing, "parse", _parse_red)
    frame, face = _scene(400)
    frame[370:470, 370:430] = NECK                                       # cou sous l'ancienne tête
    frame[275:325, 495:525] = NECK                                       # joue de 3/4 qui dépasse, au-dessus du menton
    clean, _ = _scene(-500)
    engine = head.HeadEngine(head.SourceHead.from_photo(frame, face.kps, 1.5), scale=1.5, smoothing=0.0)
    engine.plate = head.Plate(clean, 0, 0)
    out = engine.apply(frame, face)
    neck = out[400:470, 380:420].astype(int)
    assert np.abs(neck - NECK).sum(axis=2).mean() < 20                   # le cou n'est pas remplacé par le décor
    cheek = (slice(285, 315), slice(502, 518))
    assert np.median(np.abs(out[cheek].astype(int) - clean[cheek].astype(int))) < 8   # la joue, si


def test_head_level_is_refused_with_the_gpu_option():
    from fastapi import HTTPException

    from app.api.routes_jobs import check_level

    with pytest.raises(HTTPException) as err:
        check_level("head", use_gpu=True, active_mappings=1)
    assert "carte graphique" in err.value.detail
