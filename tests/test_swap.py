"""Recollage maison du visage généré : zone couverte, masque doux, couleur ramenée sur celle du visage d'origine."""
import cv2
import numpy as np

from src import swap


def _lab_shift(img: np.ndarray, shift) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32) + np.asarray(shift, np.float32)
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def test_color_shift_brings_fake_back_to_original():
    rng = np.random.default_rng(0)
    original = np.clip(np.full((128, 128, 3), (80, 110, 160), np.int16) + rng.integers(-6, 7, (128, 128, 3)),
                       0, 255).astype(np.uint8)
    fake = _lab_shift(original, (-8, -4, -6))                     # inswapper plus sombre et plus gris
    shift = swap.color_shift(fake, original)
    assert np.allclose(shift, (8, 4, 6), atol=1.5)
    fixed = swap._shift_lab(fake, shift)
    zone = swap.color_zone()
    assert np.abs(fixed[zone].astype(int).mean(axis=0) - original[zone].astype(int).mean(axis=0)).max() < 2


def test_color_shift_is_bounded():
    original = np.full((128, 128, 3), (80, 110, 160), np.uint8)
    fake = np.full((128, 128, 3), (160, 60, 40), np.uint8)       # couleur absurde : correction bornée
    shift = swap.color_shift(fake, original)
    assert (np.abs(shift) <= swap.MAX_SHIFT + 1e-6).all()


def test_paste_covers_only_the_face_area_with_soft_edges():
    frame = np.zeros((300, 400, 3), np.uint8)
    fake = np.full((swap.SIZE, swap.SIZE, 3), 200, np.uint8)
    M = np.array([[0.5, 0, -50], [0, 0.5, -40]], np.float64)     # visage de 256 px en (100, 80) → espace aligné 128
    out = swap.paste(frame, fake, M, swap.paste_mask(0.1, 0.02))
    assert out[80 + 128, 100 + 128].min() > 190                  # centre du visage : visage généré
    assert out[:70].max() == 0 and out[:, :90].max() == 0         # hors de la zone : intact
    edge = out[80 + 128, 100 + 20:100 + 40, 0]                    # bord : transition progressive
    assert edge.min() < 100 and edge.max() > 100
    assert frame.max() == 0                                       # l'image d'entrée n'est pas modifiée


def test_color_fix_leaves_the_background_of_the_square_alone():
    img = np.full((swap.SIZE, swap.SIZE, 3), (60, 170, 60), np.uint8)       # fond vert uni
    out = swap._shift_lab(img, np.array([15, 5, 5], np.float32), swap.face_zone())
    assert np.abs(out[:6, :6].astype(int) - img[:6, :6].astype(int)).max() <= 1      # coins : intacts
    assert np.abs(out[70, 64].astype(int) - img[70, 64].astype(int)).max() > 5       # centre du visage : corrigé


def test_occlusion_mask_keeps_what_hides_the_face(monkeypatch):
    """Objet devant la bouche (classé « fond » par la segmentation) : on n'y recolle pas le visage généré."""
    from src import parsing, swap

    def fake_parse(img):
        labels = np.full(img.shape[:2], parsing.SKIN, np.uint8)
        h, w = labels.shape
        labels[int(h * 0.6):, : w // 2] = parsing.BACKGROUND          # main devant le bas gauche du visage
        return labels

    monkeypatch.setattr(parsing, "_session", lambda: None)
    monkeypatch.setattr(parsing, "parse", fake_parse)
    frame = np.zeros((300, 300, 3), np.uint8)
    M = np.array([[1.0, 0, -50], [0, 1.0, -50]], np.float32)
    m = swap.occlusion_mask(frame, M, 128)
    assert m.shape == (128, 128)
    assert m[110:, :40].max() < 0.05 and m[20:60, 20:100].min() > 0.95


def test_occlusion_mask_off_without_segmentation(monkeypatch):
    from src import parsing, swap
    from src.faces import ModelsMissing

    def missing():
        raise ModelsMissing("absent")

    monkeypatch.setattr(parsing, "_session", missing)
    assert swap.occlusion_mask(np.zeros((64, 64, 3), np.uint8), np.eye(2, 3, dtype=np.float32), 128) is None
