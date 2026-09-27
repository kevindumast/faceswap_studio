"""Recherche des personnes d'un passage : répartition des images et repérage des doublons probables."""
import numpy as np

from src.scan import Person, _flag_probable_duplicates, sample_times


def _person(emb, seen: int, score: float) -> Person:
    return Person(t=float(seen), box=[0, 0, 0.1, 0.1], score=score, height_px=100, crop=np.zeros((2, 2, 3), np.uint8),
                  embedding_sum=np.asarray(emb, np.float64), times=set(range(seen)))


def test_sample_times_cover_the_passage():
    times = sample_times(200.0, 220.0, 12)
    assert len(times) == 12
    assert 200.0 < times[0] < 201 and 219 < times[-1] < 220.0
    assert all(b > a for a, b in zip(times, times[1:]))


def test_blurry_single_sighting_similar_to_main_person_is_flagged():
    rng = np.random.default_rng(0)
    a = rng.normal(size=512)
    a /= np.linalg.norm(a)
    b = rng.normal(size=512)
    b /= np.linalg.norm(b)
    profile_of_a = a * 0.2 + rng.normal(size=512) / np.sqrt(512)          # ressemble un peu à A (~0,2)
    people = [_person(a, seen=7, score=0.84), _person(b, seen=5, score=0.84), _person(profile_of_a, seen=1, score=0.49)]
    _flag_probable_duplicates(people)
    assert [p.maybe_same for p in people] == [None, None, 0]


def test_clear_secondary_character_is_not_flagged():
    """Vu une seule fois mais net (confiance élevée) : c'est un vrai personnage secondaire, on le garde."""
    rng = np.random.default_rng(1)
    a = rng.normal(size=512)
    a /= np.linalg.norm(a)
    other = a * 0.2 + rng.normal(size=512) / np.sqrt(512)
    people = [_person(a, seen=6, score=0.9), _person(other, seen=1, score=0.85)]
    _flag_probable_duplicates(people)
    assert people[1].maybe_same is None


def _face(bbox, score=0.8):
    from insightface.app.common import Face

    x1, y1, x2, y2 = bbox
    kps = np.array([[x1, y1], [x2, y1], [(x1 + x2) / 2, (y1 + y2) / 2], [x1, y2], [x2, y2]], np.float32)
    return Face(bbox=np.array(bbox, np.float32), kps=kps, det_score=np.float32(score))


def test_scan_keeps_every_sighting_and_every_analysed_frame(monkeypatch):
    """Chaque apparition d'une personne est gardée (pour la montrer sur chaque image), et toutes les images analysées
    sont renvoyées, même sans visage : on peut y indiquer un visage oublié."""
    from src import scan

    rng = np.random.default_rng(2)
    a, b = rng.normal(size=512), rng.normal(size=512)
    calls = []

    def fake_detect(img, size=None, thresh=None):
        calls.append(1)
        k = len(calls) % 3          # une image sur trois : personne ; sinon A seule, ou A et B
        return [] if k == 0 else [_face([10, 10, 60, 70])] if k == 1 else [_face([10, 10, 60, 70]), _face([120, 20, 160, 70])]

    def fake_embed(img, face):
        face.embedding = a if face.bbox[0] < 100 else b

    monkeypatch.setattr(scan.media, "extract_frame", lambda src, t, h=None: np.zeros((100, 200, 3), np.uint8))
    monkeypatch.setattr(scan, "detect_boxes", fake_detect)
    monkeypatch.setattr(scan, "embed", fake_embed)
    people, frames = scan.scan_passage(None, 0.0, 12.0)
    assert len(frames) == 12
    assert [len(p.seen_at) for p in people] == [8, 4]
    assert sorted(t for t, _ in people[0].seen_at) == sorted(people[0].times)
    assert people[1].seen_at[0][1] == [0.6, 0.2, 0.8, 0.7]           # boîte normalisée (image de 200 × 100)


def test_region_detection_zooms_and_maps_back_to_the_full_frame(monkeypatch):
    from src import faces

    frame = np.zeros((560, 1280, 3), np.uint8)
    crops = []

    def fake_detect(img, size=None, thresh=None):
        crops.append(img.shape[:2])
        # coordonnées dans le recadrage : un visage au centre de la zone, un autre hors de la zone
        return [_face([0, 0, 20, 20], 0.9), _face([60, 60, 100, 110])]

    monkeypatch.setattr(faces, "detect_boxes", fake_detect)
    monkeypatch.setattr(faces, "embed", lambda img, f: setattr(f, "embedding", np.ones(512)))
    face = faces.detect_in_region(frame, (0.3, 0.3, 0.35, 0.45), 640, 0.3)
    assert crops == [(210, 210)]                                        # zone de 64 × 84 px, recadrée avec de la marge
    assert face.bbox.tolist() == [371, 165, 411, 215]                   # rendue dans l'image entière
    assert face.kps[0].tolist() == [371, 165]
    assert face.normed_embedding is not None

    monkeypatch.setattr(faces, "detect_boxes", lambda img, size=None, thresh=None: [_face([0, 0, 20, 20], 0.9)])
    assert faces.detect_in_region(frame, (0.3, 0.3, 0.35, 0.45), 640, 0.3) is None


def test_reference_falls_back_to_zoomed_detection(monkeypatch):
    """Visage ajouté à la main, introuvable sur l'image entière : la référence du rendu est prise en zoomant."""
    from pathlib import Path
    from types import SimpleNamespace

    from src import pipeline

    emb = np.ones(512, np.float32) / np.sqrt(512)
    monkeypatch.setattr(pipeline.media, "extract_frame", lambda clip, t: np.zeros((360, 640, 3), np.uint8))
    monkeypatch.setattr(pipeline, "detect", lambda frame: [])
    monkeypatch.setattr(pipeline, "detect_in_region", lambda frame, box, size, thresh: SimpleNamespace(normed_embedding=emb))
    assert pipeline._reference_from_target(Path("clip.mp4"), {"t": 1.0, "box": [0.1, 0.1, 0.2, 0.2]}, (640, 360), {}) is emb
