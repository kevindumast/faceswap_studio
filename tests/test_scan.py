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
