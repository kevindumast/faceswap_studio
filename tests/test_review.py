"""Vérification avant l'assemblage : pistes de visages, problèmes signalés, décisions, recalcul des seules images choisies."""
import numpy as np
import pytest


def _face(x: float, m: int = -1, y: float = 100.0, size: float = 100.0, r=None) -> dict:
    kps = [x + 30, y + 40, x + 70, y + 40, x + 50, y + 60, x + 35, y + 80, x + 65, y + 80]
    f = {"b": [x, y, x + size, y + size], "k": kps, "s": 0.9, "m": m}
    if r is not None:
        f["r"] = r
    return f


def _plan(tmp_path, frames, mappings=None, fps=25.0):
    from src.review import FramePlan

    mappings = mappings or [{"label": "Visage 1", "person": "A", "active": True},
                            {"label": "Visage 2", "person": "B", "active": True}]
    return FramePlan(tmp_path / "plan.json", (640, 360), fps, mappings, frames)


def _entry(*faces, x=5.0):
    return {"f": list(faces), "x": x}


def test_tracks_link_faces_and_stop_at_scene_cuts(tmp_path):
    from src.review import build_tracks

    frames = [_entry(_face(100 + 2 * i, m=0)) for i in range(10)]
    frames.append(None)                                      # copie de l'image précédente
    frames += [_entry(_face(125, m=0)) for _ in range(3)]
    frames[5] = _entry()                                     # visage non détecté sur une image : même piste
    frames += [_entry(_face(125, m=-1), x=80.0)]             # changement de plan : autre personne au même endroit
    frames += [_entry(_face(125, m=-1)) for _ in range(9)]
    tracks = build_tracks(_plan(tmp_path, frames))
    assert len(tracks) == 2
    first, second = tracks
    assert (first.start, first.end) == (0, 13) and first.gaps() == [5]
    assert second.start == 14 and all(f["m"] == -1 for _, _, f in second.faces)


def test_issues_lost_mixed_missed(tmp_path):
    from src.review import analyze

    lost = [_entry(_face(100, m=0 if i % 5 else -1)) for i in range(20)]           # perdu 1 image sur 5
    mixed = [_entry(_face(100, m=0 if i < 12 else 1), x=80.0 if i == 0 else 5.0) for i in range(20)]
    missed = [_entry(_face(300, m=-1, r=[0.2, 1]), x=80.0 if i == 0 else 5.0) for i in range(20)]
    extra = [_entry(_face(300, m=-1, size=20, r=[0.05, 0]), x=80.0 if i == 0 else 5.0) for i in range(20)]
    plan = _plan(tmp_path, lost + mixed + missed + extra)
    _, issues = analyze(plan)
    kinds = [(x["kind"], x["minor"], x["suggest"], x["preselect"]) for x in issues]
    assert kinds == [("lost", False, "A", "A"), ("mixed", False, "A", "A"), ("missed", False, "B", None), ("missed", True, None, None)]
    lost_issue = issues[0]
    assert lost_issue["unreplaced"] == 4 and lost_issue["replaced"] == {"A": 16}
    assert lost_issue["bar"][0] == [0, 0, "none"] and lost_issue["bar"][1] == [1, 4, "A"]


def test_fully_replaced_track_is_not_an_issue(tmp_path):
    from src.review import analyze

    _, issues = analyze(_plan(tmp_path, [_entry(_face(100, m=0)) for _ in range(30)]))
    assert issues == []


def test_decisions_mark_frames_and_fill_detection_gaps(tmp_path):
    from src.review import analyze, apply_decisions

    frames = [_entry(_face(100 + i, m=0 if i < 6 else -1)) for i in range(12)]
    frames[8] = _entry()                                      # trou de détection au milieu du visage perdu
    plan = _plan(tmp_path, frames)
    _, issues = analyze(plan)
    (issue,) = issues
    count = apply_decisions(plan, [{"issue": issue["id"], "action": "assign", "person": "A"}], smoothing=0.0)
    assert count == 6                                         # 5 images d'origine + le trou
    assert all(f["m"] == 0 and f["v"] for i, e in enumerate(plan.frames) for f in e["f"])
    pending = plan.frames[8]["f"][0]
    assert pending["p"] == 1 and "k" in pending               # trou court : points clés interpolés en secours
    assert pending["b"][0] == pytest.approx(108, abs=0.1)
    assert [i for i, e in enumerate(plan.frames) if e.get("dirty")] == [6, 7, 8, 9, 10, 11]
    plan.save()
    _, issues = analyze(plan)
    assert issues == []                                       # décision prise : plus signalé


def test_keep_and_remove(tmp_path):
    from src.review import analyze, apply_decisions

    frames = [_entry(_face(100, m=0 if i < 3 else -1)) for i in range(10)]
    frames += [_entry(_face(400, m=-1, r=[0.3, 1]), x=80.0 if i == 0 else 5.0) for i in range(10)]
    plan = _plan(tmp_path, frames)
    _, issues = analyze(plan)
    lost, missed = issues
    assert apply_decisions(plan, [{"issue": missed["id"], "action": "keep"}], 0.6) == 0
    assert apply_decisions(plan, [{"issue": lost["id"], "action": "remove"}], 0.6) == 3
    assert all(f["m"] == -1 for e in plan.frames for f in e["f"])
    assert analyze(plan)[1] == []


def test_stale_decision_is_refused(tmp_path):
    from src.review import DecisionError, apply_decisions

    plan = _plan(tmp_path, [_entry(_face(100, m=-1)) for _ in range(10)])
    with pytest.raises(DecisionError):
        apply_decisions(plan, [{"issue": 99, "action": "keep"}], 0.6)
    with pytest.raises(DecisionError):
        apply_decisions(plan, [{"issue": 0, "action": "assign", "person": "Z"}], 0.6)


class _Paint:
    """Stratégie factice : peint le cadre du visage en blanc, et garde un état de couleur comme le vrai swap."""
    samples = 0

    def __init__(self):
        from src.swap import SwapState

        self.swap_state = SwapState()

    def apply(self, frame, face):
        out = frame.copy()
        x1, y1, x2, y2 = (int(v) for v in face.bbox)
        out[y1:y2, x1:x2] = 255
        self.swap_state.shift = np.array([1.0, 2.0, 3.0], np.float32)
        return out


def test_recompute_only_touches_chosen_frames(sample_video, tmp_path, monkeypatch):
    from src import media, review
    from src.review import FramePlan, analyze, apply_decisions, recompute

    clip = tmp_path / "cut.mp4"
    media.cut_segment(sample_video, clip, 1.0, 2.0, (640, 360), False)          # 25 images
    swapped = tmp_path / "swapped.mp4"
    media.ffmpeg("-i", str(clip), "-c", "copy", str(swapped))                    # « rendu » : aucun visage remplacé
    frames = [_entry(_face(100, m=-1, r=[0.4, 0])) for _ in range(25)]
    frames[3] = None                                                             # copie de l'image 2
    frames[10] = _entry()                                                        # trou de détection
    plan = _plan(tmp_path, frames)
    plan.save()
    monkeypatch.setattr(review, "_find", lambda *a, **k: None)                  # pas de détecteur : interpolation
    _, (issue,) = analyze(plan)
    apply_decisions(plan, [{"issue": issue["id"], "action": "assign", "person": "A"}], 0.0)
    plan.save()

    made = []

    def make(m):
        made.append(m)
        return _Paint()

    assert recompute(clip, swapped, plan, make) == 24                           # 25 images moins la copie
    assert made == [0]
    fixed = FramePlan.load(tmp_path)
    assert not any(e and e.get("dirty") for e in fixed.frames)
    assert fixed.frames[10]["f"][0]["k"] and "p" not in fixed.frames[10]["f"][0]
    assert fixed.frames[5]["f"][0]["c"] == [1.0, 2.0, 3.0]
    import cv2

    cap = cv2.VideoCapture(str(swapped))
    means = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        means.append(float(img[110:190, 110:190].mean()))
    cap.release()
    assert len(means) == 25 and min(means) > 245                                # la copie (image 3) suit aussi


def test_plan_recorded_by_swap_segment_and_resumed(sample_video, tmp_path):
    """Le journal suit le rendu image par image, y compris après une pause (jamais plus court que les morceaux)."""
    from src import media
    from src.identity import SourceFace
    from src.levels import PersonAssets
    from src.pipeline import Checkpoint, FaceMapping, Paused, RenderError, swap_segment
    from src.review import FramePlan

    clip = tmp_path / "clip.mp4"
    media.cut_segment(sample_video, clip, 1.0, 2.0, (640, 360), False)          # 25 images, mire sans visage
    person = PersonAssets(source=SourceFace(np.ones(512, np.float32) / np.sqrt(512)))
    plan = FramePlan(tmp_path / "plan.json", (640, 360), 25.0, [{"label": "Visage 1", "person": "A", "active": True}])

    def pause_at_10(stage, done, total):
        if done >= 10:
            raise Paused()

    with pytest.raises(Paused):
        swap_segment(clip, [FaceMapping(person)], "face", tmp_path / "swapped.mp4", progress=pause_at_10,
                     checkpoint=Checkpoint.load(tmp_path), plan=plan)
    saved = FramePlan.load(tmp_path)
    assert len(saved.frames) == Checkpoint.load(tmp_path).frames_done == 10
    with pytest.raises(RenderError):                                             # aucun visage dans la mire
        swap_segment(clip, [FaceMapping(person)], "face", tmp_path / "swapped.mp4",
                     checkpoint=Checkpoint.load(tmp_path), plan=saved)
    assert len(saved.frames) == 25
    assert all(e is None or e["f"] == [] for e in saved.frames)
    assert any(e and "x" in e for e in saved.frames)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from app.api.main import app

    with TestClient(app) as c:
        yield c


def test_review_api_flow(client, sample_video, make_session):
    """En vérification : analyse, choix → recalcul (worker), retour, assemblage refusé tant que tout n'est pas calculé."""
    import time

    from app import db
    from src import media

    set_id, ids = make_session({"Kevin": [np.ones(512)]})
    vid = db.new_id()
    db.insert("videos", id=vid, kind="upload", title="mire", status="ready", progress=1, source=str(sample_video),
              info=media.probe(sample_video).to_dict(), created_at=time.time())
    job = client.post("/api/jobs", json={"video_id": vid, "face_set_id": set_id, "start": 1, "end": 7, "consent": True,
                                         "review": True, "mappings": [{"t": 1, "box": [0.1, 0.1, 0.3, 0.4], "person": ids["Kevin"]}]}).json()
    assert job["params"]["review"] is True
    work = db.folder("jobs", job["id"])
    work.mkdir(parents=True, exist_ok=True)
    media.cut_segment(sample_video, work / "cut.mp4", 1.0, 2.0, (640, 360), False)
    media.ffmpeg("-i", str(work / "cut.mp4"), "-c", "copy", str(work / "swapped.mp4"))
    from src.review import FramePlan

    frames = [_entry(_face(100, m=0 if i % 4 else -1)) for i in range(25)]
    FramePlan(work / "plan.json", (640, 360), 25.0, [{"label": "Visage 1", "person": ids["Kevin"], "active": True}], frames).save()
    db.update("jobs", job["id"], status="review", stage="review")

    r = client.get(f"/api/jobs/{job['id']}/review").json()
    assert r["frames"] == 25 and r["pending"] == 0 and len(r["boxes"]) == 25
    (issue,) = r["issues"]
    assert issue["kind"] == "lost" and issue["preselect"] == ids["Kevin"]
    crop = client.get(f"/api/jobs/{job['id']}/review/crop.jpg", params={"frame": issue["thumb"]["frame"], "x1": 100, "y1": 100, "x2": 200, "y2": 200})
    assert crop.status_code == 200 and crop.headers["content-type"] == "image/jpeg"
    assert client.get(f"/api/jobs/{job['id']}/swapped.mp4").status_code == 200

    stale = client.post(f"/api/jobs/{job['id']}/review", json={"version": "0", "decisions": []})
    assert stale.status_code == 409
    queued = client.post(f"/api/jobs/{job['id']}/review", json={
        "version": r["version"], "decisions": [{"issue": issue["id"], "action": "assign", "person": ids["Kevin"]}]}).json()
    assert queued["status"] == "queued" and queued["params"]["review_step"] == "fix" and queued["pausable"] is False
    back = client.post(f"/api/jobs/{job['id']}/cancel").json()                    # pas encore commencé : retour
    assert back["status"] == "review" and "review_step" not in back["params"]
    r = client.get(f"/api/jobs/{job['id']}/review").json()
    assert r["pending"] == 7 and r["issues"] == []                                # choix gardés, à recalculer
    refused = client.post(f"/api/jobs/{job['id']}/assemble")
    assert refused.status_code == 409 and "recalcul" in refused.json()["detail"]

    db.update("jobs", job["id"], status="done")
    assert client.get(f"/api/jobs/{job['id']}").json()["reviewable"] is True
    assert client.post(f"/api/jobs/{job['id']}/reopen").json()["status"] == "review"
    client.post(f"/api/jobs/{job['id']}/cancel")
    assert client.get(f"/api/jobs/{job['id']}").json()["status"] == "cancelled"


def test_removal_starts_again_from_the_original(sample_video, tmp_path, monkeypatch):
    """Ajout : posé sur l'image déjà rendue (les autres visages ne sont pas refaits) ; retrait : repart de l'original."""
    import cv2

    from src import media
    from src.review import FramePlan, analyze, apply_decisions, recompute

    clip = tmp_path / "cut.mp4"
    media.cut_segment(sample_video, clip, 1.0, 2.0, (640, 360), False)          # 25 images
    swapped = tmp_path / "swapped.mp4"
    media.ffmpeg("-i", str(clip), "-c", "copy", str(swapped))
    # Visage de gauche remplacé par erreur (sur toute la séquence), visage de droite jamais remplacé.
    frames = [_entry(_face(100, m=0), _face(400, m=-1, r=[0.4, 1])) for _ in range(25)]
    plan = _plan(tmp_path, frames)
    plan.save()
    applied = []

    class Count(_Paint):
        def apply(self, frame, face):
            applied.append(int(face.bbox[0]))
            return super().apply(frame, face)

    tracks, (missed,) = analyze(plan)
    left = next(t for t in tracks if t.faces[0][2]["b"][0] == 100)
    apply_decisions(plan, [{"issue": missed["id"], "action": "assign", "person": "B"}], 0.0)
    recompute(clip, swapped, plan, lambda m: Count())
    assert set(applied) == {400}                                                 # visage de gauche pas refait
    plan = FramePlan.load(tmp_path)
    applied.clear()
    apply_decisions(plan, [{"issue": left.id, "action": "remove"}], 0.0)
    assert all(e.get("orig") for e in plan.frames)
    recompute(clip, swapped, plan, lambda m: Count())
    assert set(applied) == {400}                                                 # refait depuis l'original : seul le droit reste
    cap = cv2.VideoCapture(str(swapped))
    ok, img = cap.read()
    cap.release()
    assert ok and img[110:190, 410:490].mean() > 245 and img[110:190, 110:190].mean() < 245
