"""Vérification avant l'assemblage : journal du rendu image par image, visages mal suivis, corrections ciblées.

Pendant le rendu, chaque image note ses visages détectés (boîte, points clés, score), l'association qui a remplacé
chacun (-1 : aucune), la ressemblance mesurée avec les cibles et l'état de couleur du swap (plan.json).

Après le rendu, les visages sont reliés d'une image à l'autre en « pistes » : un même visage dans un même plan
(recouvrement des boîtes, trous de détection courts tolérés, jamais à travers un changement de plan). Une piste
remplacée en partie (visage perdu en route), remplacée par deux personnes, ou jamais remplacée est signalée.
Pour chacune, on choisit : remplacer par une personne sur toute la piste, ne rien remplacer, ou laisser tel quel.
Seules les images concernées sont recalculées depuis l'extrait d'origine ; le reste de la vidéo est recopié.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import cv2
import numpy as np

from . import media
from .config import load_config
from .faces import detect_boxes, detect_in_region, iou

PLAN = "plan.json"
STATS = "stats.json"


def _settings() -> dict:
    d = {"scene_cut": 30.0, "link_gap_s": 0.3, "min_face_px": 32, "min_s": 0.2, "suggest_similarity": 0.15,
         "interpolate_s": 0.2, "det_thresh": 0.2, "crf": 14}
    d.update(load_config().get("review", {}) or {})
    return d


# --------------------------------------------------------------------------------------------------------------
# Journal du rendu


@dataclass
class FramePlan:
    """plan.json : une entrée par image de l'extrait, None pour une copie de l'image précédente.

    Entrée : {"f": [visage…], "x": écart avec l'image précédente (changement de plan), "dirty": 1 si à recalculer,
              "orig": 1 si le recalcul doit repartir de l'image d'origine (un remplacement retiré ou changé)}.
    Visage : {"b": boîte px, "k": 5 points clés px, "s": score, "m": association qui le remplace (-1 : aucune),
              "r": [ressemblance, association la plus proche], "c": couleur du swap après cette image,
              "v": 1 si une décision a été prise, "p": 1 si à retrouver au recalcul (trou de détection),
              "n": 1 si remplacement ajouté (posé sur l'image déjà rendue, sans refaire les autres visages)}.
    """
    path: Path
    size: tuple[int, int]
    fps: float
    mappings: list[dict]               # {"label", "person", "active"} dans l'ordre des associations du rendu
    frames: list[dict | None] = field(default_factory=list)

    @classmethod
    def load(cls, work_dir: Path) -> "FramePlan | None":
        path = work_dir / PLAN
        if not path.is_file():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(path, tuple(data["size"]), float(data["fps"]), data["mappings"], data["frames"])

    def save(self) -> None:
        data = {"size": list(self.size), "fps": self.fps, "mappings": self.mappings, "frames": self.frames}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
        tmp.replace(self.path)

    @property
    def version(self) -> str:
        """Change à chaque enregistrement : une décision prise sur une ancienne analyse est refusée."""
        return str(self.path.stat().st_mtime_ns) if self.path.is_file() else "0"

    def truncate(self, count: int) -> None:
        """Reprise après une pause : on oublie les images qui ne sont pas dans les morceaux enregistrés."""
        del self.frames[count:]

    def add_duplicate(self) -> None:
        self.frames.append(None)

    def add(self, faces: list, picks: list, references: dict[int, np.ndarray], diff: float | None) -> None:
        """faces : visages détectés ; picks : (visage, index d'association, stratégie) remplacés ;
        references : empreinte de chaque cible suivie, pour noter la ressemblance des visages déjà analysés."""
        chosen = {id(face): (m, strategy) for face, m, strategy in picks}
        records = []
        for face in faces:
            m, strategy = chosen.get(id(face), (-1, None))
            rec = {"b": [round(float(v), 1) for v in face.bbox[:4]],
                   "k": [round(float(v), 1) for v in np.asarray(face.kps, np.float32).ravel()],
                   "s": round(float(getattr(face, "det_score", 0.0) or 0.0), 2), "m": m}
            if face.embedding is not None and references:
                sims = {k: float(np.dot(face.normed_embedding, ref)) for k, ref in references.items()}
                best = max(sims, key=sims.get)
                rec["r"] = [round(sims[best], 3), best]
            shift = _shift_of(strategy)
            if shift is not None:
                rec["c"] = [round(float(v), 2) for v in shift]
            records.append(rec)
        entry: dict = {"f": records}
        if diff is not None:
            entry["x"] = round(diff, 1)
        self.frames.append(entry)

    def view(self) -> list[tuple[int, list[dict]]]:
        """(image source, visages) pour chaque image : une copie reprend les visages (mêmes objets) de l'image répétée."""
        out, src, faces = [], 0, []
        for i, entry in enumerate(self.frames):
            if entry is not None:
                src, faces = i, entry["f"]
            out.append((src, faces))
        return out

    def cuts(self) -> set[int]:
        """Images qui ouvrent un nouveau plan : écart bien plus fort que celui des images voisines."""
        threshold = float(_settings()["scene_cut"])
        diffs = [e.get("x", 0.0) if e else 0.0 for e in self.frames]
        out = set()
        for i, d in enumerate(diffs):
            if d <= threshold:
                continue
            around = [diffs[j] for j in (i - 2, i - 1, i + 1, i + 2) if 0 <= j < len(diffs)]
            if d > 2.5 * max(float(np.median(around)) if around else 0.0, 4.0):
                out.add(i)
        return out


def _shift_of(strategy) -> np.ndarray | None:
    state = getattr(strategy, "swap_state", None)
    return None if state is None else state.shift


# --------------------------------------------------------------------------------------------------------------
# Pistes et problèmes


@dataclass
class Track:
    """Un même visage dans un même plan : (image, image source, visage) dans l'ordre, trous de détection compris."""
    id: int
    faces: list[tuple[int, int, dict]] = field(default_factory=list)

    @property
    def start(self) -> int:
        return self.faces[0][0]

    @property
    def end(self) -> int:
        return self.faces[-1][0]

    def at(self) -> dict[int, dict]:
        return {i: f for i, _, f in self.faces}

    def gaps(self) -> list[int]:
        seen = {i for i, _, _ in self.faces}
        return [i for i in range(self.start, self.end + 1) if i not in seen]


def build_tracks(plan: FramePlan) -> list[Track]:
    """Relie les visages d'une image à l'autre (recouvrement des boîtes), jamais à travers un changement de plan."""
    cfg = _settings()
    max_gap = max(1, round(float(cfg["link_gap_s"]) * plan.fps))
    cuts = plan.cuts()
    tracks: list[Track] = []
    open_: list[Track] = []
    for i, (src, faces) in enumerate(plan.view()):
        if i in cuts:
            open_ = []
        open_ = [t for t in open_ if i - t.end <= max_gap]
        pairs = []
        for t in open_:
            last = t.faces[-1][2]
            for k, f in enumerate(faces):
                overlap = iou(last["b"], f["b"])
                same = last["m"] >= 0 and last["m"] == f["m"]
                if overlap > 0.3 or (same and overlap > 0.05):
                    pairs.append((overlap + (0.5 if same else 0.0), t.id, k, t))
        pairs.sort(key=lambda p: (-p[0], p[1], p[2]))
        linked_t, linked_f = set(), set()
        for _, tid, k, t in pairs:
            if tid in linked_t or k in linked_f:
                continue
            t.faces.append((i, src, faces[k]))
            linked_t.add(tid)
            linked_f.add(k)
        for k, f in enumerate(faces):
            if k not in linked_f:
                t = Track(len(tracks), [(i, src, f)])
                tracks.append(t)
                open_.append(t)
    return tracks


def _height(face: dict) -> float:
    return face["b"][3] - face["b"][1]


def _runs(values: list) -> list[list]:
    """[[début, fin, valeur]] des suites de valeurs égales (fin incluse), pour la barre d'une piste."""
    runs: list[list] = []
    for i, v in enumerate(values):
        if runs and runs[-1][2] == v and runs[-1][1] == i - 1:
            runs[-1][1] = i
        else:
            runs.append([i, i, v])
    return runs


def issue_of(track: Track, mappings: list[dict], fps: float) -> dict | None:
    """Ce qui cloche sur une piste (None : rien, ou décision déjà prise). Personnes = identifiants de la bibliothèque."""
    cfg = _settings()
    faces = [f for _, _, f in track.faces]
    if all(f.get("v") for f in faces):
        return None
    person_of = [m["person"] for m in mappings]
    by_person: dict[str, int] = {}
    for f in faces:
        if f["m"] >= 0:
            by_person[person_of[f["m"]]] = by_person.get(person_of[f["m"]], 0) + 1
    unreplaced = sum(1 for f in faces if f["m"] < 0)
    gaps = track.gaps()
    height = max(_height(f) for f in faces)
    length = track.end - track.start + 1
    minor = False
    if len(by_person) > 1:
        kind = "mixed"
        suggest = max(by_person, key=by_person.get)
    elif by_person:
        (person, count), = by_person.items()
        # Trous de détection entre deux images remplacées : visage qui réapparaît d'origine une fraction de seconde.
        swapped = [i for i, _, f in track.faces if f["m"] >= 0]
        holes = [g for g in gaps if swapped[0] < g < swapped[-1]]
        if not unreplaced and not holes:
            return None
        kind, suggest = "lost", person
    else:
        if length < float(cfg["min_s"]) * fps:
            return None
        kind = "missed"
        scores: dict[int, list[float]] = {}
        for f in faces:
            if "r" in f:
                scores.setdefault(int(f["r"][1]), []).append(float(f["r"][0]))
        best = max(scores, key=lambda m: float(np.mean(scores[m])), default=None)
        suggest = None
        if best is not None and float(np.mean(scores[best])) >= float(cfg["suggest_similarity"]):
            suggest = person_of[best]
        minor = suggest is None or height < float(cfg["min_face_px"])
    # Vignette : le plus grand visage non remplacé (celui qu'il faut reconnaître), sinon le plus grand.
    pool = [(i, f) for i, _, f in track.faces if f["m"] < 0] or [(i, f) for i, _, f in track.faces]
    thumb, face = max(pool, key=lambda p: _height(p[1]))
    at = track.at()
    states = []
    for i in range(track.start, track.end + 1):
        f = at.get(i)
        states.append("gap" if f is None else person_of[f["m"]] if f["m"] >= 0 else "none")
    replaced = sum(by_person.values())
    return {
        "id": track.id, "kind": kind, "minor": minor, "start": track.start, "end": track.end, "frames": length,
        "replaced": by_person, "unreplaced": unreplaced, "undetected": len(gaps), "height": round(height),
        "suggest": suggest,
        # Décision proposée d'office : seulement quand la piste est déjà surtout remplacée par cette personne.
        "preselect": suggest if kind != "missed" and by_person.get(suggest, 0) >= 0.5 * max(1, replaced + unreplaced) else None,
        "thumb": {"frame": thumb, "box": face["b"]},
        "bar": [[a - track.start, b - track.start, s] for a, b, s in _runs(states)],
    }


def analyze(plan: FramePlan) -> tuple[list[Track], list[dict]]:
    tracks = build_tracks(plan)
    issues = [x for x in (issue_of(t, plan.mappings, plan.fps) for t in tracks) if x is not None]
    issues.sort(key=lambda x: (x["start"], x["id"]))
    return tracks, issues


def summary(plan: FramePlan) -> dict:
    """Compte des problèmes restants (avertissements de fin de rendu)."""
    _, issues = analyze(plan)
    major = [x for x in issues if not x["minor"]]
    return {"lost": sum(x["kind"] in ("lost", "mixed") for x in major), "missed": sum(x["kind"] == "missed" for x in major),
            "minor": len(issues) - len(major)}


def warnings_of(plan: FramePlan) -> list[str]:
    s = summary(plan)
    out = []
    if s["lost"]:
        out.append(f"{s['lost']} visage{'s' if s['lost'] > 1 else ''} perdu{'s' if s['lost'] > 1 else ''} par moments "
                   "(il réapparaît d'origine quelques images) : « Revoir les images » pour les corriger.")
    if s["missed"]:
        out.append(f"{s['missed']} visage{'s' if s['missed'] > 1 else ''} ressemblant à une cible non "
                   f"remplacé{'s' if s['missed'] > 1 else ''} : « Revoir les images » pour choisir.")
    return out


def frame_boxes(plan: FramePlan, tracks: list[Track]) -> list[list[list]]:
    """Pour chaque image : [x1, y1, x2, y2 (fractions), piste, association] de chaque visage, pour l'aperçu."""
    w, h = plan.size
    track_of = {id(f): t.id for t in tracks for _, _, f in t.faces}
    out = []
    for _, faces in plan.view():
        out.append([[round(f["b"][0] / w, 4), round(f["b"][1] / h, 4), round(f["b"][2] / w, 4), round(f["b"][3] / h, 4),
                     track_of.get(id(f), -1), f["m"]] for f in faces])
    return out


# --------------------------------------------------------------------------------------------------------------
# Décisions


class DecisionError(ValueError):
    pass


def mapping_for(plan: FramePlan, track: Track, person: str) -> int:
    """Association à utiliser pour remplacer une piste par cette personne : celle qui la remplace déjà en partie,
    sinon la première association de cette personne suivie pendant le rendu."""
    used = [f["m"] for _, _, f in track.faces if f["m"] >= 0 and plan.mappings[f["m"]]["person"] == person]
    if used:
        return max(set(used), key=used.count)
    candidates = [k for k, m in enumerate(plan.mappings) if m["person"] == person]
    if not candidates:
        raise DecisionError("Cette personne ne fait pas partie de ce rendu.")
    active = [k for k in candidates if plan.mappings[k].get("active", True)]
    return (active or candidates)[0]


def apply_decisions(plan: FramePlan, decisions: list[dict], smoothing: float) -> int:
    """Applique les choix au plan (sans rien calculer) ; renvoie le nombre d'images à recalculer.

    decision : {"issue": id de piste, "action": "assign" | "remove" | "keep", "person": id (assign)}.
    """
    tracks = {t.id: t for t in build_tracks(plan)}
    cfg = _settings()
    dirty: set[int] = set()
    original: set[int] = set()     # images à refaire depuis l'original (un remplacement disparaît ou change)
    for d in decisions:
        track = tracks.get(int(d["issue"]))
        if track is None:
            raise DecisionError("Cette vérification n'est plus à jour : recharge la page.")
        action = d.get("action")
        if action == "keep":
            for _, _, f in track.faces:
                f["v"] = 1
        elif action == "remove":
            for _, src, f in track.faces:
                if f["m"] >= 0:
                    f["m"] = -1
                    f.pop("c", None)
                    f.pop("n", None)
                    dirty.add(src)
                    original.add(src)
                f["v"] = 1
        elif action == "assign":
            m = mapping_for(plan, track, str(d.get("person")))
            dirty |= _assign(plan, track, m, smoothing, round(float(cfg["interpolate_s"]) * plan.fps), original)
        else:
            raise DecisionError(f"Action inconnue : {action}")
    for i in dirty:
        plan.frames[i]["dirty"] = 1
    for i in original:
        plan.frames[i]["orig"] = 1
    return len(dirty)


def _assign(plan: FramePlan, track: Track, m: int, smoothing: float, interpolate: int, original: set[int]) -> set[int]:
    from .temporal import TargetTracker

    dirty = set()
    # Points clés lissés dans le temps comme au rendu (les visages déjà remplacés servent de repère, inchangés).
    smoother = TargetTracker(np.zeros(1, np.float32), 1.0, smoothing)
    for _, src, f in track.faces:
        if "k" not in f:                   # visage encore à retrouver (recalcul interrompu) : il le sera au prochain
            f["m"], f["v"], f["n"] = m, 1, 1
            dirty.add(src)
            continue
        view = SimpleNamespace(bbox=np.asarray(f["b"], np.float32), kps=np.asarray(f["k"], np.float32).reshape(5, 2))
        if f["m"] == m:
            smoother.prev_kps, smoother.prev_bbox = view.kps, view.bbox
        else:
            smoother.smooth(view)
            if f["m"] >= 0:
                original.add(src)          # remplacé par quelqu'un d'autre : à refaire depuis l'original
            else:
                f["n"] = 1
            f["b"] = [round(float(v), 1) for v in view.bbox]
            f["k"] = [round(float(v), 1) for v in np.asarray(view.kps).ravel()]
            f["m"] = m
            f.pop("c", None)
            dirty.add(src)
        f["v"] = 1
    # Images où le visage n'a pas été détecté : il sera cherché à la position attendue au recalcul (seuil plus bas) ;
    # sur un trou court, à défaut, points clés interpolés entre les deux images voisines.
    known = [(i, f) for i, _, f in track.faces]
    for g in track.gaps():
        if plan.frames[g] is None:
            continue
        before = max((p for p in known if p[0] < g), key=lambda p: p[0])
        after = min((p for p in known if p[0] > g), key=lambda p: p[0])
        a = (g - before[0]) / (after[0] - before[0])
        box = (1 - a) * np.asarray(before[1]["b"]) + a * np.asarray(after[1]["b"])
        pending = {"b": [round(float(v), 1) for v in box], "s": 0.0, "m": m, "p": 1, "v": 1, "n": 1}
        if after[0] - before[0] - 1 <= interpolate:
            kps = (1 - a) * np.asarray(before[1]["k"]) + a * np.asarray(after[1]["k"])
            pending["k"] = [round(float(v), 1) for v in kps]
        plan.frames[g]["f"].append(pending)
        dirty.add(g)
    return dirty


# --------------------------------------------------------------------------------------------------------------
# Recalcul des images corrigées


def _face(f: dict) -> SimpleNamespace:
    return SimpleNamespace(bbox=np.asarray(f["b"], np.float32), kps=np.asarray(f["k"], np.float32).reshape(5, 2),
                           det_score=f.get("s", 0.0), embedding=None)


def _find(frame: np.ndarray, pending: dict, size: tuple[int, int], thresh: float) -> dict | None:
    """Visage attendu à la position d'un trou de détection : image entière à seuil bas, puis zoom sur la zone."""
    box = pending["b"]
    faces = detect_boxes(frame, thresh=thresh)
    best = max(faces, key=lambda f: iou(f.bbox, box), default=None)
    if best is None or iou(best.bbox, box) < 0.3:
        w, h = size
        sc = load_config().get("scan", {})
        best = detect_in_region(frame, [box[0] / w, box[1] / h, box[2] / w, box[3] / h], int(sc.get("det_size", 640)), thresh)
    if best is None:
        return None
    return {"b": [round(float(v), 1) for v in best.bbox[:4]],
            "k": [round(float(v), 1) for v in np.asarray(best.kps, np.float32).ravel()],
            "s": round(float(best.det_score), 2)}


def _prepare(clip: Path, plan: FramePlan, strategies: dict[int, object]) -> None:
    """Pré-passe des stratégies qui en ont une (teint, décor) : images où l'association remplace déjà un visage."""
    todo = {m: s for m, s in strategies.items() if getattr(s, "samples", 0) > 0}
    if not todo:
        return
    wanted: dict[int, list[tuple[int, dict]]] = {}
    for m, s in todo.items():
        spots = [(i, f) for i, e in enumerate(plan.frames) if e for f in e["f"] if f["m"] == m and "k" in f]
        if spots:
            picks = np.linspace(0, len(spots) - 1, num=min(s.samples, len(spots))).round().astype(int)
            for k in sorted(set(picks.tolist())):
                wanted.setdefault(spots[k][0], []).append((m, spots[k][1]))
    samples: dict[int, list] = {m: [] for m in todo}
    cap = cv2.VideoCapture(str(clip))
    try:
        for i in range(max(wanted, default=-1) + 1):
            if not cap.grab():
                break
            if i in wanted:
                ok, frame = cap.retrieve()
                if ok:
                    for m, f in wanted[i]:
                        samples[m].append((frame, _face(f)))
    finally:
        cap.release()
    for m, s in todo.items():
        s.prepare(samples[m])


def recompute(clip: Path, swapped: Path, plan: FramePlan, make: Callable[[int], object],
              progress: Callable[[str, int, int], None] | None = None) -> int:
    """Recalcule les images marquées dans le plan et réécrit swapped.mp4.

    Un remplacement seulement ajouté est posé sur l'image déjà rendue (les autres visages ne sont pas recalculés) ;
    un remplacement retiré ou changé fait refaire l'image depuis l'original, tous ses visages compris.

    make(m) : stratégie neuve de l'association m. Une stratégie par piste (son état de couleur ne se mélange pas
    à celui d'un autre visage), repartant de l'état noté au rendu pour l'image précédente de la piste.
    """
    report = progress or (lambda *_: None)
    cfg = _settings()
    rcfg = load_config().render
    dirty = [i for i, e in enumerate(plan.frames) if e and e.get("dirty")]
    if not dirty:
        return 0
    base: dict[int, object] = {}
    for i in dirty:
        for f in plan.frames[i]["f"]:
            if f["m"] >= 0 and f["m"] not in base:
                base[f["m"]] = make(f["m"])
    _prepare(clip, plan, base)

    tracks = build_tracks(plan)
    where: dict[int, tuple[Track, int]] = {}       # id(visage) → (piste, rang dans la piste)
    for t in tracks:
        for k, (_, _, f) in enumerate(t.faces):
            where.setdefault(id(f), (t, k))
    per_track: dict[int, tuple[object, int]] = {}  # piste → (stratégie, dernière image calculée)

    def strategy_for(f: dict, i: int):
        t, k = where.get(id(f), (None, 0))
        key = t.id if t is not None else -1 - id(f)
        strategy, last = per_track.get(key, (None, -2))
        if strategy is None:
            strategy = copy.deepcopy(base[f["m"]])
        prev = next((p for _, _, p in reversed(t.faces[:k]) if p["m"] == f["m"]), None) if t is not None else None
        if prev is not None and last != i - 1 and hasattr(strategy, "swap_state"):
            c = prev.get("c")
            strategy.swap_state.shift = None if c is None else np.asarray(c, np.float32)
        per_track[key] = (strategy, i)
        return strategy

    size = plan.size
    total = len(plan.frames)
    thresh = float(cfg["det_thresh"])
    tmp = swapped.with_name("swapped_fix.mp4")
    cap_o, cap_s = cv2.VideoCapture(str(clip)), cv2.VideoCapture(str(swapped))
    info = media.probe(swapped)
    writer = media.FrameWriter(tmp, size, info.fps_str, int(cfg["crf"]), str(rcfg.preset))
    last_out, last_dirty = None, False
    try:
        for i in range(total):
            ok_o, orig = cap_o.read()
            ok_s, cur = cap_s.read()
            if not (ok_o and ok_s):
                break
            entry = plan.frames[i]
            if entry is None:
                out = last_out if last_dirty and last_out is not None else cur
            elif entry.get("dirty"):
                kept = []
                for f in entry["f"]:
                    if f.get("p"):
                        found = _find(orig, f, size, thresh)
                        if found is not None:
                            f.update(found)
                        elif "k" not in f:
                            continue               # introuvable et trou trop long pour interpoler : laissé d'origine
                        f.pop("p", None)
                    kept.append(f)
                entry["f"] = kept
                from_original = bool(entry.get("orig"))
                out = orig if from_original else cur
                for f in kept:
                    if f["m"] >= 0 and (from_original or f.get("n")):
                        strategy = strategy_for(f, i)
                        out = strategy.apply(out, _face(f))
                        shift = _shift_of(strategy)
                        if shift is not None:
                            f["c"] = [round(float(v), 2) for v in shift]
                    f.pop("n", None)
                entry.pop("dirty", None)
                entry.pop("orig", None)
                last_out, last_dirty = out, True
            else:
                out, last_dirty = cur, False
            writer.write(out)
            report("fix", i + 1, total)
        writer.close()
    except BaseException:
        writer.abort()
        tmp.unlink(missing_ok=True)
        raise
    finally:
        cap_o.release()
        cap_s.release()
    tmp.replace(swapped)
    plan.save()
    return len(dirty)


def save_stats(work_dir: Path, stats) -> None:
    from dataclasses import asdict

    (work_dir / STATS).write_text(json.dumps(asdict(stats)), encoding="utf-8")


def load_stats(work_dir: Path):
    from .pipeline import RenderStats

    path = work_dir / STATS
    return RenderStats(**json.loads(path.read_text(encoding="utf-8"))) if path.is_file() else RenderStats()


def needs_review(work_dir: Path) -> bool:
    plan = FramePlan.load(work_dir)
    if plan is None:
        return False
    s = summary(plan)
    return bool(s["lost"] or s["missed"])
