"""Worker de rendu : process séparé qui dépile les jobs SQLite et appelle src.pipeline.

Lancement : python -m app.worker.main
Pour un GPU distant, il suffit de lancer ce même worker sur la machine GPU avec data/ partagé.
"""
from __future__ import annotations

import time
import traceback
from pathlib import Path

from app import db
from app.api.routes_faces import load_person_assets
from app.worker.zerogpu_client import RemoteCancelled, ZeroGPUClient, wait_until_ready
from src import media, review
from src.character import Reference, choose_reference, face_ratio, recompose
from src.config import accelerator, load_config
from src.levels import CHARACTER, FACE, make_strategy
from src.pipeline import (Cancelled, Checkpoint, FaceMapping, Paused, RenderOptions, RenderStats, assemble, cut, finish,
                          relative_targets, render_frames)


def remote_payload(params: dict, mappings: list[FaceMapping], start: float, length: float) -> dict:
    """Ce qui part au Space : empreintes et teint des personnes (pas leurs photos), cibles en temps relatif à l'extrait."""
    people = {}
    local = relative_targets(mappings, start, length)
    items = []
    for m, fm in zip(params["mappings"], local):
        pid = m["person"]
        if pid not in people:
            people[pid] = {"embedding": fm.person.source.normed_embedding.astype(float).tolist(),
                           "tone": fm.person.tone.to_dict() if fm.person.tone is not None else None}
        items.append({"t": fm.target["t"], "box": fm.target["box"], "person": pid, "label": fm.label})
    return {"level": params.get("level") or FACE, "stabilize": bool(params["stabilize"]), "mappings": items, "people": people}


# Étapes annoncées par le Space du niveau 4 (desc de sa progression) → étapes affichées dans l'appli.
CHARACTER_STEPS = {"squelette": "pose", "silhouette": "mask", "génération": "generate"}


def remote_stages(progress, steps: dict[str, str] | None = None):
    """Relaie les étapes d'un appel au Space : file d'attente, attente d'un GPU, et (niveau 4) ses propres étapes."""

    def on_stage(stage: str, done: int, total: int, desc: str | None) -> None:
        if stage == "progress":
            if steps and desc in steps:
                progress(steps[desc], done, total)
        else:
            progress(stage, done, total)

    return on_stage


def wake_space(kind: str, job_id: str, progress) -> None:
    """Réveille le Space s'il dort et attend qu'il soit prêt ; étape « wake » : secondes écoulées / durée typique."""
    wait_until_ready(kind, on_wait=lambda elapsed, expected: progress("wake", int(elapsed), int(expected)),
                     should_cancel=lambda: db.job_status(job_id) == "cancelling")


def run_remote(job: dict, source: Path, opts: RenderOptions, mappings: list[FaceMapping], out: Path, progress) -> RenderStats:
    """Découpe et assemblage sur le PC, remplacement des visages sur le Space ZeroGPU."""
    stats = RenderStats()
    t0 = time.perf_counter()
    progress("cut", 0, 1)
    segment = cut(source, opts, out)
    silent = out / "cut_silent.mp4"          # le son ne part pas : il est remis à l'assemblage
    media.ffmpeg("-i", str(segment.path), "-an", "-c:v", "copy", str(silent))
    progress("cut", 1, 1)

    payload = remote_payload(job["params"], mappings, opts.start, opts.end - opts.start)
    try:
        wake_space("faces", job["id"], progress)
        client = ZeroGPUClient.from_settings()
        remote = client.swap(silent, payload, out / "swapped.mp4",
                             on_progress=lambda done, total: progress("swap", done, total),
                             should_cancel=lambda: db.job_status(job["id"]) == "cancelling",
                             on_stage=remote_stages(progress))
    except RemoteCancelled as exc:
        raise Cancelled() from exc
    silent.unlink(missing_ok=True)
    stats.frames, stats.swapped, stats.reused = remote.get("frames", 0), remote.get("swapped", 0), remote.get("reused", 0)
    stats.warnings = list(remote.get("warnings", []))
    gpu_seconds = float(remote.get("gpu_seconds", 0))
    today = time.strftime("%Y-%m-%d")
    db.set_meta(f"zerogpu_used:{today}", str(float(db.get_meta(f"zerogpu_used:{today}") or 0) + gpu_seconds))

    progress("assemble", 0, 1)
    assemble(source, segment, out / "swapped.mp4", opts, out / "result.mp4", out)
    progress("assemble", 1, 1)
    stats.seconds = time.perf_counter() - t0
    stats.sec_per_frame = stats.seconds / max(1, stats.frames)
    stats.sec_per_computed = gpu_seconds / max(1, stats.frames - stats.reused)   # temps GPU = ce qui coûte du quota
    return stats


def _reference_warning(name: str, framing: str) -> str | None:
    if framing == "portrait":
        return (f"{name} : pas de photo en pied, le corps et les habits ont été inventés. "
                "Ajoute une photo en pied dans la bibliothèque pour un meilleur résultat.")
    if framing == "half":
        return (f"{name} : photo à mi-corps seulement, le bas du corps (pantalon, chaussures) a été inventé. "
                "Une photo en pied, de la tête aux pieds, donne un meilleur résultat.")
    return None


def run_character(job: dict, source: Path, opts: RenderOptions, mappings: list[FaceMapping], out: Path,
                  progress) -> RenderStats:
    """Niveau 4 : découpe et assemblage sur le PC, personnes entières générées sur le Space « niveau 4 ».

    Wan-Animate remplace une personne par passage : avec deux personnes, le 2e passage part de la vidéo où la 1re est
    déjà remplacée (recollée en pleine résolution entre les deux). Chaque passage consomme son propre temps de GPU.
    Plusieurs personnes : masque qui suit la silhouette (grille) plutôt qu'un rectangle, pour ne pas abîmer la voisine.
    """
    params = job["params"]
    ccfg = load_config().levels.character
    rcfg = load_config().render
    stats = RenderStats()
    t0 = time.perf_counter()
    progress("cut", 0, 1)
    segment = cut(source, opts, out)
    current = out / "cut_silent.mp4"          # le son ne part pas : il est remis à l'assemblage
    media.ffmpeg("-i", str(segment.path), "-an", "-c:v", "copy", str(current))
    # Photo choisie à la main dans la bibliothèque, sinon la plus en pied.
    references = [Reference(m.person.reference, face_ratio(m.person.reference)) if m.person.reference
                  else choose_reference(m.person.photos) for m in mappings]
    for i, ref in enumerate(references):
        warning = _reference_warning(f"Personne {i + 1}" if len(mappings) > 1 else "Cette personne", ref.framing)
        if warning:
            stats.warnings.append(warning)
    progress("cut", 1, 1)

    count = len(mappings)
    grid = [1, 1] if count == 1 else list(ccfg.get("mask_grid_multi", [4, 8]))
    local = relative_targets(mappings, opts.start, opts.end - opts.start)
    resolution, steps = params.get("resolution", "360p"), int(ccfg.get("steps", 6))
    gpu_seconds = 0.0
    for k, (mapping, reference) in enumerate(zip(local, references), start=1):
        # Étapes d'un passage : « generate@2/2 » = 2e personne sur 2 (rien d'ajouté s'il n'y en a qu'une).
        step = progress if count == 1 else (lambda stage, done, total, k=k: progress(f"{stage}@{k}/{count}", done, total))
        payload = {"t": mapping.target["t"], "box": mapping.target["box"], "resolution": resolution, "steps": steps,
                   "seed": 42, "mask_grid": grid}
        generated, mask = out / f"generated_{k}.mp4", out / f"generated_mask_{k}.mp4"
        try:
            wake_space("character", job["id"], step)
            client = ZeroGPUClient.from_settings("character")
            remote = client.replace(current, reference.path, payload, generated, mask,
                                    should_cancel=lambda: db.job_status(job["id"]) == "cancelling",
                                    on_stage=remote_stages(step, CHARACTER_STEPS))
        except RemoteCancelled as exc:
            raise Cancelled() from exc
        used = float(remote.get("gpu_seconds", 0))
        gpu_seconds += used
        today = time.strftime("%Y-%m-%d")
        db.set_meta(f"zerogpu_used:{today}", str(float(db.get_meta(f"zerogpu_used:{today}") or 0) + used))
        prefix = f"Personne {k} : " if count > 1 else ""
        stats.warnings += [prefix + w for w in remote.get("warnings", [])]

        # Recollage en pleine résolution : sortie finale, ou point de départ du passage suivant (qualité élevée).
        last = k == count
        target = out / "swapped.mp4" if last else out / f"pass_{k}.mp4"
        stats.frames = recompose(generated, mask, current, target, segment.fps_str,
                                 int(rcfg.crf) if last else 12, str(rcfg.preset), float(ccfg.get("feather", 0.015)))
        for f in (generated, mask, current):
            f.unlink(missing_ok=True)
        current = target

    progress("assemble", 0, 1)
    stats.swapped = stats.frames
    assemble(source, segment, out / "swapped.mp4", opts, out / "result.mp4", out)
    progress("assemble", 1, 1)
    stats.seconds = time.perf_counter() - t0
    stats.sec_per_frame = stats.seconds / max(1, stats.frames)
    stats.sec_per_computed = gpu_seconds / max(1, stats.frames)
    # Temps GPU réel par seconde d'extrait et par personne, ramené à 6 étapes : recale l'estimation avant rendu.
    length = max(0.1, opts.end - opts.start)
    per_pass = gpu_seconds / count
    db.set_meta(f"character_gpu_s:{resolution}", str(max(1.0, (per_pass - 45) / length * 6 / steps)))
    return stats


def job_options(params: dict) -> RenderOptions:
    return RenderOptions(start=params["start"], end=params["end"], output=params["output"],
                         stabilize=params["stabilize"], ai_label=params["ai_label"], level=params.get("level") or FACE,
                         limit_fps=bool(params.get("limit_fps", False)), restore=bool(params.get("restore", False)))


def back_to_review(job_id: str, params: dict, **extra) -> None:
    """Fin d'une étape de la vérification (ou étape annulée) : le rendu attend de nouveau les choix de l'utilisateur."""
    params = {k: v for k, v in params.items() if k != "review_step"}
    db.update("jobs", job_id, status="review", stage="review", params=params, **extra)


def run_review_step(job: dict, step: str, source: Path, out: Path, progress) -> None:
    """Rendu arrêté avant l'assemblage : recalcul des images corrigées (fix), ou assemblage final (assemble)."""
    params = job["params"]
    opts = job_options(params)
    plan = review.FramePlan.load(out)
    if plan is None:
        raise RuntimeError("Journal du rendu introuvable : relance le rendu.")
    if step == "fix":
        people: dict = {}

        def make(m: int):
            pid = plan.mappings[m]["person"]
            if pid not in people:
                people[pid] = load_person_assets(job["face_set_id"], pid, opts.level)
            return make_strategy(opts.level, people[pid], opts.restore)

        t0 = time.perf_counter()
        try:
            review.recompute(out / "cut.mp4", out / "swapped.mp4", plan, make, progress)
        except Cancelled:
            back_to_review(job["id"], params)
            return
        stats = review.load_stats(out)
        stats.seconds += time.perf_counter() - t0
        review.save_stats(out, stats)
        back_to_review(job["id"], params, error=None)
        return

    t0 = time.perf_counter()
    try:
        segment = cut(source, opts, out, reuse=True)
        finish(source, segment, opts, out, out / "result.mp4", progress)
    except Cancelled:
        back_to_review(job["id"], params)
        return
    stats = review.load_stats(out)
    stats.seconds += time.perf_counter() - t0
    review.save_stats(out, stats)
    now = time.time()
    params = {k: v for k, v in params.items() if k != "review_step"}
    # « Temps total » affiché = calcul du rendu + corrections + assemblage (pas le temps passé à vérifier).
    db.update("jobs", job["id"], status="done", params=params, started_at=now - stats.seconds, finished_at=now,
              sec_per_frame=stats.sec_per_frame, warnings=stats.warnings + review.warnings_of(plan),
              done=stats.frames, total=stats.frames)


def run_job(job: dict) -> None:
    job_id = job["id"]
    out = db.folder("jobs", job_id)
    out.mkdir(parents=True, exist_ok=True)
    video = db.get("videos", job["video_id"])
    if video is None:
        raise RuntimeError("La vidéo source a été supprimée.")
    params = job["params"]
    level = params.get("level") or FACE
    opts = job_options(params)
    last = [0.0]

    def progress(stage: str, done: int, total: int) -> None:
        now = time.time()
        if stage in ("swap", "fix") and done not in (1, total) and now - last[0] < 0.5:
            return
        last[0] = now
        status = db.job_status(job_id)
        if status == "cancelling":
            raise Cancelled()
        if status == "pausing":
            raise Paused()
        db.update("jobs", job_id, stage=stage, done=done, total=total)
        db.set_meta("worker_heartbeat", str(now))

    if params.get("review_step"):
        run_review_step(job, params["review_step"], Path(video["source"]), out, progress)
        return

    set_id = job["face_set_id"]
    if params.get("mappings"):
        # Données préparées une seule fois par personne, même si elle remplace plusieurs visages.
        people = {p: load_person_assets(set_id, p, level) for p in {m["person"] for m in params["mappings"]}}
        mappings = [
            FaceMapping(people[m["person"]], target={"t": m["t"], "box": m["box"]},
                        label=f"Visage {i + 1} (personne {m['person']})", person_id=m["person"])
            for i, m in enumerate(params["mappings"])
        ]
    else:  # ancien format : toutes les photos → un seul visage
        mappings = [FaceMapping(load_person_assets(set_id, None, level), target=params.get("target"))]

    if level == CHARACTER:
        # Niveau 4 : uniquement sur le Space « niveau 4 » (la case GPU est obligatoire, vérifiée à la création).
        stats, engine = run_character(job, Path(video["source"]), opts, mappings, out, progress), "zerogpu"
    elif params.get("use_gpu"):
        # Case « Utiliser le GPU » cochée pour CE rendu : calcul sur le Space ZeroGPU, jamais de repli sur le CPU.
        stats, engine = run_remote(job, Path(video["source"]), opts, mappings, out, progress), "zerogpu"
    else:
        source = Path(video["source"])
        segment, stats = render_frames(source, mappings, opts, out, progress, preview=out / "preview.jpg")
        engine = accelerator()
        remember_speed(level, engine, stats, len(mappings))
        plan = review.FramePlan.load(out)
        if params.get("review") and plan is not None and review.needs_review(out):
            # Visages mal suivis : arrêt avant l'assemblage, l'utilisateur choisit quoi corriger.
            Checkpoint.load(out).clear()
            back_to_review(job_id, params, sec_per_frame=stats.sec_per_frame, warnings=stats.warnings,
                           done=stats.frames, total=stats.frames)
            return
        finish(source, segment, opts, out, out / "result.mp4", progress)
        if plan is None:
            (out / "swapped.mp4").unlink(missing_ok=True)   # sans journal, rien à revoir : inutile de le garder
        # swapped.mp4 et le journal restent (jusqu'au ménage automatique) : « Revoir les images » après coup.
        warnings = stats.warnings + (review.warnings_of(plan) if plan is not None else [])
        db.update("jobs", job_id, status="done", finished_at=time.time(), sec_per_frame=stats.sec_per_frame,
                  warnings=warnings, done=stats.frames, total=stats.frames)
        return
    (out / "swapped.mp4").unlink(missing_ok=True)
    remember_speed(level, engine, stats, len(mappings))
    db.update("jobs", job_id, status="done", finished_at=time.time(), sec_per_frame=stats.sec_per_frame,
              warnings=stats.warnings, done=stats.frames, total=stats.frames)


def remember_speed(level: str, engine: str, stats: RenderStats, faces: int) -> None:
    """Vitesse ramenée à un seul visage et mémorisée par (niveau, moteur), pour que les estimations restent justes
    (même formule que le frontend : chaque visage en plus ≈ +75 %)."""
    factor = 1 + 0.75 * max(0, faces - 1)
    # Par image réellement calculée : les copies sautées ne doivent pas rendre les estimations trop optimistes.
    db.set_meta(f"spf:{level}:{engine}", str(stats.sec_per_computed / factor))


def main() -> None:
    db.init()
    for j in db.all_rows("jobs", 500):
        if j["status"] in ("running", "cancelling", "pausing") and j["params"].get("review_step"):
            # Recalcul des corrections ou assemblage interrompu : les corrections restent marquées, à relancer.
            back_to_review(j["id"], j["params"], error="Interrompu (worker ou PC redémarré) : relance le recalcul "
                           "ou l'assemblage.")
        elif j["status"] in ("running", "cancelling", "pausing"):
            if j["params"].get("use_gpu") or j["status"] == "cancelling":
                db.update("jobs", j["id"], status="error", error="Interrompu (worker redémarré).", finished_at=time.time())
            else:  # rendu local : le point de reprise permet de continuer où il s'était arrêté
                db.update("jobs", j["id"], status="paused", error="Interrompu (worker ou PC redémarré) : clique sur Reprendre. "
                          "Le calcul repart du dernier point enregistré (au plus ~2 min perdues).")
    print("Worker prêt, en attente de jobs…", flush=True)
    while True:
        db.set_meta("worker_heartbeat", str(time.time()))
        job = db.claim_next_job()
        if job is None:
            time.sleep(1)
            continue
        print(f"→ job {job['id']} ({job['params']['start']:.1f}–{job['params']['end']:.1f} s)", flush=True)
        try:
            run_job(job)
            print(f"✓ job {job['id']}", flush=True)
        except Cancelled:
            db.update("jobs", job["id"], status="cancelled", finished_at=time.time())
            print(f"× job {job['id']} annulé", flush=True)
        except Paused:
            db.update("jobs", job["id"], status="paused")
            print(f"‖ job {job['id']} en pause", flush=True)
        except Exception as exc:
            traceback.print_exc()
            if job["params"].get("review_step"):   # la vérification reste possible : seule cette étape a échoué
                back_to_review(job["id"], job["params"], error=str(exc)[:500])
            else:
                db.update("jobs", job["id"], status="error", error=str(exc)[:500], finished_at=time.time())


if __name__ == "__main__":
    main()
