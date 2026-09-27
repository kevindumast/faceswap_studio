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
from app.worker.zerogpu_client import RemoteCancelled, ZeroGPUClient
from src import media
from src.character import choose_reference, recompose
from src.config import accelerator, load_config
from src.levels import CHARACTER, FACE
from src.pipeline import Cancelled, FaceMapping, Paused, RenderOptions, RenderStats, assemble, cut, relative_targets, render


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
    client = ZeroGPUClient.from_settings()
    try:
        remote = client.swap(silent, payload, out / "swapped.mp4",
                             on_progress=lambda done, total: progress("swap", done, total),
                             should_cancel=lambda: db.job_status(job["id"]) == "cancelling")
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


def run_character(job: dict, source: Path, opts: RenderOptions, mapping: FaceMapping, out: Path, progress) -> RenderStats:
    """Niveau 4 : découpe et assemblage sur le PC, personne entière générée sur le Space « niveau 4 »."""
    params = job["params"]
    ccfg = load_config().levels.character
    stats = RenderStats()
    t0 = time.perf_counter()
    progress("cut", 0, 1)
    segment = cut(source, opts, out)
    silent = out / "cut_silent.mp4"          # le son ne part pas : il est remis à l'assemblage
    media.ffmpeg("-i", str(segment.path), "-an", "-c:v", "copy", str(silent))
    reference = choose_reference(mapping.person.photos)
    if not reference.full_body:
        stats.warnings.append("Pas de photo en pied pour cette personne : le corps et les habits ont été inventés. "
                              "Ajoute une photo en pied dans la bibliothèque pour un meilleur résultat.")
    progress("cut", 1, 1)

    local = relative_targets([mapping], opts.start, opts.end - opts.start)[0]
    payload = {"t": local.target["t"], "box": local.target["box"], "resolution": params.get("resolution", "360p"),
               "steps": int(ccfg.get("steps", 6)), "seed": 42}
    client = ZeroGPUClient.from_settings("character")
    generated, mask = out / "generated.mp4", out / "generated_mask.mp4"
    try:
        remote = client.replace(silent, reference.path, payload, generated, mask,
                                on_progress=lambda done, total: progress("swap", done, total),
                                should_cancel=lambda: db.job_status(job["id"]) == "cancelling")
    except RemoteCancelled as exc:
        raise Cancelled() from exc
    silent.unlink(missing_ok=True)
    gpu_seconds = float(remote.get("gpu_seconds", 0))
    today = time.strftime("%Y-%m-%d")
    db.set_meta(f"zerogpu_used:{today}", str(float(db.get_meta(f"zerogpu_used:{today}") or 0) + gpu_seconds))
    stats.warnings += list(remote.get("warnings", []))

    progress("assemble", 0, 1)
    rcfg = load_config().render
    stats.frames = recompose(generated, mask, segment.path, out / "swapped.mp4", segment.fps_str, int(rcfg.crf),
                             str(rcfg.preset), float(ccfg.get("feather", 0.015)))
    stats.swapped = stats.frames
    assemble(source, segment, out / "swapped.mp4", opts, out / "result.mp4", out)
    for f in (generated, mask):
        f.unlink(missing_ok=True)
    progress("assemble", 1, 1)
    stats.seconds = time.perf_counter() - t0
    stats.sec_per_frame = stats.seconds / max(1, stats.frames)
    stats.sec_per_computed = gpu_seconds / max(1, stats.frames)
    # Temps GPU réel par seconde d'extrait, ramené à 6 étapes : recale l'estimation affichée avant le rendu.
    length = max(0.1, opts.end - opts.start)
    db.set_meta(f"character_gpu_s:{payload['resolution']}", str(max(1.0, (gpu_seconds - 45) / length * 6 / payload["steps"])))
    return stats


def run_job(job: dict) -> None:
    job_id = job["id"]
    out = db.folder("jobs", job_id)
    out.mkdir(parents=True, exist_ok=True)
    video = db.get("videos", job["video_id"])
    if video is None:
        raise RuntimeError("La vidéo source a été supprimée.")
    params = job["params"]
    level = params.get("level") or FACE
    opts = RenderOptions(start=params["start"], end=params["end"], output=params["output"],
                         stabilize=params["stabilize"], ai_label=params["ai_label"], level=level,
                         limit_fps=bool(params.get("limit_fps", False)))
    set_id = job["face_set_id"]
    if params.get("mappings"):
        # Données préparées une seule fois par personne, même si elle remplace plusieurs visages.
        people = {p: load_person_assets(set_id, p, level) for p in {m["person"] for m in params["mappings"]}}
        mappings = [
            FaceMapping(people[m["person"]], target={"t": m["t"], "box": m["box"]}, label=f"Visage {i + 1} (personne {m['person']})")
            for i, m in enumerate(params["mappings"])
        ]
    else:  # ancien format : toutes les photos → un seul visage
        mappings = [FaceMapping(load_person_assets(set_id, None, level), target=params.get("target"))]
    last = [0.0]

    def progress(stage: str, done: int, total: int) -> None:
        now = time.time()
        if stage == "swap" and done not in (1, total) and now - last[0] < 0.5:
            return
        last[0] = now
        status = db.job_status(job_id)
        if status == "cancelling":
            raise Cancelled()
        if status == "pausing":
            raise Paused()
        db.update("jobs", job_id, stage=stage, done=done, total=total)
        db.set_meta("worker_heartbeat", str(now))

    if level == CHARACTER:
        # Niveau 4 : uniquement sur le Space « niveau 4 » (la case GPU est obligatoire, vérifiée à la création).
        stats, engine = run_character(job, Path(video["source"]), opts, mappings[0], out, progress), "zerogpu"
    elif params.get("use_gpu"):
        # Case « Utiliser le GPU » cochée pour CE rendu : calcul sur le Space ZeroGPU, jamais de repli sur le CPU.
        stats, engine = run_remote(job, Path(video["source"]), opts, mappings, out, progress), "zerogpu"
    else:
        stats = render(
            source=Path(video["source"]),
            mappings=mappings,
            opts=opts,
            work_dir=out,
            out=out / "result.mp4",
            progress=progress,
            preview=out / "preview.jpg",
        )
        engine = accelerator()
    (out / "swapped.mp4").unlink(missing_ok=True)
    # Vitesse ramenée à un seul visage et mémorisée par (niveau, moteur), pour que les estimations restent justes
    # (même formule que le frontend : chaque visage en plus ≈ +75 %).
    factor = 1 + 0.75 * max(0, len(mappings) - 1)
    # Par image réellement calculée : les copies sautées ne doivent pas rendre les estimations trop optimistes.
    db.set_meta(f"spf:{level}:{engine}", str(stats.sec_per_computed / factor))
    db.update("jobs", job_id, status="done", finished_at=time.time(), sec_per_frame=stats.sec_per_frame,
              warnings=stats.warnings, done=stats.frames, total=stats.frames)


def main() -> None:
    db.init()
    for j in db.all_rows("jobs", 500):
        if j["status"] in ("running", "cancelling", "pausing"):
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
            db.update("jobs", job["id"], status="error", error=str(exc)[:500], finished_at=time.time())


if __name__ == "__main__":
    main()
