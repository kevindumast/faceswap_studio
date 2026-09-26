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
from src.levels import FACE
from src.pipeline import Cancelled, FaceMapping, RenderOptions, render


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
                         stabilize=params["stabilize"], ai_label=params["ai_label"], level=level)
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
        if db.job_status(job_id) == "cancelling":
            raise Cancelled()
        db.update("jobs", job_id, stage=stage, done=done, total=total)
        db.set_meta("worker_heartbeat", str(now))

    stats = render(
        source=Path(video["source"]),
        mappings=mappings,
        opts=opts,
        work_dir=out,
        out=out / "result.mp4",
        progress=progress,
        preview=out / "preview.jpg",
    )
    (out / "swapped.mp4").unlink(missing_ok=True)
    # Vitesse ramenée à un seul visage et mémorisée par (niveau, moteur), pour que les estimations restent justes
    # (même formule que le frontend : chaque visage en plus ≈ +75 %).
    factor = 1 + 0.75 * max(0, len(mappings) - 1)
    db.set_meta(f"spf:{level}:cpu", str(stats.sec_per_frame / factor))
    db.update("jobs", job_id, status="done", finished_at=time.time(), sec_per_frame=stats.sec_per_frame,
              warnings=stats.warnings, done=stats.frames, total=stats.frames)


def main() -> None:
    db.init()
    for j in db.all_rows("jobs", 500):
        if j["status"] in ("running", "cancelling"):
            db.update("jobs", j["id"], status="error", error="Interrompu (worker redémarré).", finished_at=time.time())
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
        except Exception as exc:
            traceback.print_exc()
            db.update("jobs", job["id"], status="error", error=str(exc)[:500], finished_at=time.time())


if __name__ == "__main__":
    main()
