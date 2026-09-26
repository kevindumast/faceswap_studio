"""API FastAPI. Lancement : python -m app.api.main  (écoute 127.0.0.1 uniquement)."""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import db
from src import media
from src.config import ROOT, load_config

from . import routes_faces, routes_jobs, routes_videos

WEB_DIST = ROOT / "app" / "web" / "dist"


@asynccontextmanager
async def lifespan(_: FastAPI):
    cfg = load_config()
    db.init()
    # Préparations interrompues par un redémarrage : on les marque en erreur plutôt que de les laisser tourner à vide.
    for v in db.all_rows("videos", 500):
        if v["status"] in ("downloading", "preparing"):
            db.update("videos", v["id"], status="error", error="Interrompu par un redémarrage du serveur.")
    db.cleanup(float(cfg.retention_hours))
    yield


app = FastAPI(title="Faceswap Studio", lifespan=lifespan)
app.include_router(routes_videos.router)
app.include_router(routes_faces.router)
app.include_router(routes_jobs.router)


@app.get("/api/status")
def status() -> dict:
    cfg = load_config()
    models = cfg.path("models")
    try:
        media.binary("ffmpeg")
        ffmpeg_ok = True
    except media.MediaError:
        ffmpeg_ok = False
    heartbeat = float(db.get_meta("worker_heartbeat") or 0)
    return {
        "device": cfg.device,
        "ffmpeg": ffmpeg_ok,
        "models": (models / cfg.models.inswapper).is_file() and any((models / cfg.models.detector_pack).glob("*.onnx")),
        "worker": time.time() - heartbeat < 10,
        "segment": {"min_s": cfg.segment.min_s, "max_s": cfg.segment.max_s},
        "photos_max": cfg.photos.max,
        "upload_max_mb": cfg.upload.max_mb,
        "video_ext": cfg.upload.video_ext,
        "sec_per_frame": float(db.get_meta("sec_per_frame_single") or 0)
        or db.last_sec_per_frame()
        or cfg.render.default_sec_per_frame,
        "example_url": cfg.youtube.example_url,
        "youtube_max_duration_s": cfg.youtube.max_duration_s,
    }


# Frontend buildé (npm run build) servi à la racine, avec fallback SPA.
if WEB_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        file = (WEB_DIST / path).resolve()
        if path and file.is_file() and file.is_relative_to(WEB_DIST):
            return FileResponse(file)
        return FileResponse(WEB_DIST / "index.html")


def main() -> None:
    import uvicorn

    cfg = load_config()
    uvicorn.run("app.api.main:app", host=cfg.server.host, port=int(cfg.server.port))


if __name__ == "__main__":
    main()
