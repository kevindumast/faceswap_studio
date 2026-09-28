"""API FastAPI. Lancement : python -m app.api.main  (écoute 127.0.0.1 uniquement)."""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import db
from src import media
from src.config import ROOT, load_config
from src.hardware import engine_status
from src.levels import FACE

from . import routes_faces, routes_jobs, routes_models, routes_people, routes_settings, routes_videos

WEB_DIST = ROOT / "app" / "web" / "dist"


@asynccontextmanager
async def lifespan(_: FastAPI):
    cfg = load_config()
    db.init()
    # Préparations interrompues par un redémarrage : on les marque en erreur plutôt que de les laisser tourner à vide.
    for v in db.all_rows("videos", 500):
        if v["status"] in ("downloading", "preparing"):
            db.update("videos", v["id"], status="error", error="Interrompu par un redémarrage du serveur.")
    for group in routes_models.GROUP_MB:  # téléchargements de modèles interrompus
        if db.get_meta(f"download:{group}"):
            db.set_meta(f"download:{group}", "")
    db.cleanup(float(cfg.retention_hours))
    yield


app = FastAPI(title="Faceswap Studio", lifespan=lifespan)
app.include_router(routes_videos.router)
app.include_router(routes_faces.router)
app.include_router(routes_jobs.router)
app.include_router(routes_models.router)
app.include_router(routes_people.router)
app.include_router(routes_settings.router)


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
    engine = engine_status()
    return {
        "device": engine["accelerator"],
        "engine": engine,
        "ffmpeg": ffmpeg_ok,
        "models": (models / cfg.models.inswapper).is_file() and any((models / cfg.models.detector_pack).glob("*.onnx")),
        "worker": time.time() - heartbeat < 10,
        "segment": {"min_s": cfg.segment.min_s, "max_s": cfg.segment.max_s},
        "photos_max": cfg.photos.max,
        "fps_cap": float(cfg.render.get("fps_cap", 30)),
        "upload_max_mb": cfg.upload.max_mb,
        "video_ext": cfg.upload.video_ext,
        "sec_per_frame": routes_models.sec_per_frame(FACE),
        "levels": routes_models.levels_status(routes_jobs.AVAILABLE_LEVELS),
        "restore": routes_models.restore_status(),
        "gpu": {
            "configured": routes_jobs.gpu_configured(),
            "space": db.get_meta("zerogpu_space") or None,
            "used_today_s": round(float(db.get_meta(f"zerogpu_used:{time.strftime('%Y-%m-%d')}") or 0)),
            "free_quota_s": int(cfg.get("zerogpu", {}).get("free_quota_min", 5)) * 60,
            "sec_per_frame": {lvl: routes_models.sec_per_frame(lvl, "zerogpu") for lvl in ("face", "face_tone")},
            "character": routes_models.character_status(),
        },
        "example_url": cfg.youtube.example_url,
        "youtube_max_duration_s": cfg.youtube.max_duration_s,
    }


# Frontend buildé (npm run build) servi à la racine, avec fallback SPA.
if WEB_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        # Une route /api inconnue doit rester une erreur JSON, pas la page du site (sinon le front lit du HTML).
        if path == "api" or path.startswith("api/"):
            raise HTTPException(404, "Route d'API inconnue : l'API tourne peut-être avec une ancienne version, relance-la.")
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
