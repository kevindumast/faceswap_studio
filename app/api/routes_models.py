"""Niveaux disponibles et installation des modèles depuis l'interface (téléchargement en tâche de fond)."""
from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException

from app import db
from src import models
from src.config import load_config
from src.hardware import engine_status
from src.levels import CHARACTER, FACE, FACE_TONE, HEAD, MODEL_GROUPS

from .common import background

router = APIRouter(prefix="/api/models", tags=["models"])

# Taille approximative à télécharger par groupe (affichée sur le bouton « Installer »).
GROUP_MB = {"base": 850, "tone": 94, "head": 540, "restore": 360}


def _download_state(group: str) -> dict | None:
    raw = db.get_meta(f"download:{group}")
    return json.loads(raw) if raw else None


def sec_per_frame(level: str, engine: str | None = None) -> float:
    """Vitesse mesurée sur cette machine pour ce niveau et ce moteur (ramenée à un visage), sinon l'estimation de config.yaml.

    Chaque moteur (cpu, dml, cuda) a sa propre mesure : un rendu sur carte graphique ne fausse pas l'estimation CPU.
    """
    engine = engine or engine_status()["accelerator"]
    measured = db.get_meta(f"spf:{level}:{engine}")
    if measured:
        return float(measured)
    if level == FACE and engine == "cpu":  # mesures faites avant l'existence des niveaux
        legacy = db.get_meta("sec_per_frame_single") or db.last_sec_per_frame()
        if legacy:
            return float(legacy)
    defaults = load_config().get("levels", {}).get(level, {})
    key = {"cpu": "sec_per_frame", "zerogpu": "sec_per_frame_zerogpu"}.get(engine, "sec_per_frame_gpu")
    return float(defaults.get(key) or defaults.get("sec_per_frame", 2.2))


def character_status() -> dict:
    """Niveau 4 : Space branché ? + de quoi estimer le temps de GPU (mesures réelles dès le 1er rendu)."""
    from app.worker.zerogpu_client import character_configured

    ccfg = load_config().get("levels", {}).get(CHARACTER, {})
    defaults = ccfg.get("gpu_s_per_block_step", {"360p": 16, "480p": 36})
    per_block_step = {res: float(db.get_meta(f"character_block_step_s:{res}") or defaults.get(res, 16))
                      for res in ("360p", "480p")}
    return {"configured": character_configured(), "space": db.get_meta("zerogpu_character_space") or None,
            "max_s": float(ccfg.get("max_s", 10)), "steps": int(ccfg.get("steps", 4)),
            "gpu_s_per_block_step": per_block_step, "max_people": int(ccfg.get("max_people", 2))}


def levels_status(available: tuple[str, ...]) -> dict:
    engine = engine_status()["accelerator"]
    out = {}
    for level in (FACE, FACE_TONE, HEAD, CHARACTER):
        groups = MODEL_GROUPS.get(level, ())
        missing = [g for g in groups if not models.is_ready(g)]
        downloads = {g: _download_state(g) for g in missing}
        out[level] = {
            "available": level in available,
            "gpu_only": level == CHARACTER,
            "ready": level in available and not missing,
            "missing_groups": missing,
            "install_mb": sum(GROUP_MB.get(g, 0) for g in missing),
            "installing": next((d for d in downloads.values() if d and d.get("running")), None),
            "install_error": next((d["error"] for d in downloads.values() if d and d.get("error")), None),
            "sec_per_frame": sec_per_frame(level, engine),
        }
    return out


def restore_status() -> dict:
    """Option « netteté » (niveaux 1 et 2) : modèle à part, pas lié à un niveau (même écran d'installation)."""
    download = _download_state("restore")
    return {
        "ready": models.is_ready("restore"),
        "install_mb": GROUP_MB["restore"],
        "installing": download if download and download.get("running") else None,
        "install_error": download["error"] if download and download.get("error") else None,
    }


def _run_download(group: str) -> None:
    last = [0.0]

    def on_progress(frac: float) -> None:
        if frac - last[0] >= 0.01 or frac >= 1:
            last[0] = frac
            db.set_meta(f"download:{group}", json.dumps({"running": True, "progress": round(frac, 3)}))

    try:
        models.ensure(group, on_progress)
        db.set_meta(f"download:{group}", json.dumps({"running": False, "progress": 1}))
    except Exception as exc:
        db.set_meta(f"download:{group}", json.dumps({"running": False, "progress": 0, "error": str(exc)[:300]}))


@router.post("/{group}/download")
def download(group: str) -> dict:
    if group not in GROUP_MB:
        raise HTTPException(404, "Groupe de modèles inconnu.")
    state = _download_state(group)
    if not (state and state.get("running")) and not models.is_ready(group):
        db.set_meta(f"download:{group}", json.dumps({"running": True, "progress": 0}))
        background.submit(_run_download, group)
    return {"group": group, "ready": models.is_ready(group), "state": _download_state(group)}
