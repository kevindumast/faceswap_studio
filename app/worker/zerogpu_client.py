"""Client du Space GPU (ZeroGPU) : envoie un extrait, suit la progression, récupère le résultat.

Utilisé seulement quand un rendu a été lancé avec la case « Utiliser le GPU » cochée : jamais de bascule automatique,
ni du CPU vers le GPU, ni l'inverse en cas d'échec (l'utilisateur relance lui-même sur son PC).
"""
from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Callable

from app import db
from src.netfix import setup_tls

setup_tls()  # certificat absent ou proxy d'entreprise : vérification avec le magasin du système


class ZeroGPUError(RuntimeError):
    pass


class RemoteCancelled(Exception):
    pass


def settings() -> dict:
    return {"space": db.get_meta("zerogpu_space") or "", "token": db.get_meta("zerogpu_token") or "",
            "key": db.get_meta("zerogpu_key") or ""}


def configured() -> bool:
    s = settings()
    return bool(s["space"] and s["key"])


def friendly(exc: Exception) -> str:
    """Messages compréhensibles pour les erreurs les plus fréquentes."""
    msg = str(exc)
    low = msg.lower()
    if "quota" in low:
        return "Quota ZeroGPU épuisé pour aujourd'hui (5 min/jour en gratuit, 40 en PRO). Relance sur ce PC ou réessaie demain."
    if "app_key" in low:
        return "Clé APP_KEY refusée par le Space : relance scripts/deploy_space.py --save ou recopie la clé."
    if "401" in msg or "403" in msg or "unauthorized" in low or "not found" in low or "404" in msg:
        return "Space introuvable ou jeton refusé : vérifie le nom du Space et ton jeton Hugging Face (Moteur → ZeroGPU)."
    if "sleep" in low or "building" in low or "starting" in low or "runtime_error" in low:
        return "Le Space démarre ou s'est mis en veille : attends 1 à 2 min puis relance le rendu."
    if "gpu task aborted" in low or "duration" in low:
        return "Le GPU a été coupé avant la fin (durée maximale dépassée) : essaie un passage plus court."
    return f"Erreur du Space GPU : {msg[:300]}"


class ZeroGPUClient:
    def __init__(self, space: str, token: str, key: str):
        from gradio_client import Client

        self.key = key
        self.downloads = Path(tempfile.mkdtemp(prefix="zerogpu_"))
        try:
            self.client = Client(space, token=token or None, verbose=False, download_files=str(self.downloads))
        except Exception as exc:
            raise ZeroGPUError(friendly(exc)) from exc

    @classmethod
    def from_settings(cls) -> "ZeroGPUClient":
        s = settings()
        if not (s["space"] and s["key"]):
            raise ZeroGPUError("ZeroGPU n'est pas configuré (Moteur → ZeroGPU).")
        return cls(s["space"], s["token"], s["key"])

    def health(self) -> dict:
        try:
            return json.loads(self.client.predict(self.key, api_name="/health"))
        except Exception as exc:
            raise ZeroGPUError(friendly(exc)) from exc

    def swap(self, clip: Path, payload: dict, out: Path, on_progress: Callable[[int, int], None] | None = None,
             should_cancel: Callable[[], bool] | None = None, poll: float = 1.0) -> dict:
        """Envoie l'extrait (sans son) et écrit la vidéo remplacée dans `out`. Renvoie les stats du Space."""
        from gradio_client import handle_file

        try:
            job = self.client.submit(handle_file(str(clip)), json.dumps(payload), self.key, api_name="/swap")
        except Exception as exc:
            raise ZeroGPUError(friendly(exc)) from exc
        while not job.done():
            if should_cancel and should_cancel():
                job.cancel()
                raise RemoteCancelled()
            status = job.status()
            units = getattr(status, "progress_data", None) or []
            if units and on_progress:
                unit = units[-1]
                if getattr(unit, "index", None) is not None and getattr(unit, "length", None):
                    on_progress(int(unit.index), int(unit.length))
            time.sleep(poll)
        try:
            video, stats = job.result()
        except Exception as exc:
            raise ZeroGPUError(friendly(exc)) from exc
        path = video.get("path") if isinstance(video, dict) else video
        shutil.copy(path, out)
        shutil.rmtree(self.downloads, ignore_errors=True)
        return json.loads(stats) if isinstance(stats, str) else (stats or {})
