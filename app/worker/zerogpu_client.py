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


StageCallback = Callable[[str, int, int, "str | None"], None]
QUEUED = ("STARTING", "JOINING_QUEUE", "IN_QUEUE", "QUEUE_FULL", "SENDING_DATA")


class ZeroGPUError(RuntimeError):
    pass


class RemoteCancelled(Exception):
    pass


def settings() -> dict:
    """Deux Spaces possibles, même jeton et même clé : niveaux 1-2 (« space ») et niveau 4 (« character_space »)."""
    return {"space": db.get_meta("zerogpu_space") or "", "token": db.get_meta("zerogpu_token") or "",
            "key": db.get_meta("zerogpu_key") or "", "character_space": db.get_meta("zerogpu_character_space") or ""}


def configured() -> bool:
    s = settings()
    return bool(s["space"] and s["key"])


def character_configured() -> bool:
    s = settings()
    return bool(s["character_space"] and s["key"])


# --- Cycle de vie du Space : lecture de son état et réveil (métadonnées Hugging Face, sans quota GPU) ---------------

STARTING = {"APP_STARTING", "BUILDING", "RUNNING_BUILDING", "RUNNING_APP_STARTING"}
ASLEEP = {"SLEEPING", "PAUSED", "STOPPED"}
BROKEN = {"BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR", "NO_APP_FILE"}
EXPECTED_WAKE_S = {"faces": 180, "character": 1800}     # réveil typique : ~3 min ; ~30 min pour les 57 Go du niveau 4
WAKE_TIMEOUT_S = {"faces": 900, "character": 3600}
_states: dict[str, tuple[float, dict]] = {}


def space_name(kind: str) -> str:
    s = settings()
    return s["character_space"] if kind == "character" else s["space"]


def phase(stage: str | None) -> str:
    if stage == "RUNNING":
        return "ready"
    if stage in STARTING:
        return "starting"
    if stage in ASLEEP:
        return "asleep"
    if stage in BROKEN:
        return "error"
    return "unknown"


def _runtime(space: str):
    from huggingface_hub import HfApi

    try:
        return HfApi(token=settings()["token"] or None).get_space_runtime(space)
    except Exception as exc:
        raise ZeroGPUError(friendly(exc)) from exc


def state(kind: str = "faces", max_age: float = 5.0) -> dict:
    """État du Space pour l'appli. Le début du réveil est mémorisé en base : le chrono survit aux rechargements."""
    space = space_name(kind)
    if not space:
        return {"space": None, "phase": "unconfigured", "stage": None, "hardware": None, "error": None,
                "waking_since": None, "expected_s": EXPECTED_WAKE_S.get(kind, 180)}
    cached = _states.get(kind)
    if cached and time.time() - cached[0] < max_age and cached[1]["space"] == space:
        return cached[1]
    rt = _runtime(space)
    ph = phase(rt.stage)
    key = f"space_waking:{kind}"
    since = float(db.get_meta(key) or 0) or None
    if ph == "starting" and since is None:
        since = time.time()
        db.set_meta(key, str(since))
    elif since and (ph in ("ready", "error") or (ph == "asleep" and time.time() - since > 120)):
        since = None               # prêt, en erreur, ou réveil demandé qui n'a pas pris : chrono remis à zéro
        db.set_meta(key, "")
    out = {"space": space, "phase": ph, "stage": rt.stage, "hardware": rt.hardware,
           "error": (rt.raw or {}).get("errorMessage"), "waking_since": since,
           "expected_s": EXPECTED_WAKE_S.get(kind, 180)}
    _states[kind] = (time.time(), out)
    return out


def wake(kind: str = "faces") -> dict:
    """Réveille un Space endormi ou planté (redémarrage, sans reconstruction). Sans effet s'il tourne ou démarre."""
    st = state(kind, max_age=0)
    if st["phase"] in ("asleep", "error"):
        from huggingface_hub import HfApi

        try:
            HfApi(token=settings()["token"] or None).restart_space(st["space"])
        except Exception as exc:
            raise ZeroGPUError(friendly(exc)) from exc
        db.set_meta(f"space_waking:{kind}", str(time.time()))
        _states.pop(kind, None)
    return state(kind, max_age=0)


def wait_until_ready(kind: str, on_wait: Callable[[float, float], None],
                     should_cancel: Callable[[], bool] | None = None, poll: float = 10.0) -> None:
    """Avant un rendu : réveille le Space s'il dort, puis attend qu'il soit prêt (on_wait(écoulé, durée typique)).

    Attendre ne consomme pas de quota : le GPU n'est attribué qu'au moment du calcul.
    """
    t0 = time.time()
    woke = False
    while True:
        st = state(kind, max_age=0)
        if st["phase"] == "ready":
            return
        if st["phase"] == "unconfigured":
            raise ZeroGPUError("Le Space n'est pas configuré (Moteur → ZeroGPU).")
        if st["phase"] in ("asleep", "error"):
            if woke and st["phase"] == "error":
                raise ZeroGPUError(f"Le Space est en erreur ({st['stage']}) : "
                                   f"{st['error'] or 'voir ses journaux sur Hugging Face'}.")
            if not woke:
                wake(kind)
                woke = True
        if should_cancel and should_cancel():
            raise RemoteCancelled()
        on_wait(time.time() - (st["waking_since"] or t0), st["expected_s"])
        if time.time() - t0 > WAKE_TIMEOUT_S.get(kind, 900):
            raise ZeroGPUError("Le Space ne s'est pas réveillé à temps : regarde son état sur Hugging Face, puis relance.")
        time.sleep(poll)


def friendly(exc: Exception) -> str:
    """Messages compréhensibles pour les erreurs les plus fréquentes."""
    msg = str(exc)
    low = msg.lower()
    if "handshake failure" in low or "read operation timed out" in low or "certificate verify failed" in low:
        return ("Connexion au Space refusée par le réseau (filtre du réseau d'entreprise sur *.hf.space ?). "
                "Hugging Face voit le Space prêt, mais ce PC ne peut pas l'atteindre : fais autoriser *.hf.space "
                "par ton service informatique, ou utilise l'appli depuis un autre réseau.")
    if "quota" in low:
        return "Quota ZeroGPU épuisé pour aujourd'hui (5 min/jour en gratuit, 40 en PRO). Relance sur ce PC ou réessaie demain."
    if "app_key" in low:
        return "Clé APP_KEY refusée par le Space : relance scripts/deploy_space.py --save ou recopie la clé."
    if "401" in msg or "403" in msg or "unauthorized" in low or "not found" in low or "404" in msg:
        return "Space introuvable ou jeton refusé : vérifie le nom du Space et ton jeton Hugging Face (Moteur → ZeroGPU)."
    if "sleep" in low or "building" in low or "starting" in low or "runtime_error" in low:
        return ("Le Space démarre ou s'est mis en veille : attends 1 à 2 min puis relance le rendu "
                "(jusqu'à 30 min au premier démarrage du Space niveau 4, le temps de charger le modèle).")
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
    def from_settings(cls, kind: str = "faces") -> "ZeroGPUClient":
        s = settings()
        space = s["character_space"] if kind == "character" else s["space"]
        if not (space and s["key"]):
            what = "Le Space du niveau 4" if kind == "character" else "ZeroGPU"
            raise ZeroGPUError(f"{what} n'est pas configuré (Moteur → ZeroGPU).")
        return cls(space, s["token"], s["key"])

    def health(self) -> dict:
        try:
            return json.loads(self.client.predict(self.key, api_name="/health"))
        except Exception as exc:
            raise ZeroGPUError(friendly(exc)) from exc

    def swap(self, clip: Path, payload: dict, out: Path, on_progress: Callable[[int, int], None] | None = None,
             should_cancel: Callable[[], bool] | None = None, poll: float = 1.0, on_stage: StageCallback | None = None) -> dict:
        """Envoie l'extrait (sans son) et écrit la vidéo remplacée dans `out`. Renvoie les stats du Space."""
        from gradio_client import handle_file

        video, stats = self._run("/swap", (handle_file(str(clip)), json.dumps(payload), self.key), on_progress,
                                 should_cancel, poll, on_stage)
        shutil.copy(_path(video), out)
        shutil.rmtree(self.downloads, ignore_errors=True)
        return json.loads(stats) if isinstance(stats, str) else (stats or {})

    def replace(self, clip: Path, reference: Path, payload: dict, out: Path, mask_out: Path,
                on_progress: Callable[[int, int], None] | None = None, should_cancel: Callable[[], bool] | None = None,
                poll: float = 2.0, on_stage: StageCallback | None = None) -> dict:
        """Niveau 4 : extrait (sans son) + photo de la personne → vidéo générée (30 i/s) et masque de la zone refaite."""
        from gradio_client import handle_file

        video, mask, stats = self._run(
            "/replace", (handle_file(str(clip)), handle_file(str(reference)), json.dumps(payload), self.key),
            on_progress, should_cancel, poll, on_stage)
        shutil.copy(_path(video), out)
        shutil.copy(_path(mask), mask_out)
        shutil.rmtree(self.downloads, ignore_errors=True)
        return json.loads(stats) if isinstance(stats, str) else (stats or {})

    def _run(self, api_name: str, args: tuple, on_progress, should_cancel, poll: float, on_stage=None):
        """Soumet un appel, relaie la progression, annule si demandé ; renvoie le résultat brut du Space.

        on_stage(étape, fait, total, détail) : « queue » (rang, taille de la file du Space), « gpu » (calcul lancé,
        en attente d'un GPU libre chez ZeroGPU ou en préparation), « progress » (étape annoncée par le Space).
        """
        try:
            job = self.client.submit(*args, api_name=api_name)
        except Exception as exc:
            raise ZeroGPUError(friendly(exc)) from exc
        while not job.done():
            if should_cancel and should_cancel():
                job.cancel()
                raise RemoteCancelled()
            status = job.status()
            code = getattr(getattr(status, "code", None), "value", None)
            units = getattr(status, "progress_data", None) or []
            unit = units[-1] if units else None
            if unit is not None and getattr(unit, "index", None) is not None and getattr(unit, "length", None):
                if on_progress:
                    on_progress(int(unit.index), int(unit.length))
                if on_stage:
                    on_stage("progress", int(unit.index), int(unit.length), getattr(unit, "desc", None))
            elif on_stage and code in QUEUED:
                on_stage("queue", int(getattr(status, "rank", None) or 0), int(getattr(status, "queue_size", None) or 0), None)
            elif on_stage and code in ("PROCESSING", "ITERATING"):
                on_stage("gpu", 0, 0, None)
            time.sleep(poll)
        try:
            return job.result()
        except Exception as exc:
            raise ZeroGPUError(friendly(exc)) from exc


def _path(file) -> str:
    return file.get("path") if isinstance(file, dict) else file
