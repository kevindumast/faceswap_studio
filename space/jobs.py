"""Suivi d'un calcul par requêtes courtes, commun aux deux Spaces (copié dans chacun par deploy_space.py).

Un proxy d'entreprise (Cisco Umbrella) retient les réponses en flux de *.hf.space jusqu'à leur fin et les coupe au
bout de ~3 min : le flux de suivi de gradio_client n'arrive jamais, et sa coupure fait annuler le calcul par Gradio.
L'appli lance donc le calcul (/call/…), puis lit son état ici toutes les quelques secondes (/run/status) et récupère
le résultat une fois terminé. L'état est un petit fichier JSON par calcul, dans /tmp : la fonction GPU de ZeroGPU
tourne dans un autre processus que le serveur Gradio, mais sur le même disque.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Callable

ROOT = Path(os.environ.get("FACESWAP_JOBS", "/tmp/faceswap_jobs"))
MAX_AGE_S = 6 * 3600


class Cancelled(Exception):
    """Annulation demandée par l'appli (bouton « Annuler »)."""


def _path(job: str, suffix: str = ".json") -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", job or ""):
        raise ValueError("Identifiant de calcul invalide.")
    return ROOT / f"{job}{suffix}"


def write(job: str | None, **state) -> None:
    """Remplace l'état du calcul (écriture atomique : la lecture ne voit jamais un fichier à moitié écrit)."""
    if not job:
        return
    ROOT.mkdir(parents=True, exist_ok=True)
    path = _path(job)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps({**state, "t": time.time()}), encoding="utf-8")
    os.replace(tmp, path)


def read(job: str) -> dict:
    """État connu du calcul ; « unknown » tant que le Space ne l'a pas commencé (file d'attente de Gradio)."""
    try:
        return json.loads(_path(job).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"state": "unknown"}


_last_write: dict[str, float] = {}


def progress(job: str | None, done: int, total: int, desc: str, every: float = 0.5) -> None:
    """Avancement du calcul (au plus 2 écritures par seconde) ; lève Cancelled si l'appli a demandé l'arrêt."""
    check_cancel(job)
    if job and (done >= total or time.time() - _last_write.get(job, 0) >= every):
        write(job, state="running", done=done, total=total, desc=desc)
        _last_write[job] = time.time()


def request_cancel(job: str) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    _path(job, ".cancel").touch()


def check_cancel(job: str | None) -> None:
    """À appeler à chaque étape du calcul : lève Cancelled si l'appli a demandé l'arrêt."""
    if job and _path(job, ".cancel").exists():
        raise Cancelled()


def run(job: str | None, fn: Callable, *args):
    """Exécute le calcul (fonction GPU) en tenant son état à jour : attente d'un GPU, fin, erreur.

    Les refus de ZeroGPU (quota, durée) arrivent avant que la fonction GPU ne démarre : ils sont notés ici aussi.
    Une annulation remonte du processus GPU comme une erreur ordinaire : l'appli, qui l'a demandée, ne lit plus l'état.
    """
    cleanup()
    try:
        check_cancel(job)
        write(job, state="waiting")
        result = fn(*args)
    except BaseException as exc:
        write(job, state="error", error=str(exc) or type(exc).__name__)
        raise
    write(job, state="done")
    return result


def cleanup(max_age: float = MAX_AGE_S) -> None:
    """Efface les états des calculs de plus de 6 h."""
    if not ROOT.is_dir():
        return
    limit = time.time() - max_age
    for path in ROOT.iterdir():
        try:
            if path.stat().st_mtime < limit:
                path.unlink()
        except OSError:
            pass
