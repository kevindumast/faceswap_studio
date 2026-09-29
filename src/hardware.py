"""Description du matériel de calcul, pour l'affichage (« Carte graphique · MX450 + Iris Xe »)."""
from __future__ import annotations

import subprocess
import sys
from functools import lru_cache

from .config import DeviceUnavailable, accelerator

LABELS = {"cpu": "CPU", "dml": "Carte graphique", "cuda": "Carte NVIDIA (CUDA)"}
_VIRTUAL = ("mirror", "virtual", "basic display", "remote", "parsec", "meta")


@lru_cache(maxsize=1)
def gpu_names() -> tuple[str, ...]:
    """Cartes graphiques physiques de la machine (Windows : via CIM ; ailleurs : liste vide)."""
    if sys.platform != "win32":
        return ()
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_VideoController).Name"],
            capture_output=True, text=True, timeout=10, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return ()
    names = [n.strip() for n in out.splitlines() if n.strip()]
    return tuple(n for n in names if not any(v in n.lower() for v in _VIRTUAL))


def _directml_overwritten() -> bool:
    """onnxruntime-directml installé mais sans DirectML : les deux paquets fournissent le même module, et une
    réinstallation d'onnxruntime (CPU) a remplacé ses fichiers."""
    from importlib.metadata import PackageNotFoundError, version

    import onnxruntime

    try:
        version("onnxruntime-directml")
    except PackageNotFoundError:
        return False
    return "DmlExecutionProvider" not in onnxruntime.get_available_providers()


def short_name(name: str) -> str:
    """« NVIDIA GeForce MX450 » → « MX450 », « Intel(R) Iris(R) Xe Graphics » → « Iris Xe »."""
    clean = name.replace("(R)", "").replace("(TM)", "").replace("NVIDIA", "").replace("GeForce", "")
    clean = clean.replace("Intel", "").replace("AMD", "").replace("Radeon", "Radeon ").replace("Graphics", "")
    return " ".join(clean.split()) or name


def engine_status() -> dict:
    """Moteur effectivement utilisé pour les rendus locaux, ou l'erreur si la config demande un moteur absent."""
    try:
        acc = accelerator()
        error = None
    except DeviceUnavailable as exc:
        acc, error = "cpu", str(exc)
    if acc == "cpu" and error is None and _directml_overwritten():
        error = ("La version carte graphique d'onnxruntime a été écrasée par la version CPU (réinstallation des "
                 "dépendances, « uv run »…). Arrête l'API et le worker, puis lance scripts\\activer_gpu.ps1.")
    gpus = [short_name(n) for n in gpu_names()] if acc != "cpu" else []
    return {"accelerator": acc, "label": LABELS[acc], "gpus": gpus, "error": error}
