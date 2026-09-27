"""Chargement de config.yaml. Le même fichier sert en local (CPU) et en cloud (GPU)."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent


class Config(dict):
    """dict avec accès par attribut (cfg.segment.max_s)."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        return Config(value) if isinstance(value, dict) else value

    def path(self, key: str) -> Path:
        p = Path(self["paths"][key])
        return p if p.is_absolute() else ROOT / p


@lru_cache(maxsize=1)
def load_config() -> Config:
    path = Path(os.environ.get("FACESWAP_CONFIG", ROOT / "config.yaml"))
    with open(path, encoding="utf-8") as f:
        return Config(yaml.safe_load(f))


_ACCELERATORS = {"cuda": "CUDAExecutionProvider", "dml": "DmlExecutionProvider"}
_PACKAGES = {"cuda": "onnxruntime-gpu[cuda,cudnn]", "dml": "onnxruntime-directml"}


class DeviceUnavailable(RuntimeError):
    pass


def accelerator(cfg: Config | None = None) -> str:
    """Moteur effectif : cuda, dml ou cpu.

    `device: auto` prend le meilleur disponible (CUDA, puis DirectML, puis CPU). Un moteur demandé explicitement
    mais absent lève une erreur : jamais de bascule silencieuse vers un CPU dix fois plus lent.
    """
    import onnxruntime

    cfg = cfg or load_config()
    wanted = str(cfg.get("device", "cpu"))
    available = onnxruntime.get_available_providers()
    if wanted == "auto":
        return next((name for name, prov in _ACCELERATORS.items() if prov in available), "cpu")
    if wanted in _ACCELERATORS and _ACCELERATORS[wanted] not in available:
        raise DeviceUnavailable(
            f"config.yaml demande device: {wanted}, mais ce moteur n'est pas installé "
            f"(uv pip install {_PACKAGES[wanted]}). Mettez device: cpu ou device: auto."
        )
    return wanted if wanted in _ACCELERATORS else "cpu"


@lru_cache(maxsize=1)
def _preload_cuda() -> None:
    import onnxruntime

    if hasattr(onnxruntime, "preload_dlls"):  # DLL CUDA / cuDNN installées par pip
        onnxruntime.preload_dlls()


def providers(cfg: Config | None = None, role: str = "swap") -> list:
    """Moteurs onnxruntime pour un modèle. role = "swap" (le plus lourd) ou "analysis" (détection, identité, segmentation).

    DirectML avec `dml_placement: split` (portable avec deux cartes) : le swap garde la carte la plus puissante et toute
    sa mémoire dédiée, l'analyse part sur la carte intégrée (mémoire partagée). Sur une carte dédiée de 2 Go, charger
    tous les modèles au même endroit la sature et divise la vitesse du swap par deux.
    """
    import onnxruntime

    onnxruntime.set_default_logger_severity(3)  # masque les warnings « initializer inutilisé » d'inswapper
    cfg = cfg or load_config()
    acc = accelerator(cfg)
    if acc == "cuda":
        _preload_cuda()
        return [("CUDAExecutionProvider", {"device_id": int(cfg.get("cuda_device_id", 0))}), "CPUExecutionProvider"]
    if acc == "dml":
        # Pas d'index de carte en dur : Windows classe les cartes par puissance.
        split = cfg.get("dml_placement", "single") == "split" and role == "analysis"
        preference = "minimum_power" if split else "high_performance"
        return [("DmlExecutionProvider", {"performance_preference": preference, "device_filter": "gpu"}),
                "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]
