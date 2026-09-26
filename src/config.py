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


def providers(cfg: Config | None = None) -> list[str]:
    import onnxruntime

    onnxruntime.set_default_logger_severity(3)  # masque les warnings « initializer inutilisé » d'inswapper
    cfg = cfg or load_config()
    if cfg.device == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]
