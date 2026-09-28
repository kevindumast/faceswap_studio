"""Téléchargement des poids par groupe (base, tone…), utilisable en CLI et depuis l'API avec progression."""
from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path
from typing import Callable

from .config import load_config

Progress = Callable[[float], None]


class DownloadError(RuntimeError):
    pass


def _fetch(url: str, dst: Path, on_progress: Progress | None = None) -> None:
    tmp = dst.with_suffix(dst.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "faceswap-studio"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as f:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while chunk := resp.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total and on_progress:
                on_progress(done / total)
    tmp.replace(dst)


def _fetch_first(urls: list[str], dst: Path, min_bytes: int, on_progress: Progress | None) -> None:
    errors = []
    for url in urls:
        try:
            _fetch(url, dst, on_progress)
        except Exception as exc:  # miroir mort : on essaie le suivant
            errors.append(f"{url.split('/')[2]} : {exc}")
            continue
        if dst.stat().st_size >= min_bytes:
            return
        errors.append(f"{url.split('/')[2]} : fichier trop petit")
        dst.unlink()
    raise DownloadError(f"{dst.name} introuvable ({' ; '.join(errors)}). Ajoutez une URL valide dans config.yaml.")


def group_files(group: str) -> list[Path]:
    cfg = load_config()
    models = cfg.path("models")
    if group == "base":
        return [models / cfg.models.detector_pack / "det_10g.onnx", models / cfg.models.detector_pack / "w600k_r50.onnx",
                models / cfg.models.inswapper]
    if group == "tone":
        return [models / cfg.models.parser]
    if group == "head":
        from .head import FILES, model_path

        return [model_path(name) for name in FILES]
    if group == "restore":
        return [models / cfg.models.restore]
    raise KeyError(group)


def is_ready(group: str) -> bool:
    return all(p.is_file() for p in group_files(group))


def ensure(group: str, on_progress: Progress | None = None) -> None:
    """Télécharge ce qui manque dans le groupe. Progression globale 0..1."""
    cfg = load_config()
    models = cfg.path("models")
    models.mkdir(parents=True, exist_ok=True)
    report = on_progress or (lambda _: None)
    if group == "base":
        pack = models / cfg.models.detector_pack
        if not any(pack.glob("*.onnx")):
            zip_path = models / f"{pack.name}.zip"
            _fetch(cfg.models.buffalo_l_url, zip_path, lambda f: report(f * 0.3))
            pack.mkdir(exist_ok=True)
            with zipfile.ZipFile(zip_path) as z:
                for member in z.namelist():
                    if member.endswith(".onnx"):
                        (pack / Path(member).name).write_bytes(z.read(member))
            zip_path.unlink()
        swapper = models / cfg.models.inswapper
        if not swapper.is_file():
            _fetch_first(list(cfg.models.inswapper_urls), swapper, 400 << 20, lambda f: report(0.3 + f * 0.7))
    elif group == "tone":
        parser = models / cfg.models.parser
        if not parser.is_file():
            _fetch_first(list(cfg.models.parser_urls), parser, 50 << 20, report)
    elif group == "head":
        from .head import FILES, model_path

        urls = cfg.models.get("head_urls", {})
        sizes = {"feature_extractor": 3, "motion_extractor": 107, "generator": 212, "stitcher": 0.2, "lama": 198}
        total, done = sum(sizes.values()), 0.0
        for name in FILES:
            dst = model_path(name)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.is_file():
                share = sizes[name] / total
                _fetch_first([u.format(file=dst.name) for u in urls.get(name, urls.get("live_portrait", []))], dst,
                             int(sizes[name] * 0.8 * (1 << 20)), lambda f, d=done, s=share: report(d + f * s))
            done += sizes[name] / total
    elif group == "restore":
        path = models / cfg.models.restore
        if not path.is_file():
            _fetch_first(list(cfg.models.get("restore_urls", [])), path, 300 << 20, report)
    else:
        raise KeyError(group)
    report(1.0)
