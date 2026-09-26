"""Télécharge les poids dans models/ : pack buffalo_l (détection + ArcFace) et inswapper_128.

Usage : python scripts/download_models.py
Licence : modèles InsightFace = usage non commercial uniquement.
"""
from __future__ import annotations

import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config  # noqa: E402


def fetch(url: str, dst: Path) -> None:
    tmp = dst.with_suffix(dst.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "faceswap-studio"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as f:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while chunk := resp.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {dst.name} : {done / total:6.1%} ({done >> 20} / {total >> 20} Mo)", end="", flush=True)
    print()
    tmp.replace(dst)


def main() -> None:
    cfg = load_config()
    models = cfg.path("models")
    models.mkdir(parents=True, exist_ok=True)

    pack = models / cfg.models.detector_pack
    if any(pack.glob("*.onnx")):
        print(f"✓ {pack.name} déjà présent")
    else:
        zip_path = models / f"{pack.name}.zip"
        print(f"↓ {pack.name}")
        fetch(cfg.models.buffalo_l_url, zip_path)
        pack.mkdir(exist_ok=True)
        with zipfile.ZipFile(zip_path) as z:
            for member in z.namelist():
                if member.endswith(".onnx"):
                    (pack / Path(member).name).write_bytes(z.read(member))
        zip_path.unlink()
        print(f"✓ {pack.name} : {', '.join(p.name for p in pack.glob('*.onnx'))}")

    swapper = models / cfg.models.inswapper
    if swapper.is_file():
        print(f"✓ {swapper.name} déjà présent")
        return
    for url in cfg.models.inswapper_urls:
        print(f"↓ {swapper.name} depuis {url.split('/')[2]}")
        try:
            fetch(url, swapper)
        except Exception as exc:  # miroir mort : on essaie le suivant
            print(f"  échec : {exc}")
            continue
        if swapper.stat().st_size < 400 << 20:
            print("  fichier trop petit, miroir suspect")
            swapper.unlink()
            continue
        print(f"✓ {swapper.name}")
        return
    sys.exit("Aucun miroir n'a fonctionné : ajoutez une URL valide dans config.yaml (models.inswapper_urls).")


if __name__ == "__main__":
    main()
