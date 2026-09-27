"""Crée ou met à jour ton Space GPU privé sur Hugging Face (ZeroGPU) et le branche à l'appli.

Usage :
  .venv\\Scripts\\python.exe scripts\\deploy_space.py --space ton-pseudo/faceswap-gpu --save

- Le jeton Hugging Face (droit « write ») est demandé s'il n'est pas passé par --token ou HF_TOKEN.
- Le Space est créé en PRIVÉ, sur le matériel ZeroGPU ; une clé secrète APP_KEY est générée et enregistrée dans ses secrets.
- --save enregistre Space + jeton + clé dans l'appli (data/app.db) : la case « Utiliser le GPU » devient disponible.
- Relancer le script met simplement le code à jour (la clé existante est conservée si --save l'a enregistrée).

C'est une action visible sur ton compte Hugging Face : elle n'est jamais lancée par l'appli elle-même.
"""
from __future__ import annotations

import argparse
import getpass
import os
import re
import secrets
import shutil
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app import db  # noqa: E402
from src.netfix import drop_broken_cert_vars  # noqa: E402

drop_broken_cert_vars()  # sinon huggingface_hub (httpx) plante sur un certificat absent

ZEROGPU = "zero-a10g"  # identifiant Hugging Face du matériel ZeroGPU


def build_bundle(dst: Path) -> None:
    """Dossier envoyé au Space : app Gradio + code du pipeline + config GPU (modèles et données dans /tmp)."""
    for name in ("app.py", "README.md", "requirements.txt", "packages.txt"):
        shutil.copy2(ROOT / "space" / name, dst / name)
    shutil.copytree(ROOT / "src", dst / "src", ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    cfg["device"] = "cuda"
    cfg["paths"] = {"models": "/tmp/faceswap/models", "data": "/tmp/faceswap/data"}
    cfg["ffmpeg"], cfg["ffprobe"] = "ffmpeg", "ffprobe"
    (dst / "config.space.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--space", required=True, help="identifiant du Space, ex. ton-pseudo/faceswap-gpu")
    ap.add_argument("--token", help="jeton Hugging Face (sinon HF_TOKEN, sinon demandé)")
    ap.add_argument("--save", action="store_true", help="enregistrer les réglages dans l'appli")
    args = ap.parse_args()

    if not re.fullmatch(r"[\w.-]+/[\w.-]+", args.space):
        sys.exit("--space doit être de la forme ton-pseudo/nom-du-space")
    token = args.token or os.environ.get("HF_TOKEN") or getpass.getpass("Jeton Hugging Face (write) : ").strip()
    if not token:
        sys.exit("Jeton requis : https://huggingface.co/settings/tokens")

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    user = api.whoami()["name"]
    print(f"Compte Hugging Face : {user}")

    db.init()
    saved_key = db.get_meta("zerogpu_key") if db.get_meta("zerogpu_space") == args.space else None
    key = saved_key or secrets.token_urlsafe(24)

    print(f"Space {args.space} (privé, ZeroGPU)…")
    api.create_repo(args.space, repo_type="space", space_sdk="gradio", private=True, exist_ok=True)
    api.add_space_secret(args.space, "APP_KEY", key, description="Clé partagée avec l'appli Faceswap Studio")

    with tempfile.TemporaryDirectory() as tmp:
        build_bundle(Path(tmp))
        api.upload_folder(repo_id=args.space, repo_type="space", folder_path=tmp,
                          commit_message="Déploiement Faceswap Studio GPU")
    try:
        api.request_space_hardware(args.space, ZEROGPU)
    except Exception as exc:  # compte sans accès ZeroGPU (ex. compte de moins de 30 jours)
        print(f"! Matériel ZeroGPU refusé : {exc}\n  Vérifie ton compte (e-mail vérifié, plus de 30 jours) ou passe en PRO.")

    if args.save:
        db.set_meta("zerogpu_space", args.space)
        db.set_meta("zerogpu_token", token)
        db.set_meta("zerogpu_key", key)
        print("✓ Réglages enregistrés dans l'appli.")
    else:
        print(f"Clé APP_KEY à coller dans l'appli (Moteur → ZeroGPU) : {key}")
    print(f"✓ Code envoyé. Construction en cours : https://huggingface.co/spaces/{args.space}")
    print("  Compte 5 à 15 min au premier déploiement, puis « Tester » dans l'appli (Moteur → ZeroGPU).")


if __name__ == "__main__":
    main()
