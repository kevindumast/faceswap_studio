"""Télécharge les poids dans models/.

Usage :
  python scripts/download_models.py                # base : détection + ArcFace + inswapper (niveau 1)
  python scripts/download_models.py --level tone   # + segmentation du visage (niveau 2 : teint)
  python scripts/download_models.py --all

Licence : modèles InsightFace = usage non commercial uniquement.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import models  # noqa: E402

GROUPS = {"base": "détection + ArcFace + inswapper", "tone": "segmentation du visage (BiSeNet)"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", choices=[g for g in GROUPS if g != "base"], help="groupe supplémentaire")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    groups = list(GROUPS) if args.all else ["base"] + ([args.level] if args.level else [])
    for group in groups:
        if models.is_ready(group):
            print(f"✓ {group} ({GROUPS[group]}) déjà présent")
            continue
        print(f"↓ {group} ({GROUPS[group]})")
        try:
            models.ensure(group, lambda f: print(f"\r  {f:6.1%}", end="", flush=True))
        except models.DownloadError as exc:
            sys.exit(f"\n{exc}")
        print(f"\n✓ {group}")


if __name__ == "__main__":
    main()
