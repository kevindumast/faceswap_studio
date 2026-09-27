"""Reprend les personnes d'une ancienne session de visages (avant la bibliothèque) dans la bibliothèque.

Usage : python scripts/importer_ancienne_session.py <id_session> [<id_session> …]
Les personnes sont créées sous des noms par défaut (« Personne 1 »…) : renomme-les ensuite dans « Personnes ».
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import db, library  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for set_id in sys.argv[1:]:
        d = db.folder("faces", set_id)
        if not (d / "set.json").is_file():
            print(f"✗ {set_id} : session introuvable")
            continue
        created = library.import_legacy_set(d)
        if not created:
            print(f"= {set_id} : déjà au nouveau format, rien à reprendre")
            continue
        for pid in created:
            person = library.load(pid)
            print(f"✓ {set_id} → {person['name']} ({len(person['photos'])} photo(s))")


if __name__ == "__main__":
    main()
