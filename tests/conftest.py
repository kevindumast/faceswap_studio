"""Chaque session de tests travaille dans un data/ temporaire (jamais dans le vrai data/)."""
import os
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session", autouse=True)
def isolated_config(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("faceswap")
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    cfg["paths"]["data"] = str(tmp / "data")
    cfg["paths"]["models"] = str(ROOT / "models")
    path = tmp / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    os.environ["FACESWAP_CONFIG"] = str(path)
    from src.config import load_config

    load_config.cache_clear()
    yield tmp


@pytest.fixture
def make_session():
    """Crée des personnes de bibliothèque aux empreintes connues + une session de vidéo qui les utilise.

    make({"Kevin": [emb, …], "Pote": [emb]}) → (set_id, {"Kevin": person_id, "Pote": person_id})
    """
    import json
    import time

    import numpy as np

    def make(embeddings: dict) -> tuple[str, dict[str, str]]:
        from app import db, library

        ids = {}
        for label, embs in embeddings.items():
            person = library.create(label)
            d = library.root() / person["id"]
            for e in embs:
                ph = db.new_id()
                np.save(d / f"{ph}.npy", (np.asarray(e) / np.linalg.norm(e)).astype(np.float32))
                person["photos"].append({"id": ph, "name": f"{ph}.jpg", "added_at": time.time()})
            library._save(person)
            ids[label] = person["id"]
        set_id = db.new_id()
        d = db.folder("faces", set_id)
        d.mkdir(parents=True)
        (d / "set.json").write_text(json.dumps({"people": list(ids.values()), "rejected": []}), encoding="utf-8")
        return set_id, ids

    return make


@pytest.fixture(scope="session")
def sample_video(isolated_config) -> Path:
    """Vidéo de synthèse de 8 s à 25 i/s avec son (mire + bip), générée par ffmpeg."""
    from src import media

    out = isolated_config / "sample.mp4"
    media.ffmpeg("-f", "lavfi", "-i", "testsrc2=size=640x360:rate=25:duration=8",
                 "-f", "lavfi", "-i", "sine=frequency=440:duration=8",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out))
    return out
