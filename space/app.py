"""Moteur GPU privé de Faceswap Studio, à héberger sur un Space Hugging Face ZeroGPU.

Il n'a pas d'interface à lui : l'appli locale l'appelle (gradio_client) quand la case « Utiliser le GPU » est cochée.
Il reçoit un extrait déjà découpé (sans son) + les empreintes des personnes, et renvoie l'extrait remplacé.
Découpe, son, étiquette IA et réinsertion dans la vidéo complète restent sur le PC.

Chaque appel doit fournir la clé APP_KEY (secret du Space), en plus du Space privé.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from importlib import metadata
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("FACESWAP_CONFIG", str(HERE / "config.space.yaml"))
VERSION = "1"
APP_KEY = os.environ.get("APP_KEY", "")


def _fix_onnxruntime() -> None:
    """insightface réinstalle d'office onnxruntime (CPU), qui masque onnxruntime-gpu : on remet la version GPU."""
    installed = {d.metadata["Name"].lower() for d in metadata.distributions()}
    if "onnxruntime" in installed and "onnxruntime-gpu" in installed:
        pip = [sys.executable, "-m", "pip", "--disable-pip-version-check", "-q"]
        subprocess.run([*pip, "uninstall", "-y", "onnxruntime"], check=False)
        subprocess.run([*pip, "install", "--force-reinstall", "--no-deps", "onnxruntime-gpu"], check=True)


if os.environ.get("SPACE_ID"):  # seulement sur Hugging Face, jamais sur le PC
    _fix_onnxruntime()

from src.netfix import drop_broken_cert_vars  # noqa: E402

drop_broken_cert_vars()

import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402

try:  # ZeroGPU : le GPU n'est attribué que pendant les fonctions décorées
    import spaces

    gpu = spaces.GPU
except ImportError:  # en local : décorateur sans effet
    def gpu(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda fn: fn

from src import media, models  # noqa: E402
from src.identity import SourceFace  # noqa: E402
from src.levels import FACE, FACE_TONE, PersonAssets  # noqa: E402
from src.pipeline import FaceMapping, swap_segment  # noqa: E402
from src.tone import ToneStats  # noqa: E402

LEVELS = [FACE, FACE_TONE]

# Poids téléchargés au démarrage, hors GPU (ne consomme pas de quota).
for group in ("base", "tone"):
    models.ensure(group)


def _check(key: str) -> None:
    if not APP_KEY or key != APP_KEY:
        raise gr.Error("Clé APP_KEY invalide.")


def health(key: str) -> str:
    _check(key)
    return json.dumps({"version": VERSION, "levels": LEVELS, "zerogpu": bool(os.environ.get("SPACE_ID"))})


def _duration(clip: str, payload: str, key: str, progress=None) -> int:
    """Durée de GPU réservée (secondes) : proportionnelle au nombre d'images et de visages, bornée."""
    try:
        info = media.probe(Path(clip))
        frames = info.duration * info.fps
        faces = max(1, len(json.loads(payload).get("mappings", [])))
    except Exception:
        return 120
    return int(min(600, 45 + frames * 0.08 * faces))


@gpu(duration=_duration)
def swap(clip: str, payload: str, key: str, progress=gr.Progress()) -> tuple[str, str]:
    _check(key)
    data = json.loads(payload)
    if data.get("level") not in LEVELS:
        raise gr.Error(f"Niveau non disponible sur ce Space : {data.get('level')}")
    people = {
        pid: PersonAssets(
            source=SourceFace(np.asarray(p["embedding"], np.float32)),
            tone=ToneStats.from_dict(p["tone"]) if p.get("tone") else None,
        )
        for pid, p in data["people"].items()
    }
    mappings = [FaceMapping(people[m["person"]], {"t": m["t"], "box": m["box"]}, m.get("label", "Visage"))
                for m in data["mappings"]]
    out = Path(tempfile.mkdtemp()) / "swapped.mp4"
    t0 = time.time()

    def report(stage: str, done: int, total: int) -> None:
        progress((done, total), desc=stage, unit="images")

    stats = swap_segment(Path(clip), mappings, data["level"], out, stabilize=bool(data.get("stabilize", True)),
                         progress=report)
    result = {"frames": stats.frames, "swapped": stats.swapped, "reused": stats.reused,
              "warnings": stats.warnings, "gpu_seconds": round(time.time() - t0, 1)}
    return str(out), json.dumps(result)


with gr.Blocks(title="Faceswap Studio · GPU") as demo:
    gr.Markdown("### Moteur GPU privé de Faceswap Studio\nCe Space est appelé par l'appli sur ton PC ; il n'y a rien à faire ici.")
    key_in = gr.Textbox(visible=False)
    clip_in = gr.File(visible=False)
    payload_in = gr.Textbox(visible=False)
    health_out = gr.Textbox(visible=False)
    clip_out = gr.File(visible=False)
    stats_out = gr.Textbox(visible=False)
    gr.Button(visible=False).click(health, [key_in], [health_out], api_name="health")
    gr.Button(visible=False).click(swap, [clip_in, payload_in, key_in], [clip_out, stats_out], api_name="swap")

demo.queue(default_concurrency_limit=1)

if __name__ == "__main__":
    demo.launch(server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"))
