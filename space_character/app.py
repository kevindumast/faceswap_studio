"""Niveau 4 de Faceswap Studio : remplacer toute la personne (corps, habits, gestuelle), sur un Space ZeroGPU privé.

Wan2.2-Animate-14B (Apache 2.0, équipe Wan d'Alibaba) en mode « replacement » : le squelette, le visage et la
silhouette de la personne choisie sont extraits de l'extrait, puis la personne de la photo est générée à sa place,
avec la lumière de la scène (LoRA de rééclairage). Le code officiel est téléchargé à un commit fixé, sans modification
de ses fichiers ; seuls trois points sont adaptés ici :
- la personne remplacée est celle choisie dans l'appli (targeting.py), pas la plus grande de l'image ;
- l'attention passe par PyTorch (SDPA) au lieu de flash-attn, absent sur ZeroGPU ;
- le texte (fixe) est encodé une fois au démarrage, sur CPU : l'encodeur T5 (11 Go) n'occupe jamais le GPU.

Reçoit : l'extrait coupé (sans son), une photo de la personne (en pied de préférence), l'instant et le cadre du visage
à remplacer. Renvoie : la vidéo générée (30 i/s) + le masque de la silhouette, pour que l'appli recolle la personne en
pleine résolution sur l'image d'origine. Chaque appel doit fournir la clé APP_KEY (secret du Space).
"""
from __future__ import annotations

import gc
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import types
import urllib.request
from importlib import metadata
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
VERSION = "1"
APP_KEY = os.environ.get("APP_KEY", "")
WAN_COMMIT = "1ea34ff48f87168174e12956e200b1d908b1c5ff"          # github.com/Wan-Video/Wan2.2
WAN_ROOT = Path(os.environ.get("WAN_ROOT", "/tmp/wan2.2"))
SAM2_COMMIT = "0e78a118995e66bb27d78518c4bd9a3e95b4e266"         # celui des requirements officiels de Wan-Animate
SAM2_ROOT = Path(os.environ.get("SAM2_ROOT", "/tmp/sam2"))
CKPT = Path(os.environ.get("WAN_CKPT", "/tmp/Wan2.2-Animate-14B"))
WEIGHTS = [  # ~57 Go : tout le dépôt sauf les poids inutiles (xlm-roberta complet, SAM2 plus petits, doublon du LoRA)
    "config.json", "diffusion_pytorch_model*", "Wan2.1_VAE.pth", "models_clip_*.pth", "models_t5_*.pth",
    "google/umt5-xxl/*", "xlm-roberta-large/*.json", "xlm-roberta-large/*.model", "relighting_lora.ckpt",
    "process_checkpoint/det/*", "process_checkpoint/pose2d/*", "process_checkpoint/sam2/sam2_hiera_large.pt",
]
FPS = 30                                               # cadence de travail de Wan-Animate
RESOLUTIONS = {"360p": (640, 360), "480p": (832, 480)}  # surface visée, le format de la vidéo est conservé
MAX_SECONDS = 10.5
DRY_RUN = os.environ.get("FACESWAP_SPACE_DRY_RUN") == "1"   # vérification locale : code et imports, sans poids ni GPU


def gpu_seconds(seconds: float, resolution: str, steps: int) -> float:
    """Temps de GPU attendu (même formule que l'appli, config.yaml → levels.character)."""
    per_s = {"360p": 12.0, "480p": 26.0}.get(resolution, 12.0)
    return 45 + seconds * per_s * steps / 6


def _fix_onnxruntime() -> None:
    """Si un paquet a tiré onnxruntime (CPU), il masque onnxruntime-gpu : on remet la version GPU."""
    installed = {d.metadata["Name"].lower() for d in metadata.distributions()}
    if "onnxruntime" in installed and "onnxruntime-gpu" in installed:
        pip = [sys.executable, "-m", "pip", "--disable-pip-version-check", "-q"]
        subprocess.run([*pip, "uninstall", "-y", "onnxruntime"], check=False)
        subprocess.run([*pip, "install", "--force-reinstall", "--no-deps", "onnxruntime-gpu"], check=True)


def _fetch_github(repo: str, commit: str, dest: Path) -> None:
    """Code source d'un dépôt GitHub à un commit fixé (téléchargé une fois par démarrage du Space)."""
    marker = dest / ".commit"
    if marker.is_file() and marker.read_text().strip() == commit:
        return
    tmp = Path(tempfile.mkdtemp())
    archive = tmp / "src.tar.gz"
    urllib.request.urlretrieve(f"https://codeload.github.com/{repo}/tar.gz/{commit}", archive)
    with tarfile.open(archive) as tar:
        tar.extractall(tmp, **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))
    shutil.rmtree(dest, ignore_errors=True)
    shutil.move(str(next(p for p in tmp.iterdir() if p.is_dir())), dest)
    marker.write_text(commit)


def _fetch_code() -> None:
    """Wan2.2 (importable sans son __init__, qui charge tous les modèles Wan) et SAM2 (pur Python, rien à compiler :
    son extension CUDA est facultative et demanderait le kit CUDA complet pour s'installer)."""
    _fetch_github("Wan-Video/Wan2.2", WAN_COMMIT, WAN_ROOT)
    _fetch_github("facebookresearch/sam2", SAM2_COMMIT, SAM2_ROOT)
    sys.path.insert(0, str(WAN_ROOT / "wan" / "modules" / "animate" / "preprocess"))
    sys.path.append(str(SAM2_ROOT))
    pkg = types.ModuleType("wan")
    pkg.__path__ = [str(WAN_ROOT / "wan")]
    sys.modules["wan"] = pkg


if os.environ.get("SPACE_ID"):  # seulement sur Hugging Face
    _fix_onnxruntime()
_fetch_code()

try:  # ZeroGPU : le GPU n'est attribué que pendant les fonctions décorées (import avant torch)
    import spaces

    gpu = spaces.GPU
except ImportError:  # hors ZeroGPU : décorateur sans effet
    def gpu(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda fn: fn

import cv2  # noqa: E402
import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402

try:
    import onnxruntime

    onnxruntime.preload_dlls()  # bibliothèques CUDA/cuDNN installées avec torch
except Exception:
    pass

if not DRY_RUN:
    print("Poids Wan2.2-Animate-14B…", flush=True)
    snapshot_download("Wan-AI/Wan2.2-Animate-14B", local_dir=str(CKPT), allow_patterns=WEIGHTS)

import wan.animate as wan_animate  # noqa: E402
from wan.configs import WAN_CONFIGS  # noqa: E402

# Prétraitement officiel (dossier preprocess) : squelette, visage, silhouette.
from decord import VideoReader  # noqa: E402
from human_visualization import draw_aapose_by_meta_new  # noqa: E402
from pose2d import Pose2d  # noqa: E402
from pose2d_utils import AAPoseMeta, load_pose_metas_from_kp2ds_seq  # noqa: E402
from process_pipepline import ProcessPipeline  # noqa: E402
from sam_utils import build_sam2_video_predictor  # noqa: E402
from utils import get_aug_mask, get_face_bboxes, get_frame_indices, get_mask_body_img, padding_resize, resize_by_area  # noqa: E402

from targeting import pick_track  # noqa: E402

HALF = (torch.float16, torch.bfloat16)


def sdpa_attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0.0, softmax_scale=None, q_scale=None, causal=False,
                   window_size=(-1, -1), deterministic=False, dtype=torch.bfloat16, version=None):
    """Même contrat que wan.modules.attention.flash_attention ([B, L, N, C]), via l'attention intégrée à PyTorch."""
    out_dtype = q.dtype
    if q_scale is not None:
        q = q * q_scale
    q, k, v = (x if x.dtype in HALF else x.to(dtype) for x in (q, k, v))
    mask = None
    lk = k.size(1)
    if k_lens is not None and bool((k_lens < lk).any()):  # clés de remplissage à ignorer
        mask = (torch.arange(lk, device=k.device)[None, :] < k_lens.to(k.device)[:, None])[:, None, None, :]
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask,
                                         dropout_p=dropout_p, is_causal=causal, scale=softmax_scale)
    return out.transpose(1, 2).contiguous().to(out_dtype)


def sdpa_flash_attn_func(q, k, v, *args, **kwargs):
    """Même contrat que flash_attn.flash_attn_func ([B, S, H, D]), utilisé par l'adaptateur de visage."""
    return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)


for _name, _mod in list(sys.modules.items()):
    if _name.startswith("wan.") and hasattr(_mod, "flash_attention"):
        _mod.flash_attention = sdpa_attention
    if _name.startswith("wan.") and hasattr(_mod, "flash_attn_func"):
        _mod.flash_attn_func = sdpa_flash_attn_func

CFG = WAN_CONFIGS["animate-14B"]


class FixedPrompts:
    """Le texte de Wan-Animate est fixe : encodé une fois ici (CPU, hors quota), puis l'encodeur T5 est libéré."""

    def __init__(self, encoder, prompts):
        with torch.no_grad():
            self.cache = {p: [t.cpu() for t in encoder([p], torch.device("cpu"))] for p in prompts}

    def __call__(self, texts, device):
        return [t.to(device) for t in self.cache[texts[0]]]


MODEL = MASKER = None
if not DRY_RUN:
    print("Chargement du modèle…", flush=True)
    MODEL = wan_animate.WanAnimate(config=CFG, checkpoint_dir=str(CKPT), device_id=0, rank=0, t5_cpu=True,
                                   init_on_cpu=False, convert_model_dtype=True, use_relighting_lora=True)
    print("Encodage du texte (une fois)…", flush=True)
    torch.set_num_threads(os.cpu_count() or 4)
    MODEL.text_encoder = FixedPrompts(MODEL.text_encoder, [MODEL.sample_prompt, MODEL.sample_neg_prompt])
    gc.collect()
    MASKER = ProcessPipeline.__new__(ProcessPipeline)   # seulement pour sa méthode get_mask (SAM2 guidé par le squelette)
    MASKER.predictor = build_sam2_video_predictor("sam2_hiera_l.yaml",
                                                  str(CKPT / "process_checkpoint/sam2/sam2_hiera_large.pt"))
print("Prêt.", flush=True)


class Report:
    """Progression unique pour l'appli : squelette 12 %, silhouette 8 %, génération 80 % (étapes de débruitage)."""

    def __init__(self, progress, frames: int, gen_steps: int):
        self.progress, self.frames, self.gen_steps, self.gen_done = progress, max(1, frames), max(1, gen_steps), 0

    def __call__(self, stage: str, done: int = 0) -> None:
        if stage == "squelette":
            frac = 0.12 * done / self.frames
        elif stage == "silhouette":
            frac = 0.12
        else:
            frac = 0.20 + 0.80 * done / self.gen_steps
        self.progress((min(999, int(frac * 1000)), 1000), desc=stage, unit="‰")

    def step(self) -> None:
        self.gen_done += 1
        self("génération", self.gen_done)


REPORT: Report | None = None


class StepTicker:
    """Remplace tqdm dans wan.animate : chaque étape de débruitage fait avancer la progression."""

    def __init__(self, iterable, *args, **kwargs):
        self.iterable = iterable

    def __iter__(self):
        for item in self.iterable:
            yield item
            if REPORT is not None:
                REPORT.step()


wan_animate.tqdm = StepTicker


def _check(key: str) -> None:
    if not APP_KEY or key != APP_KEY:
        raise gr.Error("Clé APP_KEY invalide.")


def read_frames(path: str, resolution: str) -> list[np.ndarray]:
    """Images RGB à 30 i/s, redimensionnées à la surface de travail (comme le prétraitement officiel)."""
    reader = VideoReader(path)
    frame_num, video_fps = len(reader), reader.get_avg_fps()
    target = int(frame_num / video_fps * FPS)
    frames = reader.get_batch(get_frame_indices(frame_num, video_fps, target, FPS)).asnumpy()
    w, h = RESOLUTIONS[resolution]
    return [resize_by_area(f, w * h, divisor=16) for f in frames]


def targeted_pose(pose2d, frames, anchor: int, face_box, report) -> tuple[list, int]:
    """Squelettes de LA personne choisie : silhouettes détectées partout, puis suivies depuis le visage choisi."""
    images = pose2d.load_images(list(frames))
    h, w = images[0].shape[:2]
    candidates: list[np.ndarray] = []

    def record(bboxes, shape_raw):
        candidates[-1] = np.array([b[:5] for b in bboxes], dtype=float)
        return 0

    pose2d.detector.sorted_func = record
    for i, img in enumerate(images):
        candidates.append(np.zeros((0, 5)))
        x, shape = pose2d.detector.preprocess(img)
        pose2d.detector(x[None], shape[None])
        if i % 10 == 0:
            report("squelette", i // 2)
    track = pick_track(candidates, anchor, face_box)
    if all(b is None for b in track):
        raise gr.Error("Aucune personne détectée dans ce passage.")
    kp2ds = []
    for i, (img, box) in enumerate(zip(images, track)):
        x, center, scale = pose2d.model.preprocess(img, box)
        kp2ds.append(pose2d.model(x[None], center[None], scale[None]))
        if i % 10 == 0:
            report("squelette", len(images) // 2 + i // 2)
    seen = sum(1 for c in candidates if len(c))
    return load_pose_metas_from_kp2ds_seq(np.concatenate(kp2ds, 0), width=w, height=h), seen


def write_rgb(path: Path, frames, fps: int = FPS, crf: int = 14) -> None:
    h, w = frames[0].shape[:2]
    proc = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
                             "-crf", str(crf), "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(np.ascontiguousarray(f, dtype=np.uint8).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise gr.Error("Encodage vidéo échoué sur le Space.")


def health(key: str) -> str:
    _check(key)
    return json.dumps({"version": VERSION, "levels": ["character"], "zerogpu": bool(os.environ.get("SPACE_ID")),
                       "resolutions": list(RESOLUTIONS)})


def _duration(clip: str, ref: str, payload: str, key: str, progress=None) -> int:
    """Durée de GPU réservée : estimation + 30 % de marge (le quota ne décompte que le temps réellement utilisé)."""
    try:
        data = json.loads(payload)
        seconds = min(MAX_SECONDS, VideoReader(clip).get_frame_timestamp(-1)[-1])
        return int(min(600, 1.3 * gpu_seconds(seconds, data.get("resolution", "360p"), int(data.get("steps", 6)))))
    except Exception:
        return 240


@gpu(duration=_duration, size="large")
def replace(clip: str, ref: str, payload: str, key: str, progress=gr.Progress()) -> tuple[str, str, str]:
    global REPORT
    _check(key)
    t0 = time.time()
    data = json.loads(payload)
    resolution = data.get("resolution", "360p")
    if resolution not in RESOLUTIONS:
        raise gr.Error(f"Résolution inconnue : {resolution}")
    steps, seed = int(data.get("steps", 6)), int(data.get("seed", 42))
    work = Path(tempfile.mkdtemp())

    frames = read_frames(clip, resolution)
    if len(frames) > MAX_SECONDS * FPS:
        raise gr.Error("Extrait trop long pour le niveau 4 (10 s maximum).")
    h, w = frames[0].shape[:2]
    clips = math.ceil((MODEL.get_valid_len(len(frames), CFG.frame_num, overlap=1) - 1) / (CFG.frame_num - 1))
    report = REPORT = Report(progress, len(frames), clips * steps)
    anchor = min(len(frames) - 1, round(float(data["t"]) * FPS))
    x1, y1, x2, y2 = data["box"]
    face_box = (x1 * w, y1 * h, x2 * w, y2 * h)

    # 1. Squelette + visage de la personne choisie (ONNX sur le GPU, créé pendant l'appel).
    pose2d = Pose2d(checkpoint=str(CKPT / "process_checkpoint/pose2d/vitpose_h_wholebody.onnx"),
                    detector_checkpoint=str(CKPT / "process_checkpoint/det/yolov10m.onnx"))
    metas, seen = targeted_pose(pose2d, frames, anchor, face_box, report)
    del pose2d
    faces, last = [], None
    for idx, meta in enumerate(metas):
        fx1, fx2, fy1, fy2 = get_face_bboxes(meta["keypoints_face"][:, :2], scale=1.3, image_shape=(h, w))
        crop = frames[idx][fy1:fy2, fx1:fx2]
        last = cv2.resize(crop, (512, 512)) if crop.size else (last if last is not None else np.zeros((512, 512, 3), np.uint8))
        faces.append(last)

    refer = cv2.imread(ref)
    if refer is None:
        raise gr.Error("Photo de la personne illisible.")
    cv2.imwrite(str(work / "src_ref.png"), refer)
    refer = padding_resize(refer[..., ::-1], h, w)
    poses = [draw_aapose_by_meta_new(np.zeros_like(refer), AAPoseMeta.from_humanapi_meta(m)) for m in metas]

    # 2. Silhouette (SAM2 guidé par le squelette), fond sans la personne.
    report("silhouette")
    masks = MASKER.get_mask(frames, 400, metas)
    backgrounds, aug_masks = [], []
    aug = np.zeros((h, w), np.uint8)
    for frame, mask in zip(frames, masks):
        _, body = get_mask_body_img(frame, mask, iterations=3, k=7)
        if body.any():  # silhouette vide sur cette image : on garde la zone précédente
            aug = get_aug_mask(body, w_len=1, h_len=1)   # rectangle autour de la personne : zone régénérée
        backgrounds.append(frame * (1 - aug[:, :, None]))
        aug_masks.append(aug)
    for name, seq in (("src_face", faces), ("src_pose", poses), ("src_bg", backgrounds),
                      ("src_mask", [np.repeat((m * 255)[:, :, None], 3, axis=2) for m in aug_masks])):
        write_rgb(work / f"{name}.mp4", seq)
    del masks, backgrounds
    gc.collect()
    torch.cuda.empty_cache()

    # 3. Génération : la personne de la photo, à la place, avec la lumière de la scène.
    video = MODEL.generate(str(work), replace_flag=True, clip_len=CFG.frame_num, refert_num=1, shift=CFG.sample_shift,
                           sample_solver="dpm++", sampling_steps=steps, guide_scale=1.0, seed=seed, offload_model=False)
    REPORT = None
    out = ((video.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()
    result, mask_path = work / "result.mp4", work / "mask.mp4"
    write_rgb(result, list(out))
    write_rgb(mask_path, [np.repeat((m * 255)[:, :, None], 3, axis=2) for m in aug_masks[: len(out)]])

    warnings = []
    if seen < 0.8 * len(frames):
        warnings.append(f"Personne non détectée sur {len(frames) - seen}/{len(frames)} images : "
                        "sa dernière position connue a été gardée (occultation, sortie du cadre).")
    stats = {"frames": len(out), "fps": FPS, "width": w, "height": h, "resolution": resolution, "steps": steps,
             "warnings": warnings, "gpu_seconds": round(time.time() - t0, 1)}
    return str(result), str(mask_path), json.dumps(stats)


with gr.Blocks(title="Faceswap Studio · Niveau 4") as demo:
    gr.Markdown("### Moteur « personne entière » de Faceswap Studio\nCe Space privé est appelé par l'appli sur ton PC ; "
                "il n'y a rien à faire ici.")
    key_in = gr.Textbox(visible=False)
    clip_in = gr.File(visible=False)
    ref_in = gr.File(visible=False)
    payload_in = gr.Textbox(visible=False)
    health_out = gr.Textbox(visible=False)
    video_out = gr.File(visible=False)
    mask_out = gr.File(visible=False)
    stats_out = gr.Textbox(visible=False)
    gr.Button(visible=False).click(health, [key_in], [health_out], api_name="health")
    gr.Button(visible=False).click(replace, [clip_in, ref_in, payload_in, key_in], [video_out, mask_out, stats_out],
                                   api_name="replace")

demo.queue(default_concurrency_limit=1)

if __name__ == "__main__":
    demo.launch(server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"))
