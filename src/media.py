"""Tout ce qui touche à ffmpeg / ffprobe : probe, proxy, filmstrip, découpe, écriture, assemblage."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .config import load_config

# Pas de fenêtre console qui clignote quand l'API lance ffmpeg sous Windows.
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class MediaError(RuntimeError):
    pass


@lru_cache(maxsize=None)
def binary(name: str) -> str:
    """Résout ffmpeg / ffprobe : config, PATH, puis dossier d'installation winget."""
    configured = load_config().get(name, name)
    found = shutil.which(configured)
    if found:
        return found
    winget = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
    if winget.is_dir():
        for exe in winget.glob(f"*FFmpeg*/**/bin/{name}.exe"):
            return str(exe)
    raise MediaError(f"{name} introuvable. Installez-le (winget install Gyan.FFmpeg) ou renseignez son chemin dans config.yaml.")


def run(args: list[str]) -> subprocess.CompletedProcess:
    proc = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace", creationflags=_NO_WINDOW)
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-12:])
        raise MediaError(f"{Path(args[0]).stem} a échoué :\n{tail}")
    return proc


def ffmpeg(*args: str) -> subprocess.CompletedProcess:
    return run([binary("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y", *args])


@dataclass
class VideoInfo:
    duration: float
    width: int
    height: int
    fps: float
    fps_str: str        # fraction exacte, ex. "30000/1001"
    has_audio: bool
    codec: str

    def to_dict(self) -> dict:
        return asdict(self)


def probe(path: Path) -> VideoInfo:
    out = run([binary("ffprobe"), "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)]).stdout
    data = json.loads(out)
    video = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    if video is None:
        raise MediaError("Aucune piste vidéo dans ce fichier.")
    rate = video.get("avg_frame_rate") or "0/1"
    if rate in ("0/0", "0/1"):
        rate = video.get("r_frame_rate") or "30/1"
    frac = Fraction(rate)
    if frac <= 0 or frac > 240:
        frac = Fraction(30)
    duration = float(data.get("format", {}).get("duration") or video.get("duration") or 0)
    width, height = int(video["width"]), int(video["height"])
    rotation = _rotation(video)
    if rotation in (90, 270):
        width, height = height, width
    return VideoInfo(
        duration=duration,
        width=width,
        height=height,
        fps=float(frac),
        fps_str=f"{frac.numerator}/{frac.denominator}",
        has_audio=any(s.get("codec_type") == "audio" for s in data.get("streams", [])),
        codec=video.get("codec_name", "?"),
    )


def _rotation(stream: dict) -> int:
    for side in stream.get("side_data_list", []) or []:
        if "rotation" in side:
            return abs(int(side["rotation"])) % 360
    return abs(int(stream.get("tags", {}).get("rotate", 0))) % 360


def _even(x: float) -> int:
    return max(2, int(round(x / 2)) * 2)


def scaled_size(info: VideoInfo, max_height: int) -> tuple[int, int]:
    if info.height <= max_height:
        return _even(info.width), _even(info.height)
    return _even(info.width * max_height / info.height), max_height


def make_proxy(src: Path, dst: Path, info: VideoInfo, max_height: int,
               on_progress: Callable[[float], None] | None = None) -> None:
    """H.264 lisible par tous les navigateurs, keyframes serrées pour un seek fluide dans le trimmer."""
    w, h = scaled_size(info, max_height)
    gop = max(1, round(info.fps / 2))
    args = [binary("ffmpeg"), "-hide_banner", "-loglevel", "error", "-nostats", "-progress", "pipe:1", "-y",
            "-i", str(src), "-vf", f"scale={w}:{h}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-g", str(gop), "-keyint_min", str(gop), "-sc_threshold", "0",
            "-movflags", "+faststart"]
    args += ["-c:a", "aac", "-b:a", "128k"] if info.has_audio else ["-an"]
    proc = subprocess.Popen([*args, str(dst)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            encoding="utf-8", errors="replace", creationflags=_NO_WINDOW)
    for line in proc.stdout:
        if on_progress and line.startswith("out_time_us=") and info.duration > 0:
            value = line.split("=", 1)[1].strip()
            if value.isdigit():
                on_progress(min(int(value) / 1e6 / info.duration, 1.0))
    err = proc.stderr.read()
    if proc.wait() != 0:
        raise MediaError(f"Conversion de la vidéo échouée : {err[-600:]}")


def make_filmstrip(proxy: Path, out_jpg: Path, out_json: Path, info: VideoInfo, count: int, tile_h: int) -> dict:
    """Sprite de vignettes (grille 10 colonnes) pour la timeline + métadonnées de positionnement."""
    cols = 10
    rows = max(1, -(-count // cols))
    tile_w = _even(tile_h * info.width / info.height)
    interval = max(info.duration / count, 0.04)
    ffmpeg("-skip_frame", "nokey", "-i", str(proxy), "-an",
           "-vf", f"fps=1/{interval:.4f},scale={tile_w}:{tile_h},tile={cols}x{rows}",
           "-frames:v", "1", "-q:v", "4", str(out_jpg))
    meta = {"count": count, "cols": cols, "rows": rows, "tile_w": tile_w, "tile_h": tile_h, "interval": interval}
    out_json.write_text(json.dumps(meta), encoding="utf-8")
    return meta


def extract_frame(src: Path, t: float, max_height: int | None = None) -> np.ndarray:
    """Une frame exacte au temps t (BGR)."""
    vf = [] if max_height is None else ["-vf", f"scale=-2:'min({max_height},ih)'"]
    proc = subprocess.run(
        [binary("ffmpeg"), "-hide_banner", "-loglevel", "error", "-ss", f"{max(t, 0):.3f}", "-i", str(src),
         *vf, "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
        capture_output=True, creationflags=_NO_WINDOW,
    )
    if proc.returncode != 0 or not proc.stdout:
        raise MediaError(f"Impossible d'extraire la frame à {t:.2f} s.")
    frame = cv2.imdecode(np.frombuffer(proc.stdout, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise MediaError("Frame illisible.")
    return frame


def cut_segment(src: Path, dst: Path, start: float, end: float, size: tuple[int, int], has_audio: bool) -> None:
    """Découpe précise à la frame (seek d'entrée + réencodage), à la taille de travail."""
    w, h = size
    args = ["-ss", f"{start:.3f}", "-i", str(src), "-t", f"{end - start:.3f}", "-vf", f"scale={w}:{h},setsar=1",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p"]
    args += ["-c:a", "aac", "-b:a", "192k"] if has_audio else ["-an"]
    ffmpeg(*args, str(dst))


class FrameWriter:
    """Encode des frames BGR à la volée via un pipe vers ffmpeg (pas de frames sur disque)."""

    def __init__(self, dst: Path, size: tuple[int, int], fps_str: str, crf: int, preset: str):
        w, h = size
        self.proc = subprocess.Popen(
            [binary("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y",
             "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", fps_str, "-i", "-",
             "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p", str(dst)],
            stdin=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=_NO_WINDOW,
        )
        self.size = size

    def write(self, frame: np.ndarray) -> None:
        if (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv2.resize(frame, self.size)
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        self.proc.stdin.close()
        err = self.proc.stderr.read().decode("utf-8", "replace")
        if self.proc.wait() != 0:
            raise MediaError(f"Encodage échoué : {err[-800:]}")

    def abort(self) -> None:
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        self.proc.kill()

    def __enter__(self) -> "FrameWriter":
        return self

    def __exit__(self, exc_type, *_):
        if exc_type is None:
            self.close()
        else:
            self.abort()


def render_label(dst: Path, frame_height: int, text: str = "Contenu modifié par IA") -> Path:
    """PNG semi-transparent de l'étiquette IA (PIL gère les accents, drawtext + Windows moins bien)."""
    from PIL import Image, ImageDraw, ImageFont

    size = max(14, frame_height // 34)
    font = None
    for candidate in ("C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arialbd.ttf", "DejaVuSans-Bold.ttf"):
        try:
            font = ImageFont.truetype(candidate, size)
            break
        except OSError:
            continue
    font = font or ImageFont.load_default()
    left, top, right, bottom = font.getbbox(text)
    pad_x, pad_y = size // 2, size // 3
    img = Image.new("RGBA", (right - left + 2 * pad_x, bottom - top + 2 * pad_y), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([0, 0, img.width - 1, img.height - 1], radius=size // 2, fill=(0, 0, 0, 140))
    draw.text((pad_x - left, pad_y - top), text, font=font, fill=(255, 255, 255, 235))
    img.save(dst)
    return dst


def _encode_args(crf: int, preset: str) -> list[str]:
    return ["-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart"]


def assemble_segment(swapped: Path, cut: Path, dst: Path, has_audio: bool, label: Path | None, crf: int, preset: str) -> None:
    """Sortie « extrait seul » : vidéo swappée + audio d'origine de l'extrait (jamais retraité)."""
    args = ["-i", str(swapped), "-i", str(cut)]
    if label:
        args += ["-i", str(label), "-filter_complex", "[0:v][2:v]overlay=W-w-24:H-h-24[v]", "-map", "[v]"]
    else:
        args += ["-map", "0:v"]
    if has_audio:
        args += ["-map", "1:a:0", "-c:a", "copy"]
    ffmpeg(*args, *_encode_args(crf, preset), "-shortest" if has_audio else "-an", str(dst))


def assemble_full(source: Path, swapped: Path, dst: Path, info: VideoInfo, size: tuple[int, int], start: float,
                  end: float, label: Path | None, crf: int, preset: str) -> None:
    """Sortie « vidéo complète » : avant + extrait swappé + après, audio d'origine complet."""
    w, h = size
    frame = 1 / info.fps
    parts, chains = [], []
    has_pre = start > frame
    has_post = end < info.duration - frame
    splits = int(has_pre) + int(has_post)
    base = f"[0:v]scale={w}:{h},setsar=1,fps={info.fps_str}"
    if splits == 2:
        chains.append(f"{base},split=2[s0][s1]")
        pre_in, post_in = "[s0]", "[s1]"
    elif splits == 1:
        chains.append(f"{base}[s0]")
        pre_in = post_in = "[s0]"
    if has_pre:
        chains.append(f"{pre_in}trim=end={start:.4f},setpts=PTS-STARTPTS[pre]")
        parts.append("[pre]")
    chains.append(f"[1:v]scale={w}:{h},setsar=1,fps={info.fps_str},setpts=PTS-STARTPTS[mid]")
    parts.append("[mid]")
    if has_post:
        chains.append(f"{post_in}trim=start={end:.4f},setpts=PTS-STARTPTS[post]")
        parts.append("[post]")
    out = "[cat]"
    chains.append(f"{''.join(parts)}concat=n={len(parts)}:v=1:a=0{out}")
    inputs = ["-i", str(source), "-i", str(swapped)]
    if label:
        inputs += ["-i", str(label)]
        chains.append(f"{out}[2:v]overlay=W-w-24:H-h-24[v]")
        out = "[v]"
    args = [*inputs, "-filter_complex", ";".join(chains), "-map", out]
    if info.has_audio:
        args += ["-map", "0:a:0", "-c:a", "aac", "-b:a", "192k"]
    ffmpeg(*args, *_encode_args(crf, preset), str(dst))


def make_poster(src: Path, dst: Path, t: float = 0.0, max_height: int = 360) -> None:
    ffmpeg("-ss", f"{max(t, 0):.3f}", "-i", str(src), "-vf", f"scale=-2:'min({max_height},ih)'", "-frames:v", "1", "-q:v", "4", str(dst))
