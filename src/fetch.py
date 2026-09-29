"""Récupération d'une vidéo depuis une URL (YouTube ou autre site supporté par yt-dlp)."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, urlencode, urlparse

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

from .media import binary

URL_RE = re.compile(r"^https?://\S+$", re.I)


class FetchError(RuntimeError):
    pass


def _clean_error(exc: Exception) -> str:
    msg = re.sub(r"\x1b\[[0-9;]*m", "", str(exc)).replace("ERROR: ", "").strip()
    if "Sign in to confirm" in msg or "not a bot" in msg:
        return "YouTube demande une vérification (anti-bot). Réessayez plus tard ou importez le fichier directement."
    if "Private video" in msg or "unavailable" in msg.lower():
        return "Vidéo privée ou indisponible."
    return f"{msg[:300]} (si l'erreur persiste : uv pip install -U yt-dlp)"


def normalize_url(url: str) -> str:
    """Garde seulement la vidéo : retire playlist / radio (list=, start_radio=, index=…) des liens YouTube."""
    url = url.strip()
    if not URL_RE.match(url):
        raise FetchError("URL invalide.")
    parsed = urlparse(url)
    host = parsed.netloc.lower().removeprefix("www.").removeprefix("m.")
    if host == "youtu.be":
        return f"https://www.youtube.com/watch?v={parsed.path.strip('/')}"
    if host in ("youtube.com", "music.youtube.com"):
        if parsed.path.startswith("/shorts/"):
            return f"https://www.youtube.com/watch?v={parsed.path.split('/')[2]}"
        video = parse_qs(parsed.query).get("v")
        if video:
            return "https://www.youtube.com/watch?" + urlencode({"v": video[0]})
    return url


def video_info(url: str) -> dict:
    url = normalize_url(url)
    try:
        with YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True, "noplaylist": True}) as ydl:
            info = ydl.extract_info(url, download=False)
    except DownloadError as exc:
        raise FetchError(_clean_error(exc)) from exc
    if info.get("_type") == "playlist":
        raise FetchError("C'est une playlist : collez le lien d'une seule vidéo.")
    if info.get("is_live"):
        raise FetchError("Les directs ne sont pas supportés.")
    return {
        "id": info.get("id"),
        "title": info.get("title") or "Vidéo",
        "duration": float(info.get("duration") or 0),
        "thumbnail": info.get("thumbnail"),
        "uploader": info.get("uploader") or info.get("channel"),
        "webpage_url": info.get("webpage_url") or url,
    }


def download(url: str, out_dir: Path, max_res: int, on_progress: Callable[[float], None] | None = None) -> Path:
    """Télécharge la meilleure qualité ≤ max_res (petit côté) en mp4 (vidéo + audio fusionnés). Progression 0..1."""
    url = normalize_url(url)
    out_dir.mkdir(parents=True, exist_ok=True)
    state = {"streams_done": 0}

    def hook(d: dict) -> None:
        if not on_progress:
            return
        if d["status"] == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            frac = (d.get("downloaded_bytes") or 0) / total if total else 0
            # Vidéo puis audio : on pondère 85 % / 15 %.
            base, span = (0.0, 0.85) if state["streams_done"] == 0 else (0.85, 0.15)
            on_progress(min(base + span * frac, 0.99))
        elif d["status"] == "finished":
            state["streams_done"] += 1

    opts = {
        # Tri plutôt que filtre [height<=N] : « res » est le petit côté, comme le « p » de YouTube. Un filtre sur la
        # hauteur prenait du 360×640 pour un Short vertical au lieu du 720×1280. À résolution égale : H.264 + AAC
        # (décodage rapide et sûr), puis le meilleur débit.
        "format": "bv*+ba/b",
        "format_sort": [f"res:{int(max_res)}", "+codec:avc:m4a", "tbr"],
        "merge_output_format": "mp4",
        "outtmpl": str(out_dir / "source.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "ffmpeg_location": str(Path(binary("ffmpeg")).parent),
        "progress_hooks": [hook],
    }
    try:
        with YoutubeDL(opts) as ydl:
            ydl.download([url])
    except DownloadError as exc:
        raise FetchError(_clean_error(exc)) from exc
    files = sorted(out_dir.glob("source.*"), key=lambda p: p.stat().st_size, reverse=True)
    files = [f for f in files if f.suffix not in (".part", ".ytdl")]
    if not files:
        raise FetchError("Téléchargement terminé mais aucun fichier trouvé.")
    return files[0]
