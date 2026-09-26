"""Utilitaires partagés par les routes."""
from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile

from app import db

# Téléchargements et conversions ffmpeg : hors du thread de requête, 2 en parallèle max.
background = ThreadPoolExecutor(max_workers=2, thread_name_prefix="prep")


def not_found(what: str = "Élément") -> HTTPException:
    return HTTPException(404, f"{what} introuvable.")


def get_or_404(table: str, obj_id: str, what: str) -> dict:
    if not obj_id.isalnum():
        raise not_found(what)
    row = db.get(table, obj_id)
    if row is None:
        raise not_found(what)
    return row


async def save_upload(upload: UploadFile, dst: Path, max_mb: float) -> int:
    """Copie en streaming avec limite de taille. Supprime le fichier partiel en cas de dépassement."""
    limit = int(max_mb * 1024 * 1024)
    size = 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(dst, "wb") as f:
            while chunk := await upload.read(1 << 20):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(413, f"Fichier trop lourd (max {max_mb:g} Mo).")
                f.write(chunk)
    except BaseException:
        dst.unlink(missing_ok=True)
        raise
    return size


def jpeg_data_url(img: np.ndarray, quality: int = 85) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
