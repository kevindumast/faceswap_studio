"""Stockage : une base SQLite (file de jobs + métadonnées) et des dossiers par objet dans data/.

L'API et le worker sont deux process ; ils ne communiquent que par cette base.
"""
from __future__ import annotations

import json
import shutil
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from src.config import load_config

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,              -- upload | url
    title TEXT,
    url TEXT,
    status TEXT NOT NULL,            -- downloading | preparing | ready | error
    progress REAL DEFAULT 0,
    error TEXT,
    source TEXT,                     -- chemin du fichier original
    info TEXT,                       -- JSON VideoInfo
    filmstrip TEXT,                  -- JSON positions du sprite
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    video_id TEXT NOT NULL,
    face_set_id TEXT NOT NULL,
    params TEXT NOT NULL,            -- JSON RenderOptions
    status TEXT NOT NULL,            -- queued | running | cancelling | cancelled | done | error | paused | review
    stage TEXT,                      -- cut | swap | assemble
    done INTEGER DEFAULT 0,
    total INTEGER DEFAULT 0,
    sec_per_frame REAL,
    error TEXT,
    warnings TEXT,
    created_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""

JSON_FIELDS = {"info", "filmstrip", "params", "warnings"}


def data_dir() -> Path:
    return load_config().path("data")


def db_path() -> Path:
    return data_dir() / "app.db"


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def folder(kind: str, obj_id: str) -> Path:
    """data/videos/<id>, data/faces/<id>, data/jobs/<id>. L'id est validé pour éviter toute traversée."""
    if not obj_id.isalnum():
        raise ValueError("id invalide")
    return data_dir() / kind / obj_id


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(db_path(), timeout=15)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init() -> None:
    data_dir().mkdir(parents=True, exist_ok=True)
    with connect() as c:
        c.execute("PRAGMA journal_mode=WAL")
        c.executescript(SCHEMA)


def _row(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    out = dict(row)
    for key in JSON_FIELDS & out.keys():
        if out[key]:
            out[key] = json.loads(out[key])
    return out


def _encode(values: dict[str, Any]) -> dict[str, Any]:
    return {k: json.dumps(v) if k in JSON_FIELDS and v is not None else v for k, v in values.items()}


def insert(table: str, **values: Any) -> None:
    values = _encode(values)
    cols = ", ".join(values)
    marks = ", ".join("?" * len(values))
    with connect() as c:
        c.execute(f"INSERT INTO {table} ({cols}) VALUES ({marks})", list(values.values()))


def update(table: str, obj_id: str, **values: Any) -> None:
    values = _encode(values)
    sets = ", ".join(f"{k} = ?" for k in values)
    with connect() as c:
        c.execute(f"UPDATE {table} SET {sets} WHERE id = ?", [*values.values(), obj_id])


def get(table: str, obj_id: str) -> dict | None:
    with connect() as c:
        return _row(c.execute(f"SELECT * FROM {table} WHERE id = ?", (obj_id,)).fetchone())


def all_rows(table: str, limit: int = 50) -> list[dict]:
    with connect() as c:
        rows = c.execute(f"SELECT * FROM {table} ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [_row(r) for r in rows]


def delete(table: str, obj_id: str) -> None:
    with connect() as c:
        c.execute(f"DELETE FROM {table} WHERE id = ?", (obj_id,))


def claim_next_job() -> dict | None:
    """Passe atomiquement le plus ancien job `queued` en `running`."""
    with connect() as c:
        row = c.execute(
            "UPDATE jobs SET status = 'running', started_at = ? WHERE id = ("
            " SELECT id FROM jobs WHERE status = 'queued' ORDER BY created_at LIMIT 1"
            ") RETURNING *",
            (time.time(),),
        ).fetchone()
        return _row(row)


def jobs_ahead(created_at: float) -> int:
    """Rendus en cours ou en attente créés avant celui-ci (le worker les traite un par un ; ceux en pause ne comptent pas)."""
    with connect() as c:
        row = c.execute(
            "SELECT COUNT(*) AS n FROM jobs WHERE status IN ('queued', 'running', 'cancelling', 'pausing') AND created_at < ?",
            (created_at,),
        ).fetchone()
    return row["n"]


def job_status(job_id: str) -> str | None:
    with connect() as c:
        row = c.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return row["status"] if row else None


def last_sec_per_frame() -> float | None:
    with connect() as c:
        row = c.execute(
            "SELECT sec_per_frame FROM jobs WHERE status = 'done' AND sec_per_frame IS NOT NULL "
            "ORDER BY finished_at DESC LIMIT 1"
        ).fetchone()
    return row["sec_per_frame"] if row else None


def set_meta(key: str, value: str) -> None:
    with connect() as c:
        c.execute("INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                  (key, value))


def get_meta(key: str) -> str | None:
    with connect() as c:
        row = c.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def cleanup(retention_hours: float) -> int:
    """Supprime vidéos, visages et jobs plus vieux que la rétention (fichiers + lignes)."""
    limit = time.time() - retention_hours * 3600
    removed = 0
    with connect() as c:
        old_videos = [r["id"] for r in c.execute("SELECT id FROM videos WHERE created_at < ?", (limit,))]
        old_jobs = [r["id"] for r in c.execute(
            "SELECT id FROM jobs WHERE created_at < ? AND status NOT IN ('queued', 'running', 'pausing', 'paused', 'review')", (limit,))]
        c.execute("DELETE FROM videos WHERE created_at < ?", (limit,))
        c.execute("DELETE FROM jobs WHERE created_at < ? AND status NOT IN ('queued', 'running', 'pausing', 'paused', 'review')", (limit,))
        kept_sets = {r["face_set_id"] for r in c.execute("SELECT face_set_id FROM jobs")}   # personnes des rendus gardés
    for kind, ids in (("videos", old_videos), ("jobs", old_jobs)):
        for obj_id in ids:
            shutil.rmtree(folder(kind, obj_id), ignore_errors=True)
            removed += 1
    faces_root = data_dir() / "faces"
    if faces_root.is_dir():
        for d in faces_root.iterdir():
            if d.is_dir() and d.name not in kept_sets and d.stat().st_mtime < limit:
                shutil.rmtree(d, ignore_errors=True)
                removed += 1
    return removed
