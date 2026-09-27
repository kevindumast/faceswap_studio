"""Réglages du moteur : Space ZeroGPU (nom, jeton Hugging Face, clé APP_KEY).

Le jeton et la clé restent côté serveur (data/app.db) : l'API ne les renvoie jamais au navigateur, seulement « enregistré ».
Configurer ZeroGPU n'active rien : ça rend seulement disponible la case « Utiliser le GPU » de chaque rendu.
"""
from __future__ import annotations

import re
import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app import db
from app.worker import zerogpu_client as zg

router = APIRouter(prefix="/api/settings", tags=["settings"])

SPACE_RE = re.compile(r"^([\w.-]+/[\w.-]+|https?://\S+)$")


class ZeroGPUIn(BaseModel):
    space: str
    token: str | None = None   # None = inchangé, "" = effacé
    key: str | None = None


def public() -> dict:
    s = zg.settings()
    token = s["token"] or ""
    return {"space": s["space"] or None, "token_set": bool(token), "key_set": bool(s["key"]),
            # 4 derniers caractères seulement, pour reconnaître le jeton sans l'exposer
            "token_hint": f"hf_…{token[-4:]}" if len(token) > 8 else None,
            "configured": zg.configured(),
            # dernier « Tester la connexion » réussi avec ces réglages (effacé à chaque modification)
            "tested": bool(s["space"]) and db.get_meta("zerogpu_tested") == s["space"]}


@router.get("")
def get_settings() -> dict:
    return {"zerogpu": public()}


@router.put("/zerogpu")
def save_zerogpu(body: ZeroGPUIn) -> dict:
    space = body.space.strip()
    if not SPACE_RE.match(space):
        raise HTTPException(422, "Nom de Space attendu : ton-pseudo/nom-du-space.")
    db.set_meta("zerogpu_space", space)
    db.set_meta("zerogpu_tested", "")
    if body.token is not None:
        db.set_meta("zerogpu_token", body.token.strip())
    if body.key is not None:
        db.set_meta("zerogpu_key", body.key.strip())
    return {"zerogpu": public()}


@router.delete("/zerogpu")
def clear_zerogpu() -> dict:
    for key in ("zerogpu_space", "zerogpu_token", "zerogpu_key", "zerogpu_tested"):
        db.set_meta(key, "")
    return {"zerogpu": public()}


@router.post("/zerogpu/test")
def test_zerogpu() -> dict:
    """Vérifie que le Space répond et accepte la clé (ne consomme pas de quota GPU)."""
    t0 = time.perf_counter()
    try:
        health = zg.ZeroGPUClient.from_settings().health()
    except zg.ZeroGPUError as exc:
        db.set_meta("zerogpu_tested", "")
        raise HTTPException(502, str(exc)) from exc
    db.set_meta("zerogpu_tested", zg.settings()["space"])
    return {"ok": True, "latency_ms": round((time.perf_counter() - t0) * 1000), **health}
