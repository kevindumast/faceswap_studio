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
    character_space: str | None = None   # Space du niveau 4 (même jeton, même clé) ; None = inchangé, "" = effacé


def public() -> dict:
    s = zg.settings()
    token = s["token"] or ""
    return {"space": s["space"] or None, "token_set": bool(token), "key_set": bool(s["key"]),
            # 4 derniers caractères seulement, pour reconnaître le jeton sans l'exposer
            "token_hint": f"hf_…{token[-4:]}" if len(token) > 8 else None,
            "configured": zg.configured(),
            # dernier « Tester la connexion » réussi avec ces réglages (effacé à chaque modification)
            "tested": bool(s["space"]) and db.get_meta("zerogpu_tested") == s["space"],
            "character_space": s["character_space"] or None,
            "character_configured": zg.character_configured(),
            "character_tested": bool(s["character_space"]) and db.get_meta("zerogpu_character_tested") == s["character_space"]}


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
    if body.character_space is not None:
        character = body.character_space.strip()
        if character and not SPACE_RE.match(character):
            raise HTTPException(422, "Nom du Space niveau 4 attendu : ton-pseudo/nom-du-space.")
        db.set_meta("zerogpu_character_space", character)
        db.set_meta("zerogpu_character_tested", "")
    if body.token is not None:
        db.set_meta("zerogpu_token", body.token.strip())
    if body.key is not None:
        db.set_meta("zerogpu_key", body.key.strip())
    return {"zerogpu": public()}


@router.delete("/zerogpu")
def clear_zerogpu() -> dict:
    for key in ("zerogpu_space", "zerogpu_token", "zerogpu_key", "zerogpu_tested", "zerogpu_character_space",
                "zerogpu_character_tested"):
        db.set_meta(key, "")
    return {"zerogpu": public()}


@router.get("/zerogpu/state")
def space_state(kind: str = "faces") -> dict:
    """État en direct du Space (endormi, démarrage + chrono, prêt, erreur) : métadonnées Hugging Face, sans quota."""
    try:
        return zg.state(kind)
    except zg.ZeroGPUError as exc:
        raise HTTPException(502, str(exc)) from exc


@router.post("/zerogpu/wake")
def wake_space(kind: str = "faces") -> dict:
    """Réveille le Space en avance (ou le redémarre s'il a planté). Sans quota : le GPU n'est pris qu'au calcul."""
    try:
        return zg.wake(kind)
    except zg.ZeroGPUError as exc:
        raise HTTPException(502, str(exc)) from exc


@router.post("/zerogpu/test")
def test_zerogpu(kind: str = "faces") -> dict:
    """Vérifie que le Space (niveaux 1-2, ou niveau 4 avec kind=character) répond et accepte la clé, sans quota GPU."""
    meta, field = ("zerogpu_character_tested", "character_space") if kind == "character" else ("zerogpu_tested", "space")
    t0 = time.perf_counter()
    try:
        health = zg.ZeroGPUClient.from_settings(kind).health()
    except zg.ZeroGPUError as exc:
        db.set_meta(meta, "")
        raise HTTPException(502, str(exc)) from exc
    db.set_meta(meta, zg.settings()[field])
    zg.refresh_pro(max_age=0)   # abonnement PRO pris ou arrêté entre-temps : quota affiché à jour tout de suite
    return {"ok": True, "latency_ms": round((time.perf_counter() - t0) * 1000), "pro": zg.account_pro(), **health}
