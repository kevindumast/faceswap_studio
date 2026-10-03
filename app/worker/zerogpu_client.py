"""Client du Space GPU (ZeroGPU) : envoie un extrait, suit la progression, récupère le résultat.

Utilisé seulement quand un rendu a été lancé avec la case « Utiliser le GPU » cochée : jamais de bascule automatique,
ni du CPU vers le GPU, ni l'inverse en cas d'échec (l'utilisateur relance lui-même sur son PC).
"""
from __future__ import annotations

import concurrent.futures
import datetime
import json
import re
import shutil
import tempfile
import time
import urllib.parse
import uuid
from pathlib import Path
from typing import Callable

from app import db
from src.netfix import setup_tls

setup_tls()  # certificat absent ou proxy d'entreprise : vérification avec le magasin du système


StageCallback = Callable[[str, int, int, "str | None"], None]
QUEUED = ("STARTING", "JOINING_QUEUE", "IN_QUEUE", "QUEUE_FULL", "SENDING_DATA")


class ZeroGPUError(RuntimeError):
    pass


class RemoteCancelled(Exception):
    pass


def settings() -> dict:
    """Deux Spaces possibles, même jeton et même clé : niveaux 1-2 (« space ») et niveau 4 (« character_space »)."""
    return {"space": db.get_meta("zerogpu_space") or "", "token": db.get_meta("zerogpu_token") or "",
            "key": db.get_meta("zerogpu_key") or "", "character_space": db.get_meta("zerogpu_character_space") or ""}


def configured() -> bool:
    s = settings()
    return bool(s["space"] and s["key"])


def character_configured() -> bool:
    s = settings()
    return bool(s["character_space"] and s["key"])


# --- Cycle de vie du Space : lecture de son état et réveil (métadonnées Hugging Face, sans quota GPU) ---------------

STARTING = {"APP_STARTING", "BUILDING", "RUNNING_BUILDING", "RUNNING_APP_STARTING"}
ASLEEP = {"SLEEPING", "PAUSED", "STOPPED"}
BROKEN = {"BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR", "NO_APP_FILE"}
EXPECTED_WAKE_S = {"faces": 180, "character": 1800}     # réveil typique : ~3 min ; ~30 min pour les 57 Go du niveau 4
WAKE_TIMEOUT_S = {"faces": 900, "character": 3600}
_states: dict[str, tuple[float, dict]] = {}


def space_name(kind: str) -> str:
    s = settings()
    return s["character_space"] if kind == "character" else s["space"]


def phase(stage: str | None) -> str:
    if stage == "RUNNING":
        return "ready"
    if stage in STARTING:
        return "starting"
    if stage in ASLEEP:
        return "asleep"
    if stage in BROKEN:
        return "error"
    return "unknown"


def _runtime(space: str):
    from huggingface_hub import HfApi

    try:
        return HfApi(token=settings()["token"] or None).get_space_runtime(space)
    except Exception as exc:
        raise space_error(exc) from exc


def state(kind: str = "faces", max_age: float = 5.0) -> dict:
    """État du Space pour l'appli. Le début du réveil est mémorisé en base : le chrono survit aux rechargements."""
    space = space_name(kind)
    if not space:
        return {"space": None, "phase": "unconfigured", "stage": None, "hardware": None, "error": None,
                "waking_since": None, "expected_s": EXPECTED_WAKE_S.get(kind, 180)}
    cached = _states.get(kind)
    if cached and time.time() - cached[0] < max_age and cached[1]["space"] == space:
        return cached[1]
    rt = _runtime(space)
    refresh_pro()
    ph = phase(rt.stage)
    key = f"space_waking:{kind}"
    since = float(db.get_meta(key) or 0) or None
    if ph == "starting" and since is None:
        since = time.time()
        db.set_meta(key, str(since))
    elif since and (ph in ("ready", "error") or (ph == "asleep" and time.time() - since > 120)):
        since = None               # prêt, en erreur, ou réveil demandé qui n'a pas pris : chrono remis à zéro
        db.set_meta(key, "")
    out = {"space": space, "phase": ph, "stage": rt.stage, "hardware": rt.hardware,
           "error": (rt.raw or {}).get("errorMessage"), "waking_since": since,
           "expected_s": EXPECTED_WAKE_S.get(kind, 180)}
    _states[kind] = (time.time(), out)
    return out


def account_pro() -> bool | None:
    """Compte Hugging Face PRO (quota ZeroGPU de 40 min/jour au lieu de 5) ? None tant que ce n'est pas connu."""
    try:
        q = json.loads(db.get_meta("zerogpu_pro") or "null")
    except ValueError:
        return None
    return q.get("pro") if q else None


def refresh_pro(max_age: float = 3600) -> None:
    """Relit le statut PRO avec le jeton de l'appli, au plus une fois par heure (erreur : on garde l'ancien)."""
    try:
        q = json.loads(db.get_meta("zerogpu_pro") or "null")
    except ValueError:
        q = None
    token = settings()["token"]
    if not token or (q and time.time() - q.get("at", 0) < max_age):
        return
    from huggingface_hub import HfApi

    try:
        pro = bool(HfApi(token=token).whoami().get("isPro"))
    except Exception:
        return
    db.set_meta("zerogpu_pro", json.dumps({"pro": pro, "at": time.time()}))


def wake(kind: str = "faces") -> dict:
    """Réveille un Space endormi ou planté (redémarrage, sans reconstruction). Sans effet s'il tourne ou démarre."""
    st = state(kind, max_age=0)
    if st["phase"] in ("asleep", "error"):
        from huggingface_hub import HfApi

        try:
            HfApi(token=settings()["token"] or None).restart_space(st["space"])
        except Exception as exc:
            raise space_error(exc) from exc
        db.set_meta(f"space_waking:{kind}", str(time.time()))
        _states.pop(kind, None)
    return state(kind, max_age=0)


def wait_until_ready(kind: str, on_wait: Callable[[float, float], None],
                     should_cancel: Callable[[], bool] | None = None, poll: float = 10.0) -> None:
    """Avant un rendu : réveille le Space s'il dort, puis attend qu'il soit prêt (on_wait(écoulé, durée typique)).

    Attendre ne consomme pas de quota : le GPU n'est attribué qu'au moment du calcul.
    """
    t0 = time.time()
    woke = False
    while True:
        st = state(kind, max_age=0)
        if st["phase"] == "ready":
            return
        if st["phase"] == "unconfigured":
            raise ZeroGPUError("Le Space n'est pas configuré (Moteur → ZeroGPU).")
        if st["phase"] in ("asleep", "error"):
            if woke and st["phase"] == "error":
                raise ZeroGPUError(f"Le Space est en erreur ({st['stage']}) : "
                                   f"{st['error'] or 'voir ses journaux sur Hugging Face'}.")
            if not woke:
                wake(kind)
                woke = True
        if should_cancel and should_cancel():
            raise RemoteCancelled()
        on_wait(time.time() - (st["waking_since"] or t0), st["expected_s"])
        if time.time() - t0 > WAKE_TIMEOUT_S.get(kind, 900):
            raise ZeroGPUError("Le Space ne s'est pas réveillé à temps : regarde son état sur Hugging Face, puis relance.")
        time.sleep(poll)


def friendly(exc: Exception) -> str:
    """Messages compréhensibles pour les erreurs les plus fréquentes."""
    if isinstance(exc, concurrent.futures.CancelledError):   # gradio_client : flux de suivi fermé avant la fin
        return ("Connexion avec le Space coupée pendant le calcul : le Space a abandonné le rendu. Le proxy du réseau "
                "d'entreprise retient les réponses en flux de *.hf.space puis les coupe (~3 min) ; depuis un autre "
                "réseau (partage de connexion), le rendu passe.")
    msg = str(exc) or type(exc).__name__   # certaines erreurs réseau (httpx.ReadError…) n'ont pas de texte
    low = msg.lower()
    if "handshake failure" in low or "read operation timed out" in low or "certificate verify failed" in low:
        return ("Connexion au Space refusée par le réseau (filtre du réseau d'entreprise sur *.hf.space ?). "
                "Hugging Face voit le Space prêt, mais ce PC ne peut pas l'atteindre : fais autoriser *.hf.space "
                "par ton service informatique, ou utilise l'appli depuis un autre réseau.")
    if "larger than the maximum allowed" in low:   # refus avant calcul : réservation trop longue pour le compte
        asked = re.search(r"duration \((\d+)s\)", msg)
        return (f"ZeroGPU refuse de réserver {asked.group(1) + ' s' if asked else 'autant'} de GPU d'un coup : c'est plus "
                "que le maximum autorisé pour ton compte (gratuit). Rien n'a été décompté du quota. Passe en 360p ou "
                "raccourcis le passage (en PRO : tâches jusqu'à 40 min).")
    if "quota" in low:
        q = quota_info(msg) or {}
        detail = (f" : ce rendu réserve {q['requested_s']} s de GPU, il en reste {q['left_s']} s"
                  if q.get("left_s") is not None else " pour l'instant")
        when = f"Réessaie {retry_text(q['wait_s'])}" if q.get("wait_s") is not None else "Réessaie demain"
        return (f"Quota ZeroGPU épuisé{detail} (5 min/jour en gratuit, 40 en PRO). {when}, ou raccourcis le passage "
                "(niveaux 1-2 : relance aussi sur ce PC).")
    if "app_key" in low:
        return "Clé APP_KEY refusée par le Space : relance scripts/deploy_space.py --save ou recopie la clé."
    if "401" in msg or "403" in msg or "unauthorized" in low or "not found" in low or "404" in msg:
        return "Space introuvable ou jeton refusé : vérifie le nom du Space et ton jeton Hugging Face (Moteur → ZeroGPU)."
    if "sleep" in low or "building" in low or "starting" in low or "runtime_error" in low:
        return ("Le Space démarre ou s'est mis en veille : attends 1 à 2 min puis relance le rendu "
                "(jusqu'à 30 min au premier démarrage du Space niveau 4, le temps de charger le modèle).")
    if "gpu task aborted" in low:
        return "Le GPU a été coupé avant la fin (durée maximale dépassée) : essaie un passage plus court."
    return f"Erreur du Space GPU : {msg[:300]}"


# --- Quota ZeroGPU : rien ne permet de le lire sans réserver de GPU ; seuls les refus en donnent l'état exact ---------

def quota_info(msg: str) -> dict | None:
    """Chiffres d'un refus de quota : « (204s requested vs. 95s left). Try again in 2:14:05 » (None : pas de chiffres)."""
    need = re.search(r"\((\d+)s requested vs\. (\d+)s left\)", msg)
    wait = re.search(r"try again in (?:(\d+) days?, )?(\d+):(\d{2}):(\d{2})", msg.lower())
    if not (need or wait):
        return None
    return {"requested_s": int(need[1]) if need else None, "left_s": int(need[2]) if need else None,
            "wait_s": int(wait[1] or 0) * 86400 + int(wait[2]) * 3600 + int(wait[3]) * 60 + int(wait[4]) if wait else None}


def clock_text(ts: float, now: float | None = None) -> str:
    """Heure locale d'un instant à venir : « vers 18 h 20 », « demain vers 18 h 20 », « le 02/10 vers 18 h 20 »."""
    at = datetime.datetime.fromtimestamp(ts)
    today = datetime.datetime.fromtimestamp(time.time() if now is None else now).date()
    hm = f"vers {at.hour} h {at.minute:02d}"
    if at.date() == today:
        return hm
    if at.date() == today + datetime.timedelta(days=1):
        return f"demain {hm}"
    return f"le {at:%d/%m} {hm}"


def retry_text(wait_s: int, now: float | None = None) -> str:
    """« dans 19 h 29 (demain vers 18 h 20) » : délai annoncé par ZeroGPU, et l'heure qu'il donne."""
    now = time.time() if now is None else now
    minutes = max(1, wait_s // 60)
    delay = f"{minutes // 60} h {minutes % 60:02d}" if minutes >= 60 else f"{minutes} min"
    return f"dans {delay} ({clock_text(now + wait_s, now)})"


def record_quota(msg: str) -> None:
    """Retient ce qu'un refus dit du quota (reste, heure du prochain essai) : l'appli l'affiche jusqu'à cette heure."""
    q = quota_info(msg)
    if q and q["wait_s"] is not None:
        now = time.time()
        db.set_meta("zerogpu_quota", json.dumps({"left_s": q["left_s"], "retry_at": now + q["wait_s"], "seen_at": now}))


def quota_status() -> dict | None:
    """Dernier état connu du quota, tant que l'heure du prochain essai n'est pas passée (sinon None)."""
    try:
        q = json.loads(db.get_meta("zerogpu_quota") or "null")
    except ValueError:
        return None
    return q if q and q.get("retry_at", 0) > time.time() else None


def consume_quota(seconds: float) -> None:
    """Après un calcul réussi : le reste connu baisse d'autant."""
    q = quota_status()
    if q and q.get("left_s") is not None and seconds > 0:
        q["left_s"] = max(0, round(q["left_s"] - seconds))
        db.set_meta("zerogpu_quota", json.dumps(q))


def space_error(exc: BaseException) -> ZeroGPUError:
    """Erreur renvoyée par le Space : message compréhensible, et état du quota retenu si ZeroGPU l'a donné."""
    record_quota(str(exc))
    return ZeroGPUError(friendly(exc))


def _stats(stats) -> dict:
    result = json.loads(stats) if isinstance(stats, str) else (stats or {})
    consume_quota(float(result.get("gpu_seconds") or 0))
    return result


class ZeroGPUClient:
    def __init__(self, space: str, token: str, key: str):
        from gradio_client import Client

        self.key = key
        self.downloads = Path(tempfile.mkdtemp(prefix="zerogpu_"))
        try:
            self.client = Client(space, token=token or None, verbose=False, download_files=str(self.downloads))
        except Exception as exc:
            raise space_error(exc) from exc
        # Space à jour : suivi par requêtes courtes (passe les proxys qui retiennent les réponses en flux) ;
        # sinon (Space pas encore redéployé) : flux continu de gradio_client.
        self.short_requests = _has_endpoint(self.client, "/status")

    @classmethod
    def from_settings(cls, kind: str = "faces") -> "ZeroGPUClient":
        s = settings()
        space = s["character_space"] if kind == "character" else s["space"]
        if not (space and s["key"]):
            what = "Le Space du niveau 4" if kind == "character" else "ZeroGPU"
            raise ZeroGPUError(f"{what} n'est pas configuré (Moteur → ZeroGPU).")
        return cls(space, s["token"], s["key"])

    def health(self) -> dict:
        try:
            return json.loads(self.client.predict(self.key, api_name="/health"))
        except Exception as exc:
            raise space_error(exc) from exc

    def swap(self, clip: Path, payload: dict, out: Path, on_progress: Callable[[int, int], None] | None = None,
             should_cancel: Callable[[], bool] | None = None, poll: float = 1.0, on_stage: StageCallback | None = None) -> dict:
        """Envoie l'extrait (sans son) et écrit la vidéo remplacée dans `out`. Renvoie les stats du Space."""
        video, stats = self._run("/swap", [clip], payload, on_progress, should_cancel, poll, on_stage)
        shutil.copy(_path(video), out)
        shutil.rmtree(self.downloads, ignore_errors=True)
        return _stats(stats)

    def replace(self, clip: Path, reference: Path, payload: dict, out: Path, mask_out: Path,
                on_progress: Callable[[int, int], None] | None = None, should_cancel: Callable[[], bool] | None = None,
                poll: float = 2.0, on_stage: StageCallback | None = None) -> dict:
        """Niveau 4 : extrait (sans son) + photo de la personne → vidéo générée (30 i/s) et masque de la zone refaite."""
        video, mask, stats = self._run("/replace", [clip, reference], payload, on_progress, should_cancel, poll,
                                       on_stage)
        shutil.copy(_path(video), out)
        shutil.copy(_path(mask), mask_out)
        shutil.rmtree(self.downloads, ignore_errors=True)
        return _stats(stats)

    def _run(self, api_name: str, files: list[Path], payload: dict, on_progress, should_cancel, poll: float,
             on_stage=None):
        """Soumet un appel, relaie la progression, annule si demandé ; renvoie le résultat brut du Space.

        on_stage(étape, fait, total, détail) : « queue » (rang, taille de la file du Space), « gpu » (calcul lancé,
        en attente d'un GPU libre chez ZeroGPU ou en préparation), « progress » (étape annoncée par le Space).
        """
        if self.short_requests:
            return self._run_short(api_name, files, payload, on_progress, should_cancel, poll, on_stage)
        from gradio_client import handle_file

        args = (*(handle_file(str(f)) for f in files), json.dumps(payload), self.key)
        try:
            job = self.client.submit(*args, api_name=api_name)
        except Exception as exc:
            raise space_error(exc) from exc
        while not job.done():
            if should_cancel and should_cancel():
                job.cancel()
                raise RemoteCancelled()
            status = job.status()
            code = getattr(getattr(status, "code", None), "value", None)
            units = getattr(status, "progress_data", None) or []
            unit = units[-1] if units else None
            if unit is not None and getattr(unit, "index", None) is not None and getattr(unit, "length", None):
                if on_progress:
                    on_progress(int(unit.index), int(unit.length))
                if on_stage:
                    on_stage("progress", int(unit.index), int(unit.length), getattr(unit, "desc", None))
            elif on_stage and code in QUEUED:
                on_stage("queue", int(getattr(status, "rank", None) or 0), int(getattr(status, "queue_size", None) or 0), None)
            elif on_stage and code in ("PROCESSING", "ITERATING"):
                on_stage("gpu", 0, 0, None)
            time.sleep(poll)
        try:
            return job.result()
        except Exception as exc:
            raise space_error(exc) from exc

    # --- Suivi par requêtes courtes --------------------------------------------------------------------------------
    # Un proxy d'entreprise peut retenir les réponses en flux (le suivi de gradio_client) jusqu'à leur fin et les
    # couper au bout de ~3 min, ce qui fait annuler le calcul par Gradio. Ici, chaque requête est courte : envoi des
    # fichiers, lancement (/call/…), état toutes les `poll` s (/run/status, hors file d'attente), puis résultat une
    # fois le calcul fini. Une coupure passagère ne coûte qu'une lecture d'état : le calcul continue sur le Space.

    def _run_short(self, api_name: str, files: list[Path], payload: dict, on_progress, should_cancel, poll: float,
                   on_stage=None):
        import httpx

        job = uuid.uuid4().hex
        base = self.client.src_prefixed
        with httpx.Client(headers=self.client.headers, timeout=httpx.Timeout(120, connect=30)) as http:
            try:
                data = [self._upload(http, base, f) for f in files]
                r = http.post(f"{base}call{api_name}", json={"data": [*data, json.dumps({**payload, "job": job}), self.key]})
                r.raise_for_status()
                event = r.json()["event_id"]
            except Exception as exc:
                raise space_error(exc) from exc
            state = self._follow(http, base, job, on_progress, should_cancel, poll, on_stage)
            if state["state"] == "error":
                try:   # vide les messages que Gradio garde pour ce calcul
                    self._result(http, f"{base}call{api_name}/{event}")
                except Exception:
                    pass
                raise space_error(RuntimeError(state.get("error") or ""))
            try:
                outputs = self._result(http, f"{base}call{api_name}/{event}")
                return tuple(self._download(http, base, o) if isinstance(o, dict) and o.get("url") else o
                             for o in outputs)
            except ZeroGPUError:
                raise
            except Exception as exc:
                raise space_error(exc) from exc

    def _follow(self, http, base: str, job: str, on_progress, should_cancel, poll: float, on_stage,
                max_unreachable_s: float = 120, max_unknown_s: float = 900) -> dict:
        """Lit l'état du calcul jusqu'à sa fin (« done » ou « error ») et relaie sa progression."""
        started = False
        unreachable_since = None
        unknown_since = time.time()
        while True:
            if should_cancel and should_cancel():
                try:  # le Space s'arrête à sa prochaine étape et libère le GPU
                    http.post(f"{base}run/cancel", json={"data": [job, self.key]}, timeout=20)
                except Exception:
                    pass
                raise RemoteCancelled()
            try:
                r = http.post(f"{base}run/status", json={"data": [job, self.key]}, timeout=30)
                r.raise_for_status()
                st = json.loads(r.json()["data"][0])
                unreachable_since = None
            except Exception as exc:  # coupure passagère : le calcul continue sur le Space, on relit plus tard
                unreachable_since = unreachable_since or time.time()
                if time.time() - unreachable_since > max_unreachable_s:
                    raise space_error(exc) from exc
                time.sleep(poll)
                continue
            state = st.get("state")
            if state in ("done", "error"):
                return st
            if state == "unknown":   # pas encore commencé : file d'attente du Space (un calcul à la fois)
                if started:
                    raise ZeroGPUError("Le Space a redémarré pendant le calcul : relance le rendu.")
                if time.time() - unknown_since > max_unknown_s:
                    raise ZeroGPUError("Le Space n'a pas commencé le calcul (file d'attente bloquée ?) : relance le rendu.")
                if on_stage:
                    on_stage("queue", 0, 0, None)
            else:
                started = True
                done, total = int(st.get("done") or 0), int(st.get("total") or 0)
                if state == "running" and total:
                    if on_progress:
                        on_progress(done, total)
                    if on_stage:
                        on_stage("progress", done, total, st.get("desc"))
                elif on_stage:   # « waiting » : en attente d'un GPU chez ZeroGPU
                    on_stage("gpu", 0, 0, None)
            time.sleep(poll)

    def _upload(self, http, base: str, path: Path) -> dict:
        with open(path, "rb") as f:
            r = http.post(f"{base}upload", files=[("files", (Path(path).name, f))], timeout=_timeout(300))
        r.raise_for_status()
        return {"path": r.json()[0], "orig_name": Path(path).name, "meta": {"_type": "gradio.FileData"}}

    def _result(self, http, url: str) -> list:
        """Sorties d'un calcul terminé (événement « complete » du flux /call, court puisque le calcul est fini)."""
        r = http.get(url, timeout=_timeout(120))
        r.raise_for_status()
        event, data = None, None
        for block in r.text.split("\n\n"):
            fields = dict(line.split(": ", 1) for line in block.splitlines() if ": " in line)
            if fields.get("event") in ("complete", "error"):
                event, data = fields["event"], fields.get("data")
        if event != "complete":
            raise space_error(RuntimeError(data or "le Space n'a pas renvoyé de résultat"))
        return json.loads(data)

    def _download(self, http, base: str, file: dict) -> str:
        dest = self.downloads / (file.get("orig_name") or Path(file["path"]).name)
        with http.stream("GET", urllib.parse.urljoin(base, file["url"]), timeout=_timeout(300)) as r:
            r.raise_for_status()
            with open(dest, "wb") as out:
                for chunk in r.iter_bytes():
                    out.write(chunk)
        return str(dest)


def _timeout(seconds: float):
    import httpx

    return httpx.Timeout(seconds, connect=30)


def _has_endpoint(client, name: str) -> bool:
    try:
        return name in client.view_api(print_info=False, return_format="dict")["named_endpoints"]
    except Exception:   # client de test, ou Space sans description d'API
        return False


def _path(file) -> str:
    return file.get("path") if isinstance(file, dict) else file
