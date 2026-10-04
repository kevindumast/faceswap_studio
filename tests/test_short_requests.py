"""Suivi des calculs ZeroGPU par requêtes courtes (proxy d'entreprise qui retient les réponses en flux) :
état tenu par les Spaces (space/jobs.py) et client de l'appli contre un faux Space HTTP."""
import importlib.util
import json
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def jobs(tmp_path, monkeypatch):
    """space/jobs.py importé par son chemin (copié tel quel dans chaque Space), états rangés dans tmp_path."""
    spec = importlib.util.spec_from_file_location("space_jobs", ROOT / "space" / "jobs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path / "jobs")
    return module


# --- Côté Space ----------------------------------------------------------------------------------------------------

def test_run_records_done_and_errors(jobs):
    assert jobs.read("a1") == {"state": "unknown"}
    assert jobs.run("a1", lambda x: x * 2, 21) == 42
    assert jobs.read("a1")["state"] == "done"
    with pytest.raises(RuntimeError):
        jobs.run("b2", lambda: (_ for _ in ()).throw(RuntimeError("The requested GPU duration (340s) is larger")))
    assert jobs.read("b2") == {"state": "error", "error": "The requested GPU duration (340s) is larger",
                               "t": jobs.read("b2")["t"]}
    assert jobs.run(None, lambda: "sans suivi") == "sans suivi"          # ancien client : pas d'identifiant


def test_progress_is_throttled_and_cancel_stops_the_next_step(jobs):
    jobs.progress("c3", 1, 10, "squelette")
    jobs.progress("c3", 2, 10, "squelette")                               # moins de 0,5 s après : pas réécrit
    assert jobs.read("c3")["done"] == 1
    jobs.progress("c3", 10, 10, "génération")                             # dernière étape : toujours écrite
    assert jobs.read("c3")["done"] == 10 and jobs.read("c3")["desc"] == "génération"
    jobs.request_cancel("c3")
    with pytest.raises(jobs.Cancelled):
        jobs.progress("c3", 3, 10, "génération")
    with pytest.raises(jobs.Cancelled):                                    # annulé avant même de commencer
        jobs.run("c3", lambda: "jamais")


def test_job_ids_cannot_escape_the_folder(jobs):
    with pytest.raises(ValueError):
        jobs.read("../../etc/passwd")


def test_both_spaces_use_short_requests():
    for app in (ROOT / "space" / "app.py", ROOT / "space_character" / "app.py"):
        code = app.read_text(encoding="utf-8")
        assert 'api_name="status", queue=False' in code and 'api_name="cancel", queue=False' in code
        assert "jobs.run(json.loads(payload).get(\"job\")" in code
    deploy = (ROOT / "scripts" / "deploy_space.py").read_text(encoding="utf-8")
    assert 'ROOT / "space" / "jobs.py"' in deploy


# --- Côté appli : client contre un faux Space ----------------------------------------------------------------------

BASE = "https://kevin-c.hf.space/gradio_api/"


class FakeSpace:
    """Répond comme Gradio 6 : envoi de fichiers, /call/replace, /run/status (états successifs), résultat, fichiers."""

    def __init__(self, states):
        self.states, self.sent, self.status_calls, self.cancelled = list(states), [], 0, False

    def __call__(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        assert request.headers.get("x-hf-authorization") == "Bearer tok"   # quota ZeroGPU compté sur ton compte
        if url == BASE + "upload":
            name = b"ref.jpg" if b"ref.jpg" in request.content else b"clip.mp4"
            return httpx.Response(200, json=[f"/tmp/gradio/{name.decode()}"])
        if url == BASE + "call/replace":
            self.sent = json.loads(request.content)["data"]
            return httpx.Response(200, json={"event_id": "ev1"})
        if url == BASE + "run/status":
            self.status_calls += 1
            state = self.states.pop(0) if len(self.states) > 1 else self.states[0]
            if state == "coupure":
                return httpx.Response(502, text="proxy")
            return httpx.Response(200, json={"data": [json.dumps(state)]})
        if url == BASE + "run/cancel":
            self.cancelled = True
            return httpx.Response(200, json={"data": ["ok"]})
        if url == BASE + "call/replace/ev1":
            outputs = [{"path": "/tmp/v.mp4", "url": "/gradio_api/file=/tmp/v.mp4", "orig_name": "result.mp4"},
                       {"path": "/tmp/m.mp4", "url": BASE + "file=/tmp/m.mp4", "orig_name": "mask.mp4"},
                       json.dumps({"frames": 150, "gpu_seconds": 98.5})]
            return httpx.Response(200, text="event: heartbeat\ndata: null\n\n"
                                            f"event: complete\ndata: {json.dumps(outputs)}\n\n")
        if url.endswith("file=/tmp/v.mp4"):
            return httpx.Response(200, content=b"video")
        if url.endswith("file=/tmp/m.mp4"):
            return httpx.Response(200, content=b"mask")
        return httpx.Response(404)


def _client(monkeypatch, space: FakeSpace):
    import gradio_client

    from app.worker.zerogpu_client import ZeroGPUClient

    class FakeGradioClient:
        src_prefixed = BASE
        headers = {"x-hf-authorization": "Bearer tok"}

        def __init__(self, *a, **k):
            pass

        def view_api(self, print_info, return_format):
            return {"named_endpoints": {"/replace": {}, "/status": {}, "/cancel": {}, "/health": {}}}

    real_client = httpx.Client
    monkeypatch.setattr(gradio_client, "Client", FakeGradioClient)
    monkeypatch.setattr(httpx, "Client", lambda **k: real_client(transport=httpx.MockTransport(space), **k))
    return ZeroGPUClient("kevin/c", "tok", "key")


def _call(client, tmp_path, **kw):
    (tmp_path / "clip.mp4").write_bytes(b"clip")
    (tmp_path / "ref.jpg").write_bytes(b"photo")
    return client.replace(tmp_path / "clip.mp4", tmp_path / "ref.jpg", {"t": 1.0}, tmp_path / "out.mp4",
                          tmp_path / "mask_out.mp4", poll=0, **kw)


def test_short_requests_follow_the_job_and_download_the_result(monkeypatch, tmp_path):
    space = FakeSpace([{"state": "unknown"}, {"state": "waiting"}, "coupure",
                       {"state": "running", "done": 250, "total": 1000, "desc": "squelette"},
                       {"state": "running", "done": 600, "total": 1000, "desc": "génération"}, {"state": "done"}])
    client = _client(monkeypatch, space)
    assert client.short_requests
    stages, progress = [], []
    stats = _call(client, tmp_path, on_progress=lambda d, t: progress.append((d, t)),
                  on_stage=lambda s, d, t, desc: stages.append((s, desc)))
    assert stats["gpu_seconds"] == 98.5
    assert (tmp_path / "out.mp4").read_bytes() == b"video" and (tmp_path / "mask_out.mp4").read_bytes() == b"mask"
    assert stages == [("queue", None), ("gpu", None), ("progress", "squelette"), ("progress", "génération")]
    assert progress == [(250, 1000), (600, 1000)]                         # la coupure passagère est ignorée
    clip, ref, payload, key = space.sent
    assert clip["path"].endswith("clip.mp4") and clip["meta"] == {"_type": "gradio.FileData"}
    assert ref["path"].endswith("ref.jpg") and key == "key"
    assert json.loads(payload)["t"] == 1.0 and len(json.loads(payload)["job"]) == 32


def test_short_requests_report_the_space_error(monkeypatch, tmp_path):
    from app.worker.zerogpu_client import ZeroGPUError

    space = FakeSpace([{"state": "waiting"},
                       {"state": "error", "error": "The requested GPU duration (340s) is larger than the maximum allowed"}])
    with pytest.raises(ZeroGPUError, match="refuse de réserver 340 s"):
        _call(_client(monkeypatch, space), tmp_path)


def test_short_requests_detect_a_space_restart(monkeypatch, tmp_path):
    from app.worker.zerogpu_client import ZeroGPUError

    space = FakeSpace([{"state": "running", "done": 100, "total": 1000, "desc": "squelette"}, {"state": "unknown"}])
    with pytest.raises(ZeroGPUError, match="redémarré"):
        _call(_client(monkeypatch, space), tmp_path)


def test_short_requests_cancel_asks_the_space_to_stop(monkeypatch, tmp_path):
    from app.worker.zerogpu_client import RemoteCancelled

    space = FakeSpace([{"state": "running", "done": 100, "total": 1000, "desc": "squelette"}])
    polls = iter([False, False, True])
    with pytest.raises(RemoteCancelled):
        _call(_client(monkeypatch, space), tmp_path, should_cancel=lambda: next(polls))
    assert space.cancelled and space.status_calls == 2
