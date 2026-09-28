"""Carte graphique réinitialisée par Windows en plein rendu : sessions recréées, image refaite, rendu mené à bout."""
from types import SimpleNamespace

import numpy as np
import pytest

from src import pipeline


class FakeOrtError(RuntimeError):
    """Imite onnxruntime.capi.onnxruntime_pybind11_state.RuntimeException."""


FakeOrtError.__module__ = "onnxruntime.capi.onnxruntime_pybind11_state"


class Passthrough:
    samples = 0

    def apply(self, frame, face):
        return frame


def _face():
    return SimpleNamespace(bbox=np.array([10, 10, 80, 90], np.float32),
                           kps=np.array([[30, 40], [60, 40], [45, 55], [32, 70], [58, 70]], np.float32),
                           embedding=None, normed_embedding=None)


def _embed(frame, face):
    face.embedding = face.normed_embedding = np.ones(512, np.float32) / np.sqrt(512)


def _person():
    from src.identity import SourceFace
    from src.levels import PersonAssets

    return PersonAssets(source=SourceFace(np.ones(512, np.float32)))


def test_one_gpu_failure_is_retried(monkeypatch, sample_video, tmp_path):
    calls = {"detect": 0, "reset": 0}

    def detect(frame):
        calls["detect"] += 1
        if calls["detect"] == 5:
            raise FakeOrtError()
        return [_face()]

    monkeypatch.setattr(pipeline, "detect_boxes", detect)
    monkeypatch.setattr(pipeline, "embed", _embed)
    monkeypatch.setattr(pipeline, "make_strategy", lambda level, person, restore=False: Passthrough())
    monkeypatch.setattr(pipeline, "reset_sessions", lambda: calls.__setitem__("reset", calls["reset"] + 1))
    monkeypatch.setattr(pipeline.time, "sleep", lambda s: None)
    stats = pipeline.swap_segment(sample_video, [pipeline.FaceMapping(_person())], "face", tmp_path / "out.mp4")
    assert calls["reset"] == 1
    assert stats.swapped == stats.frames - stats.reused
    assert any("réinitialisée 1 fois" in w for w in stats.warnings)


def test_other_errors_are_not_swallowed(monkeypatch, sample_video, tmp_path):
    def detect(frame):
        raise ValueError("bug")

    monkeypatch.setattr(pipeline, "detect_boxes", detect)
    monkeypatch.setattr(pipeline, "make_strategy", lambda level, person, restore=False: Passthrough())
    with pytest.raises(ValueError):
        pipeline.swap_segment(sample_video, [pipeline.FaceMapping(_person())], "face", tmp_path / "out.mp4")
