"""Choix du moteur de calcul local (auto / cpu / dml / cuda) et répartition DirectML entre deux cartes."""
import onnxruntime
import pytest

from src import config


@pytest.fixture
def engine(monkeypatch):
    """Simule les moteurs installés et la config, sans dépendre de la machine."""
    def setup(available: list[str], **cfg):
        monkeypatch.setattr(onnxruntime, "get_available_providers", lambda: available)
        monkeypatch.setattr(config, "_preload_cuda", lambda: None)
        base = config.Config({"device": "auto", "cuda_device_id": 0, "dml_placement": "split"})
        base.update(cfg)
        return base
    return setup


def test_auto_prefers_cuda_then_dml_then_cpu(engine):
    assert config.accelerator(engine(["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"])) == "cuda"
    assert config.accelerator(engine(["DmlExecutionProvider", "CPUExecutionProvider"])) == "dml"
    assert config.accelerator(engine(["CPUExecutionProvider"])) == "cpu"


def test_explicit_engine_missing_raises_instead_of_silent_cpu(engine):
    with pytest.raises(config.DeviceUnavailable, match="onnxruntime-directml"):
        config.accelerator(engine(["CPUExecutionProvider"], device="dml"))


def test_explicit_cpu_ignores_gpu(engine):
    assert config.accelerator(engine(["DmlExecutionProvider", "CPUExecutionProvider"], device="cpu")) == "cpu"
    assert config.providers(engine(["DmlExecutionProvider", "CPUExecutionProvider"], device="cpu")) == ["CPUExecutionProvider"]


def test_dml_split_puts_swap_on_strongest_card(engine):
    cfg = engine(["DmlExecutionProvider", "CPUExecutionProvider"])
    swap = config.providers(cfg, role="swap")[0]
    analysis = config.providers(cfg, role="analysis")[0]
    assert swap == ("DmlExecutionProvider", {"performance_preference": "high_performance", "device_filter": "gpu"})
    assert analysis[1]["performance_preference"] == "minimum_power"


def test_dml_single_keeps_everything_on_strongest_card(engine):
    cfg = engine(["DmlExecutionProvider", "CPUExecutionProvider"], dml_placement="single")
    assert config.providers(cfg, role="analysis")[0][1]["performance_preference"] == "high_performance"


def test_cuda_uses_configured_device(engine):
    cfg = engine(["CUDAExecutionProvider", "CPUExecutionProvider"], cuda_device_id=1)
    assert config.providers(cfg)[0] == ("CUDAExecutionProvider", {"device_id": 1})
