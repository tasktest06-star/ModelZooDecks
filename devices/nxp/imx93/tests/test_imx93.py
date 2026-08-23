"""Tests for i.MX 93 Ethos-U65 inference engine."""
import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[4]))

from devices.nxp.imx93.inference import IMX93InferenceEngine


@pytest.fixture
def rgb():
    return np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)

@pytest.fixture
def audio():
    return np.random.randn(16000).astype(np.float32)

@pytest.fixture
def cls_engine():
    return IMX93InferenceEngine("mobilenetv2_100", simulate_latency=False)

@pytest.fixture
def kws_engine():
    return IMX93InferenceEngine("ds_cnn_l", simulate_latency=False)

@pytest.fixture
def det_engine():
    return IMX93InferenceEngine("nanodet_plus_320", simulate_latency=False)


class TestIMX93Engine:
    def test_unsupported_raises(self):
        with pytest.raises(ValueError):
            IMX93InferenceEngine("yolov8m")

    def test_cls_keys(self, cls_engine, rgb):
        r = cls_engine.run(rgb)
        for k in ["task", "top5", "device", "latency_ms", "vela_speedup", "backend"]:
            assert k in r

    def test_device_name(self, cls_engine, rgb):
        assert cls_engine.run(rgb)["device"] == "i.MX 93"

    def test_backend_ethos(self, cls_engine, rgb):
        assert cls_engine.run(rgb)["backend"] == "ethos-u65"

    def test_vela_speedup_positive(self, cls_engine, rgb):
        assert cls_engine.run(rgb)["vela_speedup"] > 1.0

    def test_kws_keyword_valid(self, kws_engine, audio):
        from devices.nxp.imx93.inference import KEYWORDS
        r = kws_engine.run(audio)
        assert r["keyword"] in KEYWORDS

    def test_det_keys(self, det_engine, rgb):
        r = det_engine.run(rgb)
        assert "detections" in r and "num_detections" in r

    def test_batch_run(self, cls_engine, rgb):
        assert len(cls_engine.batch_run([rgb, rgb])) == 2

    def test_all_models_instantiate(self):
        for name in IMX93InferenceEngine.SUPPORTED_MODELS:
            eng = IMX93InferenceEngine(name, simulate_latency=False)
            assert eng.model_name == name

    def test_profile_tops(self, cls_engine):
        assert cls_engine.profile()["tops"] == 1.0

    def test_profile_speedup_gt1(self, cls_engine):
        assert cls_engine.profile()["vela_speedup"] > 1.0
