"""Tests for RT1170 TFLite inference engine."""
import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[4]))

from devices.nxp.rt1170.inference import RT1170InferenceEngine
from devices.nxp.common.preprocess import preprocess_microspeech


@pytest.fixture
def rgb224():
    return np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)

@pytest.fixture
def audio():
    return np.random.randn(16000).astype(np.float32)

@pytest.fixture
def cls_engine():
    return RT1170InferenceEngine("mobilenetv2_100", simulate_latency=False)

@pytest.fixture
def kws_engine():
    return RT1170InferenceEngine("microspeech_kws", simulate_latency=False)

@pytest.fixture
def det_engine():
    return RT1170InferenceEngine("ssdlite_mobilenetv2", simulate_latency=False)


class TestPreprocessing:
    def test_microspeech_shape(self, audio):
        feat = preprocess_microspeech(audio)
        assert feat.shape == (1, 49, 40, 1)

    def test_microspeech_dtype(self, audio):
        feat = preprocess_microspeech(audio)
        assert feat.dtype == np.float32


class TestRT1170Engine:
    def test_unsupported_raises(self):
        with pytest.raises(ValueError):
            RT1170InferenceEngine("vgg16")

    def test_classification_keys(self, cls_engine, rgb224):
        r = cls_engine.run(rgb224)
        for k in ["task", "top5", "top1_class", "device", "latency_ms"]:
            assert k in r

    def test_device_name(self, cls_engine, rgb224):
        assert cls_engine.run(rgb224)["device"] == "RT1170"

    def test_top5_length(self, cls_engine, rgb224):
        assert len(cls_engine.run(rgb224)["top5"]) == 5

    def test_kws_keys(self, kws_engine, audio):
        r = kws_engine.run(audio)
        for k in ["task", "keyword", "confidence", "top3"]:
            assert k in r

    def test_kws_keyword_valid(self, kws_engine, audio):
        r = kws_engine.run(audio)
        assert r["keyword"] in RT1170InferenceEngine.KEYWORDS

    def test_kws_confidence_range(self, kws_engine, audio):
        r = kws_engine.run(audio)
        assert 0.0 <= r["confidence"] <= 1.0

    def test_detection_keys(self, det_engine, rgb224):
        r = det_engine.run(rgb224)
        assert "detections" in r and "num_detections" in r

    def test_batch_run_length(self, cls_engine, rgb224):
        assert len(cls_engine.batch_run([rgb224, rgb224, rgb224])) == 3

    def test_all_models_instantiate(self):
        for name in RT1170InferenceEngine.SUPPORTED_MODELS:
            eng = RT1170InferenceEngine(name, simulate_latency=False)
            assert eng.model_name == name

    def test_profile_npu_false(self, cls_engine):
        assert cls_engine.profile()["npu"] is False
