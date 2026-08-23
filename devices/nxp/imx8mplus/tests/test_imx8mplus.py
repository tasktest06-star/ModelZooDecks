"""Tests for i.MX 8M Plus NPU inference engine."""
import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[4]))

from devices.nxp.imx8mplus.inference import IMX8MPlusInferenceEngine


@pytest.fixture
def rgb():
    return np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)

@pytest.fixture
def audio():
    return np.random.randn(16000).astype(np.float32)

@pytest.fixture
def face_img():
    return np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)

@pytest.fixture
def cls_engine():
    return IMX8MPlusInferenceEngine("resnet50", simulate_latency=False)

@pytest.fixture
def det_engine():
    return IMX8MPlusInferenceEngine("yolov8m", simulate_latency=False)

@pytest.fixture
def face_engine():
    return IMX8MPlusInferenceEngine("facenet512", simulate_latency=False)

@pytest.fixture
def asr_engine():
    return IMX8MPlusInferenceEngine("wav2letter", simulate_latency=False)


class TestIMX8MPlusEngine:
    def test_unsupported_raises(self):
        with pytest.raises(ValueError):
            IMX8MPlusInferenceEngine("shufflenet")

    def test_cls_keys(self, cls_engine, rgb):
        r = cls_engine.run(rgb)
        for k in ["task", "top5", "top1_class", "device", "latency_ms"]:
            assert k in r

    def test_device_name(self, cls_engine, rgb):
        assert cls_engine.run(rgb)["device"] == "i.MX 8M Plus"

    def test_backend_nxp(self, cls_engine, rgb):
        assert cls_engine.run(rgb)["backend"] == "nxp-npu"

    def test_top5_length(self, cls_engine, rgb):
        assert len(cls_engine.run(rgb)["top5"]) == 5

    def test_detection_keys(self, det_engine, rgb):
        r = det_engine.run(rgb)
        assert "detections" in r and "num_detections" in r

    def test_face_embedding_dim(self, face_engine, face_img):
        r = face_engine.run(face_img)
        assert r["embedding_dim"] == 512
        assert len(r["embedding"]) == 8

    def test_face_task_name(self, face_engine, face_img):
        assert face_engine.run(face_img)["task"] == "face_recognition"

    def test_asr_keys(self, asr_engine, audio):
        r = asr_engine.run(audio)
        assert "prediction" in r and "wer_pct" in r

    def test_asr_wer_value(self, asr_engine, audio):
        r = asr_engine.run(audio)
        assert r["wer_pct"] == pytest.approx(7.2)

    def test_batch_run_count(self, cls_engine, rgb):
        assert len(cls_engine.batch_run([rgb, rgb, rgb])) == 3

    def test_all_models_instantiate(self):
        for name in IMX8MPlusInferenceEngine.SUPPORTED_MODELS:
            eng = IMX8MPlusInferenceEngine(name, simulate_latency=False)
            assert eng.model_name == name

    def test_profile_tops(self, cls_engine):
        assert cls_engine.profile()["tops"] == 2.3

    def test_profile_ram(self, cls_engine):
        assert cls_engine.profile()["ram_gb"] == 4
