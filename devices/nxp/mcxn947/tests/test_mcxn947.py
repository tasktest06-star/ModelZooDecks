"""Tests for MCX N947 TFLite Micro inference engine."""
import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[4]))

from devices.nxp.mcxn947.inference import MCXInferenceEngine
from devices.nxp.common.preprocess import preprocess_classification
from devices.nxp.common.postprocess import softmax, topk


@pytest.fixture
def rgb96():
    return np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8)

@pytest.fixture
def engine():
    return MCXInferenceEngine("mobilenetv1_025", simulate_latency=False)


class TestPreprocessing:
    def test_output_shape_96(self, rgb96):
        out = preprocess_classification(rgb96, (96, 96))
        assert out.shape == (1, 96, 96, 3)

    def test_output_dtype_float32(self, rgb96):
        out = preprocess_classification(rgb96, (96, 96))
        assert out.dtype == np.float32

    def test_output_shape_224(self):
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        out = preprocess_classification(img, (224, 224))
        assert out.shape == (1, 224, 224, 3)


class TestPostprocessing:
    def test_softmax_sum(self):
        logits = np.random.randn(1001).astype(np.float32)
        assert abs(softmax(logits).sum() - 1.0) < 1e-5

    def test_topk_length(self):
        probs = softmax(np.random.randn(1001).astype(np.float32))
        assert len(topk(probs, k=5)) == 5

    def test_topk_sorted(self):
        probs = softmax(np.random.randn(1001).astype(np.float32))
        t = topk(probs, k=5)
        scores = [x["score"] for x in t]
        assert scores == sorted(scores, reverse=True)


class TestMCXEngine:
    def test_unsupported_model_raises(self):
        with pytest.raises(ValueError):
            MCXInferenceEngine("resnet50")

    def test_result_keys(self, engine, rgb96):
        r = engine.run(rgb96)
        for k in ["task", "top5", "device", "model", "latency_ms", "ram_used_kb"]:
            assert k in r

    def test_device_name(self, engine, rgb96):
        assert engine.run(rgb96)["device"] == "MCX-N947"

    def test_top5_length(self, engine, rgb96):
        assert len(engine.run(rgb96)["top5"]) == 5

    def test_top5_scores_valid(self, engine, rgb96):
        for item in engine.run(rgb96)["top5"]:
            assert 0.0 <= item["score"] <= 1.0

    def test_batch_run(self, engine, rgb96):
        results = engine.batch_run([rgb96, rgb96])
        assert len(results) == 2

    def test_profile_keys(self, engine):
        p = engine.profile()
        assert "sram_kb" in p and "latency_ms" in p and "runtime" in p

    def test_npu_is_false(self, engine):
        assert engine.profile()["npu"] is False

    def test_ram_budget_respected(self):
        eng = MCXInferenceEngine("mobilenetv1_025", simulate_latency=False)
        assert eng.meta["ram_kb"] <= MCXInferenceEngine.RAM_BUDGET_KB
