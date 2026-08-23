"""Tests for AM62A inference engine."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.preprocess import letterbox, normalize_imagenet, preprocess_classification, preprocess_detection, preprocess_segmentation
from common.postprocess import softmax, topk_classes, decode_yolox_output, argmax_segmentation
from am62a.inference import AM62AInferenceEngine, SUPPORTED_MODELS


class TestPreprocessing:
    def test_letterbox_output_shape(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        out, scale, dw, dh = letterbox(img, (416, 416))
        assert out.shape == (416, 416, 3)

    def test_letterbox_scale(self):
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        _, scale, _, _ = letterbox(img, (200, 200))
        assert abs(scale - 1.0) < 1e-6

    def test_normalize_imagenet_shape(self):
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        assert normalize_imagenet(img).shape == (1, 224, 224, 3)

    def test_normalize_imagenet_dtype(self):
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        assert normalize_imagenet(img).dtype == np.float32

    def test_preprocess_classification_shape(self):
        img = np.zeros((300, 400, 3), dtype=np.uint8)
        assert preprocess_classification(img, (224, 224)).shape == (1, 224, 224, 3)

    def test_preprocess_detection_shape(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        tensor, scale, dw, dh = preprocess_detection(img, (320, 320))
        assert tensor.shape == (1, 3, 320, 320)

    def test_preprocess_segmentation_shape(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        assert preprocess_segmentation(img, (512, 512)).shape == (1, 3, 512, 512)


class TestPostprocessing:
    def test_softmax_sums_to_one(self):
        assert abs(softmax(np.array([1.0, 2.0, 3.0])).sum() - 1.0) < 1e-5

    def test_softmax_nonnegative(self):
        assert (softmax(np.random.randn(100)) >= 0).all()

    def test_topk_length(self):
        assert len(topk_classes(np.random.dirichlet(np.ones(1000)), k=5)) == 5

    def test_topk_sorted(self):
        results = topk_classes(np.random.dirichlet(np.ones(1000)), k=5)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_topk_keys(self):
        result = topk_classes(np.random.dirichlet(np.ones(10)), k=1)[0]
        assert "class_id" in result and "score" in result

    def test_decode_yolox_low_conf(self):
        assert decode_yolox_output(np.zeros((1, 100, 85), dtype=np.float32), (320, 320)) == []

    def test_decode_yolox_high_conf(self):
        output = np.zeros((1, 1, 85), dtype=np.float32)
        output[0, 0, 4] = 0.9; output[0, 0, 5] = 0.9; output[0, 0, 2] = 10.0; output[0, 0, 3] = 10.0
        assert len(decode_yolox_output(output, (320, 320))) >= 1

    def test_argmax_segmentation(self):
        output = np.zeros((1, 21, 64, 64), dtype=np.float32)
        output[0, 5, :, :] = 1.0
        mask = argmax_segmentation(output)
        assert mask.shape == (64, 64) and mask.dtype == np.uint8 and (mask == 5).all()


class TestAM62AEngine:
    def test_unsupported_model_raises(self):
        with pytest.raises(ValueError):
            AM62AInferenceEngine("nonexistent_model_lite")

    def test_classification_result_keys(self):
        eng = AM62AInferenceEngine("mobilenet_v2_lite", simulate_latency=False)
        result = eng.run(np.zeros((224, 224, 3), dtype=np.uint8))
        for key in ("device", "model", "task", "latency_ms", "top5", "top1_class"):
            assert key in result

    def test_detection_result_keys(self):
        eng = AM62AInferenceEngine("yolox_pico_lite", simulate_latency=False)
        result = eng.run(np.zeros((320, 320, 3), dtype=np.uint8))
        for key in ("device", "model", "task", "latency_ms", "detections"):
            assert key in result

    def test_top5_length(self):
        eng = AM62AInferenceEngine("mobilenet_v2_lite", simulate_latency=False)
        assert len(eng.run(np.zeros((224, 224, 3), dtype=np.uint8))["top5"]) == 5

    def test_top5_scores_sum_le_one(self):
        eng = AM62AInferenceEngine("mobilenet_v2_lite", simulate_latency=False)
        total = sum(r["score"] for r in eng.run(np.zeros((224, 224, 3), dtype=np.uint8))["top5"])
        assert total <= 1.0 + 1e-5

    def test_device_name(self):
        eng = AM62AInferenceEngine("mobilenet_v2_lite", simulate_latency=False)
        assert eng.run(np.zeros((224, 224, 3), dtype=np.uint8))["device"] == "AM62A"

    def test_batch_run_length(self):
        eng = AM62AInferenceEngine("mobilenet_v2_lite", simulate_latency=False)
        assert len(eng.batch_run([np.zeros((224, 224, 3), dtype=np.uint8)] * 3)) == 3

    def test_profile_keys(self):
        eng = AM62AInferenceEngine("mobilenet_v2_lite", simulate_latency=False)
        for key in ("device", "tops", "ram_gb", "model", "task", "nominal_latency_ms"):
            assert key in eng.profile()

    def test_warmup_flag_set(self):
        eng = AM62AInferenceEngine("mobilenet_v2_lite", simulate_latency=False)
        assert not eng._warmed_up
        eng.warmup(n=1)
        assert eng._warmed_up

    def test_inference_count_increments(self):
        eng = AM62AInferenceEngine("mobilenet_v2_lite", simulate_latency=False)
        eng.run(np.zeros((224, 224, 3), dtype=np.uint8))
        eng.run(np.zeros((224, 224, 3), dtype=np.uint8))
        assert eng._inference_count == 2

    def test_all_supported_models_instantiate(self):
        for name in SUPPORTED_MODELS:
            assert AM62AInferenceEngine(name, simulate_latency=False).model_name == name

    def test_all_supported_models_run(self):
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        for name in SUPPORTED_MODELS:
            assert "latency_ms" in AM62AInferenceEngine(name, simulate_latency=False).run(img)
