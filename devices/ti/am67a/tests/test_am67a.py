"""Tests for AM67A inference engine."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from am67a.inference import AM67AInferenceEngine, SUPPORTED_MODELS

class TestAM67AEngine:
    def test_unsupported_model_raises(self):
        with pytest.raises(ValueError): AM67AInferenceEngine("nonexistent_model")
    def test_device_name(self):
        assert AM67AInferenceEngine("yolox_s_lite", simulate_latency=False).run(np.zeros((640,640,3),dtype=np.uint8))["device"] == "AM67A"
    def test_tops_value(self):
        assert AM67AInferenceEngine("yolox_s_lite", simulate_latency=False).TOPS == 4.0
    def test_detection_result_keys(self):
        result = AM67AInferenceEngine("yolox_s_lite", simulate_latency=False).run(np.zeros((640,640,3),dtype=np.uint8))
        for k in ("device","model","task","latency_ms","detections"): assert k in result
    def test_classification_result_keys(self):
        result = AM67AInferenceEngine("mobilenet_v3_large_lite", simulate_latency=False).run(np.zeros((224,224,3),dtype=np.uint8))
        for k in ("device","model","task","latency_ms","top5","top1_class"): assert k in result
    def test_segmentation_result_keys(self):
        assert "class_map_shape" in AM67AInferenceEngine("bisenetv2_lite", simulate_latency=False).run(np.zeros((512,512,3),dtype=np.uint8))
    def test_depth_result_keys(self):
        assert "depth_map_shape" in AM67AInferenceEngine("fastdepth_lite", simulate_latency=False).run(np.zeros((320,256,3),dtype=np.uint8))
    def test_batch_run_length(self):
        assert len(AM67AInferenceEngine("yolox_tiny_lite", simulate_latency=False).batch_run([np.zeros((416,416,3),dtype=np.uint8)]*4)) == 4
    def test_profile_keys(self):
        p = AM67AInferenceEngine("yolox_s_lite", simulate_latency=False).profile()
        for k in ("device","tops","ram_gb","model","task","nominal_latency_ms"): assert k in p
    def test_inference_count(self):
        eng = AM67AInferenceEngine("mobilenet_v3_large_lite", simulate_latency=False); img = np.zeros((224,224,3),dtype=np.uint8)
        eng.run(img); eng.run(img); assert eng._inference_count == 2
    def test_warmup_resets_count(self):
        eng = AM67AInferenceEngine("mobilenet_v3_large_lite", simulate_latency=False); eng.warmup(n=2)
        assert eng._inference_count == 0 and eng._warmed_up
    def test_all_models_run(self):
        img = np.zeros((640,640,3),dtype=np.uint8)
        for name in SUPPORTED_MODELS: assert "latency_ms" in AM67AInferenceEngine(name, simulate_latency=False).run(img)
