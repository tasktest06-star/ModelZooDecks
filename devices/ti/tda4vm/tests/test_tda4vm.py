"""Tests for TDA4VM inference engine."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tda4vm.inference import TDA4VMInferenceEngine, SUPPORTED_MODELS

class TestTDA4VMEngine:
    def test_unsupported_model_raises(self):
        with pytest.raises(ValueError): TDA4VMInferenceEngine("not_a_model")
    def test_device_name(self):
        assert TDA4VMInferenceEngine("yolox_s_lite", simulate_latency=False).run(np.zeros((640,640,3),dtype=np.uint8))["device"] == "TDA4VM"
    def test_safety_label(self):
        assert TDA4VMInferenceEngine("yolox_s_lite", simulate_latency=False).run(np.zeros((640,640,3),dtype=np.uint8))["safety"] == "ASIL-B"
    def test_tops_value(self):
        assert TDA4VMInferenceEngine.TOPS == 8.0
    def test_peoplenet_keys(self):
        result = TDA4VMInferenceEngine("peoplenet_lite", simulate_latency=False).run(np.zeros((544,960,3),dtype=np.uint8))
        for k in ("device","safety","model","task","latency_ms","detections"): assert k in result
    def test_segmentation_keys(self):
        result = TDA4VMInferenceEngine("deeplabv3plus_lite", simulate_latency=False).run(np.zeros((512,512,3),dtype=np.uint8))
        assert "class_map_shape" in result and len(result["class_map_shape"]) == 2
    def test_profile_safety(self):
        p = TDA4VMInferenceEngine("peoplenet_lite", simulate_latency=False).profile()
        assert p["safety"] == "ASIL-B"
        for k in ("device","tops","ram_gb","model","task","nominal_latency_ms"): assert k in p
    def test_batch_run(self):
        assert len(TDA4VMInferenceEngine("yolox_s_lite", simulate_latency=False).batch_run([np.zeros((640,640,3),dtype=np.uint8)]*3)) == 3
    def test_warmup_flag(self):
        eng = TDA4VMInferenceEngine("yolox_s_lite", simulate_latency=False); eng.warmup(n=1)
        assert eng._warmed_up and eng._inference_count == 0
    def test_all_models_run(self):
        img = np.zeros((960,960,3),dtype=np.uint8)
        for name in SUPPORTED_MODELS: assert "latency_ms" in TDA4VMInferenceEngine(name, simulate_latency=False).run(img)
