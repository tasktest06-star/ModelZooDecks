"""Tests for AM69A inference engine."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from am69a.inference import AM69AInferenceEngine, SUPPORTED_MODELS

class TestAM69AEngine:
    def test_unsupported_model_raises(self):
        with pytest.raises(ValueError): AM69AInferenceEngine("not_a_model")
    def test_device_name(self):
        assert AM69AInferenceEngine("rtmdet_m_lite", simulate_latency=False).run(np.zeros((640,640,3),dtype=np.uint8))["device"] == "AM69A"
    def test_tops_value(self):
        assert AM69AInferenceEngine("rtmdet_m_lite", simulate_latency=False).TOPS == 32.0
    def test_detection_keys(self):
        assert "detections" in AM69AInferenceEngine("rtmdet_l_lite", simulate_latency=False).run(np.zeros((640,640,3),dtype=np.uint8))
    def test_pose_keypoints_17(self):
        assert AM69AInferenceEngine("yoloxpose_l_lite", simulate_latency=False).run(np.zeros((640,640,3),dtype=np.uint8))["num_keypoints"] == 17
    def test_classification_top5(self):
        assert len(AM69AInferenceEngine("swin_tiny", simulate_latency=False).run(np.zeros((224,224,3),dtype=np.uint8))["top5"]) == 5
    def test_segmentation_map_2d(self):
        result = AM69AInferenceEngine("deeplabv3plus_lite", simulate_latency=False).run(np.zeros((512,512,3),dtype=np.uint8))
        assert len(result["class_map_shape"]) == 2
    def test_depth_shape(self):
        assert "depth_map_shape" in AM69AInferenceEngine("midas_small_lite", simulate_latency=False).run(np.zeros((256,256,3),dtype=np.uint8))
    def test_profile_keys(self):
        p = AM69AInferenceEngine("rtmdet_m_lite", simulate_latency=False).profile()
        for k in ("device","tops","ram_gb","model","task","nominal_latency_ms"): assert k in p
    def test_all_models_run(self):
        img = np.zeros((640,640,3),dtype=np.uint8)
        for name in SUPPORTED_MODELS: assert "latency_ms" in AM69AInferenceEngine(name, simulate_latency=False).run(img)
