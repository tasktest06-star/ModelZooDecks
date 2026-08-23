"""Tests for AM68A inference engine."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from am68a.inference import AM68AInferenceEngine, SUPPORTED_MODELS

class TestAM68AEngine:
    def test_unsupported_model_raises(self):
        with pytest.raises(ValueError): AM68AInferenceEngine("not_a_model")
    def test_device_name(self):
        assert AM68AInferenceEngine("efficientnet_b0_lite", simulate_latency=False).run(np.zeros((224,224,3),dtype=np.uint8))["device"] == "AM68A"
    def test_tops_value(self):
        assert AM68AInferenceEngine("yolox_m_lite", simulate_latency=False).TOPS == 8.0
    def test_classification_top5(self):
        result = AM68AInferenceEngine("efficientnet_b0_lite", simulate_latency=False).run(np.zeros((224,224,3),dtype=np.uint8))
        assert len(result["top5"]) == 5 and "top1_class" in result
    def test_detection_keys(self):
        result = AM68AInferenceEngine("yolox_m_lite", simulate_latency=False).run(np.zeros((640,640,3),dtype=np.uint8))
        assert "detections" in result and "num_detections" in result
    def test_segmentation_map_shape(self):
        result = AM68AInferenceEngine("deeplabv3plus_lite", simulate_latency=False).run(np.zeros((512,512,3),dtype=np.uint8))
        assert "class_map_shape" in result and len(result["class_map_shape"]) == 2
    def test_pose_keypoints(self):
        result = AM68AInferenceEngine("yoloxpose_s_lite", simulate_latency=False).run(np.zeros((640,640,3),dtype=np.uint8))
        assert "keypoints" in result and result["num_keypoints"] == 17
    def test_depth_map_shape(self):
        assert "depth_map_shape" in AM68AInferenceEngine("midas_small_lite", simulate_latency=False).run(np.zeros((256,256,3),dtype=np.uint8))
    def test_batch_run(self):
        assert len(AM68AInferenceEngine("efficientnet_b0_lite", simulate_latency=False).batch_run([np.zeros((224,224,3),dtype=np.uint8)]*5)) == 5
    def test_profile_keys(self):
        p = AM68AInferenceEngine("yolox_l_lite", simulate_latency=False).profile()
        for k in ("device","tops","ram_gb","model","task","nominal_latency_ms"): assert k in p
    def test_all_models_run(self):
        img = np.zeros((640,640,3),dtype=np.uint8)
        for name in SUPPORTED_MODELS: assert "latency_ms" in AM68AInferenceEngine(name, simulate_latency=False).run(img)
