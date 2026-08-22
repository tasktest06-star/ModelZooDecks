"""Tests for NXP eIQ model combination pipelines."""

import numpy as np
import pytest

REGISTRY = "config/model_registry.yaml"


class TestLowLightFaceRecognition:
    @pytest.fixture
    def pipeline(self):
        from combinations.low_light_face_recognition import (
            LowLightFaceRecognitionPipeline, FaceRecognitionConfig
        )
        return LowLightFaceRecognitionPipeline(FaceRecognitionConfig(), REGISTRY)

    def test_process_frame_returns_output(self, pipeline):
        from combinations.low_light_face_recognition import FaceRecognitionOutput
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        out = pipeline.process_frame(frame)
        assert isinstance(out, FaceRecognitionOutput)
        assert out.n_faces == 0
        assert out.enhanced_frame is not None

    def test_enhance_image_preserves_shape(self, pipeline):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        enhanced, ms = pipeline.enhance_image(frame)
        assert enhanced.shape == frame.shape
        assert ms >= 0

    def test_crop_face_valid_bbox(self, pipeline):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        crop = pipeline._crop_face(frame, [100, 50, 200, 150])
        assert crop.shape == (160, 160, 3)

    def test_crop_face_invalid_bbox(self, pipeline):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        crop = pipeline._crop_face(frame, [80, 80, 10, 10])
        assert crop.shape == (160, 160, 3)

    def test_match_identity_no_db(self, pipeline):
        emb = np.random.randn(512).astype(np.float32)
        identity, conf = pipeline.match_identity(emb)
        assert identity is None

    def test_enroll_and_match(self, pipeline):
        face = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
        pipeline.enroll_identity("alice", face)
        assert "alice" in pipeline.identity_db

    def test_embedding_shape(self, pipeline):
        crop = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
        emb, ms = pipeline.extract_embedding(crop)
        assert emb.shape == (512,)
        assert ms >= 0


class TestDriverMonitoring:
    @pytest.fixture
    def pipeline(self):
        from combinations.driver_monitoring import DriverMonitoringPipeline, DriverMonitorConfig
        return DriverMonitoringPipeline(DriverMonitorConfig(), REGISTRY)

    def test_process_frame_no_face(self, pipeline):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        state = pipeline.process_frame(frame)
        assert not state.face_detected
        assert state.alert_level == "normal"

    def test_process_with_audio(self, pipeline):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        audio = np.zeros(16000, dtype=np.float32)
        state = pipeline.process_frame(frame, audio)
        assert "kws_ms" in state.latency_ms

    def test_drowsiness_accumulates(self, pipeline):
        from combinations.driver_monitoring import DROWSINESS_EMOTIONS
        drowsy_emotion = list(DROWSINESS_EMOTIONS)[0]
        for _ in range(pipeline.config.drowsiness_consecutive_frames + 1):
            drowsy = pipeline._assess_drowsiness(drowsy_emotion)
        assert drowsy

    def test_distraction_at_high_yaw(self, pipeline):
        pose = {"yaw": 45.0, "pitch": 0.0, "roll": 0.0}
        distracted, on_road = pipeline._assess_distraction(pose)
        assert distracted
        assert not on_road

    def test_alert_level_logic(self, pipeline):
        from combinations.driver_monitoring import DriverState
        state = DriverState(is_drowsy=True, is_distracted=True)
        level = pipeline._compute_alert_level(state)
        assert level == "critical"

    def test_latency_keys_present(self, pipeline):
        state = pipeline.process_frame(np.zeros((480, 640, 3), dtype=np.uint8))
        assert "face_det_ms" in state.latency_ms
        assert "total_ms" in state.latency_ms


class TestSmartVideoAnalytics:
    @pytest.fixture
    def pipeline(self):
        from combinations.smart_video_analytics import SmartVideoAnalyticsPipeline, VideoAnalyticsConfig
        return SmartVideoAnalyticsPipeline(VideoAnalyticsConfig(), REGISTRY)

    def test_process_frame_returns_output(self, pipeline):
        from combinations.smart_video_analytics import VideoAnalyticsOutput
        out = pipeline.process_frame(np.zeros((720, 1280, 3), dtype=np.uint8))
        assert isinstance(out, VideoAnalyticsOutput)
        assert "total_ms" in out.latency_ms

    def test_enhance_passthrough(self, pipeline):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        enhanced, ms = pipeline.enhance(frame)
        assert enhanced.shape == frame.shape
        assert ms == 0.0

    def test_depth_map_shape(self, pipeline):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        depth, ms = pipeline.estimate_depth(frame)
        assert depth is not None
        assert depth.shape == (480, 640)

    def test_segmentation_mask_shape(self, pipeline):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mask, ms = pipeline.segment(frame)
        assert mask is not None
        assert mask.shape == (480, 640)

    def test_object_depth_lookup(self, pipeline):
        depth = np.ones((100, 100), dtype=np.float32) * 0.5
        d = pipeline._get_object_depth(depth, [10, 10, 50, 50], (100, 100))
        assert 0.0 <= d <= 1.0

    def test_no_crash_with_empty_detections(self, pipeline):
        out = pipeline.process_frame(np.zeros((240, 320, 3), dtype=np.uint8))
        assert out.objects == []
