"""Tests for MAX78002 inference engine."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.preprocess import preprocess_image, preprocess_audio_segment, compute_mfcc
from common.postprocess import softmax, topk_classes, decode_kws_output, energy_estimate_uj
from max78002.inference import MAX78002InferenceEngine, SUPPORTED_MODELS, KWS_LABELS


class TestPreprocessing:
    def test_preprocess_image_shape(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        out = preprocess_image(img, (96, 96))
        assert out.shape == (1, 3, 96, 96)

    def test_preprocess_image_range(self):
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        out = preprocess_image(img, (64, 64))
        assert out.dtype == np.float32
        assert float(out.max()) <= 127.0

    def test_mfcc_shape(self):
        waveform = np.zeros(16000, dtype=np.float32)
        mfcc = compute_mfcc(waveform)
        assert mfcc.shape == (1, 64, 101)

    def test_audio_segment_pads(self):
        short = np.zeros(8000, dtype=np.int16)
        out = preprocess_audio_segment(short, pad=True)
        assert out.shape[2] == 101


class TestPostprocessing:
    def test_softmax_sums_to_one(self):
        logits = np.array([1.0, 2.0, 3.0])
        assert abs(softmax(logits).sum() - 1.0) < 1e-5

    def test_topk_with_labels(self):
        probs = np.array([0.1, 0.7, 0.2])
        results = topk_classes(probs, k=2, labels=["a", "b", "c"])
        assert results[0]["label"] == "b"
        assert results[0]["score"] == pytest.approx(0.7, abs=1e-5)

    def test_decode_kws_keys(self):
        logits = np.random.randn(1, 12)
        result = decode_kws_output(logits, KWS_LABELS)
        for key in ("keyword", "confidence", "class_id", "probs"):
            assert key in result

    def test_decode_kws_confidence_range(self):
        logits = np.random.randn(1, 12)
        result = decode_kws_output(logits, KWS_LABELS)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_energy_estimate(self):
        energy = energy_estimate_uj(latency_ms=0.15, power_mw=15.0)
        assert abs(energy - 2.25) < 0.01


class TestMAX78002Engine:
    def test_unsupported_raises(self):
        with pytest.raises(ValueError):
            MAX78002InferenceEngine("unknown_model")

    def test_device_name(self):
        eng = MAX78002InferenceEngine("keyword_spotting_ds_cnn", simulate_latency=False)
        waveform = np.zeros(16000, dtype=np.float32)
        result = eng.run_audio(waveform)
        assert result["device"] == "MAX78002"

    def test_kws_keys(self):
        eng = MAX78002InferenceEngine("keyword_spotting_ds_cnn", simulate_latency=False)
        waveform = np.zeros(16000, dtype=np.float32)
        result = eng.run_audio(waveform)
        for key in ("device", "model", "task", "latency_ms", "energy_uj", "keyword", "confidence"):
            assert key in result

    def test_kws_confidence_range(self):
        eng = MAX78002InferenceEngine("keyword_spotting_ds_cnn", simulate_latency=False)
        waveform = np.zeros(16000, dtype=np.float32)
        result = eng.run_audio(waveform)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_vww_image_keys(self):
        eng = MAX78002InferenceEngine("visual_wake_words", simulate_latency=False)
        img = np.zeros((96, 96, 3), dtype=np.uint8)
        result = eng.run_image(img)
        for key in ("device", "model", "task", "latency_ms", "energy_uj", "top1"):
            assert key in result

    def test_detection_keys(self):
        eng = MAX78002InferenceEngine("face_detection_fpn", simulate_latency=False)
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        result = eng.run_image(img)
        assert "detections" in result

    def test_anomaly_keys(self):
        eng = MAX78002InferenceEngine("anomaly_detection", simulate_latency=False)
        ts = np.random.randn(128).astype(np.float32)
        result = eng.run_sensor(ts)
        for key in ("device", "model", "task", "anomaly_score", "is_anomaly", "energy_uj"):
            assert key in result

    def test_energy_uj_positive(self):
        eng = MAX78002InferenceEngine("keyword_spotting_ds_cnn", simulate_latency=False)
        waveform = np.zeros(16000, dtype=np.float32)
        result = eng.run_audio(waveform)
        assert result["energy_uj"] > 0

    def test_profile_keys(self):
        eng = MAX78002InferenceEngine("visual_wake_words", simulate_latency=False)
        p = eng.profile()
        for key in ("device", "vendor", "sram_kb", "efficiency_tops_w", "model",
                    "task", "nominal_latency_ms", "nominal_energy_uj"):
            assert key in p

    def test_warmup(self):
        eng = MAX78002InferenceEngine("keyword_spotting_ds_cnn", simulate_latency=False)
        eng.warmup(n=2)
        assert eng._warmed_up
        assert eng._inference_count == 0

    def test_all_models_instantiate(self):
        for name in SUPPORTED_MODELS:
            eng = MAX78002InferenceEngine(name, simulate_latency=False)
            assert eng.model_name == name
