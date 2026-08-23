"""Tests for MAX32690 inference engine."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from max32690.inference import MAX32690InferenceEngine, SUPPORTED_MODELS


class TestMAX32690Engine:
    def test_unsupported_raises(self):
        with pytest.raises(ValueError):
            MAX32690InferenceEngine("unknown_model")

    def test_device_name(self):
        eng = MAX32690InferenceEngine("anomaly_detection", simulate_latency=False)
        ts = np.random.randn(64).astype(np.float32)
        result = eng.run(ts.reshape(1, 64))
        assert result["device"] == "MAX32690"

    def test_kws_keys(self):
        eng = MAX32690InferenceEngine("keyword_spotting_simple", simulate_latency=False)
        waveform = np.zeros(16000, dtype=np.float32)
        result = eng.run_audio(waveform)
        for key in ("device", "model", "task", "latency_ms", "energy_uj", "keyword", "confidence"):
            assert key in result

    def test_kws_confidence_range(self):
        eng = MAX32690InferenceEngine("keyword_spotting_simple", simulate_latency=False)
        waveform = np.zeros(16000, dtype=np.float32)
        result = eng.run_audio(waveform)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_anomaly_keys(self):
        eng = MAX32690InferenceEngine("anomaly_detection", simulate_latency=False)
        ts = np.random.randn(64).astype(np.float32)
        result = eng.run(ts.reshape(1, 64))
        for key in ("device", "model", "task", "anomaly_score", "is_anomaly"):
            assert key in result

    def test_classification_keys(self):
        eng = MAX32690InferenceEngine("vibration_classifier", simulate_latency=False)
        ts = np.random.randn(256).astype(np.float32)
        result = eng.run(ts.reshape(1, 256))
        for key in ("device", "model", "task", "top1", "probs"):
            assert key in result

    def test_energy_uj_positive(self):
        eng = MAX32690InferenceEngine("anomaly_detection", simulate_latency=False)
        ts = np.zeros((1, 64), dtype=np.float32)
        result = eng.run(ts)
        assert result["energy_uj"] > 0

    def test_profile_keys(self):
        eng = MAX32690InferenceEngine("vibration_classifier", simulate_latency=False)
        p = eng.profile()
        for key in ("device", "vendor", "cpu", "sram_kb", "model", "task", "nominal_latency_ms"):
            assert key in p

    def test_warmup(self):
        eng = MAX32690InferenceEngine("anomaly_detection", simulate_latency=False)
        eng.warmup(n=1)
        assert eng._warmed_up

    def test_all_models_instantiate(self):
        for name in SUPPORTED_MODELS:
            eng = MAX32690InferenceEngine(name, simulate_latency=False)
            assert eng.model_name == name
