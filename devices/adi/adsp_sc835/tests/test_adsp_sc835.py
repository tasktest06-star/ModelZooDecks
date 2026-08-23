"""Tests for ADSP-SC835 inference engine."""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from adsp_sc835.inference import ADSPSC835InferenceEngine, SUPPORTED_MODELS, GENRE_LABELS


class TestADSPSC835Engine:
    def test_unsupported_raises(self):
        with pytest.raises(ValueError):
            ADSPSC835InferenceEngine("unknown_model")

    def test_device_name_speech(self):
        eng = ADSPSC835InferenceEngine("rnnoise_speech_enhancement", simulate_latency=False)
        frame = np.zeros(512, dtype=np.float32)
        result = eng.run_speech_enhancement(frame)
        assert result["device"] == "ADSP-SC835"

    def test_speech_enhancement_keys(self):
        eng = ADSPSC835InferenceEngine("rnnoise_speech_enhancement", simulate_latency=False)
        frame = np.random.randn(512).astype(np.float32)
        result = eng.run_speech_enhancement(frame)
        for key in ("device", "model", "task", "enhanced_frame", "latency_ms", "energy_uj"):
            assert key in result

    def test_enhanced_frame_shape(self):
        eng = ADSPSC835InferenceEngine("rnnoise_speech_enhancement", simulate_latency=False)
        frame = np.zeros(512, dtype=np.float32)
        result = eng.run_speech_enhancement(frame)
        assert len(result["enhanced_frame"]) == 512

    def test_asr_keys(self):
        eng = ADSPSC835InferenceEngine("wav2letter_asr", simulate_latency=False)
        features = np.zeros((1, 80, 200), dtype=np.float32)
        result = eng.run_asr(features)
        for key in ("device", "model", "task", "transcript", "latency_ms", "energy_uj"):
            assert key in result

    def test_asr_transcript_is_string(self):
        eng = ADSPSC835InferenceEngine("wav2letter_asr", simulate_latency=False)
        features = np.zeros((1, 80, 200), dtype=np.float32)
        result = eng.run_asr(features)
        assert isinstance(result["transcript"], str)

    def test_genre_classification_keys(self):
        eng = ADSPSC835InferenceEngine("genre_net", simulate_latency=False)
        mel = np.random.randn(128, 128).astype(np.float32)
        result = eng.run_classification(mel)
        for key in ("device", "model", "task", "top3", "top1_label", "latency_ms", "energy_uj"):
            assert key in result

    def test_genre_top3_length(self):
        eng = ADSPSC835InferenceEngine("genre_net", simulate_latency=False)
        mel = np.random.randn(128, 128).astype(np.float32)
        result = eng.run_classification(mel)
        assert len(result["top3"]) == 3

    def test_genre_label_valid(self):
        eng = ADSPSC835InferenceEngine("genre_net", simulate_latency=False)
        mel = np.zeros((128, 128), dtype=np.float32)
        result = eng.run_classification(mel)
        assert result["top1_label"] in GENRE_LABELS

    def test_energy_uj_positive(self):
        eng = ADSPSC835InferenceEngine("rnnoise_speech_enhancement", simulate_latency=False)
        frame = np.zeros(512, dtype=np.float32)
        result = eng.run_speech_enhancement(frame)
        assert result["energy_uj"] > 0

    def test_profile_keys(self):
        eng = ADSPSC835InferenceEngine("genre_net", simulate_latency=False)
        p = eng.profile()
        for key in ("device", "vendor", "dsp", "sram_mb", "model", "task", "nominal_latency_ms"):
            assert key in p

    def test_warmup(self):
        eng = ADSPSC835InferenceEngine("genre_net", simulate_latency=False)
        eng.warmup(n=1)
        assert eng._warmed_up
        assert eng._inference_count == 0

    def test_all_models_instantiate(self):
        for name in SUPPORTED_MODELS:
            eng = ADSPSC835InferenceEngine(name, simulate_latency=False)
            assert eng.model_name == name
