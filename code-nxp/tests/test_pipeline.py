"""Unit tests for NXP eIQ MLOps pipeline modules."""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mlops.data_pipeline import (
    compute_mfcc,
    compute_mel_spectrogram,
    compute_wav2letter_features,
    letterbox,
    normalize_imagenet,
    preprocess,
    preprocess_eeg,
    preprocess_face_recognition,
    preprocess_low_light,
    preprocess_super_resolution,
    preprocess_vision_classification,
    preprocess_vision_detection,
    preprocess_vision_segmentation,
)
from mlops.evaluator import (
    AccuracyGateError,
    Evaluator,
    compute_miou,
    compute_psnr,
    compute_reconstruction_error,
    compute_wer,
    top1_accuracy,
)
from mlops.monitor import InferenceMonitor


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_config(tmp_path):
    config = {
        "platform": "imx8mplus",
        "task": "image_classification",
        "model": "mobilenetv2",
        "zoo_root": str(tmp_path),
        "vela": {
            "enabled": False,
            "accelerator_config": "ethos-u65-256",
            "optimise": "Performance",
            "system_config": "Ethos_U65_High_End",
        },
        "evaluation": {
            "gates": {
                "vision_image_classification": {"metric": "top1_accuracy", "threshold": 0.68},
                "audio_keyword_spotting": {"metric": "top1_accuracy", "threshold": 0.88},
                "audio_speech_recognition": {"metric": "wer", "threshold": 0.10, "lower_is_better": True},
                "misc_eeg_classification": {"metric": "top1_accuracy", "threshold": 0.70},
            },
            "num_eval_frames": 200,
            "fail_on_drop": True,
        },
        "monitoring": {
            "window_size": 10,
            "drift_threshold": 0.5,
            "alert_webhook": "",
            "log_dir": str(tmp_path / "logs"),
        },
        "artifact": {
            "output_dir": str(tmp_path / "deployed"),
            "bundle_format": "tgz",
            "include_recipe": True,
        },
        "docker": {"image": "nxp-model-zoo"},
    }
    cfg_path = tmp_path / "pipeline_config.yaml"
    with open(cfg_path, "w") as f:
        yaml.dump(config, f)
    return str(cfg_path), config


@pytest.fixture
def sample_image():
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_audio():
    return np.random.randn(16000).astype(np.float32)


@pytest.fixture
def sample_eeg():
    return np.random.randn(22, 1125).astype(np.float32)


# ── Vision preprocessing tests ───────────────────────────────────────────────

class TestLetterbox:
    def test_output_shape(self, sample_image):
        result = letterbox(sample_image, (320, 320))
        assert result.shape == (320, 320, 3)

    def test_square_input(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = letterbox(img, (64, 64))
        assert result.shape == (64, 64, 3)

    def test_tall_image(self):
        img = np.zeros((640, 320, 3), dtype=np.uint8)
        result = letterbox(img, (320, 320))
        assert result.shape == (320, 320, 3)


class TestNormalizeImagenet:
    def test_range(self):
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        out = normalize_imagenet(img)
        assert out.min() >= -1.01 and out.max() <= 0.01

    def test_dtype(self):
        img = np.ones((10, 10, 3), dtype=np.uint8) * 127
        out = normalize_imagenet(img)
        assert out.dtype == np.float32


class TestVisionPreprocessing:
    def test_classification_shape(self, sample_image):
        out = preprocess_vision_classification(sample_image)
        assert out.shape == (1, 224, 224, 3)

    def test_detection_letterbox(self, sample_image):
        out, scale, orig = preprocess_vision_detection(sample_image, (320, 320))
        assert out.shape == (1, 320, 320, 3)
        assert isinstance(scale, float)
        assert orig == (480, 640)

    def test_segmentation_shape(self, sample_image):
        out = preprocess_vision_segmentation(sample_image, (513, 513))
        assert out.shape == (1, 513, 513, 3)

    def test_super_resolution_shape(self, sample_image):
        out = preprocess_super_resolution(sample_image, (128, 128))
        assert out.shape == (1, 128, 128, 3)
        assert out.max() <= 1.01

    def test_low_light_shape(self, sample_image):
        out = preprocess_low_light(sample_image)
        assert out.shape[0] == 1

    def test_face_recognition_shape(self):
        face = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
        out = preprocess_face_recognition(face)
        assert out.shape == (1, 160, 160, 3)


# ── Audio preprocessing tests ────────────────────────────────────────────────

class TestAudioPreprocessing:
    def test_mfcc_shape(self, sample_audio):
        out = compute_mfcc(sample_audio)
        assert out.shape == (1, 49, 10, 1)

    def test_wav2letter_shape(self, sample_audio):
        out = compute_wav2letter_features(sample_audio)
        assert out.shape == (1, 296, 39)

    def test_mel_spectrogram_returns_array(self, sample_audio):
        out = compute_mel_spectrogram(sample_audio)
        assert isinstance(out, np.ndarray)
        assert out.ndim == 2


# ── EEG preprocessing tests ──────────────────────────────────────────────────

class TestEEGPreprocessing:
    def test_output_shape(self, sample_eeg):
        out = preprocess_eeg(sample_eeg)
        assert out.shape == (1, 1, 22, 1125)

    def test_dtype(self, sample_eeg):
        out = preprocess_eeg(sample_eeg)
        assert out.dtype == np.float32

    def test_normalization(self, sample_eeg):
        out = preprocess_eeg(sample_eeg)
        # After z-score, values should be in reasonable range
        assert abs(out.mean()) < 1.0


# ── Dispatcher tests ─────────────────────────────────────────────────────────

class TestDispatcher:
    def test_classification_dispatch(self, sample_image):
        out = preprocess("image_classification", sample_image)
        assert out.shape == (1, 224, 224, 3)

    def test_unknown_task_raises(self, sample_image):
        with pytest.raises(ValueError, match="Unknown task"):
            preprocess("nonexistent_task", sample_image)


# ── Metric function tests ─────────────────────────────────────────────────────

class TestMetrics:
    def test_top1_accuracy_perfect(self):
        logits = np.array([[0, 10, 0], [0, 0, 10]])
        labels = np.array([1, 2])
        assert top1_accuracy(logits, labels) == 1.0

    def test_top1_accuracy_zero(self):
        logits = np.array([[10, 0], [10, 0]])
        labels = np.array([1, 1])
        assert top1_accuracy(logits, labels) == 0.0

    def test_compute_miou_perfect(self):
        pred = np.array([[0, 1], [1, 0]])
        gt = np.array([[0, 1], [1, 0]])
        assert compute_miou(pred, gt, 2) == 1.0

    def test_compute_psnr_identical(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        psnr = compute_psnr(img, img)
        assert psnr == float("inf")

    def test_compute_psnr_different(self):
        a = np.zeros((10, 10, 3), dtype=np.uint8)
        b = np.ones((10, 10, 3), dtype=np.uint8) * 128
        psnr = compute_psnr(a, b)
        assert psnr > 0

    def test_compute_wer_perfect(self):
        wer = compute_wer(["hello world"], ["hello world"])
        assert wer == 0.0

    def test_compute_wer_all_wrong(self):
        wer = compute_wer(["foo bar"], ["hello world"])
        assert wer == 1.0

    def test_reconstruction_error(self):
        a = np.zeros(10, dtype=np.float32)
        b = np.ones(10, dtype=np.float32)
        err = compute_reconstruction_error(a, b)
        assert err == pytest.approx(1.0)


# ── Evaluator tests ──────────────────────────────────────────────────────────

class TestEvaluator:
    def test_gate_pass(self, sample_config):
        _, config = sample_config
        evaluator = Evaluator(config)
        metrics = {"top1_accuracy": 0.75}
        passed = evaluator.check_gate(
            "mobilenetv2", "vision", "image_classification", metrics, raise_on_fail=False
        )
        assert passed is True

    def test_gate_fail_no_raise(self, sample_config):
        _, config = sample_config
        evaluator = Evaluator(config)
        metrics = {"top1_accuracy": 0.50}
        passed = evaluator.check_gate(
            "mobilenetv2", "vision", "image_classification", metrics, raise_on_fail=False
        )
        assert passed is False

    def test_gate_fail_raises(self, sample_config):
        _, config = sample_config
        evaluator = Evaluator(config)
        metrics = {"top1_accuracy": 0.50}
        with pytest.raises(AccuracyGateError):
            evaluator.check_gate(
                "mobilenetv2", "vision", "image_classification", metrics, raise_on_fail=True
            )

    def test_gate_lower_is_better_pass(self, sample_config):
        _, config = sample_config
        evaluator = Evaluator(config)
        metrics = {"wer": 0.05}
        passed = evaluator.check_gate(
            "wav2letter", "audio", "speech_recognition", metrics, raise_on_fail=False
        )
        assert passed is True

    def test_gate_lower_is_better_fail(self, sample_config):
        _, config = sample_config
        evaluator = Evaluator(config)
        metrics = {"wer": 0.25}
        passed = evaluator.check_gate(
            "wav2letter", "audio", "speech_recognition", metrics, raise_on_fail=False
        )
        assert passed is False

    def test_missing_gate_passes(self, sample_config):
        _, config = sample_config
        evaluator = Evaluator(config)
        metrics = {"some_metric": 0.99}
        passed = evaluator.check_gate(
            "unknown_model", "vision", "monocular_depth", metrics, raise_on_fail=True
        )
        assert passed is True


# ── Monitor tests ─────────────────────────────────────────────────────────────

class TestMonitor:
    def test_latency_tracking(self, sample_config):
        _, config = sample_config
        mon = InferenceMonitor(config, "test_model")
        import time
        mon.start_inference()
        time.sleep(0.01)
        ms = mon.end_inference([np.array([[0.9, 0.1]])])
        assert ms > 0
        assert mon.mean_latency_ms > 0

    def test_fps(self, sample_config):
        _, config = sample_config
        mon = InferenceMonitor(config, "test_model")
        for _ in range(5):
            mon.latencies_ms.append(50.0)
        assert mon.fps == pytest.approx(20.0, rel=0.01)

    def test_summary_keys(self, sample_config):
        _, config = sample_config
        mon = InferenceMonitor(config, "test_model")
        s = mon.summary()
        assert "model_id" in s
        assert "mean_latency_ms" in s
        assert "fps" in s
        assert "mean_confidence" in s


# ── Model registry YAML validation ───────────────────────────────────────────

class TestModelRegistry:
    def test_registry_loads(self):
        registry_path = Path(__file__).parent.parent / "config" / "model_registry.yaml"
        if not registry_path.exists():
            pytest.skip("model_registry.yaml not found")
        with open(registry_path) as f:
            data = yaml.safe_load(f)
        assert "models" in data
        assert len(data["models"]) > 0

    def test_all_models_have_required_fields(self):
        registry_path = Path(__file__).parent.parent / "config" / "model_registry.yaml"
        if not registry_path.exists():
            pytest.skip("model_registry.yaml not found")
        with open(registry_path) as f:
            data = yaml.safe_load(f)
        required = {"task", "domain", "format", "model_path", "weight_file"}
        for model_id, meta in data["models"].items():
            missing = required - set(meta.keys())
            assert not missing, f"{model_id} missing fields: {missing}"

    def test_all_models_have_supported_platforms(self):
        registry_path = Path(__file__).parent.parent / "config" / "model_registry.yaml"
        if not registry_path.exists():
            pytest.skip("model_registry.yaml not found")
        with open(registry_path) as f:
            data = yaml.safe_load(f)
        for model_id, meta in data["models"].items():
            platforms = meta.get("supported_platforms", [])
            assert len(platforms) > 0, f"{model_id} has no supported_platforms"
