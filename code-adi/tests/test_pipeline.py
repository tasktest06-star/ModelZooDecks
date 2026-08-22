"""
Unit tests for the ADI Model Zoo MLOps pipeline modules.

No real model downloads or hardware required — all mocked/stubbed.
Run: cd code-adi && pytest tests/ -v
"""

import json
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlops.data_pipeline import (
    DataPipeline, preprocess_vision, preprocess_audio, preprocess_sensor,
    ai8x_normalize, channel_fold, compute_stft_magnitude, compute_mel_spectrogram,
    VISION_TASK_DEFAULTS, AUDIO_TASK_DEFAULTS, SENSOR_TASK_DEFAULTS,
)
from mlops.monitor import InferenceMonitor
from mlops.artifact_manager import ArtifactManager, _default_preproc, _default_postproc

# ── Inline test configs ───────────────────────────────────────────────────────

CONFIG_YAML = """
adi_modelzoo:
  version: "1.0.0"
  repo_root: "./adi-model-zoo"
  local_cache: "./model_cache"
  artifact_dir: "./artifacts"
device: MAX78002
board: MAX78002EVKIT
task: object_detection
model: feature_pyramid_net
training:
  framework: ai8x-training
  epochs: 1
  batch_size: 2
  learning_rate: 0.001
  qat: true
  qat_start_epoch: 0
  checkpoint_dir: ./checkpoints
export:
  format: ai8x
  output_path: ./exports
  fold_ratio: 1
  act_mode_8bit: true
synthesis:
  ai8xize_path: ./ai8x-synthesis
  device: MAX78002
  output_dir: ./synthesized
evaluation:
  dataset_path: ./datasets
  dataset: pascal_voc
  split: val
  num_frames: 10
  detection_threshold: 0.3
  report_dir: ./reports
  gates:
    image_classification:
      min_top1: 0.55
      max_drop_from_fp32: 0.05
    object_detection:
      min_map: 0.40
      max_drop_from_fp32: 0.05
    image_segmentation:
      min_accuracy: 0.95
    keyword_spotting:
      min_accuracy: 0.85
    audio_denoising:
      min_pesq: 2.5
    anomaly_detection:
      min_auc: 0.50
deployment:
  bundle_output: ./deploy_bundles
  target: local
monitoring:
  enabled: true
  latency_alert_ms: 50.0
  confidence_drift_threshold: 0.25
  drift_window_frames: 10
  log_dir: ./monitor_logs
  webhook_url: null
"""

REGISTRY_YAML = """
models:
  feature_pyramid_net:
    version: "1.0.0"
    task: object_detection
    domain: vision
    format: ai8x
    input_size: [256, 320]
    input_channels: 3
    num_classes: 21
    dataset: pascal_voc
    weight_file: "ai87-pascalvoc-fpndetector-qat8-q.pth.tar"
    net_def: "ai87-fpndetector.py"
    model_path: "vision_models/object_detection/feature_pyramid_net/data/model"
    size_mb: 25.16
    precision: int8
    supported_devices:
      - device: MAX78002
        board: MAX78002EVKIT
        family: CNN
    metrics:
      mAP_reported: 0.50512
    status: production
  ds_cnn:
    version: "1.0.0"
    task: keyword_spotting
    domain: audio
    format: tflite
    input_size: [1, 490]
    num_classes: 12
    dataset: google_speech_commands_v2
    weight_file: "ds_cnn_s_float32.tflite"
    weight_file_int8: "ds_cnn_s_int8.tflite"
    model_path: "audio_models/audio_keyword_spotting/ds_cnn/data/model"
    size_mb: 0.097
    precision: float32
    supported_devices:
      - device: MAX32690
        board: EvKit_V1
        family: Cortex-M
        core: CM4
    metrics:
      accuracy_reported: 0.9452
    status: production
  autoencoder_motor:
    version: "1.0.0"
    task: motor_fault_detection
    domain: sensor
    format: ai8x
    input_size: [256, 3]
    dataset: adi_motor_limerick
    weight_file: "ai85-autoencoder-samplemotordatalimerick-qat-q.pth.tar"
    net_def: "ai85net-autoencoder.py"
    model_path: "sensor_models/motor_fault_detection/autoencoder/data/model"
    size_mb: 1.59
    precision: int8
    supported_devices:
      - device: MAX78002
        board: MAX78002EVKIT
        family: CNN
    metrics:
      MSE_reported: 0.02205
    status: production
"""


@pytest.fixture
def tmp_config(tmp_path):
    cfg = tmp_path / "pipeline_config.yaml"
    reg = tmp_path / "model_registry.yaml"
    cfg.write_text(CONFIG_YAML)
    reg.write_text(REGISTRY_YAML)
    return str(cfg)


@pytest.fixture
def base_config():
    return yaml.safe_load(CONFIG_YAML)


# ── AI8X normalization tests ──────────────────────────────────────────────────

class TestAI8XNormalization:

    def test_normalize_range_8bit(self):
        img = np.array([[[0, 128, 255]]], dtype=np.uint8)
        out = ai8x_normalize(img, act_mode_8bit=True)
        assert out.min() >= -128
        assert out.max() <= 127

    def test_normalize_zero_maps_to_neg128(self):
        img = np.zeros((1, 1, 1), dtype=np.uint8)
        out = ai8x_normalize(img, act_mode_8bit=True)
        assert out[0, 0, 0] == pytest.approx(-128.0, abs=1.0)

    def test_normalize_255_maps_to_127(self):
        img = np.full((1, 1, 1), 255, dtype=np.uint8)
        out = ai8x_normalize(img, act_mode_8bit=True)
        assert out[0, 0, 0] == pytest.approx(127.0, abs=1.0)


# ── Channel fold tests ────────────────────────────────────────────────────────

class TestChannelFold:

    def test_fold_ratio_1_passthrough(self):
        img = np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8)
        out = channel_fold(img, 1)
        assert out.shape == img.shape

    def test_fold_ratio_4_shape(self):
        img = np.random.randint(0, 255, (192, 192, 3), dtype=np.uint8)
        out = channel_fold(img, 4)
        assert out.shape == (48, 48, 48)   # 192//4, 192//4, 3*4*4

    def test_fold_ratio_2_shape(self):
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        out = channel_fold(img, 2)
        assert out.shape == (32, 32, 12)


# ── Vision preprocessing tests ───────────────────────────────────────────────

class TestVisionPreprocessing:

    def test_classification_output_shape(self):
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        out = preprocess_vision(img, "image_classification")
        assert out.shape == (1, 3, 32, 32)
        assert out.dtype == np.float32

    def test_detection_output_shape_fpn(self):
        img = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        out = preprocess_vision(img, "object_detection", input_size=(256, 320))
        assert out.shape == (1, 3, 256, 320)

    def test_segmentation_with_fold(self):
        img = np.random.randint(0, 255, (192, 192, 3), dtype=np.uint8)
        out = preprocess_vision(img, "image_segmentation", input_size=(48, 48), fold_ratio=4)
        # After fold: channels = 3 * 4 * 4 = 48
        assert out.shape == (1, 48, 48, 48)

    def test_vww_grayscale(self):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        out = preprocess_vision(img, "visual_wake_word", input_size=(50, 50), grayscale=True)
        assert out.shape == (1, 1, 50, 50)

    def test_unknown_task_raises(self):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="Unknown vision task"):
            preprocess_vision(img, "unknown_task")

    def test_output_in_ai8x_range(self):
        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        out = preprocess_vision(img, "image_classification")
        assert out.min() >= -128
        assert out.max() <= 127


# ── Audio preprocessing tests ────────────────────────────────────────────────

class TestAudioPreprocessing:

    def test_stft_magnitude_shape(self):
        audio = np.random.randn(16000).astype(np.float32)
        stft = compute_stft_magnitude(audio, n_fft=512, hop_size=256)
        assert stft.ndim == 2
        assert stft.shape[1] == 257   # n_fft//2 + 1

    def test_mel_spectrogram_shape(self):
        audio = np.random.randn(16000).astype(np.float32)
        mel = compute_mel_spectrogram(audio, sample_rate=16000, n_mels=40,
                                      n_fft=640, hop_length=320, duration_s=1.0)
        assert mel.ndim == 3   # (1, n_mels, n_frames)
        assert mel.shape[1] == 40

    def test_audio_denoising_output(self):
        audio = np.random.randn(16000).astype(np.float32)
        out = preprocess_audio(audio, "audio_denoising")
        assert out.ndim == 2

    def test_kws_output(self):
        audio = np.random.randn(16000).astype(np.float32)
        out = preprocess_audio(audio, "keyword_spotting")
        assert out.ndim == 3
        assert out.shape[0] == 1

    def test_unknown_audio_task_raises(self):
        audio = np.zeros(16000, dtype=np.float32)
        with pytest.raises(ValueError):
            preprocess_audio(audio, "unknown_task")


# ── Sensor preprocessing tests ───────────────────────────────────────────────

class TestSensorPreprocessing:

    def test_motor_fault_shape(self):
        signal = np.random.randn(256, 3).astype(np.float32)
        out = preprocess_sensor(signal, "motor_fault_detection")
        assert out.shape == (1, 256, 3)

    def test_motor_fault_range(self):
        signal = np.random.randn(256, 3).astype(np.float32) * 1000
        out = preprocess_sensor(signal, "motor_fault_detection")
        assert out.min() >= -128
        assert out.max() <= 128

    def test_anomaly_detection_shape(self):
        audio = np.random.randn(16000).astype(np.float32)
        out = preprocess_sensor(audio, "anomaly_detection", sample_rate=16000)
        assert out.ndim == 3


# ── InferenceMonitor tests ────────────────────────────────────────────────────

class TestInferenceMonitor:

    def test_latency_tracking(self, base_config):
        monitor = InferenceMonitor(base_config)
        for _ in range(20):
            monitor.start_frame()
            time.sleep(0.001)
            monitor.end_frame()
        s = monitor.stats()
        assert s["frames_processed"] == 20
        assert s["latency_ms"]["mean"] > 0
        assert s["fps"] > 0

    def test_track_frame_context_manager(self, base_config):
        monitor = InferenceMonitor(base_config)
        with monitor.track_frame():
            time.sleep(0.001)
        assert monitor._frame_count == 1

    def test_detection_recording(self, base_config):
        monitor = InferenceMonitor(base_config)
        monitor.start_frame()
        monitor.end_frame()
        dets = [{"score": 0.9, "class_id": 3}, {"score": 0.6, "class_id": 7}]
        monitor.record_detections(dets)
        s = monitor.stats()
        assert s["detections_per_frame"]["mean"] == 2.0

    def test_drift_detection(self, base_config):
        base_config["monitoring"]["confidence_drift_threshold"] = 0.99
        base_config["monitoring"]["drift_window_frames"] = 5
        monitor = InferenceMonitor(base_config)
        for _ in range(10):
            monitor.start_frame()
            time.sleep(0.0001)
            monitor.end_frame()
            monitor.record_classification(0.05)
        monitor.check_drift()
        assert len(monitor._drift_events) > 0

    def test_anomaly_score_recording(self, base_config):
        monitor = InferenceMonitor(base_config)
        monitor.start_frame()
        monitor.end_frame()
        monitor.record_anomaly_score(0.01, threshold=0.05)   # low error = normal
        s = monitor.stats()
        assert s["confidence"]["mean"] == pytest.approx(1.0, abs=0.01)

    def test_save_log(self, base_config, tmp_path):
        base_config["monitoring"]["log_dir"] = str(tmp_path)
        monitor = InferenceMonitor(base_config)
        for _ in range(5):
            monitor.start_frame()
            time.sleep(0.001)
            monitor.end_frame()
            monitor.record_classification(0.85)
        log = monitor.save_log()
        assert log.exists()
        data = json.loads(log.read_text())
        assert data["frames_processed"] == 5


# ── ArtifactManager tests ─────────────────────────────────────────────────────

class TestArtifactManager:

    def test_default_preproc_classification(self):
        p = _default_preproc("vision", "image_classification")
        assert p["input_size"] == [32, 32]
        assert p["normalize"] == "ai8x"

    def test_default_preproc_segmentation_fold(self):
        p = _default_preproc("vision", "image_segmentation")
        assert p["fold_ratio"] == 4

    def test_default_postproc_detection(self):
        p = _default_postproc("vision", "object_detection")
        assert p["type"] == "nms"
        assert p["num_classes"] == 21

    def test_default_preproc_anomaly(self):
        p = _default_preproc("sensor", "anomaly_detection")
        assert "n_fft" in p

    def test_pack_unpack(self, tmp_path, base_config):
        base_config["deployment"] = {"bundle_output": str(tmp_path / "bundles")}
        am = ArtifactManager(base_config)

        fake_dir = tmp_path / "fake_model"
        fake_dir.mkdir()
        (fake_dir / "model.pth.tar").write_bytes(b"FAKE_WEIGHTS")

        bundle = am.pack(
            model_name="feature_pyramid_net",
            device="MAX78002",
            model_files=[str(fake_dir / "model.pth.tar")],
            version="1.0.0",
            domain="vision",
            task="object_detection",
            metrics={"mAP_reported": 0.50512},
        )
        assert bundle.exists()
        assert bundle.suffix == ".gz"

        dest = tmp_path / "unpacked"
        extracted = am.unpack(str(bundle), str(dest))
        assert (extracted / "feature_pyramid_net_1.0.0" / "weights" / "model.pth.tar").exists()

    def test_verify_valid_bundle(self, tmp_path, base_config):
        base_config["deployment"] = {"bundle_output": str(tmp_path / "bundles")}
        am = ArtifactManager(base_config)
        fake_dir = tmp_path / "fake"
        fake_dir.mkdir()
        (fake_dir / "w.tflite").write_bytes(b"TFL")
        bundle = am.pack("ds-cnn", "MAX32690", [str(fake_dir / "w.tflite")],
                         "1.0", "audio", "keyword_spotting")
        assert am.verify(str(bundle)) is True

    def test_list_bundles_empty(self, tmp_path, base_config):
        base_config["deployment"] = {"bundle_output": str(tmp_path / "empty")}
        am = ArtifactManager(base_config)
        assert am.list_bundles() == []


# ── Integration test ──────────────────────────────────────────────────────────

class TestIntegration:

    def test_vision_preprocess_then_monitor(self, base_config):
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        tensor = preprocess_vision(img, "object_detection", input_size=(256, 320))
        assert tensor.shape == (1, 3, 256, 320)

        monitor = InferenceMonitor(base_config)
        monitor.start_frame()
        time.sleep(0.003)
        monitor.end_frame()
        fake_detections = [{"score": float(np.random.rand()), "class_id": i}
                           for i in range(5)]
        monitor.record_detections(fake_detections)
        s = monitor.stats()
        assert s["frames_processed"] == 1
        assert s["detections_per_frame"]["mean"] == 5.0

    def test_audio_preprocess_smoke(self, base_config):
        audio = np.random.randn(22050 * 3).astype(np.float32)
        mel = preprocess_audio(audio, "audio_genre_identification", sample_rate=22050)
        assert mel.shape[0] == 1 and mel.shape[1] == 128  # frames vary with librosa version

    def test_sensor_preprocess_smoke(self, base_config):
        signal = np.random.randn(512, 3).astype(np.float32)
        out = preprocess_sensor(signal, "motor_fault_detection")
        assert out.shape == (1, 256, 3)
