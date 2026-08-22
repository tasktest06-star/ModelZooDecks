"""
Unit tests for the TI EdgeAI MLOps pipeline modules.

Tests use only the model registry YAML and mocked inference —
no actual model downloads or TI hardware required.
"""

import json
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import yaml

# Ensure local modules are importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlops.data_pipeline import DataPipeline, preprocess_image, letterbox_resize, TASK_DEFAULTS
from mlops.monitor import InferenceMonitor
from mlops.artifact_manager import ArtifactManager, _default_preproc, _default_postproc

# ── Fixtures ──────────────────────────────────────────────────────────────────

CONFIG_YAML = """
modelzoo:
  version: "11.2.0"
  base_url: "https://software-dl.ti.com/jacinto7/esd/modelzoo/11.2.0/"
  local_cache: "./model_cache"
  artifact_dir: "./artifacts"
soc: AM68A
task: detection
model: yolox-s-lite
training:
  framework: edgeai-mmdetection
  epochs: 1
  batch_size: 1
  learning_rate: 0.001
  quantization: false
  checkpoint_dir: ./checkpoints
export:
  format: onnx
  opset: 11
  input_shape: [1, 3, 640, 640]
  output_path: ./exports
compilation:
  tidl_tools_path: ./edgeai-tidl-tools
  calibration_frames: 10
  accuracy_level: 1
  tensor_bits: 8
  output_dir: ./compiled
evaluation:
  dataset_path: ./datasets
  dataset: coco
  split: val
  num_frames: 10
  detection_threshold: 0.05
  detection_top_k: 500
  report_dir: ./reports
  gates:
    classification:
      min_metric: 70.0
      max_int8_drop: 2.0
    object_detection:
      min_metric: 24.0
      max_int8_drop: 1.0
    segmentation:
      min_metric: 49.0
      max_int8_drop: 1.5
deployment:
  bundle_output: ./deploy_bundles
  target: local
  verify_after_push: false
monitoring:
  enabled: true
  latency_alert_ms: 100.0
  confidence_drift_threshold: 0.3
  drift_window_frames: 10
  log_dir: ./monitor_logs
  webhook_url: null
"""

REGISTRY_YAML = """
models:
  yolox-s-lite:
    version: "11.2.0"
    task: object_detection
    format: onnx
    input_size: [640, 640]
    model_id: "od-8220"
    gmacs: 13.43
    artifact_ids:
      AM68A: "ONR-OD-8220-yolox-s-lite-mmdet-coco-640x640"
      AM69A: "ONR-OD-8220-yolox-s-lite-mmdet-coco-640x640"
      TDA4VM: "ONR-OD-8220-yolox-s-lite-mmdet-coco-640x640"
    metrics:
      mAP_fp32: 38.6
      mAP_int8: 38.2
      int8_drop: 0.4
    dataset: coco
    num_classes: 80
    status: production
  mobilenetv2:
    version: "11.2.0"
    task: classification
    format: onnx
    input_size: [224, 224]
    model_id: "cl-6360"
    gmacs: 0.300
    artifact_ids:
      AM62A: "ONR-CL-6360-mobv2-tv-imagenet-224x224"
      AM68A: "ONR-CL-6360-mobv2-tv-imagenet-224x224"
    metrics:
      top1_fp32: 72.13
      top1_int8: 71.8
      int8_drop: 0.33
    dataset: imagenet
    status: production
"""


@pytest.fixture
def tmp_config(tmp_path):
    """Write config and registry YAML files to a temp directory."""
    cfg_file = tmp_path / "pipeline_config.yaml"
    reg_file = tmp_path / "model_registry.yaml"
    cfg_file.write_text(CONFIG_YAML)
    reg_file.write_text(REGISTRY_YAML)
    # Patch the ModelManager to load registry from tmp_path
    return str(cfg_file)


@pytest.fixture
def base_config():
    return yaml.safe_load(CONFIG_YAML)


# ── DataPipeline tests ────────────────────────────────────────────────────────

class TestDataPipeline:

    def test_task_defaults_coverage(self):
        for task in ["classification", "object_detection", "segmentation",
                     "depth_estimation", "keypoint"]:
            params = TASK_DEFAULTS[task]
            assert "input_size" in params
            assert "mean" in params
            assert "std" in params

    def test_letterbox_preserves_aspect(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = letterbox_resize(img, (416, 416))
        assert result.shape == (416, 416, 3)

    def test_letterbox_square_input(self):
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        result = letterbox_resize(img, (640, 640))
        assert result.shape == (640, 640, 3)

    def test_preprocess_classification_shape(self):
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        tensor = preprocess_image(img, "classification")
        assert tensor.ndim == 4
        assert tensor.shape[0] == 1   # batch
        assert tensor.shape[1] == 3   # channels
        assert tensor.shape[2] == 224
        assert tensor.shape[3] == 224
        assert tensor.dtype == np.float32

    def test_preprocess_detection_shape(self):
        img = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        tensor = preprocess_image(img, "object_detection")
        assert tensor.shape == (1, 3, 640, 640)
        # YOLOX: values in [0, 1]
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0 + 1e-5

    def test_preprocess_segmentation_shape(self):
        img = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
        tensor = preprocess_image(img, "segmentation")
        assert tensor.shape == (1, 3, 512, 512)

    def test_preprocess_custom_input_size(self):
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        tensor = preprocess_image(img, "classification", input_size=(256, 256))
        assert tensor.shape == (1, 3, 256, 256)

    def test_datapipeline_invalid_task(self):
        with pytest.raises(ValueError, match="Unknown task"):
            DataPipeline(task="invalid_task")

    def test_datapipeline_missing_dataset_root(self, tmp_path):
        pipeline = DataPipeline(
            task="classification",
            dataset="imagenet",
            dataset_root=str(tmp_path / "nonexistent"),
            num_frames=5,
        )
        # Should not raise — just returns empty loader
        with pytest.raises(FileNotFoundError):
            list(pipeline.get_dataloader(split="val", batch_size=1))


# ── InferenceMonitor tests ────────────────────────────────────────────────────

class TestInferenceMonitor:

    def test_basic_latency_tracking(self, base_config):
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
            time.sleep(0.002)
        assert monitor._frame_count == 1

    def test_prediction_recording_dicts(self, base_config):
        monitor = InferenceMonitor(base_config)
        preds = [{"score": 0.9, "class_id": 0}, {"score": 0.7, "class_id": 3}]
        monitor.start_frame()
        monitor.end_frame()
        monitor.record_predictions(preds)
        s = monitor.stats()
        assert s["detections_per_frame"]["mean"] == 2.0

    def test_drift_detection_fires(self, base_config):
        # Set a high confidence threshold so drift is easy to trigger
        base_config["monitoring"]["confidence_drift_threshold"] = 0.99
        base_config["monitoring"]["drift_window_frames"] = 5
        monitor = InferenceMonitor(base_config)

        alerts = []
        def capture_alert(msg):
            alerts.append(msg)

        # Push low-confidence predictions
        for _ in range(10):
            monitor.start_frame()
            time.sleep(0.0001)
            monitor.end_frame()
            monitor.record_predictions(
                [{"score": 0.1, "class_id": 0}]
            )
        monitor.check_drift()
        assert len(monitor._drift_events) > 0

    def test_classification_recording(self, base_config):
        monitor = InferenceMonitor(base_config)
        monitor.start_frame()
        monitor.end_frame()
        monitor.record_classification(top1_conf=0.95)
        s = monitor.stats()
        assert s["confidence"]["mean"] == pytest.approx(0.95, abs=0.01)

    def test_save_log(self, base_config, tmp_path):
        base_config["monitoring"]["log_dir"] = str(tmp_path)
        monitor = InferenceMonitor(base_config)
        for _ in range(3):
            monitor.start_frame()
            time.sleep(0.001)
            monitor.end_frame()
            monitor.record_classification(0.8)
        log_path = monitor.save_log()
        assert log_path.exists()
        with open(log_path) as f:
            data = json.load(f)
        assert data["frames_processed"] == 3


# ── ArtifactManager tests ─────────────────────────────────────────────────────

class TestArtifactManager:

    def test_default_preproc_classification(self):
        p = _default_preproc("classification")
        assert p["input_size"] == [224, 224]
        assert not p["letterbox"]

    def test_default_preproc_detection(self):
        p = _default_preproc("object_detection")
        assert p["input_size"] == [640, 640]
        assert p["letterbox"] is True
        assert p["normalize_0_255"] is True

    def test_default_postproc_detection(self):
        p = _default_postproc("object_detection")
        assert p["num_classes"] == 80
        assert "conf_threshold" in p

    def test_pack_and_unpack(self, tmp_path, base_config):
        base_config["deployment"] = {"bundle_output": str(tmp_path / "bundles")}
        am = ArtifactManager(base_config)

        # Create fake artifact directory
        fake_artifact = tmp_path / "fake_artifact"
        fake_artifact.mkdir()
        (fake_artifact / "network.bin").write_bytes(b"fake_tidl_data")
        (fake_artifact / "param.yaml").write_text("model: test")

        bundle = am.pack(
            model_id="yolox-s-lite",
            soc="AM68A",
            artifact_dir=str(fake_artifact),
            version="11.2.0",
            task="object_detection",
            metrics={"mAP_int8": 38.2},
        )
        assert bundle.exists()
        assert bundle.suffix == ".gz"

        # Unpack and verify
        dest = tmp_path / "unpacked"
        extracted = am.unpack(str(bundle), str(dest))
        assert (extracted / "tidl_artifacts" / "network.bin").exists()

    def test_verify_valid_bundle(self, tmp_path, base_config):
        base_config["deployment"] = {"bundle_output": str(tmp_path / "bundles")}
        am = ArtifactManager(base_config)

        fake_artifact = tmp_path / "artifact"
        fake_artifact.mkdir()
        (fake_artifact / "network.bin").write_bytes(b"data")

        bundle = am.pack(
            model_id="test-model",
            soc="AM68A",
            artifact_dir=str(fake_artifact),
            version="1.0.0",
            task="classification",
        )
        assert am.verify(str(bundle)) is True

    def test_list_bundles_empty(self, tmp_path, base_config):
        base_config["deployment"] = {"bundle_output": str(tmp_path / "bundles")}
        am = ArtifactManager(base_config)
        assert am.list_bundles() == []


# ── Integration smoke test ────────────────────────────────────────────────────

class TestIntegration:

    def test_preprocess_then_monitor(self, base_config):
        """End-to-end: preprocess an image, run 'inference', record metrics."""
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        tensor = preprocess_image(img, "object_detection")
        assert tensor.shape == (1, 3, 640, 640)

        monitor = InferenceMonitor(base_config)
        fake_output = np.random.rand(1, 100, 85).astype(np.float32)

        monitor.start_frame()
        time.sleep(0.002)
        monitor.end_frame()
        monitor.record_predictions(
            [{"score": float(fake_output[0, i, 4]), "class_id": 0}
             for i in range(10)]
        )
        s = monitor.stats()
        assert s["frames_processed"] == 1
        assert s["detections_per_frame"]["mean"] == 10.0
