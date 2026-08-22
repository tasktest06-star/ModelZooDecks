"""Tests for ExperimentTracker, DriftDetector, TIDLHPORunner."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestExperimentTracker:
    def test_init(self, tmp_path):
        from mlops.experiment_tracker import ExperimentTracker
        db = f"sqlite:///{tmp_path}/mlflow.db"
        tracker = ExperimentTracker(tracking_uri=db)
        assert tracker.experiment_name == "ti-edgeai-modelzoo"

    def test_start_run_context(self, tmp_path):
        from mlops.experiment_tracker import ExperimentTracker
        import mlflow
        db = f"sqlite:///{tmp_path}/mlflow.db"
        tracker = ExperimentTracker(tracking_uri=db)
        with tracker.start_run("mobilenet_v2", "AM68A", "image_classification") as run:
            assert run is not None
            assert mlflow.active_run() is not None
        assert mlflow.active_run() is None

    def test_log_eval_metrics(self, tmp_path):
        from mlops.experiment_tracker import ExperimentTracker
        import mlflow
        db = f"sqlite:///{tmp_path}/mlflow.db"
        tracker = ExperimentTracker(tracking_uri=db)
        with tracker.start_run("yolox_s", "AM68A", "object_detection"):
            tracker.log_eval_metrics({"map_50": 0.45, "latency_ms": 23.5})

    def test_get_best_run_empty(self, tmp_path):
        from mlops.experiment_tracker import ExperimentTracker
        db = f"sqlite:///{tmp_path}/mlflow.db"
        tracker = ExperimentTracker(
            tracking_uri=db,
            experiment_name="ti-edgeai-test-empty",
        )
        result = tracker.get_best_run()
        assert result == {}


class TestDriftDetector:
    def test_save_and_load_reference(self, tmp_path):
        from mlops.drift_detector import DriftDetector
        ref_path = tmp_path / "ref.json"
        detector = DriftDetector(reference_path=str(ref_path), output_dir=str(tmp_path))
        preds = np.array([0, 1, 1, 0, 1])
        confs = np.array([0.9, 0.8, 0.85, 0.7, 0.92])
        labels = np.array([0, 1, 1, 0, 0])
        stats = detector.save_reference(preds, confs, labels)
        assert "accuracy" in stats
        assert ref_path.exists()

    def test_no_drift_within_threshold(self, tmp_path):
        from mlops.drift_detector import DriftDetector
        ref_path = tmp_path / "ref.json"
        detector = DriftDetector(reference_path=str(ref_path), output_dir=str(tmp_path))
        detector.save_reference(
            np.ones(100, dtype=int), np.full(100, 0.9), np.ones(100, dtype=int)
        )
        result = detector.check_accuracy_drift(current_accuracy=0.99)
        assert not result["drift_detected"]

    def test_drift_detected(self, tmp_path):
        from mlops.drift_detector import DriftDetector
        ref_path = tmp_path / "ref.json"
        detector = DriftDetector(reference_path=str(ref_path), output_dir=str(tmp_path))
        detector.save_reference(
            np.ones(100, dtype=int), np.full(100, 0.9), np.ones(100, dtype=int)
        )
        result = detector.check_accuracy_drift(current_accuracy=0.90, threshold_delta=0.03)
        assert result["drift_detected"]

    def test_no_reference(self, tmp_path):
        from mlops.drift_detector import DriftDetector
        detector = DriftDetector(
            reference_path=str(tmp_path / "nonexistent.json"),
            output_dir=str(tmp_path),
        )
        result = detector.check_accuracy_drift(0.85)
        assert not result["drift_detected"]
        assert result["reason"] == "no_reference"


class TestHPORunner:
    def test_run_without_optuna(self, tmp_path):
        from mlops import hpo
        original = hpo.OPTUNA_AVAILABLE
        hpo.OPTUNA_AVAILABLE = False
        try:
            runner = hpo.TIDLHPORunner("test-study")
            result = runner.run(lambda params: 0.9)
            assert "error" in result
        finally:
            hpo.OPTUNA_AVAILABLE = original

    def test_run_with_optuna(self, tmp_path):
        import uuid
        pytest.importorskip("optuna")
        from mlops.hpo import TIDLHPORunner
        unique_name = f"test-ti-hpo-{uuid.uuid4().hex[:8]}"
        runner = TIDLHPORunner(unique_name, n_trials=3)
        result = runner.run(lambda params: params.get("num_calib_frames", 0) / 500.0)
        assert "best_params" in result
        assert "best_value" in result
        assert result["n_trials"] == 3

    def test_save_best_params(self, tmp_path):
        from mlops.hpo import TIDLHPORunner
        runner = TIDLHPORunner("test-save")
        result = {"best_params": {"tensor_bits": 8}, "best_value": 0.72, "n_trials": 5}
        out = tmp_path / "hpo_result.json"
        runner.save_best_params(result, str(out))
        loaded = json.loads(out.read_text())
        assert loaded["best_value"] == 0.72
