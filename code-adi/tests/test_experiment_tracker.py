"""Tests for ADI experiment tracker, drift detector, QAT HPO."""

import json
import os
import numpy as np
import pytest
from pathlib import Path


class TestADIExperimentTracker:
    def test_init(self, tmp_path):
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
        from mlops.experiment_tracker import ExperimentTracker
        tracker = ExperimentTracker(tracking_uri=f"file:{tmp_path}/mlruns")
        assert tracker.experiment_name == "adi-ai8x-modelzoo"

    def test_start_run(self, tmp_path):
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
        from mlops.experiment_tracker import ExperimentTracker
        import mlflow
        tracker = ExperimentTracker(tracking_uri=f"file:{tmp_path}/mlruns")
        with tracker.start_run("feature_pyramid_net", "MAX78002") as run:
            assert run is not None
        assert mlflow.active_run() is None

    def test_log_qat_params(self, tmp_path):
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
        from mlops.experiment_tracker import ExperimentTracker
        tracker = ExperimentTracker(tracking_uri=f"file:{tmp_path}/mlruns")
        with tracker.start_run("ds_cnn", "MAX32690"):
            tracker.log_qat_params(8, 8, 8, 1e-3, 64)

    def test_get_best_run_empty(self, tmp_path):
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
        from mlops.experiment_tracker import ExperimentTracker
        tracker = ExperimentTracker(
            tracking_uri=f"file:{tmp_path}/mlruns",
            experiment_name="adi-test-empty",
        )
        assert tracker.get_best_run() == {}


class TestADIDriftDetector:
    def test_save_and_check(self, tmp_path):
        from mlops.drift_detector import DriftDetector
        ref = tmp_path / "ref.json"
        d = DriftDetector(str(ref), str(tmp_path))
        d.save_reference(np.ones(50, int), np.full(50, 0.9), np.ones(50, int))
        result = d.check_accuracy_drift(0.99)
        assert not result["drift_detected"]

    def test_drift_detected(self, tmp_path):
        from mlops.drift_detector import DriftDetector
        ref = tmp_path / "ref.json"
        d = DriftDetector(str(ref), str(tmp_path))
        d.save_reference(np.ones(50, int), np.full(50, 0.9), np.ones(50, int))
        result = d.check_accuracy_drift(0.90, threshold_delta=0.03)
        assert result["drift_detected"]

    def test_no_reference(self, tmp_path):
        from mlops.drift_detector import DriftDetector
        d = DriftDetector(str(tmp_path / "none.json"), str(tmp_path))
        assert not d.check_accuracy_drift(0.80)["drift_detected"]


class TestQATHPORunner:
    def test_no_optuna_fallback(self, monkeypatch):
        from mlops import hpo
        monkeypatch.setattr(hpo, "OPTUNA_AVAILABLE", False)
        runner = hpo.QATHPORunner("test")
        result = runner.run(lambda p: 0.9)
        assert "error" in result

    def test_with_optuna(self):
        pytest.importorskip("optuna")
        import importlib
        from mlops import hpo
        importlib.reload(hpo)
        runner = hpo.QATHPORunner("adi-test-hpo", n_trials=3)
        result = runner.run(lambda p: p["learning_rate"])
        assert result["n_trials"] == 3
        assert "best_params" in result

    def test_save_params(self, tmp_path):
        from mlops.hpo import QATHPORunner
        runner = QATHPORunner("test")
        out = tmp_path / "out.json"
        runner.save_best_params({"best_value": 0.93, "best_params": {}}, str(out))
        assert json.loads(out.read_text())["best_value"] == 0.93
