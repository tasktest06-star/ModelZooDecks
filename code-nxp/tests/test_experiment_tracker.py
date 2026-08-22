"""Tests for NXP experiment tracker, drift detector, TFLite INT8 HPO."""

import json
import numpy as np
import pytest
from pathlib import Path


class TestNXPExperimentTracker:
    def test_init(self, tmp_path):
        from mlops.experiment_tracker import ExperimentTracker
        tracker = ExperimentTracker(tracking_uri=f"sqlite:///{tmp_path}/mlflow.db")
        assert tracker.experiment_name == "nxp-eiq-modelzoo"

    def test_start_run(self, tmp_path):
        from mlops.experiment_tracker import ExperimentTracker
        import mlflow
        tracker = ExperimentTracker(tracking_uri=f"sqlite:///{tmp_path}/mlflow.db")
        with tracker.start_run("mobilenetv2", "imx8mplus") as run:
            assert run is not None
        assert mlflow.active_run() is None

    def test_log_vela_stats(self, tmp_path):
        from mlops.experiment_tracker import ExperimentTracker
        tracker = ExperimentTracker(tracking_uri=f"sqlite:///{tmp_path}/mlflow.db")
        with tracker.start_run("mobilenetv2", "imx93"):
            tracker.log_vela_stats("ethos-u65-256", 0.82, 512000, 1024000)

    def test_get_best_run_empty(self, tmp_path):
        from mlops.experiment_tracker import ExperimentTracker
        tracker = ExperimentTracker(
            tracking_uri=f"sqlite:///{tmp_path}/mlflow.db",
            experiment_name="nxp-test-empty",
        )
        assert tracker.get_best_run() == {}


class TestNXPDriftDetector:
    def test_save_and_check_no_drift(self, tmp_path):
        from mlops.drift_detector import DriftDetector
        d = DriftDetector(str(tmp_path / "ref.json"), str(tmp_path))
        d.save_reference(np.ones(100, int), np.full(100, 0.9), np.ones(100, int))
        assert not d.check_accuracy_drift(0.99)["drift_detected"]

    def test_drift_detected(self, tmp_path):
        from mlops.drift_detector import DriftDetector
        d = DriftDetector(str(tmp_path / "ref.json"), str(tmp_path))
        d.save_reference(np.ones(100, int), np.full(100, 0.9), np.ones(100, int))
        assert d.check_accuracy_drift(0.90, threshold_delta=0.03)["drift_detected"]

    def test_psnr_drift(self, tmp_path):
        from mlops.drift_detector import DriftDetector
        d = DriftDetector(str(tmp_path / "ref.json"), str(tmp_path))
        d._reference_stats = {"psnr_db": 32.0}
        result = d.check_psnr_drift(current_psnr=29.0, threshold_db=1.0)
        assert result["drift_detected"]

    def test_no_reference(self, tmp_path):
        from mlops.drift_detector import DriftDetector
        d = DriftDetector(str(tmp_path / "none.json"), str(tmp_path))
        assert not d.check_accuracy_drift(0.7)["drift_detected"]


class TestTFLiteINT8HPORunner:
    def test_no_optuna_fallback(self):
        from mlops import hpo
        orig = hpo.OPTUNA_AVAILABLE
        hpo.OPTUNA_AVAILABLE = False
        try:
            runner = hpo.TFLiteINT8HPORunner("test")
            result = runner.run(lambda p: 0.72)
            assert "error" in result
        finally:
            hpo.OPTUNA_AVAILABLE = orig

    def test_with_optuna(self):
        pytest.importorskip("optuna")
        from mlops.hpo import TFLiteINT8HPORunner
        import mlops.hpo as hpo_mod
        hpo_mod.OPTUNA_AVAILABLE = True
        runner = TFLiteINT8HPORunner("nxp-test-hpo", n_trials=3)
        result = runner.run(lambda p: p["num_calib_steps"] / 500.0)
        assert result["n_trials"] == 3
        assert "best_params" in result

    def test_save_params(self, tmp_path):
        from mlops.hpo import TFLiteINT8HPORunner
        runner = TFLiteINT8HPORunner("test")
        out = tmp_path / "result.json"
        runner.save_best_params({"best_value": 0.718, "best_params": {}}, str(out))
        assert json.loads(out.read_text())["best_value"] == 0.718
