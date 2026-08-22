"""Evidently AI drift detection for TI EdgeAI Model Zoo."""

import json
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, ClassificationPreset
    from evidently.metrics import ColumnDriftMetric
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False


class DriftDetector:
    """Detects accuracy and data drift for edge model inference streams."""

    def __init__(self, reference_path: str, output_dir: str = "drift_reports"):
        self.reference_path = Path(reference_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._reference_stats = self._load_reference()

    def _load_reference(self) -> dict:
        if self.reference_path.exists():
            with open(self.reference_path) as f:
                return json.load(f)
        return {}

    def save_reference(self, predictions: np.ndarray, confidences: np.ndarray,
                       labels: np.ndarray):
        """Save reference distribution from baseline evaluation."""
        stats = {
            "mean_confidence": float(np.mean(confidences)),
            "std_confidence": float(np.std(confidences)),
            "accuracy": float(np.mean(predictions == labels)),
            "confidence_p5": float(np.percentile(confidences, 5)),
            "confidence_p95": float(np.percentile(confidences, 95)),
            "n_samples": len(predictions),
        }
        self.reference_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.reference_path, "w") as f:
            json.dump(stats, f, indent=2)
        self._reference_stats = stats
        return stats

    def check_accuracy_drift(self, current_accuracy: float,
                              threshold_delta: float = 0.03) -> dict:
        """Detect if accuracy has drifted more than threshold from reference."""
        ref_acc = self._reference_stats.get("accuracy", None)
        if ref_acc is None:
            return {"drift_detected": False, "reason": "no_reference"}
        delta = ref_acc - current_accuracy
        drift_detected = delta > threshold_delta
        return {
            "drift_detected": drift_detected,
            "reference_accuracy": ref_acc,
            "current_accuracy": current_accuracy,
            "delta": delta,
            "threshold": threshold_delta,
        }

    def check_confidence_drift(self, current_confidences: np.ndarray,
                               threshold_delta: float = 0.05) -> dict:
        """Detect if mean confidence has drifted significantly."""
        ref_mean = self._reference_stats.get("mean_confidence", None)
        if ref_mean is None:
            return {"drift_detected": False, "reason": "no_reference"}
        current_mean = float(np.mean(current_confidences))
        delta = abs(ref_mean - current_mean)
        return {
            "drift_detected": delta > threshold_delta,
            "reference_mean_confidence": ref_mean,
            "current_mean_confidence": current_mean,
            "delta": delta,
            "threshold": threshold_delta,
        }

    def generate_report(self, reference_df, current_df, report_name: str = "drift") -> Path:
        """Generate Evidently HTML drift report if available."""
        if not EVIDENTLY_AVAILABLE:
            report_path = self.output_dir / f"{report_name}_unavailable.json"
            with open(report_path, "w") as f:
                json.dump({"error": "evidently not installed"}, f)
            return report_path
        report = Report(metrics=[DataDriftPreset(), ClassificationPreset()])
        report.run(reference_data=reference_df, current_data=current_df)
        report_path = self.output_dir / f"{report_name}.html"
        report.save_html(str(report_path))
        return report_path
