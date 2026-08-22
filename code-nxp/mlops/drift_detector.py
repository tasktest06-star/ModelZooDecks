"""Drift detection for NXP eIQ Model Zoo inference streams."""

import json
from pathlib import Path
import numpy as np


class DriftDetector:
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
                       labels: np.ndarray) -> dict:
        stats = {
            "mean_confidence": float(np.mean(confidences)),
            "accuracy": float(np.mean(predictions == labels)),
            "n_samples": len(predictions),
        }
        self.reference_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.reference_path, "w") as f:
            json.dump(stats, f, indent=2)
        self._reference_stats = stats
        return stats

    def check_accuracy_drift(self, current_accuracy: float,
                              threshold_delta: float = 0.03) -> dict:
        ref_acc = self._reference_stats.get("accuracy")
        if ref_acc is None:
            return {"drift_detected": False, "reason": "no_reference"}
        delta = ref_acc - current_accuracy
        return {
            "drift_detected": delta > threshold_delta,
            "reference_accuracy": ref_acc,
            "current_accuracy": current_accuracy,
            "delta": delta,
            "threshold": threshold_delta,
        }

    def check_psnr_drift(self, current_psnr: float, threshold_db: float = 1.0) -> dict:
        """For super-resolution models, detect PSNR degradation."""
        ref_psnr = self._reference_stats.get("psnr_db")
        if ref_psnr is None:
            return {"drift_detected": False, "reason": "no_reference"}
        delta = ref_psnr - current_psnr
        return {
            "drift_detected": delta > threshold_db,
            "reference_psnr": ref_psnr,
            "current_psnr": current_psnr,
            "delta_db": delta,
        }

    def check_confidence_drift(self, current_confidences: np.ndarray,
                                threshold_delta: float = 0.05) -> dict:
        ref_mean = self._reference_stats.get("mean_confidence")
        if ref_mean is None:
            return {"drift_detected": False, "reason": "no_reference"}
        current_mean = float(np.mean(current_confidences))
        return {
            "drift_detected": abs(ref_mean - current_mean) > threshold_delta,
            "reference_mean": ref_mean,
            "current_mean": current_mean,
        }
