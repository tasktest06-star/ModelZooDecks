"""
evaluator.py — Multi-domain evaluation with accuracy gate checking.

Supports all 3 ADI Model Zoo domains:
  Vision  : Top-1 accuracy (classification), IoU (segmentation), mAP (detection), accuracy (VWW)
  Audio   : Accuracy (KWS, genre), PESQ (denoising)
  Sensor  : AUC/pAUC (anomaly), MSE (motor fault)

Gates are defined in pipeline_config.yaml under evaluation.gates.
"""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import yaml


class AccuracyGateError(Exception):
    """Raised when a model fails its accuracy gate in CI."""


# ── Metric accumulators ───────────────────────────────────────────────────────

class TopKAccuracy:
    def __init__(self, k: int = 1):
        self.k = k
        self.correct = 0
        self.total = 0

    def update(self, logits: np.ndarray, label: int) -> None:
        top_k = np.argsort(logits.ravel())[-self.k:]
        self.correct += int(label in top_k)
        self.total += 1

    @property
    def value(self) -> float:
        return self.correct / max(self.total, 1)


class BinaryAccuracy:
    def __init__(self):
        self.correct = 0
        self.total = 0

    def update(self, pred: int, label: int) -> None:
        self.correct += int(pred == label)
        self.total += 1

    @property
    def value(self) -> float:
        return self.correct / max(self.total, 1)


class MeanSquaredError:
    def __init__(self):
        self.mse_sum = 0.0
        self.total = 0

    def update(self, pred: np.ndarray, target: np.ndarray) -> None:
        self.mse_sum += float(np.mean((pred - target) ** 2))
        self.total += 1

    @property
    def value(self) -> float:
        return self.mse_sum / max(self.total, 1)


class AnomalyAUC:
    """Accumulates reconstruction errors and labels for AUC computation."""
    def __init__(self):
        self.scores = []
        self.labels = []

    def update(self, reconstruction_error: float, is_anomaly: int) -> None:
        self.scores.append(reconstruction_error)
        self.labels.append(is_anomaly)

    @property
    def value(self) -> float:
        if len(self.labels) < 2 or all(l == self.labels[0] for l in self.labels):
            return 0.5
        labels = np.array(self.labels)
        scores = np.array(self.scores)
        # Compute AUC via trapezoidal rule (no sklearn needed)
        thresholds = np.sort(np.unique(scores))[::-1]
        tpr_list, fpr_list = [0.0], [0.0]
        pos = np.sum(labels)
        neg = len(labels) - pos
        for t in thresholds:
            pred = (scores >= t).astype(int)
            tp = np.sum((pred == 1) & (labels == 1))
            fp = np.sum((pred == 1) & (labels == 0))
            tpr_list.append(tp / max(pos, 1))
            fpr_list.append(fp / max(neg, 1))
        tpr_list.append(1.0)
        fpr_list.append(1.0)
        return float(np.trapz(tpr_list, fpr_list))


class MeanIoU:
    def __init__(self, num_classes: int = 2):
        self.num_classes = num_classes
        self.intersection = np.zeros(num_classes)
        self.union = np.zeros(num_classes)

    def update(self, pred: np.ndarray, target: np.ndarray) -> None:
        for c in range(self.num_classes):
            p = pred == c
            t = target == c
            self.intersection[c] += np.sum(p & t)
            self.union[c] += np.sum(p | t)

    @property
    def value(self) -> float:
        iou = self.intersection / np.maximum(self.union, 1)
        return float(np.mean(iou))


# ── Main Evaluator ────────────────────────────────────────────────────────────

class Evaluator:
    """
    Evaluates ADI Model Zoo models across all 3 domains.

    Usage:
        ev = Evaluator("config/pipeline_config.yaml")
        results = ev.run("feature_pyramid_net", device="MAX78002", precision="int8")
        ev.check_gates(results)
        ev.save_report(results)
    """

    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.eval_cfg = self.config.get("evaluation", {})
        self.gates = self.eval_cfg.get("gates", {})
        self.report_dir = Path(self.eval_cfg.get("report_dir", "./reports"))
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        model_name: str,
        device: str,
        precision: str = "int8",
        num_frames: Optional[int] = None,
        domain: Optional[str] = None,
        task: Optional[str] = None,
    ) -> dict:
        """
        Run evaluation. Returns metrics dict.
        In this scaffolding, inference is mocked — wire in real model inference
        by replacing _mock_inference() with your AI8X or TFLite runner.
        """
        n = num_frames or self.eval_cfg.get("num_frames", 100)
        print(f"\n[Evaluator] {model_name} | device={device} | precision={precision} | n={n}")

        start = time.time()
        results = self._mock_eval_loop(model_name, domain or "vision",
                                       task or self.config.get("task", "object_detection"),
                                       n, precision)
        elapsed = time.time() - start
        results["eval_time_s"] = round(elapsed, 2)
        results["model"] = model_name
        results["device"] = device
        results["precision"] = precision
        results["timestamp"] = datetime.utcnow().isoformat()
        return results

    def _mock_eval_loop(self, model_name, domain, task, n, precision) -> dict:
        """
        Placeholder eval loop. Replace with real inference calls to:
          AI8X:    third_party.ai8x.util_ai8x_inference.AI8XInference
          TFLite:  tflite_runtime.interpreter.Interpreter
          PyTorch: torch.load + model.eval() + model(input)
        """
        # Simulate realistic metrics based on registry
        task_metrics = {
            "image_classification": {"metric": "top1_accuracy", "value": 0.64},
            "object_detection":     {"metric": "mAP",           "value": 0.50},
            "image_segmentation":   {"metric": "mean_iou",      "value": 0.98},
            "visual_wake_word":     {"metric": "accuracy",      "value": 0.77},
            "keyword_spotting":     {"metric": "accuracy",      "value": 0.93},
            "audio_genre_identification": {"metric": "accuracy", "value": 0.84},
            "audio_denoising":      {"metric": "PESQ",          "value": 2.95},
            "anomaly_detection":    {"metric": "AUC",           "value": 0.52},
            "motor_fault_detection": {"metric": "MSE",          "value": 0.022},
        }
        m = task_metrics.get(task, {"metric": "accuracy", "value": 0.80})
        latency_ms = np.random.normal(12.0, 2.0, n).clip(5, 40)
        return {
            "task": task,
            "domain": domain,
            "metric_name": m["metric"],
            "metric_value": round(m["value"] + np.random.normal(0, 0.005), 4),
            "n_frames": n,
            "latency_mean_ms": round(float(latency_ms.mean()), 2),
            "latency_p95_ms": round(float(np.percentile(latency_ms, 95)), 2),
            "fps": round(1000.0 / float(latency_ms.mean()), 1),
        }

    def run_ai8x(self, model_name: str, weight_file: str, input_tensor: np.ndarray) -> np.ndarray:
        """
        Run inference via ADI's AI8X runtime shim.
        Requires: third_party/ai8x/ on sys.path (from adi-model-zoo repo).
        """
        import sys
        sys.path.insert(0, str(Path(__file__).parents[3] / "adi-model-zoo"))
        from third_party.ai8x.util_ai8x_inference import AI8XInference
        model = AI8XInference(weight_file)
        return model.run(input_tensor)

    def run_tflite(self, weight_file: str, input_tensor: np.ndarray) -> np.ndarray:
        """Run inference via TFLite Micro (tflite_runtime or tensorflow)."""
        try:
            import tflite_runtime.interpreter as tflite
        except ImportError:
            import tensorflow as tf
            tflite = tf.lite
        interp = tflite.Interpreter(model_path=str(weight_file))
        interp.allocate_tensors()
        inp_details = interp.get_input_details()
        out_details = interp.get_output_details()
        interp.set_tensor(inp_details[0]['index'], input_tensor)
        interp.invoke()
        return interp.get_tensor(out_details[0]['index'])

    def check_gates(self, results: dict, fail_on_drop: float = 5.0) -> None:
        """
        Verify that model metrics meet minimum thresholds.
        Raises AccuracyGateError if any gate fails.
        """
        task = results.get("task", "")
        metric_name = results.get("metric_name", "")
        metric_value = results.get("metric_value", 0.0)

        gate = self.gates.get(task, {})
        if not gate:
            print(f"[Evaluator] No gate defined for task '{task}' — skipping")
            return

        failures = []
        gate_map = {
            "image_classification": ("min_top1", metric_value),
            "object_detection":     ("min_map", metric_value),
            "image_segmentation":   ("min_accuracy", metric_value),
            "visual_wake_word":     ("min_accuracy", metric_value),
            "keyword_spotting":     ("min_accuracy", metric_value),
            "audio_genre_identification": ("min_accuracy", metric_value),
            "audio_denoising":      ("min_pesq", metric_value),
            "anomaly_detection":    ("min_auc", metric_value),
        }

        if task in gate_map:
            key, val = gate_map[task]
            threshold = gate.get(key)
            if threshold is not None and val < threshold:
                failures.append(
                    f"  {metric_name}={val:.4f} < gate {key}={threshold}"
                )

        if failures:
            msg = (f"\nAccuracy gate FAILED for '{results.get('model')}' "
                   f"on {results.get('device')}:\n" + "\n".join(failures))
            print(f"[Evaluator] GATE FAIL:{msg}")
            raise AccuracyGateError(msg)

        print(f"[Evaluator] Gate PASS: {metric_name}={metric_value:.4f} "
              f"(task={task})")

    def save_report(self, results: dict) -> Path:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        fname = f"eval_{results.get('model','model')}_{results.get('device','dev')}_{ts}.json"
        out = self.report_dir / fname
        with open(out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[Evaluator] Report saved: {out}")
        return out

    def save_csv_report(self, all_results: list) -> Path:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out = self.report_dir / f"eval_summary_{ts}.csv"
        if not all_results:
            return out
        fieldnames = list(all_results[0].keys())
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"[Evaluator] CSV report: {out}")
        return out
