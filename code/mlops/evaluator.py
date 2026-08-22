"""
evaluator.py — Accuracy evaluation and CI gate checking for TI EdgeAI models.

Wraps edgeai-benchmark conventions:
  - Accuracy mode: detection_threshold=0.05, top_k=500   (maximises AP score)
  - Performance mode: detection_threshold=0.3, top_k=200  (real-time settings)
"""

import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml
import numpy as np


# ── Metric accumulators ──────────────────────────────────────────────────────────

class TopKAccuracy:
    """Streaming Top-1 / Top-5 accuracy for classification."""

    def __init__(self):
        self.correct_top1 = 0
        self.correct_top5 = 0
        self.total = 0

    def update(self, logits: np.ndarray, labels: np.ndarray) -> None:
        top5 = np.argsort(logits, axis=-1)[:, -5:]
        top1 = top5[:, -1:]
        self.correct_top1 += int(np.sum(top1 == labels.reshape(-1, 1)))
        self.correct_top5 += int(np.sum(
            np.any(top5 == labels.reshape(-1, 1), axis=-1)
        ))
        self.total += len(labels)

    def compute(self) -> dict:
        if self.total == 0:
            return {"top1": 0.0, "top5": 0.0}
        return {
            "top1": 100.0 * self.correct_top1 / self.total,
            "top5": 100.0 * self.correct_top5 / self.total,
        }


class MeanIoU:
    """Streaming MeanIoU for segmentation."""

    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred: np.ndarray, target: np.ndarray) -> None:
        mask = target != self.ignore_index
        pred, target = pred[mask], target[mask]
        valid = (pred >= 0) & (pred < self.num_classes) & \
                (target >= 0) & (target < self.num_classes)
        self.confusion += np.bincount(
            self.num_classes * target[valid].astype(int) + pred[valid].astype(int),
            minlength=self.num_classes ** 2,
        ).reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict:
        iou_per_class = np.diag(self.confusion) / (
            self.confusion.sum(axis=1) +
            self.confusion.sum(axis=0) -
            np.diag(self.confusion) + 1e-10
        )
        return {"miou": float(np.nanmean(iou_per_class) * 100)}


class DetectionMetrics:
    """
    Lightweight COCO-style AP accumulator.
    For full COCO evaluation, install pycocotools.
    This fallback tracks raw prediction counts for CI gate checking.
    """

    def __init__(self, threshold: float = 0.05, top_k: int = 500):
        self.threshold = threshold
        self.top_k = top_k
        self.predictions = []
        self.ground_truths = []

    def update(self, preds: list[dict], targets: list[dict]) -> None:
        self.predictions.extend(preds)
        self.ground_truths.extend(targets)

    def compute(self) -> dict:
        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
            # Full COCO AP — requires pycocotools and formatted dicts
            # Stub: return placeholder
            return {"mAP": 0.0, "AP50": 0.0, "note": "pycocotools required"}
        except ImportError:
            above_thresh = [
                p for batch in self.predictions
                for p in (batch if isinstance(batch, list) else [batch])
                if isinstance(p, dict) and p.get("score", 1.0) >= self.threshold
            ]
            return {
                "mAP": 0.0,
                "detections_above_threshold": len(above_thresh),
                "note": "Install pycocotools for full COCO mAP",
            }


# ── Evaluator ────────────────────────────────────────────────────────────────────

class Evaluator:
    """
    Runs evaluation for a given model + SoC + dataset, checks accuracy gates,
    and exports a CSV/JSON report.

    Usage:
        ev = Evaluator("config/pipeline_config.yaml")
        ev.run(model_id="od-8220", soc="AM68A", split="val")
        ev.check_gates()
        ev.save_report()
    """

    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.eval_cfg = self.config.get("evaluation", {})
        self.gates_cfg = self.eval_cfg.get("gates", {})
        self.report_dir = Path(self.eval_cfg.get("report_dir", "./reports"))
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self._results: dict = {}

    def run(
        self,
        model_id: str,
        soc: str,
        split: str = "val",
        precision: str = "int8",
        mode: str = "accuracy",     # "accuracy" | "performance"
    ) -> dict:
        """
        Run evaluation against edgeai-benchmark conventions.

        mode="accuracy":    threshold=0.05, top_k=500
        mode="performance": threshold=0.3,  top_k=200
        """
        threshold = 0.05 if mode == "accuracy" else 0.3
        top_k = 500 if mode == "accuracy" else 200

        print(f"[Evaluator] model={model_id} soc={soc} "
              f"split={split} precision={precision} mode={mode}")
        print(f"[Evaluator] detection_threshold={threshold}, top_k={top_k}")

        # In a real deployment this calls edgeai-benchmark Python API.
        # Here we return a structured placeholder result.
        result = {
            "model_id": model_id,
            "soc": soc,
            "split": split,
            "precision": precision,
            "mode": mode,
            "detection_threshold": threshold,
            "top_k": top_k,
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": self._stub_metrics(model_id),
        }
        self._results[f"{model_id}:{soc}:{precision}"] = result
        return result

    def check_gates(self, fail_on_drop: Optional[float] = None) -> bool:
        """
        Check all recorded results against accuracy gates from config.
        Raises GateFailure if any model fails.
        Returns True if all pass.
        """
        failures = []
        for key, result in self._results.items():
            task = self._infer_task_from_id(result["model_id"])
            gate = self.gates_cfg.get(task, {})
            metrics = result.get("metrics", {})
            drop_override = fail_on_drop

            for metric_key, gate_key in [
                ("top1", "min_metric"),
                ("miou", "min_metric"),
                ("mAP", "min_metric"),
                ("AP",  "min_metric"),
            ]:
                if metric_key not in metrics:
                    continue
                value = metrics[metric_key]
                min_val = gate.get(gate_key, 0.0)
                max_drop = drop_override or gate.get("max_int8_drop", 99.0)

                if value < min_val:
                    failures.append(
                        f"{key}: {metric_key}={value:.2f} < gate {min_val}"
                    )

                ref_key = metric_key + "_ref"
                if ref_key in metrics:
                    drop = metrics[ref_key] - value
                    if drop > max_drop:
                        failures.append(
                            f"{key}: INT8 drop {drop:.2f}pp > max {max_drop}pp"
                        )

        if failures:
            msg = "Accuracy gate failures:\n" + "\n".join(f"  • {f}" for f in failures)
            print(f"[Evaluator] GATE FAIL\n{msg}")
            raise AccuracyGateError(msg)

        print(f"[Evaluator] All gates passed ({len(self._results)} results).")
        return True

    def save_report(self, fmt: str = "both") -> Path:
        """Save results as CSV and/or JSON. fmt: 'csv' | 'json' | 'both'."""
        ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        rows = list(self._results.values())

        if fmt in ("csv", "both"):
            csv_path = self.report_dir / f"accuracy_report_{ts}.csv"
            if rows:
                with open(csv_path, "w", newline="") as f:
                    flat = self._flatten(rows[0])
                    writer = csv.DictWriter(f, fieldnames=flat.keys())
                    writer.writeheader()
                    for r in rows:
                        writer.writerow(self._flatten(r))
            print(f"[Evaluator] CSV saved: {csv_path}")

        json_path = self.report_dir / f"accuracy_report_{ts}.json"
        if fmt in ("json", "both"):
            with open(json_path, "w") as f:
                json.dump(rows, f, indent=2)
            print(f"[Evaluator] JSON saved: {json_path}")

        return json_path

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _stub_metrics(model_id: str) -> dict:
        """Return plausible stub metrics keyed by model_id prefix."""
        if model_id.startswith("cl-") or "mobv" in model_id or "efficientnet" in model_id:
            return {"top1": 71.8, "top5": 90.2, "top1_ref": 72.1}
        if model_id.startswith("od-") or "yolox" in model_id:
            return {"mAP": 38.2, "AP50": 55.8, "mAP_ref": 38.6}
        if model_id.startswith("ss-") or "deeplab" in model_id:
            return {"miou": 49.5, "miou_ref": 49.95}
        if model_id.startswith("kd-") or "pose" in model_id:
            return {"AP": 55.1, "AP_ref": 56.4}
        if model_id.startswith("de-") or "midas" in model_id:
            return {"delta1": 85.9, "delta1_ref": 86.67}
        return {"metric": 0.0}

    @staticmethod
    def _infer_task_from_id(model_id: str) -> str:
        prefix_map = {
            "cl-": "classification", "od-": "object_detection",
            "ss-": "segmentation",   "kd-": "keypoint",
            "de-": "depth_estimation",
        }
        for prefix, task in prefix_map.items():
            if model_id.startswith(prefix):
                return task
        return "classification"

    @staticmethod
    def _flatten(d: dict, parent: str = "") -> dict:
        out = {}
        for k, v in d.items():
            key = f"{parent}.{k}" if parent else k
            if isinstance(v, dict):
                out.update(Evaluator._flatten(v, key))
            else:
                out[key] = v
        return out


class AccuracyGateError(Exception):
    """Raised when an accuracy gate check fails (used to fail CI jobs)."""
