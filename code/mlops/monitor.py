"""
monitor.py — Runtime inference monitoring, drift detection, and alerting.

Tracks latency, FPS, confidence score distribution, and detection counts.
Issues drift alerts when model predictions degrade over a rolling window.
"""

import json
import time
import statistics
import urllib.request
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml
import numpy as np


class InferenceMonitor:
    """
    Collects per-frame inference telemetry and triggers drift alerts.

    Usage:
        monitor = InferenceMonitor(config)
        with monitor.track_frame():
            preds = model.infer(img)
        monitor.record_predictions(preds, conf_scores)
        monitor.check_drift()
        monitor.save_log()
    """

    def __init__(self, config: dict):
        mon_cfg = config.get("monitoring", {})
        self.enabled = mon_cfg.get("enabled", True)
        self.latency_alert_ms = mon_cfg.get("latency_alert_ms", 35.0)
        self.conf_threshold = mon_cfg.get("confidence_drift_threshold", 0.3)
        self.drift_window = mon_cfg.get("drift_window_frames", 100)
        self.log_dir = Path(mon_cfg.get("log_dir", "./monitor_logs"))
        self.webhook_url = mon_cfg.get("webhook_url", None)

        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Rolling windows
        self._latencies: deque[float] = deque(maxlen=self.drift_window)
        self._conf_scores: deque[float] = deque(maxlen=self.drift_window)
        self._det_counts: deque[int] = deque(maxlen=self.drift_window)
        self._frame_count: int = 0
        self._drift_events: list[dict] = []
        self._start_time: Optional[float] = None
        self._alerts_fired: list[str] = []

    # ── Context manager for timing ──────────────────────────────────────────────

    def start_frame(self) -> None:
        self._start_time = time.perf_counter()

    def end_frame(self) -> float:
        if self._start_time is None:
            return 0.0
        elapsed_ms = (time.perf_counter() - self._start_time) * 1000.0
        self._latencies.append(elapsed_ms)
        self._frame_count += 1
        self._start_time = None

        if elapsed_ms > self.latency_alert_ms:
            msg = (f"[Monitor] LATENCY ALERT frame={self._frame_count}: "
                   f"{elapsed_ms:.1f}ms > {self.latency_alert_ms}ms")
            print(msg)
            self._alerts_fired.append(msg)
            self._fire_webhook(msg)
        return elapsed_ms

    class _FrameTimer:
        def __init__(self, monitor: "InferenceMonitor"):
            self._m = monitor
        def __enter__(self):
            self._m.start_frame()
            return self
        def __exit__(self, *_):
            self._m.end_frame()

    def track_frame(self) -> "_FrameTimer":
        return self._FrameTimer(self)

    # ── Prediction recording ────────────────────────────────────────────────────

    def record_predictions(
        self,
        predictions,
        conf_scores: Optional[list[float]] = None,
    ) -> None:
        """
        Record prediction outcomes for this frame.

        predictions: raw model output (list of dicts, numpy array, etc.)
        conf_scores: optional list of confidence scores for detections.
                     If None and predictions is a list of dicts with 'score' key,
                     scores are extracted automatically.
        """
        if not self.enabled:
            return

        # Extract confidence scores
        if conf_scores is None:
            if isinstance(predictions, list) and predictions and \
                    isinstance(predictions[0], dict):
                conf_scores = [p.get("score", 0.0) for p in predictions]
            else:
                conf_scores = []

        # Detection count
        self._det_counts.append(len(conf_scores))

        # Record per-detection scores
        for score in conf_scores:
            self._conf_scores.append(float(score))

    def record_classification(self, top1_conf: float) -> None:
        """Convenience method for classification confidence."""
        self._conf_scores.append(float(top1_conf))
        self._det_counts.append(1)

    # ── Drift detection ─────────────────────────────────────────────────────────

    def check_drift(self) -> bool:
        """
        Check for confidence drift over the last drift_window frames.
        Returns True if drift is detected (alert fired).
        """
        if len(self._conf_scores) < self.drift_window:
            return False

        avg_conf = float(np.mean(self._conf_scores))
        if avg_conf < self.conf_threshold:
            event = {
                "type": "confidence_drift",
                "timestamp": datetime.utcnow().isoformat(),
                "avg_confidence": round(avg_conf, 4),
                "threshold": self.conf_threshold,
                "window_frames": self.drift_window,
                "frame_number": self._frame_count,
            }
            self._drift_events.append(event)
            msg = (f"[Monitor] DRIFT ALERT frame={self._frame_count}: "
                   f"avg_conf={avg_conf:.3f} < threshold={self.conf_threshold}")
            print(msg)
            self._alerts_fired.append(msg)
            self._fire_webhook(msg)
            return True
        return False

    # ── Stats ───────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        latencies = list(self._latencies)
        confs = list(self._conf_scores)
        dets = list(self._det_counts)
        return {
            "frames_processed": self._frame_count,
            "latency_ms": {
                "mean": round(statistics.mean(latencies), 2) if latencies else 0.0,
                "p50":  round(statistics.median(latencies), 2) if latencies else 0.0,
                "p95":  round(float(np.percentile(latencies, 95)), 2) if latencies else 0.0,
                "max":  round(max(latencies), 2) if latencies else 0.0,
            },
            "fps": round(1000.0 / statistics.mean(latencies), 1)
                   if latencies and statistics.mean(latencies) > 0 else 0.0,
            "confidence": {
                "mean": round(statistics.mean(confs), 4) if confs else 0.0,
                "min":  round(min(confs), 4) if confs else 0.0,
                "max":  round(max(confs), 4) if confs else 0.0,
            },
            "detections_per_frame": {
                "mean": round(statistics.mean(dets), 2) if dets else 0.0,
                "max":  max(dets) if dets else 0,
            },
            "drift_events": len(self._drift_events),
            "alerts_fired": len(self._alerts_fired),
        }

    def print_summary(self) -> None:
        s = self.stats()
        print("\n[Monitor] ── Runtime Summary ──────────────────────────")
        print(f"  Frames:       {s['frames_processed']}")
        print(f"  Latency(ms):  mean={s['latency_ms']['mean']}  "
              f"p95={s['latency_ms']['p95']}  max={s['latency_ms']['max']}")
        print(f"  Throughput:   {s['fps']} FPS")
        print(f"  Confidence:   mean={s['confidence']['mean']}  "
              f"min={s['confidence']['min']}")
        print(f"  Drift events: {s['drift_events']}")
        print(f"  Alerts fired: {s['alerts_fired']}")
        print("─────────────────────────────────────────────────────────\n")

    def save_log(self) -> Path:
        ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        log_file = self.log_dir / f"monitor_{ts}.json"
        payload = {
            "stats": self.stats(),
            "drift_events": self._drift_events,
            "alerts": self._alerts_fired,
        }
        with open(log_file, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[Monitor] Log saved: {log_file}")
        return log_file

    # ── Webhook ─────────────────────────────────────────────────────────────────

    def _fire_webhook(self, message: str) -> None:
        if not self.webhook_url:
            return
        payload = json.dumps({"text": message}).encode()
        req = urllib.request.Request(
            self.webhook_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(req, timeout=3)
        except Exception as exc:
            print(f"[Monitor] Webhook delivery failed: {exc}")
