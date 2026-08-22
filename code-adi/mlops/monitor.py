"""
monitor.py — Runtime inference monitoring for ADI Model Zoo deployments.

Tracks latency, confidence, and drift on MAX78002 / MAX32690 / ADSP-SC835.
MAX78002 CNN inference is very fast (~5-20ms), so latency thresholds are tight.
"""

from __future__ import annotations

import contextlib
import json
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional


class InferenceMonitor:
    """
    Rolling-window monitor for edge inference metrics.

    Usage:
        monitor = InferenceMonitor(config)
        with monitor.track_frame():
            output = model.run(input)
            monitor.record_output(output, domain="vision", task="object_detection")
        stats = monitor.stats()
        monitor.check_drift()
    """

    def __init__(self, config: dict):
        mon_cfg = config.get("monitoring", {})
        window = mon_cfg.get("drift_window_frames", 50)

        self.latency_alert_ms = mon_cfg.get("latency_alert_ms", 50.0)
        self.drift_threshold = mon_cfg.get("confidence_drift_threshold", 0.25)
        self.log_dir = Path(mon_cfg.get("log_dir", "./monitor_logs"))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.webhook_url = mon_cfg.get("webhook_url")
        self.device = config.get("device", "MAX78002")

        self._latencies = deque(maxlen=window)
        self._confidences = deque(maxlen=window)
        self._det_counts = deque(maxlen=window)
        self._frame_count = 0
        self._drift_events = []
        self._t_start: Optional[float] = None

    def start_frame(self) -> None:
        self._t_start = time.perf_counter()

    def end_frame(self) -> float:
        if self._t_start is None:
            return 0.0
        elapsed_ms = (time.perf_counter() - self._t_start) * 1000.0
        self._latencies.append(elapsed_ms)
        self._frame_count += 1
        self._t_start = None
        if elapsed_ms > self.latency_alert_ms:
            self._fire_alert(f"Latency spike: {elapsed_ms:.1f}ms > {self.latency_alert_ms}ms")
        return elapsed_ms

    @contextlib.contextmanager
    def track_frame(self):
        self.start_frame()
        try:
            yield self
        finally:
            self.end_frame()

    def record_output(
        self,
        output: object,
        domain: str = "vision",
        task: str = "object_detection",
    ) -> None:
        """
        Parse model output and record confidence/count metrics.
        Accepts dict list (detection), array (classification), or scalar.
        """
        if isinstance(output, (list, tuple)) and len(output) > 0:
            if isinstance(output[0], dict):
                self.record_detections(output)
                return
        if hasattr(output, '__len__') and not isinstance(output, (str, bytes)):
            arr = list(output)
            if arr:
                self.record_classification(float(max(arr)) if len(arr) > 0 else 0.5)
                return
        if isinstance(output, (int, float)):
            self.record_classification(float(output))

    def record_detections(self, detections: list) -> None:
        """detections: list of dicts with 'score' key."""
        self._det_counts.append(len(detections))
        if detections:
            avg_conf = sum(d.get("score", 0.5) for d in detections) / len(detections)
            self._confidences.append(avg_conf)
        else:
            self._confidences.append(0.0)

    def record_classification(self, top1_conf: float) -> None:
        self._confidences.append(top1_conf)
        self._det_counts.append(1)

    def record_anomaly_score(self, reconstruction_error: float, threshold: float = 0.05) -> None:
        confidence = float(reconstruction_error < threshold)
        self._confidences.append(confidence)
        self._det_counts.append(1)

    def check_drift(self) -> bool:
        if len(self._confidences) < 5:
            return False
        avg_conf = sum(self._confidences) / len(self._confidences)
        if avg_conf < self.drift_threshold:
            event = {
                "timestamp": datetime.utcnow().isoformat(),
                "avg_confidence": round(avg_conf, 4),
                "threshold": self.drift_threshold,
                "window_size": len(self._confidences),
                "device": self.device,
            }
            self._drift_events.append(event)
            self._fire_alert(
                f"Confidence drift detected on {self.device}: "
                f"avg={avg_conf:.3f} < threshold={self.drift_threshold}"
            )
            return True
        return False

    def _fire_alert(self, message: str) -> None:
        print(f"[Monitor] ALERT: {message}")
        if self.webhook_url:
            try:
                import urllib.request
                import urllib.error
                payload = json.dumps({"alert": message, "device": self.device}).encode()
                req = urllib.request.Request(
                    self.webhook_url,
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                urllib.request.urlopen(req, timeout=5)
            except Exception as e:
                print(f"[Monitor] Webhook failed: {e}")

    def stats(self) -> dict:
        def _stats(data):
            if not data:
                return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
            arr = sorted(data)
            n = len(arr)
            return {
                "mean": round(sum(arr) / n, 3),
                "p50":  round(arr[n // 2], 3),
                "p95":  round(arr[int(n * 0.95)], 3),
                "min":  round(arr[0], 3),
                "max":  round(arr[-1], 3),
            }

        lat = _stats(self._latencies)
        fps = round(1000.0 / lat["mean"], 1) if lat["mean"] > 0 else 0.0
        return {
            "device": self.device,
            "frames_processed": self._frame_count,
            "latency_ms": lat,
            "fps": fps,
            "confidence": _stats(self._confidences),
            "detections_per_frame": _stats(self._det_counts),
            "drift_events": len(self._drift_events),
        }

    def print_summary(self) -> None:
        s = self.stats()
        print(f"\n[Monitor] === Inference Summary ({self.device}) ===")
        print(f"  Frames:      {s['frames_processed']}")
        print(f"  Latency:     {s['latency_ms']['mean']}ms mean | "
              f"{s['latency_ms']['p95']}ms p95")
        print(f"  FPS:         {s['fps']}")
        print(f"  Confidence:  {s['confidence']['mean']}")
        print(f"  Drift events:{s['drift_events']}")

    def save_log(self) -> Path:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out = self.log_dir / f"monitor_{self.device}_{ts}.json"
        with open(out, "w") as f:
            json.dump({**self.stats(), "drift_log": self._drift_events}, f, indent=2)
        print(f"[Monitor] Log saved: {out}")
        return out
