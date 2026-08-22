"""Runtime monitoring: latency tracking, confidence drift, alerting."""

import json
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np


class InferenceMonitor:
    def __init__(self, config: dict, model_id: str):
        mon_cfg = config.get("monitoring", {})
        self.model_id = model_id
        self.window_size = mon_cfg.get("window_size", 100)
        self.drift_threshold = mon_cfg.get("drift_threshold", 0.60)
        self.alert_webhook = mon_cfg.get("alert_webhook", "")
        log_dir = Path(mon_cfg.get("log_dir", "logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / f"{model_id}_monitor.jsonl"

        self.latencies_ms: deque = deque(maxlen=self.window_size)
        self.confidences: deque = deque(maxlen=self.window_size)
        self._start_time: Optional[float] = None

    # ── timing ──────────────────────────────────────────────────────────────

    def start_inference(self) -> None:
        self._start_time = time.perf_counter()

    def end_inference(self, outputs: list) -> float:
        elapsed_ms = (time.perf_counter() - self._start_time) * 1000.0
        self.latencies_ms.append(elapsed_ms)
        if outputs:
            out = outputs[0]
            if out.ndim > 1:
                conf = float(np.max(out))
                self.confidences.append(conf)
                if conf < self.drift_threshold:
                    self._trigger_alert("confidence_drift", conf)
        self._log_event(elapsed_ms)
        return elapsed_ms

    # ── statistics ──────────────────────────────────────────────────────────

    @property
    def mean_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        return float(np.mean(list(self.latencies_ms)))

    @property
    def p99_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        return float(np.percentile(list(self.latencies_ms), 99))

    @property
    def fps(self) -> float:
        if not self.latencies_ms:
            return 0.0
        return 1000.0 / max(self.mean_latency_ms, 1e-9)

    @property
    def mean_confidence(self) -> float:
        if not self.confidences:
            return 0.0
        return float(np.mean(list(self.confidences)))

    def summary(self) -> dict:
        return {
            "model_id": self.model_id,
            "n_samples": len(self.latencies_ms),
            "mean_latency_ms": round(self.mean_latency_ms, 2),
            "p99_latency_ms": round(self.p99_latency_ms, 2),
            "fps": round(self.fps, 1),
            "mean_confidence": round(self.mean_confidence, 4),
            "drift_threshold": self.drift_threshold,
        }

    def print_summary(self) -> None:
        s = self.summary()
        print(
            f"[monitor] {s['model_id']} | "
            f"lat={s['mean_latency_ms']}ms p99={s['p99_latency_ms']}ms "
            f"fps={s['fps']} conf={s['mean_confidence']}"
        )

    # ── alerting ─────────────────────────────────────────────────────────────

    def _trigger_alert(self, alert_type: str, value: float) -> None:
        msg = {
            "alert": alert_type,
            "model_id": self.model_id,
            "value": round(value, 4),
            "threshold": self.drift_threshold,
        }
        print(f"[monitor] ALERT: {msg}")
        if self.alert_webhook:
            try:
                import urllib.request
                data = json.dumps(msg).encode()
                req = urllib.request.Request(
                    self.alert_webhook,
                    data=data,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                urllib.request.urlopen(req, timeout=5)
            except Exception as e:
                print(f"[monitor] Webhook failed: {e}")

    def _log_event(self, latency_ms: float) -> None:
        entry = {
            "ts": time.time(),
            "model": self.model_id,
            "latency_ms": round(latency_ms, 2),
            "mean_conf": round(self.mean_confidence, 4),
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
