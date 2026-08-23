"""MAX78002 inference engine (CNN accelerator, 442 TOPS/W, 5 MB SRAM, QAT INT4/INT8)."""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from common.preprocess import preprocess_image, preprocess_audio_segment
from common.postprocess import softmax, topk_classes, decode_kws_output, decode_detection_output

KWS_LABELS = [
    "silence", "unknown", "yes", "no", "up", "down", "left", "right",
    "on", "off", "stop", "go",
]

SUPPORTED_MODELS = {
    "keyword_spotting_ds_cnn": {
        "task": "kws",
        "input_shape": (1, 64, 101),
        "latency_ms": 0.15,
        "accuracy": 94.5,
        "energy_uj": 2.3,
        "sram_kb": 512,
        "labels": KWS_LABELS,
    },
    "visual_wake_words": {
        "task": "classification",
        "input_shape": (3, 96, 96),
        "latency_ms": 0.8,
        "accuracy": 88.0,
        "energy_uj": 12.0,
        "sram_kb": 1800,
        "labels": ["no_person", "person"],
    },
    "face_detection_fpn": {
        "task": "detection",
        "input_shape": (3, 64, 64),
        "latency_ms": 0.5,
        "mAP": 72.0,
        "energy_uj": 8.0,
        "sram_kb": 1200,
    },
    "anomaly_detection": {
        "task": "anomaly",
        "input_shape": (1, 128),
        "latency_ms": 0.05,
        "auc": 0.96,
        "energy_uj": 0.75,
        "sram_kb": 64,
    },
}

SRAM_BUDGET_KB = 5120
POWER_INFERENCE_MW = 15.0
POWER_SLEEP_MW = 0.03


class MAX78002InferenceEngine:
    """Simulated inference for MAX78002 CNN accelerator (QAT INT4/INT8, 442 TOPS/W)."""

    DEVICE = "MAX78002"
    VENDOR = "ADI"
    SRAM_KB = SRAM_BUDGET_KB
    EFFICIENCY_TOPS_W = 442

    def __init__(self, model_name: str, simulate_latency: bool = True):
        if model_name not in SUPPORTED_MODELS:
            raise ValueError(
                f"Model '{model_name}' not supported on MAX78002. "
                f"Supported: {list(SUPPORTED_MODELS)}"
            )
        info = SUPPORTED_MODELS[model_name]
        if info["sram_kb"] > SRAM_BUDGET_KB:
            raise ValueError(
                f"Model '{model_name}' requires {info['sram_kb']} KB SRAM, "
                f"exceeds budget of {SRAM_BUDGET_KB} KB"
            )
        self.model_name = model_name
        self.model_info = info
        self.simulate_latency = simulate_latency
        self._inference_count = 0
        self._warmed_up = False

    def warmup(self, n: int = 3):
        dummy = self._make_dummy_input()
        for _ in range(n):
            self._run(dummy)
        self._warmed_up = True
        self._inference_count = 0

    def _make_dummy_input(self) -> np.ndarray:
        shape = self.model_info["input_shape"]
        return np.zeros(shape, dtype=np.float32)

    def _run_forward(self, tensor: np.ndarray) -> np.ndarray:
        np.random.seed(self._inference_count % 1000)
        task = self.model_info["task"]
        if task == "kws":
            return np.random.randn(1, len(KWS_LABELS)).astype(np.float32)
        elif task == "classification":
            labels = self.model_info.get("labels", ["class_0"])
            return np.random.randn(1, len(labels)).astype(np.float32)
        elif task == "detection":
            return np.random.rand(1, 50, 6).astype(np.float32) * 0.3
        else:  # anomaly
            return np.random.rand(1, 1).astype(np.float32)

    def _run(self, tensor: np.ndarray) -> dict:
        t0 = time.perf_counter()
        raw = self._run_forward(tensor)
        if self.simulate_latency:
            target_s = self.model_info["latency_ms"] / 1000.0
            elapsed = time.perf_counter() - t0
            if elapsed < target_s:
                time.sleep(target_s - elapsed)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        self._inference_count += 1
        return {"raw": raw, "latency_ms": latency_ms}

    def run_image(self, image: np.ndarray) -> dict:
        """Run vision model (VWW or face detection) on an image."""
        task = self.model_info["task"]
        if task not in ("classification", "detection"):
            raise ValueError(f"run_image() not applicable for task='{task}'")
        h, w = self.model_info["input_shape"][1], self.model_info["input_shape"][2]
        tensor = preprocess_image(image, (h, w))
        out = self._run(tensor)
        raw, latency_ms = out["raw"], out["latency_ms"]

        result: dict = {
            "device": self.DEVICE,
            "model": self.model_name,
            "task": task,
            "latency_ms": round(latency_ms, 3),
            "energy_uj": round(latency_ms * POWER_INFERENCE_MW, 3),
            "warmup": self._warmed_up,
        }
        if task == "classification":
            labels = self.model_info.get("labels")
            probs = softmax(raw)
            result["top1"] = topk_classes(probs, k=1, labels=labels)[0]
            result["probs"] = probs.tolist()
        else:
            result["detections"] = decode_detection_output(raw, conf_thresh=0.4)
        return result

    def run_audio(self, waveform: np.ndarray, sample_rate: int = 16000) -> dict:
        """Run KWS model on a raw audio waveform."""
        if self.model_info["task"] != "kws":
            raise ValueError("run_audio() only for kws task")
        tensor = preprocess_audio_segment(waveform, sample_rate)
        out = self._run(tensor)
        raw, latency_ms = out["raw"], out["latency_ms"]
        kws = decode_kws_output(raw, self.model_info["labels"])
        return {
            "device": self.DEVICE,
            "model": self.model_name,
            "task": "kws",
            "latency_ms": round(latency_ms, 3),
            "energy_uj": round(latency_ms * POWER_INFERENCE_MW, 3),
            "warmup": self._warmed_up,
            **kws,
        }

    def run_sensor(self, time_series: np.ndarray) -> dict:
        """Run anomaly detection on a 1-D time-series window."""
        if self.model_info["task"] != "anomaly":
            raise ValueError("run_sensor() only for anomaly task")
        tensor = time_series.astype(np.float32).reshape(self.model_info["input_shape"])
        out = self._run(tensor)
        raw, latency_ms = out["raw"], out["latency_ms"]
        score = float(raw.flatten()[0])
        return {
            "device": self.DEVICE,
            "model": self.model_name,
            "task": "anomaly",
            "anomaly_score": score,
            "is_anomaly": score > 0.5,
            "latency_ms": round(latency_ms, 3),
            "energy_uj": round(latency_ms * POWER_INFERENCE_MW, 3),
            "warmup": self._warmed_up,
        }

    def profile(self) -> dict:
        return {
            "device": self.DEVICE,
            "vendor": self.VENDOR,
            "sram_kb": self.SRAM_KB,
            "efficiency_tops_w": self.EFFICIENCY_TOPS_W,
            "model": self.model_name,
            "task": self.model_info["task"],
            "nominal_latency_ms": self.model_info["latency_ms"],
            "nominal_energy_uj": self.model_info["energy_uj"],
            "model_sram_kb": self.model_info["sram_kb"],
            "inference_count": self._inference_count,
        }
