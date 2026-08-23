"""MAX32690 inference engine (Cortex-M4F @ 120 MHz, ultra-low power, TFLite Micro / CMSIS-NN)."""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from common.preprocess import preprocess_audio_segment
from common.postprocess import softmax, topk_classes, decode_kws_output

SIMPLE_KWS_LABELS = [
    "silence", "unknown", "yes", "no", "on", "off", "stop", "go",
    "up", "down", "left", "right",
]

VIBRATION_LABELS = ["normal", "unbalance", "bearing_fault", "misalignment"]

SUPPORTED_MODELS = {
    "keyword_spotting_simple": {
        "task": "kws",
        "input_shape": (1, 40, 51),
        "latency_ms": 12.0,
        "accuracy": 90.2,
        "sram_kb": 128,
        "labels": SIMPLE_KWS_LABELS,
    },
    "anomaly_detection": {
        "task": "anomaly",
        "input_shape": (1, 64),
        "latency_ms": 0.8,
        "auc": 0.94,
        "sram_kb": 32,
    },
    "vibration_classifier": {
        "task": "classification",
        "input_shape": (1, 256),
        "latency_ms": 5.0,
        "accuracy": 91.5,
        "sram_kb": 96,
        "labels": VIBRATION_LABELS,
    },
}

POWER_ACTIVE_MW = 3.5
SRAM_BUDGET_KB = 1024


class MAX32690InferenceEngine:
    """Simulated TFLite Micro / CMSIS-NN inference for MAX32690 (Cortex-M4F, 3.5 mW)."""

    DEVICE = "MAX32690"
    VENDOR = "ADI"
    SRAM_KB = SRAM_BUDGET_KB
    CPU = "Cortex-M4F"

    def __init__(self, model_name: str, simulate_latency: bool = True):
        if model_name not in SUPPORTED_MODELS:
            raise ValueError(
                f"Model '{model_name}' not supported on MAX32690. "
                f"Supported: {list(SUPPORTED_MODELS)}"
            )
        self.model_name = model_name
        self.model_info = SUPPORTED_MODELS[model_name]
        self.simulate_latency = simulate_latency
        self._inference_count = 0
        self._warmed_up = False

    def warmup(self, n: int = 2):
        for _ in range(n):
            dummy = np.zeros(self.model_info["input_shape"], dtype=np.float32)
            self._run_forward(dummy)
        self._warmed_up = True
        self._inference_count = 0

    def _run_forward(self, tensor: np.ndarray) -> np.ndarray:
        np.random.seed(self._inference_count % 1000)
        task = self.model_info["task"]
        if task == "kws":
            return np.random.randn(1, len(self.model_info["labels"])).astype(np.float32)
        elif task == "classification":
            return np.random.randn(1, len(self.model_info["labels"])).astype(np.float32)
        else:  # anomaly
            return np.random.rand(1, 1).astype(np.float32)

    def run(self, tensor: np.ndarray) -> dict:
        t0 = time.perf_counter()
        raw = self._run_forward(tensor)
        if self.simulate_latency:
            target_s = self.model_info["latency_ms"] / 1000.0
            elapsed = time.perf_counter() - t0
            if elapsed < target_s:
                time.sleep(target_s - elapsed)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        self._inference_count += 1

        task = self.model_info["task"]
        result: dict = {
            "device": self.DEVICE,
            "model": self.model_name,
            "task": task,
            "latency_ms": round(latency_ms, 3),
            "energy_uj": round(latency_ms * POWER_ACTIVE_MW, 3),
            "warmup": self._warmed_up,
        }

        if task == "kws":
            kws = decode_kws_output(raw, self.model_info["labels"])
            result.update(kws)
        elif task == "classification":
            probs = softmax(raw)
            labels = self.model_info.get("labels")
            result["top1"] = topk_classes(probs, k=1, labels=labels)[0]
            result["probs"] = probs.tolist()
        else:
            score = float(raw.flatten()[0])
            result["anomaly_score"] = score
            result["is_anomaly"] = score > 0.5

        return result

    def run_audio(self, waveform: np.ndarray, sample_rate: int = 16000) -> dict:
        tensor = preprocess_audio_segment(waveform, sample_rate)
        # Resize MFCC from (1, 64, 101) to model input (1, 40, 51)
        mfcc = tensor[0]  # (64, 101)
        target_h, target_w = self.model_info["input_shape"][1], self.model_info["input_shape"][2]
        mfcc_r = mfcc[:target_h, :target_w]
        return self.run(mfcc_r[np.newaxis])

    def profile(self) -> dict:
        return {
            "device": self.DEVICE,
            "vendor": self.VENDOR,
            "cpu": self.CPU,
            "sram_kb": self.SRAM_KB,
            "model": self.model_name,
            "task": self.model_info["task"],
            "nominal_latency_ms": self.model_info["latency_ms"],
            "model_sram_kb": self.model_info["sram_kb"],
            "inference_count": self._inference_count,
        }
