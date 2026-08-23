"""AM62A inference engine (1 TOPS MMA C7x NPU, 2 GB RAM)."""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from common.preprocess import preprocess_classification, preprocess_detection
from common.postprocess import softmax, topk_classes, decode_yolox_output

SUPPORTED_MODELS = {
    "mobilenet_v2_lite": {"task": "classification", "input": (224, 224), "latency_ms": 15, "top1": 72.3, "params_m": 3.4},
    "mobilenet_v3_small_lite": {"task": "classification", "input": (224, 224), "latency_ms": 10, "top1": 67.4, "params_m": 2.5},
    "yolox_pico_lite": {"task": "detection", "input": (320, 320), "latency_ms": 8, "mAP": 20.1, "params_m": 0.9},
    "yolox_nano_lite": {"task": "detection", "input": (416, 416), "latency_ms": 12, "mAP": 25.8, "params_m": 0.91},
}


class AM62AInferenceEngine:
    """Simulated TIDL inference for AM62A (1 TOPS)."""
    DEVICE = "AM62A"
    TOPS = 1.0
    RAM_GB = 2

    def __init__(self, model_name: str, simulate_latency: bool = True):
        if model_name not in SUPPORTED_MODELS:
            raise ValueError(f"Model '{model_name}' not supported on AM62A. Supported: {list(SUPPORTED_MODELS)}")
        self.model_name = model_name
        self.model_info = SUPPORTED_MODELS[model_name]
        self.simulate_latency = simulate_latency
        self._inference_count = 0
        self._warmed_up = False

    def warmup(self, n: int = 3):
        h, w = self.model_info["input"]
        dummy = np.zeros((h, w, 3), dtype=np.uint8)
        for _ in range(n):
            self.run(dummy)
        self._warmed_up = True
        self._inference_count = 0

    def _run_forward(self, tensor: np.ndarray) -> np.ndarray:
        np.random.seed(self._inference_count % 1000)
        if self.model_info["task"] == "classification":
            return np.random.randn(1, 1000).astype(np.float32)
        return np.random.rand(1, 8400, 85).astype(np.float32) * 0.3

    def run(self, image: np.ndarray) -> dict:
        task = self.model_info["task"]
        input_hw = self.model_info["input"]
        if task == "classification":
            tensor = preprocess_classification(image, input_hw)
        else:
            tensor, scale, dw, dh = preprocess_detection(image, input_hw)
        t0 = time.perf_counter()
        raw = self._run_forward(tensor)
        if self.simulate_latency:
            target_s = self.model_info["latency_ms"] / 1000.0
            elapsed = time.perf_counter() - t0
            if elapsed < target_s:
                time.sleep(target_s - elapsed)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        self._inference_count += 1
        result: dict = {"device": self.DEVICE, "model": self.model_name, "task": task, "latency_ms": round(latency_ms, 2), "warmup": self._warmed_up}
        if task == "classification":
            probs = softmax(raw)
            top5 = topk_classes(probs, k=5)
            result["top5"] = top5
            result["top1_class"] = top5[0]["class_id"]
            result["top1_score"] = top5[0]["score"]
        else:
            dets = decode_yolox_output(raw, input_hw)
            result["detections"] = dets
            result["num_detections"] = len(dets)
        return result

    def batch_run(self, images: list) -> list:
        return [self.run(img) for img in images]

    def profile(self) -> dict:
        return {"device": self.DEVICE, "tops": self.TOPS, "ram_gb": self.RAM_GB, "model": self.model_name, "task": self.model_info["task"], "nominal_latency_ms": self.model_info["latency_ms"], "params_m": self.model_info["params_m"], "inference_count": self._inference_count}
