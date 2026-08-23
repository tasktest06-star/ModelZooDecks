"""AM67A inference engine (4 TOPS MMA C7x dual-DSP NPU, 4 GB RAM)."""
import sys
import time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from common.preprocess import preprocess_classification, preprocess_detection, preprocess_segmentation, preprocess_depth
from common.postprocess import softmax, topk_classes, decode_yolox_output, argmax_segmentation, decode_depth_output

SUPPORTED_MODELS = {
    "yolox_tiny_lite": {"task": "detection", "input": (416, 416), "latency_ms": 12, "mAP": 29.2, "params_m": 5.1},
    "yolox_s_lite": {"task": "detection", "input": (640, 640), "latency_ms": 20, "mAP": 38.4, "params_m": 9.0},
    "mobilenet_v3_large_lite": {"task": "classification", "input": (224, 224), "latency_ms": 6, "top1": 74.8, "params_m": 5.4},
    "regnet_x_400mf_lite": {"task": "classification", "input": (224, 224), "latency_ms": 8, "top1": 74.1, "params_m": 5.2},
    "fastdepth_lite": {"task": "depth", "input": (320, 256), "latency_ms": 18, "params_m": 3.9},
    "bisenetv2_lite": {"task": "segmentation", "input": (512, 512), "latency_ms": 35, "mIoU": 61.0, "params_m": 4.8},
}

class AM67AInferenceEngine:
    """Simulated TIDL inference for AM67A (4 TOPS)."""
    DEVICE = "AM67A"; TOPS = 4.0; RAM_GB = 4

    def __init__(self, model_name: str, simulate_latency: bool = True):
        if model_name not in SUPPORTED_MODELS:
            raise ValueError(f"Model '{model_name}' not supported on AM67A. Supported: {list(SUPPORTED_MODELS)}")
        self.model_name = model_name; self.model_info = SUPPORTED_MODELS[model_name]
        self.simulate_latency = simulate_latency; self._inference_count = 0; self._warmed_up = False

    def warmup(self, n: int = 3):
        h, w = self.model_info["input"]
        dummy = np.zeros((h, w, 3), dtype=np.uint8)
        for _ in range(n): self.run(dummy)
        self._warmed_up = True; self._inference_count = 0

    def _run_forward(self, tensor):
        np.random.seed(self._inference_count % 1000)
        task = self.model_info["task"]; h, w = self.model_info["input"]
        if task == "classification": return np.random.randn(1, 1000).astype(np.float32)
        elif task == "detection": return np.random.rand(1, 8400, 85).astype(np.float32) * 0.3
        elif task == "segmentation": return np.random.rand(1, 21, h // 8, w // 8).astype(np.float32)
        else: return np.random.rand(1, 1, h, w).astype(np.float32)

    def run(self, image: np.ndarray) -> dict:
        task = self.model_info["task"]; input_hw = self.model_info["input"]
        if task == "classification": tensor = preprocess_classification(image, input_hw)
        elif task == "detection": tensor, *_ = preprocess_detection(image, input_hw)
        elif task == "segmentation": tensor = preprocess_segmentation(image, input_hw)
        else: tensor = preprocess_depth(image, input_hw)
        t0 = time.perf_counter(); raw = self._run_forward(tensor)
        if self.simulate_latency:
            elapsed = time.perf_counter() - t0
            if elapsed < self.model_info["latency_ms"] / 1000.0: time.sleep(self.model_info["latency_ms"] / 1000.0 - elapsed)
        latency_ms = (time.perf_counter() - t0) * 1000.0; self._inference_count += 1
        result = {"device": self.DEVICE, "model": self.model_name, "task": task, "latency_ms": round(latency_ms, 2), "warmup": self._warmed_up}
        if task == "classification": probs = softmax(raw); top5 = topk_classes(probs, k=5); result["top5"] = top5; result["top1_class"] = top5[0]["class_id"]
        elif task == "detection": dets = decode_yolox_output(raw, input_hw); result["detections"] = dets; result["num_detections"] = len(dets)
        elif task == "segmentation": result["class_map_shape"] = list(argmax_segmentation(raw).shape)
        else: result["depth_map_shape"] = list(decode_depth_output(raw).shape)
        return result

    def batch_run(self, images): return [self.run(img) for img in images]
    def profile(self): return {"device": self.DEVICE, "tops": self.TOPS, "ram_gb": self.RAM_GB, "model": self.model_name, "task": self.model_info["task"], "nominal_latency_ms": self.model_info["latency_ms"], "params_m": self.model_info["params_m"], "inference_count": self._inference_count}
