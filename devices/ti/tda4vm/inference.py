"""TDA4VM inference engine (8 TOPS, ASIL-B automotive SoC, 8 GB RAM).

TDA4VM targets ADAS applications. PeopleNet covers pedestrians, cyclists, vehicles at 960x544.
"""
import sys
import time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from common.preprocess import preprocess_detection, preprocess_segmentation
from common.postprocess import decode_yolox_output, argmax_segmentation

SUPPORTED_MODELS = {
    "peoplenet_lite": {"task": "detection", "input": (544, 960), "latency_ms": 28, "mAP": 85.0, "params_m": 20.9, "classes": ["person","bicycle","vehicle"]},
    "yolox_s_lite": {"task": "detection", "input": (640, 640), "latency_ms": 20, "mAP": 38.4, "params_m": 9.0},
    "deeplabv3plus_lite": {"task": "segmentation", "input": (512, 512), "latency_ms": 40, "mIoU": 65.3, "params_m": 11.2},
}

class TDA4VMInferenceEngine:
    """Simulated TIDL inference for TDA4VM (8 TOPS, ASIL-B automotive)."""
    DEVICE = "TDA4VM"; TOPS = 8.0; RAM_GB = 8; SAFETY = "ASIL-B"

    def __init__(self, model_name: str, simulate_latency: bool = True):
        if model_name not in SUPPORTED_MODELS:
            raise ValueError(f"Model '{model_name}' not supported on TDA4VM. Supported: {list(SUPPORTED_MODELS)}")
        self.model_name = model_name; self.model_info = SUPPORTED_MODELS[model_name]
        self.simulate_latency = simulate_latency; self._inference_count = 0; self._warmed_up = False

    def warmup(self, n: int = 3):
        h, w = self.model_info["input"]
        for _ in range(n): self.run(np.zeros((h,w,3),dtype=np.uint8))
        self._warmed_up = True; self._inference_count = 0

    def _run_forward(self, tensor):
        np.random.seed(self._inference_count % 1000)
        task = self.model_info["task"]; h, w = self.model_info["input"]
        if task == "detection":
            anchors = (h//8)*(w//8) + (h//16)*(w//16) + (h//32)*(w//32)
            nc = len(self.model_info.get("classes",[]) or []) or 80
            return np.random.rand(1, anchors, 5+nc).astype(np.float32)*0.3
        return np.random.rand(1, 21, h//8, w//8).astype(np.float32)

    def run(self, image: np.ndarray) -> dict:
        task = self.model_info["task"]; input_hw = self.model_info["input"]
        if task == "detection": tensor, *_ = preprocess_detection(image, input_hw)
        else: tensor = preprocess_segmentation(image, input_hw)
        t0 = time.perf_counter(); raw = self._run_forward(tensor)
        if self.simulate_latency:
            elapsed = time.perf_counter() - t0
            if elapsed < self.model_info["latency_ms"]/1000: time.sleep(self.model_info["latency_ms"]/1000 - elapsed)
        latency_ms = (time.perf_counter()-t0)*1000; self._inference_count += 1
        result = {"device": self.DEVICE, "safety": self.SAFETY, "model": self.model_name, "task": task, "latency_ms": round(latency_ms,2), "warmup": self._warmed_up}
        if task == "detection": dets = decode_yolox_output(raw, input_hw); result["detections"] = dets; result["num_detections"] = len(dets)
        else: result["class_map_shape"] = list(argmax_segmentation(raw).shape)
        return result

    def batch_run(self, images): return [self.run(img) for img in images]
    def profile(self): return {"device": self.DEVICE, "safety": self.SAFETY, "tops": self.TOPS, "ram_gb": self.RAM_GB, "model": self.model_name, "task": self.model_info["task"], "nominal_latency_ms": self.model_info["latency_ms"], "params_m": self.model_info["params_m"], "inference_count": self._inference_count}
