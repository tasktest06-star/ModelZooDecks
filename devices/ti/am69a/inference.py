"""AM69A inference engine (32 TOPS C7x x8 + MMA, 32 GB RAM)."""
import sys
import time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))
from common.preprocess import preprocess_classification, preprocess_detection, preprocess_segmentation, preprocess_depth
from common.postprocess import softmax, topk_classes, decode_yolox_output, argmax_segmentation, decode_depth_output, decode_pose_output

SUPPORTED_MODELS = {
    "yolox_m_lite": {"task": "detection", "input": (640,640), "latency_ms": 8, "mAP": 44.2, "params_m": 26.8},
    "yolox_l_lite": {"task": "detection", "input": (640,640), "latency_ms": 12, "mAP": 49.0, "params_m": 54.2},
    "rtmdet_m_lite": {"task": "detection", "input": (640,640), "latency_ms": 14, "mAP": 56.0, "params_m": 24.7},
    "rtmdet_l_lite": {"task": "detection", "input": (640,640), "latency_ms": 18, "mAP": 60.0, "params_m": 52.3},
    "deeplabv3plus_lite": {"task": "segmentation", "input": (512,512), "latency_ms": 15, "mIoU": 65.3, "params_m": 11.2},
    "yoloxpose_s_lite": {"task": "pose", "input": (640,640), "latency_ms": 10, "AP": 61.2, "params_m": 9.2},
    "yoloxpose_l_lite": {"task": "pose", "input": (640,640), "latency_ms": 20, "AP": 70.1, "params_m": 54.7},
    "fastvit_s12": {"task": "classification", "input": (256,256), "latency_ms": 3, "top1": 79.3, "params_m": 10.7},
    "swin_tiny": {"task": "classification", "input": (224,224), "latency_ms": 10, "top1": 81.2, "params_m": 28.3},
    "midas_small_lite": {"task": "depth", "input": (256,256), "latency_ms": 8, "params_m": 21.1},
}

class AM69AInferenceEngine:
    """Simulated TIDL inference for AM69A (32 TOPS, highest-end TI SoC)."""
    DEVICE = "AM69A"; TOPS = 32.0; RAM_GB = 32

    def __init__(self, model_name: str, simulate_latency: bool = True):
        if model_name not in SUPPORTED_MODELS:
            raise ValueError(f"Model '{model_name}' not supported on AM69A. Supported: {list(SUPPORTED_MODELS)}")
        self.model_name = model_name; self.model_info = SUPPORTED_MODELS[model_name]
        self.simulate_latency = simulate_latency; self._inference_count = 0; self._warmed_up = False

    def warmup(self, n: int = 3):
        h, w = self.model_info["input"]
        for _ in range(n): self.run(np.zeros((h,w,3),dtype=np.uint8))
        self._warmed_up = True; self._inference_count = 0

    def _run_forward(self, tensor):
        np.random.seed(self._inference_count % 1000)
        task = self.model_info["task"]; h, w = self.model_info["input"]
        if task == "classification": return np.random.randn(1,1000).astype(np.float32)
        elif task == "detection": return np.random.rand(1,8400,85).astype(np.float32)*0.3
        elif task == "segmentation": return np.random.rand(1,21,h//8,w//8).astype(np.float32)
        elif task == "pose": return np.random.rand(1,17,h//4,w//4).astype(np.float32)
        else: return np.random.rand(1,1,h,w).astype(np.float32)

    def run(self, image: np.ndarray) -> dict:
        task = self.model_info["task"]; input_hw = self.model_info["input"]
        if task == "classification": tensor = preprocess_classification(image, input_hw)
        elif task in ("detection","pose"): tensor, *_ = preprocess_detection(image, input_hw)
        elif task == "segmentation": tensor = preprocess_segmentation(image, input_hw)
        else: tensor = preprocess_depth(image, input_hw)
        t0 = time.perf_counter(); raw = self._run_forward(tensor)
        if self.simulate_latency:
            elapsed = time.perf_counter() - t0
            if elapsed < self.model_info["latency_ms"]/1000: time.sleep(self.model_info["latency_ms"]/1000 - elapsed)
        latency_ms = (time.perf_counter()-t0)*1000; self._inference_count += 1
        result = {"device": self.DEVICE, "model": self.model_name, "task": task, "latency_ms": round(latency_ms,2), "warmup": self._warmed_up}
        if task == "classification": probs = softmax(raw); top5 = topk_classes(probs,k=5); result["top5"] = top5; result["top1_class"] = top5[0]["class_id"]
        elif task == "detection": dets = decode_yolox_output(raw,input_hw); result["detections"] = dets; result["num_detections"] = len(dets)
        elif task == "segmentation": result["class_map_shape"] = list(argmax_segmentation(raw).shape)
        elif task == "pose": kps = decode_pose_output(raw,input_hw,input_hw); result["keypoints"] = kps; result["num_keypoints"] = len(kps)
        else: result["depth_map_shape"] = list(decode_depth_output(raw).shape)
        return result

    def batch_run(self, images): return [self.run(img) for img in images]
    def profile(self): return {"device": self.DEVICE, "tops": self.TOPS, "ram_gb": self.RAM_GB, "model": self.model_name, "task": self.model_info["task"], "nominal_latency_ms": self.model_info["latency_ms"], "params_m": self.model_info["params_m"], "inference_count": self._inference_count}
