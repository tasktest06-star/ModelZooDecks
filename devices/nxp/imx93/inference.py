"""
i.MX 93 Ethos-U65 Inference Engine (Vela-compiled TFLite).

i.MX 93 key facts:
- Cortex-A55 dual-core 1.7 GHz + Ethos-U65 1.0 TOPS NPU
- Models compiled with Vela: `vela model.tflite --accelerator-config ethos-u65-256`
- Shared_Sram mode -- 2MB command-stream SRAM on NPU
- 3-4x speedup over CPU-only TFLite on supported ops
- Run-time: TFLite Delegate calling Ethos-U65 driver
"""
import time
import numpy as np
import yaml
from pathlib import Path

from ..common.preprocess import (preprocess_classification, preprocess_detection,
                                  preprocess_microspeech)
from ..common.postprocess import softmax, topk, decode_nanodet_output

CONFIG_PATH = Path(__file__).parent / "config.yaml"

KEYWORDS = ["silence", "unknown", "yes", "no", "up", "down",
            "left", "right", "on", "off", "stop", "go"]


class IMX93InferenceEngine:
    """Simulates Ethos-U65 accelerated TFLite inference on i.MX 93."""

    DEVICE = "i.MX 93"
    TOPS   = 1.0
    RAM_GB = 2

    SUPPORTED_MODELS = {
        "nanodet_plus_320": {
            "task": "detection", "input": (320, 320),
            "latency_ms": 30, "mAP": 27.0, "vela_speedup": 3.2,
        },
        "efficientdet_lite0": {
            "task": "detection", "input": (320, 320),
            "latency_ms": 45, "mAP": 25.7, "vela_speedup": 3.8,
        },
        "mobilenetv2_100": {
            "task": "classification", "input": (224, 224),
            "latency_ms": 8, "top1": 71.8, "vela_speedup": 4.0,
        },
        "ds_cnn_l": {
            "task": "kws", "input": (49, 40),
            "latency_ms": 6, "accuracy": 95.1, "vela_speedup": 3.5,
        },
    }

    def __init__(self, model_name: str, simulate_latency: bool = True):
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"'{model_name}' not supported on i.MX 93. "
                f"Options: {list(self.SUPPORTED_MODELS.keys())}"
            )
        self.model_name       = model_name
        self.meta             = self.SUPPORTED_MODELS[model_name]
        self.simulate_latency = simulate_latency
        self._call_count      = 0

    def _vela_forward(self, tensor: np.ndarray, out_size: int = 1001) -> np.ndarray:
        """Simulate Ethos-U65 accelerated forward pass."""
        if self.simulate_latency:
            time.sleep(self.meta["latency_ms"] * 0.0001)
        np.random.seed(self._call_count % 512)
        self._call_count += 1
        return np.random.randn(out_size).astype(np.float32)

    def run(self, data: np.ndarray) -> dict:
        t0   = time.perf_counter()
        task = self.meta["task"]

        if task == "classification":
            tensor = preprocess_classification(data, self.meta["input"])
            logits = self._vela_forward(tensor, 1001)
            probs  = softmax(logits)
            top5   = topk(probs, k=5)
            result = {"task": task, "top5": top5, "top1_class": top5[0]["class_id"]}

        elif task == "detection":
            tensor, scale, dw, dh = preprocess_detection(data, self.meta["input"])
            scores = self._vela_forward(tensor, 90)
            dets   = decode_nanodet_output(scores, np.zeros(4))
            result = {"task": task, "detections": dets, "num_detections": len(dets)}

        else:  # kws
            audio_feat = preprocess_microspeech(data) if data.ndim == 1 else data
            logits     = self._vela_forward(audio_feat, len(KEYWORDS))
            probs      = softmax(logits)
            kw_idx     = int(probs.argmax())
            result     = {
                "task": task,
                "keyword": KEYWORDS[kw_idx % len(KEYWORDS)],
                "confidence": round(float(probs[kw_idx]), 4),
            }

        elapsed = (time.perf_counter() - t0) * 1000
        result.update({
            "device": self.DEVICE, "model": self.model_name,
            "latency_ms": round(elapsed, 3),
            "vela_speedup": self.meta["vela_speedup"],
            "backend": "ethos-u65",
        })
        return result

    def batch_run(self, inputs: list) -> list:
        return [self.run(x) for x in inputs]

    def profile(self) -> dict:
        return {
            "device": self.DEVICE, "tops": self.TOPS,
            "model": self.model_name, "task": self.meta["task"],
            "latency_ms": self.meta["latency_ms"],
            "vela_speedup": self.meta["vela_speedup"],
            "backend": "ethos-u65",
        }
