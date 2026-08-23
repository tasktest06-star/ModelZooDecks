"""
MCX N947 TFLite Micro Inference Engine.

MCX N947 key facts:
- ARM Cortex-M33 @ 150 MHz with DSP extension
- 512 KB SRAM -- models must fit in < 128 KB RAM
- TFLite Micro runtime (no OS, no DRAM)
- Only very small quantized models are viable
"""
import time
import numpy as np
import yaml
from pathlib import Path

from ..common.preprocess import preprocess_classification
from ..common.postprocess import softmax, topk

CONFIG_PATH = Path(__file__).parent / "config.yaml"


class MCXInferenceEngine:
    """TFLite Micro inference simulation for MCX N947."""

    DEVICE        = "MCX-N947"
    SRAM_KB       = 512
    RAM_BUDGET_KB = 128

    SUPPORTED_MODELS = {
        "mobilenetv1_025": {
            "task": "classification", "input": (96, 96),
            "latency_ms": 170, "top1": 50.8,
            "ram_kb": 128, "flash_kb": 300, "format": "tflite_micro",
        },
    }

    def __init__(self, model_name: str, simulate_latency: bool = True):
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"'{model_name}' not supported on MCX-N947. "
                f"Options: {list(self.SUPPORTED_MODELS.keys())}"
            )
        self.model_name       = model_name
        self.meta             = self.SUPPORTED_MODELS[model_name]
        self.simulate_latency = simulate_latency
        self._call_count      = 0
        if self.meta["ram_kb"] > self.RAM_BUDGET_KB:
            raise MemoryError(
                f"'{model_name}' needs {self.meta['ram_kb']}KB > "
                f"MCX-N947 budget {self.RAM_BUDGET_KB}KB"
            )

    def _tflm_forward(self, tensor: np.ndarray) -> np.ndarray:
        if self.simulate_latency:
            time.sleep(self.meta["latency_ms"] * 0.0001)
        np.random.seed(self._call_count % 256)
        self._call_count += 1
        return np.random.randn(1001).astype(np.float32)

    def run(self, image: np.ndarray) -> dict:
        t0      = time.perf_counter()
        tensor  = preprocess_classification(image, self.meta["input"])
        logits  = self._tflm_forward(tensor)
        probs   = softmax(logits)
        top5    = topk(probs, k=5)
        elapsed = (time.perf_counter() - t0) * 1000
        return {
            "task": "classification", "top5": top5,
            "top1_class": top5[0]["class_id"],
            "device": self.DEVICE, "model": self.model_name,
            "latency_ms": round(elapsed, 3),
            "ram_used_kb": self.meta["ram_kb"],
            "runtime": "tflite_micro",
        }

    def batch_run(self, images: list) -> list:
        return [self.run(img) for img in images]

    def profile(self) -> dict:
        return {
            "device": self.DEVICE,
            "model": self.model_name,
            "sram_kb": self.SRAM_KB,
            "ram_used_kb": self.meta["ram_kb"],
            "flash_kb": self.meta["flash_kb"],
            "latency_ms": self.meta["latency_ms"],
            "top1": self.meta["top1"],
            "runtime": "tflite_micro",
            "npu": False,
        }
