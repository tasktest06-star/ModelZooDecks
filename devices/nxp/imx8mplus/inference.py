"""
i.MX 8M Plus NPU (2.3 TOPS) Inference Engine.

i.MX 8M Plus key facts:
- Cortex-A53 quad-core 1.8 GHz + NXP proprietary NPU 2.3 TOPS
- 4 GB LPDDR4 -- handles large models like ResNet50, InceptionV4
- eIQ Toolkit: Neutron + NNStreamer integration
- Supports FaceNet512 (LFW 99.65%), YOLOv8M, wav2letter ASR
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


class IMX8MPlusInferenceEngine:
    """Simulates i.MX 8M Plus NPU (2.3 TOPS) inference."""

    DEVICE = "i.MX 8M Plus"
    TOPS   = 2.3
    RAM_GB = 4

    SUPPORTED_MODELS = {
        "yolov8m": {
            "task": "detection", "input": (640, 640),
            "latency_ms": 62,  "mAP": 50.2,
        },
        "resnet50": {
            "task": "classification", "input": (224, 224),
            "latency_ms": 18,  "top1": 76.1,
        },
        "inceptionv4": {
            "task": "classification", "input": (299, 299),
            "latency_ms": 45,  "top1": 79.2,
        },
        "facenet512": {
            "task": "face_recognition", "input": (160, 160),
            "latency_ms": 25,  "lfw_accuracy": 99.65,
        },
        "wav2letter": {
            "task": "asr", "input": (296, 39),
            "latency_ms": 120, "wer_pct": 7.2,
        },
        "mobilenetv3_large": {
            "task": "classification", "input": (224, 224),
            "latency_ms": 12,  "top1": 75.2,
        },
    }

    def __init__(self, model_name: str, simulate_latency: bool = True):
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"'{model_name}' not supported on i.MX 8M Plus. "
                f"Options: {list(self.SUPPORTED_MODELS.keys())}"
            )
        self.model_name       = model_name
        self.meta             = self.SUPPORTED_MODELS[model_name]
        self.simulate_latency = simulate_latency
        self._call_count      = 0

    def _npu_forward(self, tensor: np.ndarray, out_size: int = 1001) -> np.ndarray:
        if self.simulate_latency:
            time.sleep(self.meta["latency_ms"] * 0.0001)
        np.random.seed(self._call_count % 1024)
        self._call_count += 1
        return np.random.randn(out_size).astype(np.float32)

    def run(self, data: np.ndarray) -> dict:
        t0   = time.perf_counter()
        task = self.meta["task"]

        if task == "classification":
            tensor = preprocess_classification(data, self.meta["input"])
            logits = self._npu_forward(tensor, 1001)
            probs  = softmax(logits)
            top5   = topk(probs, k=5)
            result = {"task": task, "top5": top5, "top1_class": top5[0]["class_id"]}

        elif task == "detection":
            tensor, scale, dw, dh = preprocess_detection(data, self.meta["input"])
            scores = self._npu_forward(tensor, 80)
            dets   = decode_nanodet_output(scores, np.zeros(4))
            result = {"task": task, "detections": dets, "num_detections": len(dets)}

        elif task == "face_recognition":
            tensor    = preprocess_classification(data, self.meta["input"])
            embedding = self._npu_forward(tensor, 512)
            norm      = embedding / (np.linalg.norm(embedding) + 1e-8)
            result    = {
                "task": task,
                "embedding": norm.tolist()[:8],
                "embedding_dim": 512,
            }

        elif task == "asr":
            feats  = preprocess_microspeech(data) if data.ndim == 1 else data
            logits = self._npu_forward(feats, 29)
            chars  = "abcdefghijklmnopqrstuvwxyz' _|"
            pred   = chars[int(np.argmax(logits)) % len(chars)]
            result = {"task": task, "prediction": pred, "wer_pct": self.meta["wer_pct"]}

        else:
            result = {"task": task}

        elapsed = (time.perf_counter() - t0) * 1000
        result.update({
            "device": self.DEVICE, "model": self.model_name,
            "latency_ms": round(elapsed, 3),
            "backend": "nxp-npu",
        })
        return result

    def batch_run(self, inputs: list) -> list:
        return [self.run(x) for x in inputs]

    def profile(self) -> dict:
        return {
            "device": self.DEVICE, "tops": self.TOPS, "ram_gb": self.RAM_GB,
            "model": self.model_name, "task": self.meta["task"],
            "latency_ms": self.meta["latency_ms"], "backend": "nxp-npu",
        }
