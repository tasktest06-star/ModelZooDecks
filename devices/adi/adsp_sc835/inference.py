"""ADSP-SC835 inference engine (SHARC+ DSP x2 @ 1 GHz, audio AI workloads)."""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from common.preprocess import compute_mfcc
from common.postprocess import softmax, topk_classes

GENRE_LABELS = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock",
]

SUPPORTED_MODELS = {
    "rnnoise_speech_enhancement": {
        "task": "speech_enhancement",
        "input_shape": (1, 512),
        "latency_ms": 4.0,
        "pesq": 3.1,
        "sram_mb": 8,
    },
    "wav2letter_asr": {
        "task": "asr",
        "input_shape": (1, 80, 200),
        "latency_ms": 40.0,
        "wer_pct": 7.2,
        "sram_mb": 48,
    },
    "genre_net": {
        "task": "classification",
        "input_shape": (1, 128, 128),
        "latency_ms": 15.0,
        "accuracy": 87.3,
        "sram_mb": 16,
        "labels": GENRE_LABELS,
    },
}

POWER_ACTIVE_MW = 800.0
SRAM_MB = 512


class ADSPSC835InferenceEngine:
    """Simulated ONNX Runtime / SHARC inference for ADSP-SC835 audio DSP."""

    DEVICE = "ADSP-SC835"
    VENDOR = "ADI"
    SRAM_MB = SRAM_MB
    DSP = "SHARC+ x2 @ 1 GHz"

    def __init__(self, model_name: str, simulate_latency: bool = True):
        if model_name not in SUPPORTED_MODELS:
            raise ValueError(
                f"Model '{model_name}' not supported on ADSP-SC835. "
                f"Supported: {list(SUPPORTED_MODELS)}"
            )
        self.model_name = model_name
        self.model_info = SUPPORTED_MODELS[model_name]
        self.simulate_latency = simulate_latency
        self._inference_count = 0
        self._warmed_up = False

    def warmup(self, n: int = 2):
        dummy = np.zeros(self.model_info["input_shape"], dtype=np.float32)
        for _ in range(n):
            self._run_forward(dummy)
        self._warmed_up = True
        self._inference_count = 0

    def _run_forward(self, tensor: np.ndarray) -> np.ndarray:
        np.random.seed(self._inference_count % 1000)
        task = self.model_info["task"]
        if task == "speech_enhancement":
            return np.random.randn(*self.model_info["input_shape"]).astype(np.float32) * 0.5
        elif task == "asr":
            # 28 characters + blank (CTC output) for wav2letter
            seq_len = self.model_info["input_shape"][2]
            return np.random.randn(1, seq_len, 29).astype(np.float32)
        else:  # classification
            labels = self.model_info.get("labels", [])
            return np.random.randn(1, len(labels)).astype(np.float32)

    def _timed_run(self, tensor: np.ndarray) -> tuple:
        t0 = time.perf_counter()
        raw = self._run_forward(tensor)
        if self.simulate_latency:
            target_s = self.model_info["latency_ms"] / 1000.0
            elapsed = time.perf_counter() - t0
            if elapsed < target_s:
                time.sleep(target_s - elapsed)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        self._inference_count += 1
        return raw, latency_ms

    def run_speech_enhancement(self, frame: np.ndarray) -> dict:
        """Enhance a 512-sample audio frame."""
        if self.model_info["task"] != "speech_enhancement":
            raise ValueError("run_speech_enhancement() only for speech_enhancement task")
        tensor = frame.astype(np.float32).reshape(self.model_info["input_shape"])
        raw, latency_ms = self._timed_run(tensor)
        return {
            "device": self.DEVICE,
            "model": self.model_name,
            "task": "speech_enhancement",
            "enhanced_frame": raw.flatten(),
            "latency_ms": round(latency_ms, 3),
            "energy_uj": round(latency_ms * POWER_ACTIVE_MW, 3),
            "warmup": self._warmed_up,
        }

    def run_asr(self, audio_features: np.ndarray) -> dict:
        """Run ASR on mel-spectrogram features (1, 80, T)."""
        if self.model_info["task"] != "asr":
            raise ValueError("run_asr() only for asr task")
        tensor = audio_features.astype(np.float32)
        if tensor.ndim == 2:
            tensor = tensor[np.newaxis]
        raw, latency_ms = self._timed_run(tensor)
        # Greedy CTC decode (simplified — just pick argmax per frame)
        greedy = np.argmax(raw[0], axis=-1)  # (T,)
        chars = "abcdefghijklmnopqrstuvwxyz '" + "_"
        transcript = ""
        prev = -1
        for c in greedy:
            if c != prev and c != len(chars) - 1:
                transcript += chars[c] if c < len(chars) else "?"
            prev = c
        return {
            "device": self.DEVICE,
            "model": self.model_name,
            "task": "asr",
            "transcript": transcript.strip(),
            "latency_ms": round(latency_ms, 3),
            "energy_uj": round(latency_ms * POWER_ACTIVE_MW, 3),
            "warmup": self._warmed_up,
        }

    def run_classification(self, mel_spec: np.ndarray) -> dict:
        """Run genre classification on a (128, 128) mel-spectrogram."""
        if self.model_info["task"] != "classification":
            raise ValueError("run_classification() only for classification task")
        tensor = mel_spec.astype(np.float32)
        if tensor.ndim == 2:
            tensor = tensor[np.newaxis, np.newaxis]
        elif tensor.ndim == 3:
            tensor = tensor[np.newaxis]
        raw, latency_ms = self._timed_run(tensor)
        probs = softmax(raw)
        labels = self.model_info.get("labels")
        top = topk_classes(probs, k=3, labels=labels)
        return {
            "device": self.DEVICE,
            "model": self.model_name,
            "task": "classification",
            "top3": top,
            "top1_label": top[0].get("label", str(top[0]["class_id"])),
            "latency_ms": round(latency_ms, 3),
            "energy_uj": round(latency_ms * POWER_ACTIVE_MW, 3),
            "warmup": self._warmed_up,
        }

    def profile(self) -> dict:
        return {
            "device": self.DEVICE,
            "vendor": self.VENDOR,
            "dsp": self.DSP,
            "sram_mb": self.SRAM_MB,
            "model": self.model_name,
            "task": self.model_info["task"],
            "nominal_latency_ms": self.model_info["latency_ms"],
            "model_sram_mb": self.model_info["sram_mb"],
            "inference_count": self._inference_count,
        }
