"""
data_pipeline.py — Domain-aware preprocessing for ADI Model Zoo.

Covers all 3 domains and their task-specific transforms:
  Vision  : resize, AI8X normalize [-128,127], optional channel fold
  Audio   : STFT/Bark features (DTLN/RNNoise), Mel spectrogram (GenreNet/KWS)
  Sensor  : FFT slice normalization (motor fault), log-Mel (anomaly)

All preprocessing uses NumPy only (no cv2/PIL required for basic ops).
"""

from __future__ import annotations

import math
from typing import Iterator, Optional, Tuple

import numpy as np

# ── Per-task defaults ─────────────────────────────────────────────────────────

VISION_TASK_DEFAULTS = {
    "image_classification": {
        "input_size": (32, 32),
        "normalize": "ai8x",          # [-128, 127] INT8 range
        "fold_ratio": 1,
    },
    "image_segmentation": {
        "input_size": (48, 48),
        "normalize": "ai8x",
        "fold_ratio": 4,               # 3-ch → 48-ch folded
    },
    "object_detection": {
        "input_size": (256, 320),
        "normalize": "ai8x",
        "fold_ratio": 1,
    },
    "visual_wake_word": {
        "input_size": (50, 50),
        "normalize": "int8_unsigned",  # [0, 255] → int8
        "grayscale": True,
        "fold_ratio": 1,
    },
}

AUDIO_TASK_DEFAULTS = {
    "audio_denoising": {
        "sample_rate": 16000,
        "frame_size": 512,
        "hop_size": 256,
        "n_fft": 512,
    },
    "audio_genre_identification": {
        "sample_rate": 22050,
        "n_mels": 128,
        "n_fft": 2048,
        "hop_length": 512,
        "duration_s": 3.0,
    },
    "keyword_spotting": {
        "sample_rate": 16000,
        "n_mfcc": 10,
        "n_mels": 40,
        "window_ms": 40,
        "hop_ms": 20,
        "duration_s": 1.0,
    },
}

SENSOR_TASK_DEFAULTS = {
    "anomaly_detection": {
        "sample_rate": 16000,
        "n_mels": 64,
        "n_fft": 1024,
        "hop_length": 512,
        "n_frames": 196,
        "n_bins": 64,
    },
    "motor_fault_detection": {
        "window_size": 256,
        "n_axes": 3,
        "normalize": "int8",
    },
}


# ── Vision preprocessing ──────────────────────────────────────────────────────

def _resize_nearest(img: np.ndarray, h: int, w: int) -> np.ndarray:
    """Nearest-neighbour resize without external deps."""
    oh, ow = img.shape[:2]
    row_idx = (np.arange(h) * oh / h).astype(np.int32)
    col_idx = (np.arange(w) * ow / w).astype(np.int32)
    return img[row_idx][:, col_idx]


def ai8x_normalize(img: np.ndarray, act_mode_8bit: bool = True) -> np.ndarray:
    """
    Convert uint8 [0,255] image to AI8X INT8 range.
    act_mode_8bit=True  → [-128, 127] (integer)
    act_mode_8bit=False → [-1.0, 127/128] (float normalized)
    """
    scaled = img.astype(np.float32) / 255.0       # [0, 1]
    scaled = scaled - 0.5                         # [-0.5, 0.5]
    scaled = scaled * 256.0                       # [-128, 128]
    clipped = np.clip(scaled, -128, 127)
    if act_mode_8bit:
        return np.round(clipped).astype(np.float32)
    return (clipped / 128.0).astype(np.float32)


def channel_fold(img: np.ndarray, fold_ratio: int) -> np.ndarray:
    """
    Interlaced channel folding for MAX78002 spatial compression.
    Input:  (H, W, C) — uint8 or float
    Output: (H//r, W//r, C*r*r) where r = fold_ratio

    Used by U-Net (fold_ratio=4): 3-channel 192×192 → 48-channel 48×48.
    """
    if fold_ratio == 1:
        return img
    h, w, c = img.shape
    out_h, out_w = h // fold_ratio, w // fold_ratio
    out_c = c * fold_ratio * fold_ratio
    folded = np.zeros((out_h, out_w, out_c), dtype=img.dtype)
    ch_idx = 0
    for i in range(fold_ratio):
        for j in range(fold_ratio):
            folded[:, :, ch_idx * c:(ch_idx + 1) * c] = img[i::fold_ratio, j::fold_ratio, :]
            ch_idx += 1
    return folded


def preprocess_vision(
    img: np.ndarray,
    task: str,
    input_size: Optional[Tuple[int, int]] = None,
    fold_ratio: Optional[int] = None,
    grayscale: bool = False,
    act_mode_8bit: bool = True,
) -> np.ndarray:
    """
    Preprocess a uint8 HWC image for AI8X / TFLite vision models.
    Returns NCHW float32 tensor ready for inference.
    """
    if task not in VISION_TASK_DEFAULTS:
        raise ValueError(f"Unknown vision task '{task}'. "
                         f"Known: {list(VISION_TASK_DEFAULTS.keys())}")

    defaults = VISION_TASK_DEFAULTS[task]
    h, w = input_size or defaults["input_size"]
    fold = fold_ratio if fold_ratio is not None else defaults["fold_ratio"]
    use_gray = grayscale or defaults.get("grayscale", False)

    if use_gray and img.ndim == 3 and img.shape[2] == 3:
        img = (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
               ).astype(np.uint8)[:, :, np.newaxis]

    img_r = _resize_nearest(img, h * fold, w * fold) if fold > 1 else _resize_nearest(img, h, w)

    if fold > 1:
        img_r = channel_fold(img_r, fold)

    img_norm = ai8x_normalize(img_r, act_mode_8bit=act_mode_8bit)

    # HWC → CHW
    if img_norm.ndim == 3:
        tensor = img_norm.transpose(2, 0, 1)
    else:
        tensor = img_norm[np.newaxis, :, :]

    return tensor[np.newaxis].astype(np.float32)


# ── Audio preprocessing ───────────────────────────────────────────────────────

def compute_stft_magnitude(
    audio: np.ndarray,
    n_fft: int = 512,
    hop_size: int = 256,
) -> np.ndarray:
    """
    Simple STFT magnitude for DTLN preprocessing.
    Returns (n_frames, n_fft//2 + 1) float32.
    """
    n_freq = n_fft // 2 + 1
    frames = []
    window = np.hanning(n_fft).astype(np.float32)
    for start in range(0, len(audio) - n_fft + 1, hop_size):
        frame = audio[start:start + n_fft] * window
        spec = np.fft.rfft(frame, n=n_fft)
        frames.append(np.abs(spec).astype(np.float32))
    if not frames:
        return np.zeros((1, n_freq), dtype=np.float32)
    return np.stack(frames)


def compute_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int = 22050,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    duration_s: float = 3.0,
) -> np.ndarray:
    """
    Log-Mel spectrogram for GenreNet / KWS preprocessing.
    Returns (1, n_mels, n_frames) float32, fixed duration (zero-padded/truncated).
    """
    target_len = int(sample_rate * duration_s)
    if len(audio) > target_len:
        audio = audio[:target_len]
    elif len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))

    n_frames = 1 + (target_len - n_fft) // hop_length
    window = np.hanning(n_fft).astype(np.float32)

    # Build mel filterbank (triangular, Hz-space)
    hz_min, hz_max = 0.0, sample_rate / 2.0
    mel_min = 2595 * math.log10(1 + hz_min / 700)
    mel_max = 2595 * math.log10(1 + hz_max / 700)
    mel_pts = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_pts = 700 * (10 ** (mel_pts / 2595) - 1)
    bin_pts = np.floor((n_fft + 1) * hz_pts / sample_rate).astype(int)

    filterbank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        for k in range(bin_pts[m - 1], bin_pts[m]):
            filterbank[m - 1, k] = (k - bin_pts[m - 1]) / (bin_pts[m] - bin_pts[m - 1] + 1e-8)
        for k in range(bin_pts[m], bin_pts[m + 1] + 1):
            filterbank[m - 1, k] = (bin_pts[m + 1] - k) / (bin_pts[m + 1] - bin_pts[m] + 1e-8)

    mel = np.zeros((n_mels, n_frames), dtype=np.float32)
    for i in range(n_frames):
        start = i * hop_length
        frame = audio[start:start + n_fft].astype(np.float32) * window
        fft_mag = np.abs(np.fft.rfft(frame, n=n_fft))
        mel[:, i] = filterbank @ fft_mag

    log_mel = np.log(np.maximum(mel, 1e-9))
    return log_mel[np.newaxis].astype(np.float32)


def preprocess_audio(
    audio: np.ndarray,
    task: str,
    sample_rate: int = 16000,
) -> np.ndarray:
    """
    Preprocess a float32 audio array for the given task.
    Returns a tensor appropriate for the target model.
    """
    if task == "audio_denoising":
        defaults = AUDIO_TASK_DEFAULTS["audio_denoising"]
        return compute_stft_magnitude(audio, defaults["n_fft"], defaults["hop_size"])

    if task == "audio_genre_identification":
        defaults = AUDIO_TASK_DEFAULTS["audio_genre_identification"]
        return compute_mel_spectrogram(
            audio, sample_rate, defaults["n_mels"],
            defaults["n_fft"], defaults["hop_length"], defaults["duration_s"]
        )

    if task == "keyword_spotting":
        defaults = AUDIO_TASK_DEFAULTS["keyword_spotting"]
        return compute_mel_spectrogram(
            audio, sample_rate, defaults["n_mels"],
            n_fft=int(defaults["window_ms"] * sample_rate / 1000),
            hop_length=int(defaults["hop_ms"] * sample_rate / 1000),
            duration_s=defaults["duration_s"],
        )

    raise ValueError(f"Unknown audio task '{task}'")


# ── Sensor preprocessing ──────────────────────────────────────────────────────

def preprocess_sensor(
    signal: np.ndarray,
    task: str,
    sample_rate: int = 16000,
) -> np.ndarray:
    """
    Preprocess time-series sensor data.
    motor_fault_detection : expects (N, 3) raw vibration → (1, 256, 3) int8
    anomaly_detection     : expects (N,) audio → (1, 196, 64) log-mel spectrogram
    """
    if task == "motor_fault_detection":
        defaults = SENSOR_TASK_DEFAULTS["motor_fault_detection"]
        w = defaults["window_size"]
        if signal.shape[0] < w:
            signal = np.pad(signal, ((0, w - signal.shape[0]), (0, 0)))
        window = signal[:w].astype(np.float32)
        # Normalize to int8 range
        window = np.clip(window / (np.max(np.abs(window)) + 1e-9) * 127, -128, 127)
        return window[np.newaxis].astype(np.float32)   # (1, 256, 3)

    if task == "anomaly_detection":
        defaults = SENSOR_TASK_DEFAULTS["anomaly_detection"]
        mel = compute_mel_spectrogram(
            signal, sample_rate,
            n_mels=defaults["n_bins"],
            n_fft=defaults["n_fft"],
            hop_length=defaults["hop_length"],
            duration_s=defaults["n_frames"] * defaults["hop_length"] / sample_rate,
        )
        return mel   # (1, n_bins, n_frames)

    raise ValueError(f"Unknown sensor task '{task}'")


# ── Unified entry point ───────────────────────────────────────────────────────

def preprocess(
    data: np.ndarray,
    domain: str,
    task: str,
    **kwargs,
) -> np.ndarray:
    """
    Unified preprocessing entry point.
    domain: "vision" | "audio" | "sensor"
    """
    if domain == "vision":
        return preprocess_vision(data, task, **kwargs)
    if domain == "audio":
        return preprocess_audio(data, task, **kwargs)
    if domain == "sensor":
        return preprocess_sensor(data, task, **kwargs)
    raise ValueError(f"Unknown domain '{domain}'. Use: vision, audio, sensor")


# ── DataPipeline class ────────────────────────────────────────────────────────

class DataPipeline:
    """
    Minimal data loader wrapping preprocessing for eval/test loops.

    Yields (input_tensor, label) tuples. For vision: loads .jpg/.png from
    dataset_root/<split>/. For audio: loads .wav/.mp3. For sensor: loads .csv.
    """

    SUPPORTED_DOMAINS = ("vision", "audio", "sensor")
    SUPPORTED_TASKS = (
        list(VISION_TASK_DEFAULTS.keys()) +
        list(AUDIO_TASK_DEFAULTS.keys()) +
        list(SENSOR_TASK_DEFAULTS.keys())
    )

    def __init__(
        self,
        domain: str,
        task: str,
        dataset_root: Optional[str] = None,
        num_frames: int = 100,
        **preproc_kwargs,
    ):
        if domain not in self.SUPPORTED_DOMAINS:
            raise ValueError(f"Unknown domain '{domain}'. Use: {self.SUPPORTED_DOMAINS}")
        all_tasks = (list(VISION_TASK_DEFAULTS) +
                     list(AUDIO_TASK_DEFAULTS) +
                     list(SENSOR_TASK_DEFAULTS))
        if task not in all_tasks:
            raise ValueError(f"Unknown task '{task}'. Known tasks: {all_tasks}")

        self.domain = domain
        self.task = task
        self.dataset_root = dataset_root
        self.num_frames = num_frames
        self.preproc_kwargs = preproc_kwargs

    def get_dataloader(self, split: str = "val", batch_size: int = 1) -> Iterator:
        from pathlib import Path
        if not self.dataset_root:
            raise FileNotFoundError("dataset_root not set")
        root = Path(self.dataset_root) / split
        if not root.exists():
            raise FileNotFoundError(f"Dataset split directory not found: {root}")

        ext_map = {"vision": (".jpg", ".png", ".jpeg"),
                   "audio": (".wav", ".mp3"),
                   "sensor": (".csv", ".npy")}
        exts = ext_map[self.domain]
        files = sorted(f for f in root.rglob("*") if f.suffix.lower() in exts)
        files = files[:self.num_frames]

        batch_inputs, batch_labels = [], []
        for fpath in files:
            data, label = self._load_file(fpath)
            tensor = preprocess(data, self.domain, self.task, **self.preproc_kwargs)
            batch_inputs.append(tensor)
            batch_labels.append(label)
            if len(batch_inputs) == batch_size:
                yield np.concatenate(batch_inputs, axis=0), batch_labels
                batch_inputs, batch_labels = [], []
        if batch_inputs:
            yield np.concatenate(batch_inputs, axis=0), batch_labels

    def _load_file(self, fpath):
        import os
        label = fpath.parent.name
        ext = fpath.suffix.lower()
        if ext in (".jpg", ".png", ".jpeg"):
            # Pure numpy fallback (no cv2/PIL)
            try:
                from PIL import Image
                img = np.array(Image.open(fpath).convert("RGB"), dtype=np.uint8)
            except ImportError:
                img = np.zeros((64, 64, 3), dtype=np.uint8)
        elif ext in (".wav",):
            try:
                import wave
                with wave.open(str(fpath)) as wf:
                    frames = wf.readframes(wf.getnframes())
                    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                img = audio
            except Exception:
                img = np.zeros(16000, dtype=np.float32)
        elif ext == ".npy":
            img = np.load(fpath)
        elif ext == ".csv":
            img = np.loadtxt(fpath, delimiter=",", dtype=np.float32)
        else:
            img = np.zeros((64, 64, 3), dtype=np.uint8)
        return img, label
