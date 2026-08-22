"""Multi-domain data preprocessing for NXP eIQ Model Zoo."""

import io
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np


def letterbox(
    image: np.ndarray,
    target_size: Tuple[int, int],
    pad_value: int = 114,
) -> np.ndarray:
    h, w = image.shape[:2]
    th, tw = target_size
    scale = min(tw / w, th / h)
    new_w, new_h = int(w * scale), int(h * scale)
    # Resize (pure numpy — no PIL dependency required at import time)
    from PIL import Image as PILImage
    pil = PILImage.fromarray(image).resize((new_w, new_h), PILImage.BILINEAR)
    canvas = np.full((th, tw, 3), pad_value, dtype=np.uint8)
    pad_top = (th - new_h) // 2
    pad_left = (tw - new_w) // 2
    canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = np.array(pil)
    return canvas


def normalize_imagenet(image: np.ndarray) -> np.ndarray:
    """ImageNet mean/std normalization → float32 in [-1, 1] range expected by MobileNet family."""
    img = image.astype(np.float32) / 127.5 - 1.0
    return img


def normalize_float(image: np.ndarray) -> np.ndarray:
    """Scale to [0, 1]."""
    return image.astype(np.float32) / 255.0


def preprocess_vision_classification(
    image: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
    normalize: str = "imagenet",
) -> np.ndarray:
    from PIL import Image as PILImage
    pil = PILImage.fromarray(image).resize(target_size, PILImage.BILINEAR)
    arr = np.array(pil, dtype=np.float32)
    if normalize == "imagenet":
        arr = normalize_imagenet(arr)
    else:
        arr = normalize_float(arr)
    return np.expand_dims(arr, 0)  # (1, H, W, 3)


def preprocess_vision_detection(
    image: np.ndarray,
    target_size: Tuple[int, int] = (320, 320),
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    lb = letterbox(image, target_size)
    arr = lb.astype(np.float32) / 127.5 - 1.0
    scale = min(target_size[1] / image.shape[1], target_size[0] / image.shape[0])
    return np.expand_dims(arr, 0), scale, (image.shape[0], image.shape[1])


def preprocess_vision_segmentation(
    image: np.ndarray,
    target_size: Tuple[int, int] = (513, 513),
) -> np.ndarray:
    from PIL import Image as PILImage
    pil = PILImage.fromarray(image).resize(target_size, PILImage.BILINEAR)
    arr = np.array(pil, dtype=np.float32)
    arr = normalize_imagenet(arr)
    return np.expand_dims(arr, 0)


def preprocess_super_resolution(
    image: np.ndarray,
    target_size: Tuple[int, int] = (128, 128),
) -> np.ndarray:
    from PIL import Image as PILImage
    pil = PILImage.fromarray(image).resize(target_size, PILImage.BILINEAR)
    arr = np.array(pil, dtype=np.float32) / 255.0
    return np.expand_dims(arr, 0)


def preprocess_low_light(
    image: np.ndarray,
) -> np.ndarray:
    """SCI has dynamic shape — pass image at original resolution."""
    arr = image.astype(np.float32) / 255.0
    return np.expand_dims(arr, 0)


def preprocess_monocular_depth(
    image: np.ndarray,
    target_size: Tuple[int, int] = (256, 256),
) -> np.ndarray:
    from PIL import Image as PILImage
    pil = PILImage.fromarray(image).resize(target_size, PILImage.BILINEAR)
    arr = np.array(pil, dtype=np.float32)
    arr = normalize_imagenet(arr)
    return np.expand_dims(arr, 0)


def preprocess_face_recognition(
    face_crop: np.ndarray,
    target_size: Tuple[int, int] = (160, 160),
) -> np.ndarray:
    from PIL import Image as PILImage
    pil = PILImage.fromarray(face_crop).resize(target_size, PILImage.BILINEAR)
    arr = np.array(pil, dtype=np.float32)
    # Facenet: subtract 127.5 and divide by 128
    arr = (arr - 127.5) / 128.0
    return np.expand_dims(arr, 0)


# ── Audio preprocessing ──────────────────────────────────────────────────────

def compute_mfcc(
    audio: np.ndarray,
    sample_rate: int = 16000,
    n_mfcc: int = 10,
    n_frames: int = 49,
) -> np.ndarray:
    """Minimal MFCC for DS-CNN KWS: (1, n_frames, n_mfcc, 1)."""
    try:
        import librosa
        mfcc = librosa.feature.mfcc(
            y=audio.astype(np.float32),
            sr=sample_rate,
            n_mfcc=n_mfcc,
            n_fft=512,
            hop_length=160,
            win_length=400,
        )
        mfcc = mfcc[:, :n_frames]
        if mfcc.shape[1] < n_frames:
            mfcc = np.pad(mfcc, ((0, 0), (0, n_frames - mfcc.shape[1])))
        mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-6)
        return mfcc.T[np.newaxis, :, :, np.newaxis].astype(np.float32)
    except ImportError:
        frames = np.zeros((1, n_frames, n_mfcc, 1), dtype=np.float32)
        return frames


def compute_wav2letter_features(
    audio: np.ndarray,
    sample_rate: int = 16000,
    n_mfcc: int = 39,
    n_frames: int = 296,
) -> np.ndarray:
    """MFCC features for Wav2Letter: (1, n_frames, n_mfcc)."""
    try:
        import librosa
        mfcc = librosa.feature.mfcc(
            y=audio.astype(np.float32),
            sr=sample_rate,
            n_mfcc=n_mfcc,
            n_fft=512,
            hop_length=160,
        )
        mfcc = mfcc[:, :n_frames]
        if mfcc.shape[1] < n_frames:
            mfcc = np.pad(mfcc, ((0, 0), (0, n_frames - mfcc.shape[1])))
        return mfcc.T[np.newaxis].astype(np.float32)
    except ImportError:
        return np.zeros((1, n_frames, n_mfcc), dtype=np.float32)


def compute_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int = 16000,
    n_mels: int = 128,
    n_frames: int = 64,
) -> np.ndarray:
    """Mel spectrogram for anomaly detection autoencoder."""
    try:
        import librosa
        mel = librosa.feature.melspectrogram(
            y=audio.astype(np.float32),
            sr=sample_rate,
            n_mels=n_mels,
            n_fft=1024,
            hop_length=512,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_db = mel_db[:, :n_frames]
        if mel_db.shape[1] < n_frames:
            mel_db = np.pad(mel_db, ((0, 0), (0, n_frames - mel_db.shape[1])))
        mel_flat = mel_db.flatten()[np.newaxis].astype(np.float32)
        return mel_flat
    except ImportError:
        return np.zeros((1, n_mels * n_frames), dtype=np.float32)


# ── EEG preprocessing ────────────────────────────────────────────────────────

def preprocess_eeg(
    eeg_signal: np.ndarray,
    n_channels: int = 22,
    n_samples: int = 1125,
) -> np.ndarray:
    """Normalize EEG trial to (1, 1, n_channels, n_samples)."""
    if eeg_signal.ndim == 2:
        data = eeg_signal[:n_channels, :n_samples]
    else:
        data = eeg_signal
    # z-score per channel
    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True) + 1e-8
    data = (data - mean) / std
    # pad if short
    if data.shape[1] < n_samples:
        data = np.pad(data, ((0, 0), (0, n_samples - data.shape[1])))
    return data[np.newaxis, np.newaxis].astype(np.float32)  # (1,1,22,1125)


# ── Unified dispatcher ───────────────────────────────────────────────────────

def preprocess(task: str, data: np.ndarray, **kwargs) -> np.ndarray:
    """Route preprocessing by task name."""
    dispatch = {
        "image_classification": preprocess_vision_classification,
        "object_detection": lambda d, **kw: preprocess_vision_detection(d, **kw)[0],
        "semantic_segmentation": preprocess_vision_segmentation,
        "instance_segmentation": preprocess_vision_segmentation,
        "super_resolution": preprocess_super_resolution,
        "low_light_enhancement": preprocess_low_light,
        "monocular_depth": preprocess_monocular_depth,
        "face_recognition": preprocess_face_recognition,
        "pose_estimation": preprocess_vision_classification,
        "keyword_spotting": compute_mfcc,
        "speech_recognition": compute_wav2letter_features,
        "anomaly_detection": compute_mel_spectrogram,
        "eeg_classification": preprocess_eeg,
    }
    fn = dispatch.get(task)
    if fn is None:
        raise ValueError(f"Unknown task: {task}")
    return fn(data, **kwargs)
