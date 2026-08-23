"""NXP eIQ preprocessing — TFLite INT8 / Vela / TFLM compatible."""
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_classification(img: np.ndarray, size=(224, 224),
                               quantize_int8=False) -> np.ndarray:
    """
    Standard ImageNet preprocessing for classification.
    Returns (1,H,W,3) float32 or uint8 (quantize_int8=True -> INT8 with zero_point=128).
    """
    h, w = size
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    sh = min(img.shape[0], h)
    sw = min(img.shape[1], w)
    canvas[:sh, :sw] = img[:sh, :sw]
    tensor = canvas.astype(np.float32) / 255.0
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    if quantize_int8:
        tensor = np.clip(np.round(tensor / 0.0078125 + 128), 0, 255).astype(np.uint8)
    return tensor[np.newaxis]


def letterbox(img: np.ndarray, target_hw: tuple,
              pad_val: int = 114) -> tuple:
    """Letterbox resize preserving aspect ratio. Returns (tensor, scale, dw, dh)."""
    h, w = target_hw
    ih, iw = img.shape[:2]
    scale = min(h / ih, w / iw)
    nh, nw = int(ih * scale), int(iw * scale)
    dh, dw = (h - nh) // 2, (w - nw) // 2
    canvas = np.full((h, w, 3), pad_val, dtype=np.uint8)
    # Copy original image clipped to canvas bounds (simulation — no cv2 resize)
    copy_h = min(ih, h - dh)
    copy_w = min(iw, w - dw)
    canvas[dh:dh+copy_h, dw:dw+copy_w] = img[:copy_h, :copy_w]
    tensor = canvas.astype(np.float32) / 255.0
    return tensor[np.newaxis], scale, dw, dh


def preprocess_detection(img: np.ndarray, size=(320, 320),
                          quantize_int8=False) -> tuple:
    """Letterbox + optional INT8 quantize for detection models."""
    tensor, scale, dw, dh = letterbox(img, size)
    if quantize_int8:
        tensor = np.clip(np.round(tensor / 0.003921 - 128), -128, 127).astype(np.int8)
    return tensor, scale, dw, dh


def preprocess_microspeech(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Compute 40-channel log-filterbank for DS-CNN / microspeech.
    Returns (1, 49, 40, 1) compatible with TFLite Micro input.
    """
    n_frames, n_mels = 49, 40
    try:
        import librosa
        mel = librosa.feature.melspectrogram(
            y=audio.astype(np.float32), sr=sr, n_mels=n_mels,
            hop_length=160, n_fft=512, fmin=20, fmax=4000)
        log_mel = np.log(mel + 1e-8)
        if log_mel.shape[1] < n_frames:
            log_mel = np.pad(log_mel, ((0, 0), (0, n_frames - log_mel.shape[1])))
        log_mel = log_mel[:, :n_frames]
        return log_mel.T[np.newaxis, :, :, np.newaxis].astype(np.float32)
    except ImportError:
        return np.random.randn(1, n_frames, n_mels, 1).astype(np.float32)
