"""Shared preprocessing for ADI AI8X models (MAX78002 / MAX32690 / ADSP-SC835)."""
import numpy as np

# MAX78002 expects 3-channel uint8 input for vision models
MAX78002_IMAGE_SIZE = (64, 64)     # default for VWW / FaceNet lite
MAX78002_SRAM_KB = 5120            # 5 MB on-chip SRAM

# MFCC parameters for KWS (DS-CNN on MAX78002, keyword spotting on MAX32690)
SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 160
N_MELS = 64
N_FRAMES = 101  # ~1 second at 16 kHz, 10 ms hop


def preprocess_image(img: np.ndarray, size: tuple = (64, 64)) -> np.ndarray:
    """Resize to target and return NCHW float32 in [-128, 127] (ai8x convention)."""
    h, w = size
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    sh, sw = min(img.shape[0], h), min(img.shape[1], w)
    canvas[:sh, :sw] = img[:sh, :sw]
    # ai8x uses signed int8 range: scale [0,255] → [-128,127]
    tensor = (canvas.astype(np.float32) - 128.0)
    return tensor[np.newaxis].transpose(0, 3, 1, 2)  # NCHW


def compute_mfcc(waveform: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Simple mel-filterbank energy features (NumPy MFCC approximation).

    Returns shape (1, N_MELS, N_FRAMES) float32 in [-1, 1].
    """
    if waveform.dtype != np.float32:
        waveform = waveform.astype(np.float32) / 32768.0

    frames = []
    for i in range(N_FRAMES):
        start = i * HOP_LENGTH
        end = start + N_FFT
        frame = waveform[start:end] if end <= len(waveform) else np.zeros(N_FFT)
        frames.append(frame[:N_FFT])

    frames_arr = np.stack(frames, axis=0)  # (N_FRAMES, N_FFT)
    window = np.hanning(N_FFT)
    frames_arr = frames_arr * window

    spectrum = np.abs(np.fft.rfft(frames_arr, n=N_FFT)) ** 2  # (N_FRAMES, N_FFT//2+1)

    # Mel filterbank (triangular approximation)
    mel_min, mel_max = 0.0, 2595.0 * np.log10(1 + (sample_rate / 2) / 700.0)
    mel_points = np.linspace(mel_min, mel_max, N_MELS + 2)
    hz_points = 700.0 * (10 ** (mel_points / 2595.0) - 1)
    bin_points = np.floor(hz_points / sample_rate * N_FFT).astype(int)

    filterbank = np.zeros((N_MELS, N_FFT // 2 + 1), dtype=np.float32)
    for m in range(1, N_MELS + 1):
        lo, ctr, hi = bin_points[m - 1], bin_points[m], bin_points[m + 1]
        for k in range(lo, min(ctr, N_FFT // 2 + 1)):
            if ctr > lo:
                filterbank[m - 1, k] = (k - lo) / (ctr - lo)
        for k in range(ctr, min(hi, N_FFT // 2 + 1)):
            if hi > ctr:
                filterbank[m - 1, k] = (hi - k) / (hi - ctr)

    mel_energy = np.dot(spectrum, filterbank.T)  # (N_FRAMES, N_MELS)
    mel_energy = np.log(mel_energy + 1e-6)
    mel_energy = mel_energy.T  # (N_MELS, N_FRAMES)

    # Normalize to [-1, 1]
    mel_min_val, mel_max_val = mel_energy.min(), mel_energy.max()
    if mel_max_val > mel_min_val:
        mel_energy = 2.0 * (mel_energy - mel_min_val) / (mel_max_val - mel_min_val) - 1.0

    return mel_energy[np.newaxis].astype(np.float32)  # (1, N_MELS, N_FRAMES)


def preprocess_audio_segment(
    waveform: np.ndarray, sample_rate: int = SAMPLE_RATE, pad: bool = True
) -> np.ndarray:
    """Pad/trim waveform to exactly 1 second and compute MFCC."""
    target_len = sample_rate
    if len(waveform) < target_len and pad:
        waveform = np.pad(waveform, (0, target_len - len(waveform)))
    else:
        waveform = waveform[:target_len]
    return compute_mfcc(waveform, sample_rate)
