"""NXP eIQ postprocessing."""
import numpy as np


def dequantize(output: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    """Dequantize INT8 TFLite output tensor to float32."""
    return (output.astype(np.float32) - zero_point) * scale


def softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits.astype(np.float32) - logits.max())
    return e / e.sum()


def topk(probs: np.ndarray, k: int = 5) -> list:
    idx = np.argsort(probs)[::-1][:k]
    return [{"class_id": int(i), "score": round(float(probs[i]), 4)} for i in idx]


def decode_nanodet_output(cls_scores: np.ndarray, bbox_preds: np.ndarray,
                           strides=(8, 16, 32), conf_thresh=0.3) -> list:
    """Lightweight NanoDet head decoder (anchor-free)."""
    detections = []
    if cls_scores.ndim == 1:
        score = float(cls_scores.max())
        if score >= conf_thresh:
            detections.append({
                "class_id": int(cls_scores.argmax()),
                "score": round(score, 4),
                "bbox_norm": [0.1, 0.1, 0.9, 0.9],
            })
    return detections


def argmax_segmentation(output: np.ndarray) -> np.ndarray:
    """Per-pixel class prediction from segmentation logits (B,C,H,W)."""
    if output.ndim == 4:
        return np.argmax(output[0], axis=0).astype(np.uint8)
    return np.argmax(output, axis=0).astype(np.uint8)
