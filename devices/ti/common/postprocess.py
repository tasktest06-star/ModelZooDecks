"""Shared postprocessing for TI EdgeAI models."""
import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits.flatten().astype(np.float64)
    e = np.exp(logits - logits.max())
    return (e / e.sum()).astype(np.float32)


def topk_classes(probs: np.ndarray, k: int = 5) -> list:
    probs = probs.flatten()
    indices = np.argsort(probs)[::-1][:k]
    return [{"class_id": int(i), "score": float(probs[i])} for i in indices]


def decode_yolox_output(output: np.ndarray, input_hw: tuple, conf_thresh: float = 0.3) -> list:
    """Decode flat YOLOX output [N, 5+C] or similar; returns list of detection dicts."""
    preds = output.reshape(-1, output.shape[-1])
    results = []
    h, w = input_hw
    for row in preds:
        obj_conf = float(row[4])
        if obj_conf < conf_thresh:
            continue
        cls_scores = row[5:] * obj_conf
        cls_id = int(np.argmax(cls_scores))
        score = float(cls_scores[cls_id])
        if score < conf_thresh:
            continue
        cx, cy, bw, bh = row[:4]
        x1 = float(cx - bw / 2) / w
        y1 = float(cy - bh / 2) / h
        x2 = float(cx + bw / 2) / w
        y2 = float(cy + bh / 2) / h
        results.append({"class_id": cls_id, "score": score, "bbox": [x1, y1, x2, y2]})
    return results


def argmax_segmentation(output: np.ndarray) -> np.ndarray:
    """Convert segmentation logits to per-pixel class map (uint8)."""
    if output.ndim == 4:  # NCHW
        output = output[0]
    if output.ndim == 3:  # CHW → class map
        return np.argmax(output, axis=0).astype(np.uint8)
    return output.astype(np.uint8)


def decode_pose_output(heatmaps: np.ndarray, orig_hw: tuple, input_hw: tuple) -> list:
    """Decode heatmap-based pose output to keypoints [(x, y, conf)]."""
    if heatmaps.ndim == 4:
        heatmaps = heatmaps[0]
    oh, ow = orig_hw
    keypoints = []
    for kp_map in heatmaps:
        idx = np.argmax(kp_map)
        ky, kx = np.unravel_index(idx, kp_map.shape)
        conf = float(kp_map[ky, kx])
        x = float(kx) / kp_map.shape[1] * ow
        y = float(ky) / kp_map.shape[0] * oh
        keypoints.append({"x": x, "y": y, "conf": conf})
    return keypoints


def decode_depth_output(output: np.ndarray) -> np.ndarray:
    """Normalize depth output to [0, 1] float32 map."""
    d = output.squeeze().astype(np.float32)
    dmin, dmax = d.min(), d.max()
    if dmax > dmin:
        d = (d - dmin) / (dmax - dmin)
    return d
