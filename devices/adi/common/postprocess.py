"""Shared postprocessing for ADI AI8X models."""
import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits.flatten().astype(np.float64)
    e = np.exp(logits - logits.max())
    return (e / e.sum()).astype(np.float32)


def topk_classes(probs: np.ndarray, k: int = 5, labels: list = None) -> list:
    probs = probs.flatten()
    indices = np.argsort(probs)[::-1][:k]
    results = []
    for i in indices:
        entry = {"class_id": int(i), "score": float(probs[i])}
        if labels is not None and int(i) < len(labels):
            entry["label"] = labels[int(i)]
        results.append(entry)
    return results


def decode_kws_output(logits: np.ndarray, keyword_labels: list) -> dict:
    """Decode keyword spotting output to label + confidence."""
    probs = softmax(logits)
    idx = int(np.argmax(probs))
    label = keyword_labels[idx] if idx < len(keyword_labels) else f"class_{idx}"
    return {
        "keyword": label,
        "confidence": float(probs[idx]),
        "class_id": idx,
        "probs": probs.tolist(),
    }


def decode_detection_output(output: np.ndarray, conf_thresh: float = 0.4) -> list:
    """Simple anchor-free detection decoder for MAX78002 TinierSSD / FPN output."""
    preds = output.reshape(-1, output.shape[-1])
    results = []
    for row in preds:
        conf = float(row[4]) if len(row) > 4 else float(row[0])
        if conf < conf_thresh:
            continue
        if len(row) >= 5:
            x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
        else:
            x1, y1, x2, y2 = 0.0, 0.0, 1.0, 1.0
        cls_id = int(np.argmax(row[5:])) if len(row) > 5 else 0
        results.append({"class_id": cls_id, "score": conf, "bbox": [x1, y1, x2, y2]})
    return results


def energy_estimate_uj(latency_ms: float, power_mw: float) -> float:
    """Convert latency + average power to energy in microjoules."""
    return latency_ms * power_mw  # ms × mW = µJ


def battery_life_hours(energy_uj_per_inference: float, inferences_per_hour: int,
                       battery_mah: float, voltage_v: float = 3.0) -> float:
    """Estimate battery life given energy cost and inference rate."""
    battery_uj = battery_mah * 3600.0 * 1000.0 * voltage_v  # mAh → µJ
    total_uj = energy_uj_per_inference * inferences_per_hour * battery_life_hours.__defaults__[0]
    # Iterative: assume 8760 hours/year
    energy_per_hour = energy_uj_per_inference * inferences_per_hour
    if energy_per_hour <= 0:
        return float("inf")
    return battery_uj / energy_per_hour
