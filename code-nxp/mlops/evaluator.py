"""TFLite inference + accuracy evaluation for NXP eIQ models."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class AccuracyGateError(Exception):
    """Raised when a model fails its accuracy gate."""


class TFLiteInferenceEngine:
    """Thin wrapper around TFLite runtime with optional delegate support."""

    def __init__(
        self,
        model_path: str,
        delegate_path: Optional[str] = None,
    ):
        try:
            import tflite_runtime.interpreter as tflite
        except ImportError:
            import tensorflow.lite as tflite

        if delegate_path:
            ext = tflite.load_delegate(delegate_path)
            self.interpreter = tflite.Interpreter(
                model_path=model_path,
                experimental_delegates=[ext],
            )
        else:
            self.interpreter = tflite.Interpreter(model_path=model_path)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def run(self, inputs: List[np.ndarray]) -> List[np.ndarray]:
        for i, tensor in enumerate(inputs):
            self.interpreter.set_tensor(self.input_details[i]["index"], tensor)
        self.interpreter.invoke()
        return [
            self.interpreter.get_tensor(d["index"])
            for d in self.output_details
        ]

    @property
    def input_shape(self) -> Tuple:
        return tuple(self.input_details[0]["shape"])

    @property
    def input_dtype(self):
        return self.input_details[0]["dtype"]


# ── Per-task metrics ─────────────────────────────────────────────────────────

def top1_accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    preds = np.argmax(logits.reshape(logits.shape[0], -1), axis=1)
    return float(np.mean(preds == labels))


def compute_miou(pred_masks: np.ndarray, gt_masks: np.ndarray, num_classes: int) -> float:
    ious = []
    for cls in range(num_classes):
        pred_c = (pred_masks == cls)
        gt_c = (gt_masks == cls)
        intersection = (pred_c & gt_c).sum()
        union = (pred_c | gt_c).sum()
        if union == 0:
            continue
        ious.append(intersection / union)
    return float(np.mean(ious)) if ious else 0.0


def compute_psnr(output: np.ndarray, target: np.ndarray) -> float:
    mse = np.mean((output.astype(np.float32) - target.astype(np.float32)) ** 2)
    if mse == 0:
        return float("inf")
    return float(10 * np.log10(255.0 ** 2 / mse))


def compute_wer(predicted: List[str], references: List[str]) -> float:
    total_words, total_errors = 0, 0
    for pred, ref in zip(predicted, references):
        ref_words = ref.split()
        pred_words = pred.split()
        total_words += len(ref_words)
        # Simple Levenshtein on word level
        dp = list(range(len(pred_words) + 1))
        for r in ref_words:
            new_dp = [dp[0] + 1]
            for j, p in enumerate(pred_words):
                new_dp.append(min(new_dp[-1] + 1, dp[j + 1] + 1, dp[j] + (0 if r == p else 1)))
            dp = new_dp
        total_errors += dp[-1]
    return total_errors / max(total_words, 1)


def compute_embedding_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
    e1 = emb1.flatten() / (np.linalg.norm(emb1) + 1e-8)
    e2 = emb2.flatten() / (np.linalg.norm(emb2) + 1e-8)
    return float(np.dot(e1, e2))  # cosine similarity


def compute_reconstruction_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
    return float(np.mean((original - reconstructed) ** 2))


# ── Evaluator ────────────────────────────────────────────────────────────────

class Evaluator:
    def __init__(self, config: dict):
        self.config = config
        self.gates = config.get("evaluation", {}).get("gates", {})

    def _gate_key(self, domain: str, task: str) -> str:
        return f"{domain}_{task}"

    def evaluate_classification(
        self,
        model_path: str,
        samples: List[Tuple[np.ndarray, int]],
        domain: str = "vision",
    ) -> Dict[str, float]:
        engine = TFLiteInferenceEngine(model_path)
        all_logits, labels = [], []
        for inp, label in samples:
            out = engine.run([inp])
            all_logits.append(out[0])
            labels.append(label)
        logits = np.concatenate(all_logits, axis=0)
        labels = np.array(labels)
        acc = top1_accuracy(logits, labels)
        return {"top1_accuracy": acc, "num_samples": len(samples)}

    def evaluate_segmentation(
        self,
        model_path: str,
        samples: List[Tuple[np.ndarray, np.ndarray]],
        num_classes: int = 21,
    ) -> Dict[str, float]:
        engine = TFLiteInferenceEngine(model_path)
        ious = []
        for inp, gt_mask in samples:
            out = engine.run([inp])
            pred = np.argmax(out[0].squeeze(), axis=-1)
            ious.append(compute_miou(pred, gt_mask, num_classes))
        return {"mIoU": float(np.mean(ious)), "num_samples": len(samples)}

    def evaluate_super_resolution(
        self,
        model_path: str,
        samples: List[Tuple[np.ndarray, np.ndarray]],
    ) -> Dict[str, float]:
        engine = TFLiteInferenceEngine(model_path)
        psnr_vals = []
        for inp, hr_target in samples:
            out = engine.run([inp])
            sr_out = (out[0].squeeze() * 255).clip(0, 255).astype(np.uint8)
            psnr_vals.append(compute_psnr(sr_out, hr_target))
        return {"psnr_db": float(np.mean(psnr_vals)), "num_samples": len(samples)}

    def evaluate_asr(
        self,
        model_path: str,
        samples: List[Tuple[np.ndarray, str]],
        decode_fn=None,
    ) -> Dict[str, float]:
        engine = TFLiteInferenceEngine(model_path)
        predictions, references = [], []
        alphabet = list("abcdefghijklmnopqrstuvwxyz '")
        for inp, ref_text in samples:
            out = engine.run([inp])
            logits = out[0].squeeze()  # (T, vocab)
            pred_indices = np.argmax(logits, axis=-1)
            # CTC greedy decode
            decoded = []
            prev = -1
            for idx in pred_indices:
                if idx != prev and idx < len(alphabet):
                    decoded.append(alphabet[idx])
                prev = idx
            predictions.append("".join(decoded).strip())
            references.append(ref_text)
        wer = compute_wer(predictions, references)
        return {"wer": wer, "num_samples": len(samples)}

    def evaluate_anomaly_detection(
        self,
        model_path: str,
        samples: List[Tuple[np.ndarray, int]],
    ) -> Dict[str, float]:
        engine = TFLiteInferenceEngine(model_path)
        scores, labels = [], []
        for inp, label in samples:
            out = engine.run([inp])
            rec_error = compute_reconstruction_error(inp, out[0])
            scores.append(rec_error)
            labels.append(label)
        # Simple AUC approximation via rank correlation
        pairs = list(zip(scores, labels))
        pairs.sort(key=lambda x: x[0], reverse=True)
        n_pos = sum(labels)
        n_neg = len(labels) - n_pos
        tp = fp = 0
        auc = 0.0
        prev_fp = 0
        for score, label in pairs:
            if label == 1:
                tp += 1
            else:
                fp += 1
                auc += tp * (fp - prev_fp)
                prev_fp = fp
        if n_pos > 0 and n_neg > 0:
            auc = auc / (n_pos * n_neg)
        return {"auc_roc": auc, "num_samples": len(samples)}

    def evaluate_eeg(
        self,
        model_path: str,
        samples: List[Tuple[np.ndarray, int]],
    ) -> Dict[str, float]:
        return self.evaluate_classification(model_path, samples, domain="misc")

    def check_gate(
        self,
        model_id: str,
        domain: str,
        task: str,
        metrics: Dict[str, float],
        raise_on_fail: bool = True,
    ) -> bool:
        key = self._gate_key(domain, task)
        gate = self.gates.get(key)
        if gate is None:
            print(f"[gate] No gate defined for '{key}', passing.")
            return True
        metric_name = gate["metric"]
        threshold = gate["threshold"]
        lower_is_better = gate.get("lower_is_better", False)
        value = metrics.get(metric_name, metrics.get(metric_name.replace("_", "")))
        if value is None:
            print(f"[gate] Metric '{metric_name}' not in results, skipping gate.")
            return True
        passed = (value <= threshold) if lower_is_better else (value >= threshold)
        direction = "≤" if lower_is_better else "≥"
        status = "PASS" if passed else "FAIL"
        print(
            f"[gate] {model_id} | {metric_name}={value:.4f} {direction} {threshold:.4f} → {status}"
        )
        if not passed and raise_on_fail:
            raise AccuracyGateError(
                f"Model '{model_id}' failed gate: {metric_name}={value:.4f}, threshold={threshold:.4f}"
            )
        return passed
