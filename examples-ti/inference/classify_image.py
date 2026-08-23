"""Single-image classification using TI EdgeAI models via ONNX or TFLite runtime."""

import argparse
import time
import numpy as np
import cv2
from pathlib import Path
import json


IMAGENET_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
IMAGENET_STD  = np.array([58.395,  57.12,  57.375], dtype=np.float32)

MODELS = {
    "mobilenet_v2_lite":       {"input": (224, 224), "format": "tflite", "top1": 72.3},
    "mobilenet_v3_small_lite": {"input": (224, 224), "format": "tflite", "top1": 67.2},
    "mobilenet_v3_large_lite": {"input": (224, 224), "format": "tflite", "top1": 74.8},
    "efficientnet_b0_lite":    {"input": (224, 224), "format": "tflite", "top1": 76.3},
    "fastvit_s12":             {"input": (256, 256), "format": "onnx",   "top1": 79.3},
    "swin_tiny":               {"input": (224, 224), "format": "onnx",   "top1": 81.2},
    "resnet18_lite":           {"input": (224, 224), "format": "tflite", "top1": 71.5},
    "regnet_x_400mf_lite":     {"input": (224, 224), "format": "tflite", "top1": 74.1},
}

SOC_NOTES = {
    "AM62A":  "Entry-level, 1 TOPS — best with mobilenet_v2_lite / mobilenet_v3_small_lite",
    "AM67A":  "Mid-range, 4 TOPS — optimal for mobilenet_v3_large_lite / regnet families",
    "AM68A":  "High-perf, 8 TOPS — supports all models including efficientnet, fastvit",
    "AM69A":  "Premium, 32 TOPS — ideal for swin_tiny and transformer families",
    "TDA4VM": "Automotive ASIL-B, 8 TOPS — use with peoplenet_lite for safety apps",
}


def preprocess(image_path: str, input_size: tuple) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, input_size)
    img = img.astype(np.float32)
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img[np.newaxis]  # (1, H, W, 3)


def load_labels(labels_path: str = None) -> list:
    if labels_path and Path(labels_path).exists():
        with open(labels_path) as f:
            return [line.strip() for line in f]
    return [f"class_{i}" for i in range(1000)]


def run_tflite(model_path: str, input_data: np.ndarray) -> tuple:
    try:
        import tflite_runtime.interpreter as tflite
        Interpreter = tflite.Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter

    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    if inp["dtype"] == np.int8:
        scale, zero_point = inp["quantization"]
        input_data = (input_data / scale + zero_point).astype(np.int8)

    interpreter.set_tensor(inp["index"], input_data)
    t0 = time.perf_counter()
    interpreter.invoke()
    latency_ms = (time.perf_counter() - t0) * 1000

    output = interpreter.get_tensor(out["index"])
    if out["dtype"] == np.int8:
        scale, zero_point = out["quantization"]
        output = (output.astype(np.float32) - zero_point) * scale

    return output[0], latency_ms


def run_onnx(model_path: str, input_data: np.ndarray) -> tuple:
    import onnxruntime as ort
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name

    if input_data.shape[-1] == 3:
        input_data = input_data.transpose(0, 3, 1, 2)

    t0 = time.perf_counter()
    output = sess.run(None, {inp_name: input_data})[0]
    latency_ms = (time.perf_counter() - t0) * 1000

    return output[0], latency_ms


def classify(image_path: str, model_name: str, model_path: str,
             soc: str, labels_path: str = None, top_k: int = 5) -> dict:
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Choose from: {list(MODELS.keys())}")

    meta = MODELS[model_name]
    labels = load_labels(labels_path)

    print(f"\n{'='*60}")
    print(f"Model  : {model_name} (Top-1 FP32: {meta['top1']}%)")
    print(f"SoC    : {soc} — {SOC_NOTES.get(soc, 'unknown')}")
    print(f"Image  : {image_path}")
    print(f"{'='*60}")

    input_data = preprocess(image_path, meta["input"])

    if meta["format"] == "tflite":
        logits, latency_ms = run_tflite(model_path, input_data)
    else:
        logits, latency_ms = run_onnx(model_path, input_data)

    exp = np.exp(logits - logits.max())
    probs = exp / exp.sum()

    top_k_idx = np.argsort(probs)[::-1][:top_k]
    results = []
    for rank, idx in enumerate(top_k_idx, 1):
        results.append({
            "rank": rank,
            "class_id": int(idx),
            "class_name": labels[idx] if idx < len(labels) else f"class_{idx}",
            "confidence": float(probs[idx]),
        })
        print(f"  #{rank}: {results[-1]['class_name']:<40} {probs[idx]*100:.2f}%")

    print(f"\nLatency (CPU simulation): {latency_ms:.2f} ms")
    print(f"Note: On {soc} NPU, expect {latency_ms/5:.1f}–{latency_ms/3:.1f} ms")

    return {
        "model": model_name,
        "soc": soc,
        "image": image_path,
        "top_k": results,
        "latency_cpu_ms": latency_ms,
    }


def main():
    parser = argparse.ArgumentParser(description="TI EdgeAI Classification Example")
    parser.add_argument("--image",      required=True)
    parser.add_argument("--model",      default="mobilenet_v2_lite", choices=list(MODELS.keys()))
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--soc",        default="AM68A",
                        choices=["AM62A", "AM67A", "AM68A", "AM69A", "TDA4VM"])
    parser.add_argument("--labels",     default=None)
    parser.add_argument("--top_k",      type=int, default=5)
    parser.add_argument("--output",     default=None)
    args = parser.parse_args()

    if args.model_path is None:
        fmt = MODELS[args.model]["format"]
        args.model_path = f"models/{args.model}.{fmt}"

    result = classify(args.image, args.model, args.model_path,
                      args.soc, args.labels, args.top_k)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
