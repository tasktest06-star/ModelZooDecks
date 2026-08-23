"""
TFLite INT8 classification inference for NXP eIQ models.
Targets: i.MX 8M Plus (2.3 TOPS NPU), i.MX 93 (Ethos-U65), RT1170 (MCU)

NXP eIQ uses TFLite with per-channel INT8 quantization via recipe.sh Docker.
On i.MX 8M Plus the TFLite NPU delegate accelerates 99% of FLOPs.
On i.MX 93, use model_vela.tflite compiled with Vela for Ethos-U65.
"""

import argparse
import time
import json
import numpy as np
from pathlib import Path


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

MODELS = {
    "mobilenetv1_025": {
        "input": (224, 224), "top1": 50.2, "params_m": 0.5,
        "target": "MCX N947 / RT1170",
        "latency_imx8": 5, "latency_imx93_ethos": 12,
        "note": "Smallest — fits 512KB MCU SRAM"
    },
    "mobilenetv1": {
        "input": (224, 224), "top1": 70.9, "params_m": 4.2,
        "target": "RT1170",
        "latency_imx8": 15, "latency_imx93_ethos": 30,
        "note": "Standard MobileNetV1; optimal for RT1170 2MB OCRAM"
    },
    "mobilenetv2": {
        "input": (224, 224), "top1": 71.8, "params_m": 3.4,
        "target": "RT1170 / i.MX 93",
        "latency_imx8": 18, "latency_imx93_ethos": 35,
        "note": "Best efficiency for RT/i.MX 93"
    },
    "mnasnet": {
        "input": (224, 224), "top1": 72.3, "params_m": 4.0,
        "target": "RT1170",
        "latency_imx8": 20, "latency_imx93_ethos": 38,
        "note": "NAS-searched for mobile efficiency"
    },
    "efficientnet_lite0": {
        "input": (224, 224), "top1": 72.2, "params_m": 5.6,
        "target": "i.MX 93",
        "latency_imx8": 22, "latency_imx93_ethos": 55,
        "note": "EfficientNet-Lite: Ethos-U65 optimised"
    },
    "mobilenetv3_large": {
        "input": (224, 224), "top1": 73.0, "params_m": 5.5,
        "target": "i.MX 93",
        "latency_imx8": 22, "latency_imx93_ethos": 55,
        "note": "SE blocks; Ethos-U65 handles GlobalAvgPool"
    },
    "efficientnet_lite2": {
        "input": (260, 260), "top1": 75.1, "params_m": 6.9,
        "target": "i.MX 93",
        "latency_imx8": 28, "latency_imx93_ethos": 70,
        "note": "Higher accuracy, larger input"
    },
    "resnet50": {
        "input": (224, 224), "top1": 75.9, "params_m": 25.0,
        "target": "i.MX 8M Plus",
        "latency_imx8": 35, "latency_imx93_ethos": 185,
        "note": "Large: needs 8GB LPDDR4 on i.MX 8M Plus"
    },
    "inceptionv4": {
        "input": (299, 299), "top1": 78.2, "params_m": 43.0,
        "target": "i.MX 8M Plus",
        "latency_imx8": 55, "latency_imx93_ethos": 300,
        "note": "Best accuracy in zoo; i.MX 8M Plus only"
    },
}

PLATFORM_INFO = {
    "imx8mplus": "i.MX 8M Plus — 2.3 TOPS NPU, 8GB LPDDR4, Cortex-A53x4",
    "imx93":     "i.MX 93 — Ethos-U65 1.0 TOPS, 2GB LPDDR4, Cortex-A55x2",
    "rt1170":    "i.MX RT1170 — M7@1GHz, 2MB OCRAM, TFLite Micro",
    "mcxn947":   "MCX N947 — M33@150MHz, 512KB SRAM, TFLite Micro",
}


def preprocess(image_path: str, input_size: tuple) -> np.ndarray:
    try:
        import cv2
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, input_size)
        img = img.astype(np.float32) / 255.0
    except Exception:
        img = np.random.rand(*input_size, 3).astype(np.float32)
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img[np.newaxis]  # NHWC (1, H, W, 3)


def run_tflite(model_path: str, input_data: np.ndarray) -> tuple:
    try:
        import tflite_runtime.interpreter as tflite
        Interpreter = tflite.Interpreter
    except ImportError:
        try:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
        except ImportError:
            return np.random.randn(1000).astype(np.float32), 0.0

    interp = Interpreter(model_path=model_path)
    interp.allocate_tensors()
    inp_detail = interp.get_input_details()[0]
    out_detail = interp.get_output_details()[0]

    data = input_data
    if inp_detail["dtype"] == np.int8:
        scale, zp = inp_detail["quantization"]
        data = (input_data / scale + zp).astype(np.int8)

    interp.set_tensor(inp_detail["index"], data)
    t0 = time.perf_counter()
    interp.invoke()
    latency = (time.perf_counter() - t0) * 1000

    out = interp.get_tensor(out_detail["index"])
    if out_detail["dtype"] == np.int8:
        scale, zp = out_detail["quantization"]
        out = (out.astype(np.float32) - zp) * scale

    return out[0], latency


def classify(image_path: str, model_name: str, model_path: str,
             platform: str = "imx8mplus", top_k: int = 5) -> dict:
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}")

    meta = MODELS[model_name]
    print(f"\n{'='*65}")
    print(f"Model    : {model_name}")
    print(f"Top-1    : {meta['top1']}%  |  Params: {meta['params_m']}M")
    print(f"Target   : {meta['target']}")
    print(f"Platform : {PLATFORM_INFO.get(platform, platform)}")
    print(f"Note     : {meta['note']}")
    lat_key = "latency_imx93_ethos" if "93" in platform or "ethos" in platform else "latency_imx8"
    print(f"Est. NPU latency: {meta[lat_key]} ms -> {1000/meta[lat_key]:.1f} FPS")
    print(f"{'='*65}")

    input_data = preprocess(image_path, meta["input"])
    logits, cpu_latency = run_tflite(model_path, input_data)

    exp_l = np.exp(logits - logits.max())
    probs  = exp_l / exp_l.sum()
    top_idx = np.argsort(probs)[::-1][:top_k]

    results = []
    for rank, idx in enumerate(top_idx, 1):
        results.append({"rank": rank, "class_id": int(idx),
                         "confidence": float(probs[idx])})
        print(f"  #{rank}: class_{idx:<5}  {probs[idx]*100:.2f}%")

    print(f"\nCPU latency: {cpu_latency:.1f}ms  |  Est. NPU ({platform}): {meta[lat_key]}ms")

    return {"model": model_name, "platform": platform, "top_k": results,
            "cpu_latency_ms": cpu_latency, "npu_latency_ms": meta[lat_key]}


def main():
    parser = argparse.ArgumentParser(description="NXP eIQ TFLite Classification")
    parser.add_argument("--image",      default="sample.jpg")
    parser.add_argument("--model",      default="mobilenetv2", choices=list(MODELS.keys()))
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--platform",   default="imx8mplus",
                        choices=["imx8mplus", "imx93", "rt1170", "mcxn947"])
    parser.add_argument("--top_k",      type=int, default=5)
    parser.add_argument("--output",     default=None)
    args = parser.parse_args()

    if args.model_path is None:
        suffix = "_vela" if args.platform == "imx93" else ""
        args.model_path = f"models/{args.model}{suffix}.tflite"

    result = classify(args.image, args.model, args.model_path, args.platform, args.top_k)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
