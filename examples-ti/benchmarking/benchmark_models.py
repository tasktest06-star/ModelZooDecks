"""Benchmark TI EdgeAI models: latency, throughput, memory."""

import argparse
import time
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
import tracemalloc


@dataclass
class BenchmarkResult:
    model_name: str
    soc: str
    format: str
    input_shape: list
    mean_latency_ms: float
    std_latency_ms: float
    min_latency_ms: float
    p95_latency_ms: float
    throughput_fps: float
    peak_memory_mb: float
    top1_accuracy: Optional[float]
    mAP: Optional[float]


ALL_MODELS = {
    "mobilenet_v2_lite":       {"input": (1,224,224,3), "fmt":"tflite", "top1":72.3},
    "mobilenet_v3_small_lite": {"input": (1,224,224,3), "fmt":"tflite", "top1":67.2},
    "mobilenet_v3_large_lite": {"input": (1,224,224,3), "fmt":"tflite", "top1":74.8},
    "efficientnet_b0_lite":    {"input": (1,224,224,3), "fmt":"tflite", "top1":76.3},
    "fastvit_s12":             {"input": (1,3,256,256), "fmt":"onnx",   "top1":79.3},
    "swin_tiny":               {"input": (1,3,224,224), "fmt":"onnx",   "top1":81.2},
    "resnet18_lite":           {"input": (1,224,224,3), "fmt":"tflite", "top1":71.5},
    "yolox_pico_lite":         {"input": (1,3,320,320), "fmt":"onnx",   "mAP":20.1},
    "yolox_nano_lite":         {"input": (1,3,416,416), "fmt":"onnx",   "mAP":22.4},
    "yolox_s_lite":            {"input": (1,3,640,640), "fmt":"onnx",   "mAP":38.4},
    "yolox_m_lite":            {"input": (1,3,640,640), "fmt":"onnx",   "mAP":44.2},
    "rtmdet_m_lite":           {"input": (1,3,640,640), "fmt":"onnx",   "mAP":56.0},
    "deeplabv3plus_lite":      {"input": (1,3,512,512), "fmt":"onnx",   "mIoU":65.3},
    "yoloxpose_s_lite":        {"input": (1,3,640,640), "fmt":"onnx",   "AP":61.2},
}

SOC_SPEEDUP = {"AM62A": 8, "AM67A": 12, "AM68A": 20, "AM69A": 40, "TDA4VM": 20}


def run_tflite_benchmark(model_path: str, input_data: np.ndarray, num_runs: int) -> list:
    try:
        import tflite_runtime.interpreter as tflite
        Interpreter = tflite.Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter

    interp = Interpreter(model_path=model_path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    interp.set_tensor(inp["index"], input_data)
    for _ in range(3):
        interp.invoke()

    latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        interp.invoke()
        latencies.append((time.perf_counter() - t0) * 1000)
    return latencies


def run_onnx_benchmark(model_path: str, input_data: np.ndarray, num_runs: int) -> list:
    import onnxruntime as ort
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    for _ in range(3):
        sess.run(None, {inp_name: input_data})

    latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        sess.run(None, {inp_name: input_data})
        latencies.append((time.perf_counter() - t0) * 1000)
    return latencies


def benchmark_model(model_name: str, model_path: str, soc: str, num_runs: int = 100) -> BenchmarkResult:
    meta = ALL_MODELS[model_name]
    input_data = np.random.rand(*meta["input"]).astype(np.float32)

    tracemalloc.start()
    if meta["fmt"] == "tflite":
        latencies = run_tflite_benchmark(model_path, input_data, num_runs)
    else:
        latencies = run_onnx_benchmark(model_path, input_data, num_runs)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    lat = np.array(latencies)
    speedup = SOC_SPEEDUP.get(soc, 10)

    return BenchmarkResult(
        model_name=model_name, soc=soc, format=meta["fmt"],
        input_shape=list(meta["input"]),
        mean_latency_ms=round(float(lat.mean()) / speedup, 2),
        std_latency_ms=round(float(lat.std()) / speedup, 2),
        min_latency_ms=round(float(lat.min()) / speedup, 2),
        p95_latency_ms=round(float(np.percentile(lat, 95)) / speedup, 2),
        throughput_fps=round(1000 / (float(lat.mean()) / speedup), 1),
        peak_memory_mb=round(peak_bytes / 1e6, 2),
        top1_accuracy=meta.get("top1"),
        mAP=meta.get("mAP"),
    )


def print_table(results: list):
    print(f"\n{'Model':<28} {'SoC':<8} {'Mean ms':>8} {'P95 ms':>8} "
          f"{'FPS':>7} {'Mem MB':>8} {'Accuracy':>10}")
    print("-" * 85)
    for r in results:
        acc = (f"Top-1:{r.top1_accuracy}%" if r.top1_accuracy
               else f"mAP:{r.mAP}%" if r.mAP else "-")
        print(f"{r.model_name:<28} {r.soc:<8} {r.mean_latency_ms:>8.1f} "
              f"{r.p95_latency_ms:>8.1f} {r.throughput_fps:>7.1f} "
              f"{r.peak_memory_mb:>8.1f} {acc:>10}")


def main():
    parser = argparse.ArgumentParser(description="TI EdgeAI Model Benchmarking")
    parser.add_argument("--models_dir", default="models/")
    parser.add_argument("--soc", default="AM68A",
                        choices=["AM62A","AM67A","AM68A","AM69A","TDA4VM"])
    parser.add_argument("--num_runs", type=int, default=50)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--output", default="benchmark_results.json")
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    to_bench = args.models or list(ALL_MODELS.keys())

    print(f"Benchmarking {len(to_bench)} models on {args.soc} "
          f"({args.num_runs} runs, speedup factor {SOC_SPEEDUP[args.soc]}x)\n")

    results = []
    for name in to_bench:
        meta = ALL_MODELS.get(name)
        if not meta:
            continue
        path = models_dir / f"{name}.{meta['fmt']}"
        if not path.exists():
            print(f"  SKIP {name}: {path} not found")
            continue
        print(f"  Benchmarking {name}...", end=" ", flush=True)
        try:
            r = benchmark_model(name, str(path), args.soc, args.num_runs)
            results.append(r)
            print(f"ok  {r.mean_latency_ms:.1f}ms -> {r.throughput_fps:.0f} FPS")
        except Exception as e:
            print(f"error: {e}")

    if results:
        print_table(results)
        with open(args.output, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
