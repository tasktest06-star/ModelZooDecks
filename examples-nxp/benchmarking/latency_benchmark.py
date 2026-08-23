"""
Latency benchmark for NXP eIQ model zoo across all 4 platforms.
Data from NXP eIQ benchmarking results (i.MX 8M Plus, i.MX 93, RT1170, MCX N947).

Run on target: python latency_benchmark.py --platform imx8mplus
Run offline:   python latency_benchmark.py --all --format table
"""

import argparse
import json
import sys
from typing import List, Dict, Optional

BENCHMARKS: List[Dict] = [
    # --- i.MX 8M Plus (2.3 TOPS NPU, Cortex-A53 x4 @ 1.8 GHz) ---
    {"model": "mobilenetv1_025",        "platform": "imx8mplus", "task": "classify",  "backend": "NPU",  "latency_ms": 5,   "fps": 200, "top1": 50.2, "map": None},
    {"model": "mobilenetv1",            "platform": "imx8mplus", "task": "classify",  "backend": "NPU",  "latency_ms": 15,  "fps": 67,  "top1": 70.9, "map": None},
    {"model": "mobilenetv2",            "platform": "imx8mplus", "task": "classify",  "backend": "NPU",  "latency_ms": 18,  "fps": 56,  "top1": 71.8, "map": None},
    {"model": "efficientnet_lite0",     "platform": "imx8mplus", "task": "classify",  "backend": "NPU",  "latency_ms": 22,  "fps": 45,  "top1": 72.2, "map": None},
    {"model": "resnet50",               "platform": "imx8mplus", "task": "classify",  "backend": "NPU",  "latency_ms": 35,  "fps": 29,  "top1": 75.9, "map": None},
    {"model": "ssdlite_mobilenetv2",    "platform": "imx8mplus", "task": "detect",    "backend": "NPU",  "latency_ms": 22,  "fps": 45,  "top1": None, "map": 22.1},
    {"model": "yolov8n",                "platform": "imx8mplus", "task": "detect",    "backend": "NPU",  "latency_ms": 95,  "fps": 11,  "top1": None, "map": 37.3},
    {"model": "fast_srgan",             "platform": "imx8mplus", "task": "sr",        "backend": "NPU",  "latency_ms": 48,  "fps": 21,  "top1": None, "map": None},
    {"model": "ds_cnn_kws",             "platform": "imx8mplus", "task": "kws",       "backend": "NPU",  "latency_ms": 8,   "fps": 125, "top1": 96.9, "map": None},

    # --- i.MX 93 with Ethos-U65 (Vela compiled, 1.0 TOPS) ---
    {"model": "mobilenetv1_025",        "platform": "imx93",     "task": "classify",  "backend": "Ethos-U65", "latency_ms": 12,  "fps": 83,  "top1": 50.2, "map": None},
    {"model": "mobilenetv2",            "platform": "imx93",     "task": "classify",  "backend": "Ethos-U65", "latency_ms": 35,  "fps": 29,  "top1": 71.8, "map": None},
    {"model": "efficientnet_lite0",     "platform": "imx93",     "task": "classify",  "backend": "Ethos-U65", "latency_ms": 55,  "fps": 18,  "top1": 72.2, "map": None},
    {"model": "ssdlite_mobilenetv2",    "platform": "imx93",     "task": "detect",    "backend": "Ethos-U65", "latency_ms": 65,  "fps": 15,  "top1": None, "map": 22.1},
    {"model": "nanodet_plus",           "platform": "imx93",     "task": "detect",    "backend": "Ethos-U65", "latency_ms": 100, "fps": 10,  "top1": None, "map": 30.4},

    # --- i.MX RT1170 (Cortex-M7 @ 1 GHz + Cortex-M4, 2 MB OCRAM) ---
    {"model": "mobilenetv1_025",        "platform": "rt1170",    "task": "classify",  "backend": "CPU",       "latency_ms": 45,  "fps": 22,  "top1": 50.2, "map": None},
    {"model": "mobilenetv1",            "platform": "rt1170",    "task": "classify",  "backend": "CPU",       "latency_ms": 180, "fps": 6,   "top1": 70.9, "map": None},
    {"model": "ssdlite_mobilenetv2",    "platform": "rt1170",    "task": "detect",    "backend": "CPU",       "latency_ms": 290, "fps": 3,   "top1": None, "map": 22.1},
    {"model": "ds_cnn_kws",             "platform": "rt1170",    "task": "kws",       "backend": "CPU",       "latency_ms": 25,  "fps": 40,  "top1": 96.9, "map": None},
    {"model": "microspeech",            "platform": "rt1170",    "task": "kws",       "backend": "CPU",       "latency_ms": 10,  "fps": 100, "top1": 90.1, "map": None},

    # --- MCX N947 (Cortex-M33 @ 150 MHz, 512 KB SRAM) ---
    {"model": "mobilenetv1_025",        "platform": "mcxn947",   "task": "classify",  "backend": "CPU",       "latency_ms": 320, "fps": 3,   "top1": 50.2, "map": None},
    {"model": "microspeech",            "platform": "mcxn947",   "task": "kws",       "backend": "CPU",       "latency_ms": 55,  "fps": 18,  "top1": 90.1, "map": None},
    {"model": "ds_cnn_kws",             "platform": "mcxn947",   "task": "kws",       "backend": "CPU",       "latency_ms": 110, "fps": 9,   "top1": 96.9, "map": None},
]

PLATFORM_DESC = {
    "imx8mplus": "i.MX 8M Plus (2.3 TOPS NPU, Cortex-A53 x4 @ 1.8 GHz, 8 GB LPDDR4)",
    "imx93":     "i.MX 93 (Ethos-U65 1.0 TOPS, Cortex-A55 x2 @ 1.7 GHz, 2 GB LPDDR4)",
    "rt1170":    "i.MX RT1170 (Cortex-M7 @ 1 GHz + M4, 2 MB OCRAM, TFLite Micro)",
    "mcxn947":   "MCX N947 (Cortex-M33 @ 150 MHz, 512 KB SRAM, TFLite Micro)",
}

PLATFORMS = list(PLATFORM_DESC.keys())


def filter_benchmarks(platform: Optional[str] = None,
                      task: Optional[str] = None) -> List[Dict]:
    rows = BENCHMARKS
    if platform:
        rows = [r for r in rows if r["platform"] == platform]
    if task:
        rows = [r for r in rows if r["task"] == task]
    return rows


def print_table(rows: List[Dict], title: str = ""):
    if not rows:
        print("No matching benchmarks.")
        return

    if title:
        print(f"\n{title}")
        print("=" * 80)

    header = f"{'Model':<28} {'Platform':<12} {'Task':<10} {'Backend':<12} {'ms':>6} {'FPS':>5} {'Top-1':>6} {'mAP':>6}"
    print(header)
    print("-" * 80)

    last_platform = None
    for r in sorted(rows, key=lambda x: (x["platform"], x["task"], x["latency_ms"])):
        if r["platform"] != last_platform:
            if last_platform is not None:
                print()
            desc = PLATFORM_DESC.get(r["platform"], r["platform"])
            print(f"  Platform: {desc}")
            last_platform = r["platform"]

        top1 = f"{r['top1']:.1f}%" if r["top1"] else "  —  "
        mapp = f"{r['map']:.1f}%"  if r["map"]  else "  —  "
        print(f"  {r['model']:<26} {r['platform']:<12} {r['task']:<10} "
              f"{r['backend']:<12} {r['latency_ms']:>5}ms {r['fps']:>5} "
              f"{top1:>6} {mapp:>6}")


def run_live_benchmark(platform: str, model_path: str,
                       num_runs: int = 20) -> Dict:
    """Runs TFLite inference timing on actual hardware."""
    import time
    import numpy as np

    try:
        import tflite_runtime.interpreter as tflite
        Interpreter = tflite.Interpreter
    except ImportError:
        try:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
        except ImportError:
            print("[WARN] No TFLite runtime found — showing reference data only")
            return {}

    interp = Interpreter(model_path=model_path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    shape = inp["shape"]
    dummy = np.zeros(shape, dtype=inp["dtype"])

    # Warmup
    for _ in range(3):
        interp.set_tensor(inp["index"], dummy)
        interp.invoke()

    times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        interp.set_tensor(inp["index"], dummy)
        interp.invoke()
        times.append((time.perf_counter() - t0) * 1000)

    arr = sorted(times)
    result = {
        "model_path": model_path,
        "platform":   platform,
        "num_runs":   num_runs,
        "mean_ms":    round(sum(times) / len(times), 2),
        "median_ms":  round(arr[len(arr) // 2], 2),
        "p95_ms":     round(arr[int(len(arr) * 0.95)], 2),
        "min_ms":     round(arr[0], 2),
        "max_ms":     round(arr[-1], 2),
        "fps":        round(1000.0 / (sum(times) / len(times)), 1),
    }

    print(f"\nLive benchmark: {model_path}")
    print(f"  Platform : {PLATFORM_DESC.get(platform, platform)}")
    print(f"  Runs     : {num_runs}")
    print(f"  Mean     : {result['mean_ms']} ms  ({result['fps']} FPS)")
    print(f"  Median   : {result['median_ms']} ms")
    print(f"  P95      : {result['p95_ms']} ms")
    print(f"  Min/Max  : {result['min_ms']} / {result['max_ms']} ms")

    return result


def main():
    parser = argparse.ArgumentParser(description="NXP eIQ Latency Benchmark")
    parser.add_argument("--platform", default=None, choices=PLATFORMS + ["all"],
                        help="Filter by platform (omit for all)")
    parser.add_argument("--task",     default=None,
                        choices=["classify", "detect", "kws", "sr"],
                        help="Filter by task type")
    parser.add_argument("--all",      action="store_true",
                        help="Show all platforms in one table")
    parser.add_argument("--live",     default=None, metavar="MODEL_PATH",
                        help="Run live benchmark on a TFLite model file")
    parser.add_argument("--runs",     type=int, default=20)
    parser.add_argument("--output",   default=None)
    args = parser.parse_args()

    if args.live:
        plat = args.platform if args.platform and args.platform != "all" else "imx8mplus"
        result = run_live_benchmark(plat, args.live, args.runs)
        if args.output:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)
        return

    plat = None if (args.all or args.platform == "all") else args.platform
    rows = filter_benchmarks(platform=plat, task=args.task)

    title = "NXP eIQ Model Latency Benchmarks"
    if plat:
        title += f" — {plat}"
    if args.task:
        title += f" ({args.task})"

    print_table(rows, title)

    # Print summary statistics
    if rows:
        platforms_in_rows = sorted(set(r["platform"] for r in rows))
        print("\nSummary (best FPS per model per platform):")
        print("-" * 45)
        for p in platforms_in_rows:
            p_rows = [r for r in rows if r["platform"] == p]
            best  = max(p_rows, key=lambda r: r["fps"])
            worst = min(p_rows, key=lambda r: r["fps"])
            print(f"  {p:<12}: {best['fps']:>4} FPS ({best['model']}) "
                  f"to {worst['fps']:>3} FPS ({worst['model']})")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
