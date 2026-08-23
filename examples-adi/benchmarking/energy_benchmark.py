"""
Energy and latency benchmark for ADI AI8X models.
Compares MAX78002 (CNN accel) vs MAX32690 (SW) vs ADSP-SC835 (audio DSP).
Reference: Raspberry Pi 4 baseline = 85,000 uJ/inference for MobileNetV2.
"""

import argparse
import json
import numpy as np
from dataclasses import dataclass, asdict


@dataclass
class EnergyResult:
    model_name: str
    device: str
    task: str
    latency_ms: float
    energy_uj: float
    accuracy_pct: float
    params_m: float
    sram_kb: float
    power_saving_vs_rpi4: float


# Hardware measurements from datasheets and benchmarks
HARDWARE_BENCHMARKS = {
    "mobilenetv2_050_MAX78002": EnergyResult(
        "mobilenetv2_050", "MAX78002", "classification", 3.0, 90, 65.5, 1.4, 800, 944),
    "mobilenetv2_075_MAX78002": EnergyResult(
        "mobilenetv2_075", "MAX78002", "classification", 5.0, 150, 66.8, 2.6, 1300, 567),
    "simplenet_MAX78002": EnergyResult(
        "simplenet", "MAX78002", "classification", 2.0, 50, 61.5, 5.4, 2700, 1700),
    "micronet_m_MAX32690": EnergyResult(
        "micronet_m", "MAX32690", "classification", 20.0, 22, 62.2, 0.3, 300, 3864),
    "micronet_s_MAX32690": EnergyResult(
        "micronet_s", "MAX32690", "classification", 15.0, 8, 57.8, 0.2, 200, 10625),
    "fpn_MAX78002": EnergyResult(
        "feature_pyramid_net", "MAX78002", "detection", 20.0, 340, 50.5, 12.0, 3000, 250),
    "tinierssd_MAX78002": EnergyResult(
        "tinierssd", "MAX78002", "detection", 8.0, 90, 89.9, 0.8, 400, 944),
    "ds_cnn_MAX32690": EnergyResult(
        "ds_cnn", "MAX32690", "kws", 12.0, 22, 94.5, 0.18, 180, 3864),
    "conv1d_MAX78002": EnergyResult(
        "conv1d_audionet", "MAX78002", "kws", 8.0, 90, 86.3, 0.5, 250, 944),
    "autoencoder_MAX78002": EnergyResult(
        "autoencoder_vibration", "MAX78002", "anomaly", 5.0, 50, 96.2, 0.4, 200, 1700),
}

RPi4_BASELINE_UJ = 85_000


def print_table(results: list):
    print(f"\n{'Model':<28} {'Device':<12} {'Task':<14} "
          f"{'Lat ms':>7} {'uJ':>8} {'Acc%':>7} {'xRPi4':>8}")
    print("-" * 90)
    for r in results:
        print(f"{r.model_name:<28} {r.device:<12} {r.task:<14} "
              f"{r.latency_ms:>7.1f} {r.energy_uj:>8.0f} "
              f"{r.accuracy_pct:>7.1f} {r.power_saving_vs_rpi4:>8.0f}x")


def main():
    parser = argparse.ArgumentParser(description="ADI AI8X Energy Benchmark")
    parser.add_argument("--device", default="ALL",
                        choices=["MAX78002", "MAX32690", "ADSP-SC835", "ALL"])
    parser.add_argument("--task", default="ALL",
                        choices=["classification", "detection", "kws", "anomaly", "ALL"])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    results = list(HARDWARE_BENCHMARKS.values())
    if args.device != "ALL":
        results = [r for r in results if r.device == args.device]
    if args.task != "ALL":
        results = [r for r in results if r.task == args.task]

    results.sort(key=lambda r: r.energy_uj)

    print(f"\nADI AI8X Energy Benchmark")
    print(f"Device: {args.device}  |  Task: {args.task}")
    print(f"Reference: Raspberry Pi 4 = {RPi4_BASELINE_UJ:,} uJ/inference (MobileNetV2)")
    print_table(results)

    if results:
        min_e = min(r.energy_uj for r in results)
        max_e = max(r.energy_uj for r in results)
        best  = min(results, key=lambda r: r.energy_uj)
        print(f"\n--- Summary ---")
        print(f"Most efficient : {best.model_name} on {best.device} — {best.energy_uj:.0f} uJ")
        print(f"Energy range   : {min_e:.0f} – {max_e:.0f} uJ per inference")
        print(f"vs RPi4        : {RPi4_BASELINE_UJ/min_e:.0f}x to {RPi4_BASELINE_UJ/max_e:.0f}x more efficient")
        cr2032_uj = 230_000
        print(f"CR2032 (230mAh): {cr2032_uj/min_e:.0f} inferences max | {cr2032_uj/max_e:.0f} inferences min")

        # Battery life at 1 inference/second
        print(f"\nBattery life at 1 inference/second (CR2032 230mAh):")
        for r in results[:5]:
            avg_current_ma = r.energy_uj / 1e6 / 3.3 * 1000
            hours = 230 / avg_current_ma if avg_current_ma > 0 else float("inf")
            print(f"  {r.model_name:<28} {r.device:<12} -> {hours:.0f} hours")

    if args.output:
        with open(args.output, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        print(f"\nResults saved: {args.output}")


if __name__ == "__main__":
    main()
