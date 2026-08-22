"""
eval_pipeline.py — Multi-domain evaluation pipeline for ADI Model Zoo.

Runs accuracy/performance evaluation across all 3 domains and checks gates.

Usage:
    python pipelines/eval_pipeline.py \\
        --config config/pipeline_config.yaml \\
        --domain vision --task object_detection \\
        --device MAX78002 \\
        --check-gates
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from mlops.model_manager import ModelManager
from mlops.evaluator import Evaluator, AccuracyGateError


def run_evaluation(
    config_path: str,
    model_name: str,
    device: str,
    domain: str = "vision",
    task: str = "object_detection",
    check_gates: bool = False,
    fail_on_violation: bool = False,
) -> int:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    manager = ModelManager(config_path)
    ev = Evaluator(config_path)

    print(f"\n{'='*60}")
    print(f" ADI Model Zoo Evaluation")
    print(f" model={model_name}  device={device}  domain={domain}  task={task}")
    print(f"{'='*60}")

    # Verify model exists and supports device
    try:
        meta = manager.get_metadata(model_name)
    except KeyError as e:
        print(f"[EvalPipeline] ERROR: {e}")
        return 1

    supported = [d["device"] for d in meta.get("supported_devices", [])]
    if device not in supported:
        print(f"[EvalPipeline] WARN: {model_name} not listed for {device}. "
              f"Supported: {supported}")

    # Run evaluation
    results = ev.run(
        model_name,
        device=device,
        domain=domain,
        task=task,
    )

    # Print results table
    print(f"\n{'─'*50}")
    print(f"  Model:   {model_name}")
    print(f"  Device:  {device}")
    print(f"  Domain:  {domain} / {task}")
    print(f"  {results['metric_name']:20s}: {results['metric_value']:.4f}")
    print(f"  Latency: {results['latency_mean_ms']} ms mean | "
          f"{results['latency_p95_ms']} ms p95")
    print(f"  FPS:     {results['fps']}")
    print(f"{'─'*50}")

    reported = meta.get("metrics", {})
    if reported:
        print("\n  Reported vs. measured:")
        for k, v in reported.items():
            print(f"    {k}: {v} (reported) vs "
                  f"{results.get('metric_value', '?')} (measured)")

    ev.save_report(results)

    if check_gates:
        try:
            ev.check_gates(results)
        except AccuracyGateError as e:
            print(str(e))
            if fail_on_violation:
                return 2

    return 0


def run_all_models(config_path: str, device: str, check_gates: bool) -> None:
    """Run evaluation across all models in the registry for a given device."""
    manager = ModelManager(config_path)
    ev = Evaluator(config_path)
    all_results = []

    models = manager.list_models(device=device, status="production")
    print(f"\n[EvalPipeline] Evaluating {len(models)} models on {device}\n")

    for m in models:
        name = m["name"]
        domain = m.get("domain", "vision")
        task = m.get("task", "object_detection")
        try:
            results = ev.run(name, device=device, domain=domain, task=task)
            all_results.append(results)
            if check_gates:
                try:
                    ev.check_gates(results)
                except AccuracyGateError as e:
                    print(f"  GATE FAIL for {name}: {e}")
        except Exception as e:
            print(f"  ERROR evaluating {name}: {e}")

    if all_results:
        ev.save_csv_report(all_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ADI Model Zoo Evaluation Pipeline")
    parser.add_argument("--config", default="config/pipeline_config.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default="MAX78002")
    parser.add_argument("--domain", default="vision")
    parser.add_argument("--task", default=None)
    parser.add_argument("--all-models", action="store_true",
                        help="Evaluate all registry models for --device")
    parser.add_argument("--check-gates", action="store_true")
    parser.add_argument("--fail-on-violation", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.all_models:
        run_all_models(args.config, args.device, args.check_gates)
        sys.exit(0)

    model_name = args.model or cfg.get("model", "feature_pyramid_net")
    task = args.task or cfg.get("task", "object_detection")

    code = run_evaluation(
        args.config,
        model_name=model_name,
        device=args.device,
        domain=args.domain,
        task=task,
        check_gates=args.check_gates,
        fail_on_violation=args.fail_on_violation,
    )
    sys.exit(code)
