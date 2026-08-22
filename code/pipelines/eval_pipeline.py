"""
eval_pipeline.py — Standalone accuracy and performance evaluation pipeline.

Evaluates pre-compiled TI EdgeAI Model Zoo artifacts against edgeai-benchmark
conventions. Used directly by the GitHub Actions CI workflow.

Usage:
    # Evaluate all models in config
    python pipelines/eval_pipeline.py --config config/pipeline_config.yaml

    # Evaluate a specific model + SoC
    python pipelines/eval_pipeline.py --model yolox-s-lite --soc AM68A

    # CI gate check (exits with code 1 on failure)
    python pipelines/eval_pipeline.py --check-gates --fail-on-drop 2.0
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from mlops.evaluator import Evaluator, AccuracyGateError
from mlops.model_manager import ModelManager
from mlops.data_pipeline import DataPipeline


def run_evaluation(
    config_path: str,
    model_name: Optional[str] = None,
    soc: Optional[str] = None,
    mode: str = "accuracy",
    check_gates: bool = False,
    fail_on_drop: Optional[float] = None,
) -> int:
    """
    Main evaluation entry point. Returns exit code (0=pass, 1=fail).
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    soc = soc or config.get("soc", "AM68A")
    task = config.get("task", "detection")
    eval_cfg = config.get("evaluation", {})
    num_frames = eval_cfg.get("num_frames", 1000)

    manager = ModelManager(config_path)
    evaluator = Evaluator(config_path)

    # Determine which models to evaluate
    if model_name:
        models_to_eval = [model_name]
    else:
        shortlisted = manager.list_models(task=task, soc=soc)
        models_to_eval = [m["name"] for m in shortlisted]

    if not models_to_eval:
        print(f"[EvalPipeline] No models found for task={task}, soc={soc}")
        return 0

    print(f"\n[EvalPipeline] Evaluating {len(models_to_eval)} model(s) "
          f"on {soc} — mode={mode}, frames={num_frames}")
    print("─" * 60)

    # Setup data pipeline for this task
    data_pipeline = DataPipeline(
        task=task,
        dataset=eval_cfg.get("dataset", "coco"),
        dataset_root=eval_cfg.get("dataset_path", "./datasets"),
        num_frames=num_frames,
    )

    results = []
    for m_name in models_to_eval:
        try:
            meta = manager.get_metadata(m_name)
            model_id = meta.get("model_id", m_name)

            print(f"\n[EvalPipeline] Model: {m_name} (id={model_id})")

            # FP32 simulation
            fp32_result = evaluator.run(
                model_id=model_id, soc=soc,
                precision="fp32", mode=mode
            )
            print(f"  FP32: {fp32_result['metrics']}")

            # INT8 (deployed)
            int8_result = evaluator.run(
                model_id=model_id, soc=soc,
                precision="int8", mode=mode
            )
            print(f"  INT8: {int8_result['metrics']}")

            results.append({
                "model": m_name,
                "model_id": model_id,
                "fp32": fp32_result["metrics"],
                "int8": int8_result["metrics"],
            })

        except KeyError as e:
            print(f"[EvalPipeline] SKIP {m_name}: {e}")

    # Save report
    report_path = evaluator.save_report()
    print(f"\n[EvalPipeline] Report: {report_path}")

    # Print summary table
    _print_summary_table(results, task)

    # Gate check
    if check_gates:
        try:
            evaluator.check_gates(fail_on_drop=fail_on_drop)
            print("[EvalPipeline] All accuracy gates PASSED.")
            return 0
        except AccuracyGateError as e:
            print(f"[EvalPipeline] GATE FAILED:\n{e}")
            return 1

    return 0


def _print_summary_table(results: list[dict], task: str) -> None:
    metric_key = {
        "classification":  ("top1",  "Top-1 %"),
        "object_detection":("mAP",   "mAP %"),
        "segmentation":    ("miou",  "MeanIoU %"),
        "keypoint":        ("AP",    "AP %"),
        "depth_estimation":("delta1","Delta1 %"),
    }.get(task, ("metric", "Metric"))

    key, label = metric_key
    print(f"\n{'Model':<30} {'FP32 '+label:>14} {'INT8 '+label:>14} {'Drop':>8}")
    print("─" * 70)
    for r in results:
        fp32_val = r["fp32"].get(key, 0.0)
        int8_val = r["int8"].get(key, 0.0)
        drop = fp32_val - int8_val
        print(f"{r['model']:<30} {fp32_val:>14.2f} {int8_val:>14.2f} {drop:>7.2f}pp")
    print("─" * 70)


from typing import Optional

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TI EdgeAI Eval Pipeline")
    parser.add_argument("--config", default="config/pipeline_config.yaml")
    parser.add_argument("--model",  default=None, help="Model name from registry")
    parser.add_argument("--soc",    default=None, help="Target SoC")
    parser.add_argument("--mode",   default="accuracy",
                        choices=["accuracy", "performance"],
                        help="accuracy=low-threshold, performance=deployment-threshold")
    parser.add_argument("--check-gates", action="store_true",
                        help="Fail with exit code 1 if accuracy gates fail")
    parser.add_argument("--fail-on-drop", type=float, default=None,
                        help="Override max INT8 accuracy drop threshold")
    args = parser.parse_args()

    exit_code = run_evaluation(
        config_path=args.config,
        model_name=args.model,
        soc=args.soc,
        mode=args.mode,
        check_gates=args.check_gates,
        fail_on_drop=args.fail_on_drop,
    )
    sys.exit(exit_code)
