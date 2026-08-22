"""Evaluation pipeline for NXP eIQ models — runs inference and checks accuracy gates."""

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from mlops.evaluator import Evaluator, AccuracyGateError
from mlops.model_manager import ModelManager


def make_dummy_samples(task: str, n: int = 8) -> list:
    """Generate dummy samples for smoke-testing when no real dataset is available."""
    task_shapes = {
        "image_classification": ([1, 224, 224, 3], 0),
        "object_detection": ([1, 320, 320, 3], 0),
        "semantic_segmentation": ([1, 513, 513, 3], np.zeros((513, 513), dtype=np.int64)),
        "instance_segmentation": ([1, 550, 550, 3], 0),
        "super_resolution": ([1, 128, 128, 3], np.zeros((512, 512, 3), dtype=np.uint8)),
        "low_light_enhancement": ([1, 256, 256, 3], 0),
        "face_recognition": ([1, 160, 160, 3], 0),
        "pose_estimation": ([1, 192, 192, 3], 0),
        "keyword_spotting": ([1, 49, 10, 1], 0),
        "speech_recognition": ([1, 296, 39], 0),
        "anomaly_detection": ([1, 8192], 1),
        "eeg_classification": ([1, 1, 22, 1125], 0),
        "monocular_depth": ([1, 256, 256, 3], 0),
    }
    shape_info = task_shapes.get(task, ([1, 224, 224, 3], 0))
    shape, label = shape_info
    samples = []
    for _ in range(n):
        inp = np.zeros(shape, dtype=np.float32)
        if task == "speech_recognition":
            samples.append((inp, "hello world"))
        elif task == "super_resolution":
            samples.append((inp, label))
        elif task == "anomaly_detection":
            samples.append((inp, label))
        else:
            samples.append((inp, label if isinstance(label, int) else 0))
    return samples


def evaluate_model(
    config_path: str,
    model_id: str,
    platform: str = "imx8mplus",
    n_samples: int = 8,
    raise_on_gate_fail: bool = True,
) -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    manager = ModelManager(config_path)
    evaluator = Evaluator(config)
    meta = manager.get_metadata(model_id)
    task = meta["task"]
    domain = meta["domain"]

    model_path = manager.get_model_path(model_id, platform)
    if not model_path.exists():
        print(f"[eval] Model file not found: {model_path}")
        print(f"[eval] Run recipe first: python pipelines/recipe_pipeline.py --model {model_id}")
        return {"status": "model_not_found", "model_id": model_id}

    print(f"\n{'='*60}")
    print(f"Evaluating: {model_id}")
    print(f"  Task    : {task}")
    print(f"  Domain  : {domain}")
    print(f"  Platform: {platform}")
    print(f"  Model   : {model_path}")
    print(f"{'='*60}")

    samples = make_dummy_samples(task, n=n_samples)

    try:
        if task in ("image_classification", "eeg_classification", "pose_estimation"):
            metrics = evaluator.evaluate_classification(str(model_path), samples, domain=domain)
        elif task in ("semantic_segmentation", "instance_segmentation"):
            metrics = evaluator.evaluate_segmentation(str(model_path), samples)
        elif task == "super_resolution":
            metrics = evaluator.evaluate_super_resolution(str(model_path), samples)
        elif task == "speech_recognition":
            metrics = evaluator.evaluate_asr(str(model_path), samples)
        elif task == "anomaly_detection":
            metrics = evaluator.evaluate_anomaly_detection(str(model_path), samples)
        else:
            metrics = evaluator.evaluate_classification(str(model_path), samples, domain=domain)

        print(f"[eval] Metrics: {metrics}")

        gate_passed = evaluator.check_gate(
            model_id, domain, task, metrics, raise_on_fail=raise_on_gate_fail
        )
        metrics["gate_passed"] = gate_passed
        metrics["model_id"] = model_id
        metrics["status"] = "passed" if gate_passed else "gate_failed"

    except AccuracyGateError as e:
        print(f"[eval] Gate FAILED: {e}")
        metrics = {"model_id": model_id, "status": "gate_failed", "error": str(e)}
    except Exception as e:
        print(f"[eval] Evaluation error: {e}")
        metrics = {"model_id": model_id, "status": "error", "error": str(e)}

    return metrics


def evaluate_all(
    config_path: str,
    platform: str = "imx8mplus",
    domain: Optional[str] = None,
    task: Optional[str] = None,
) -> list:
    manager = ModelManager(config_path)
    model_ids = manager.list_models(domain=domain, task=task, platform=platform)
    if not model_ids:
        print(f"[eval] No models found for platform={platform} domain={domain} task={task}")
        return []
    results = []
    for model_id in model_ids:
        result = evaluate_model(config_path, model_id, platform=platform, raise_on_gate_fail=False)
        results.append(result)
    # Summary
    passed = [r for r in results if r.get("status") == "passed"]
    failed = [r for r in results if r.get("status") not in ("passed", "model_not_found")]
    not_found = [r for r in results if r.get("status") == "model_not_found"]
    print(f"\n{'='*60}")
    print(f"Evaluation Summary: {len(model_ids)} models on {platform}")
    print(f"  Passed    : {len(passed)}")
    print(f"  Failed    : {len(failed)}")
    print(f"  Not found : {len(not_found)} (run recipes first)")
    print(f"{'='*60}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NXP eIQ Evaluation Pipeline")
    parser.add_argument("--config", default="config/pipeline_config.yaml")
    parser.add_argument("--model", help="Single model ID (omit for all)")
    parser.add_argument("--platform", default="imx8mplus")
    parser.add_argument("--domain")
    parser.add_argument("--task")
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--all-models", action="store_true")
    args = parser.parse_args()

    if args.model:
        result = evaluate_model(
            args.config, args.model, platform=args.platform, n_samples=args.n_samples
        )
        print(f"\n[eval] Final: {result}")
        sys.exit(0 if result.get("status") == "passed" else 1)
    else:
        evaluate_all(
            args.config,
            platform=args.platform,
            domain=args.domain,
            task=args.task,
        )
