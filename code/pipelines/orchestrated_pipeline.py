"""Prefect 2 orchestrated pipeline for TI EdgeAI Model Zoo."""

from pathlib import Path
from typing import Optional

try:
    from prefect import flow, task, get_run_logger
    from prefect.tasks import task_input_hash
    from datetime import timedelta
    PREFECT_AVAILABLE = True
except ImportError:
    PREFECT_AVAILABLE = False
    def flow(fn=None, **kwargs):
        return fn if fn else lambda f: f
    def task(fn=None, **kwargs):
        return fn if fn else lambda f: f
    def get_run_logger():
        import logging
        return logging.getLogger(__name__)

import yaml
from mlops.experiment_tracker import ExperimentTracker
from mlops.drift_detector import DriftDetector
from mlops.model_manager import ModelManager
from mlops.evaluator import Evaluator


@task(name="load-registry", retries=2)
def load_registry(config_path: str) -> dict:
    logger = get_run_logger()
    with open(config_path) as f:
        registry = yaml.safe_load(f)
    logger.info(f"Loaded {len(registry.get('models', {}))} models from registry")
    return registry


@task(name="evaluate-model", retries=1)
def evaluate_model_task(model_id: str, soc: str, task_type: str,
                        config_path: str, tracker: ExperimentTracker) -> dict:
    logger = get_run_logger()
    logger.info(f"Evaluating {model_id} on {soc}")

    with tracker.start_run(model_id=model_id, soc=soc, task=task_type) as run:
        evaluator = Evaluator(config_path)
        try:
            results = evaluator.evaluate(model_id=model_id, soc=soc, n_samples=100)
        except Exception as e:
            logger.warning(f"Evaluation failed for {model_id}: {e}")
            results = {"error": str(e), "top1_accuracy": 0.0}

        tracker.log_eval_metrics(
            {k: v for k, v in results.items() if isinstance(v, (int, float))}
        )
        return {"model_id": model_id, "soc": soc, "run_id": run.info.run_id, **results}


@task(name="check-drift")
def check_drift_task(model_id: str, current_accuracy: float,
                     reference_path: str) -> dict:
    detector = DriftDetector(reference_path=reference_path)
    return detector.check_accuracy_drift(current_accuracy)


@task(name="register-model")
def register_model_task(run_id: str, model_id: str, soc: str,
                        accuracy: float, gate: float,
                        tracker: ExperimentTracker) -> Optional[str]:
    logger = get_run_logger()
    if accuracy < gate:
        logger.warning(f"{model_id} accuracy {accuracy:.4f} below gate {gate} — skipping registration")
        return None
    try:
        version = tracker.register_model(
            run_id=run_id,
            model_name=f"ti-edgeai-{model_id}-{soc}",
            accuracy_gate=gate,
        )
        tracker.transition_to_production(f"ti-edgeai-{model_id}-{soc}", version)
        logger.info(f"Registered {model_id} v{version} → Production")
        return version
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        return None


@flow(name="ti-edgeai-eval-pipeline", log_prints=True)
def ti_edgeai_eval_flow(
    config_path: str = "config/model_registry.yaml",
    soc: str = "AM68A",
    accuracy_gate: float = 0.68,
    tracking_uri: str = "file:./mlruns",
    models: list = None,
):
    """Full TI EdgeAI evaluation + registration Prefect flow."""
    logger = get_run_logger()
    tracker = ExperimentTracker(tracking_uri=tracking_uri)
    registry = load_registry(config_path)

    model_ids = models or list(registry.get("models", {}).keys())[:10]
    logger.info(f"Running eval for {len(model_ids)} models on {soc}")

    results = []
    for model_id in model_ids:
        model_cfg = registry.get("models", {}).get(model_id, {})
        task_type = model_cfg.get("task", "image_classification")

        result = evaluate_model_task(
            model_id=model_id,
            soc=soc,
            task_type=task_type,
            config_path=config_path,
            tracker=tracker,
        )
        results.append(result)

        drift = check_drift_task(
            model_id=model_id,
            current_accuracy=result.get("top1_accuracy", 0.0),
            reference_path=f"drift_refs/{model_id}_{soc}.json",
        )
        if drift.get("drift_detected"):
            logger.warning(f"Drift detected for {model_id}: {drift}")

        register_model_task(
            run_id=result.get("run_id", ""),
            model_id=model_id,
            soc=soc,
            accuracy=result.get("top1_accuracy", 0.0),
            gate=accuracy_gate,
            tracker=tracker,
        )

    passed = [r for r in results if r.get("top1_accuracy", 0) >= accuracy_gate]
    logger.info(f"Pipeline complete: {len(passed)}/{len(results)} models passed gate")
    return {"total": len(results), "passed": len(passed), "results": results}


if __name__ == "__main__":
    ti_edgeai_eval_flow()
