"""Prefect 2 orchestrated pipeline for ADI AI8X Model Zoo."""

import yaml

try:
    from prefect import flow, task, get_run_logger
    PREFECT_AVAILABLE = True
except ImportError:
    PREFECT_AVAILABLE = False
    def flow(fn=None, **kwargs): return fn if fn else lambda f: f
    def task(fn=None, **kwargs): return fn if fn else lambda f: f
    def get_run_logger():
        import logging; return logging.getLogger(__name__)

from mlops.experiment_tracker import ExperimentTracker
from mlops.drift_detector import DriftDetector


@task(name="load-adi-registry", retries=2)
def load_registry(config_path: str) -> dict:
    logger = get_run_logger()
    with open(config_path) as f:
        registry = yaml.safe_load(f)
    logger.info(f"Loaded {len(registry.get('models', {}))} ADI models")
    return registry


@task(name="evaluate-adi-model", retries=1)
def evaluate_model_task(model_id: str, device: str, config_path: str,
                        tracker: ExperimentTracker) -> dict:
    logger = get_run_logger()
    logger.info(f"Evaluating {model_id} on {device}")
    with tracker.start_run(model_id=model_id, device=device) as run:
        results = {"top1_accuracy": 0.0, "model_id": model_id}
        try:
            from mlops.evaluator import Evaluator
            evaluator = Evaluator(config_path)
            raw = evaluator.run(model_name=model_id, device=device, n_samples=50)
            metrics = raw.get("metrics", raw)
            results = {
                "top1_accuracy": metrics.get("top1_accuracy", metrics.get("accuracy", 0.0)),
                **metrics,
            }
        except Exception as e:
            logger.warning(f"Eval failed for {model_id}: {e}")
        tracker.log_eval_metrics(
            {k: v for k, v in results.items() if isinstance(v, (int, float))}
        )
        return {**results, "run_id": run.info.run_id}


@task(name="check-adi-drift")
def check_drift_task(model_id: str, device: str, current_accuracy: float) -> dict:
    detector = DriftDetector(reference_path=f"drift_refs/{model_id}_{device}.json")
    return detector.check_accuracy_drift(current_accuracy)


@task(name="register-adi-model")
def register_model_task(run_id: str, model_id: str, device: str,
                        accuracy: float, gate: float,
                        tracker: ExperimentTracker):
    logger = get_run_logger()
    if not run_id or accuracy < gate:
        logger.warning(f"Skipping registration for {model_id}: acc={accuracy:.4f} gate={gate}")
        return None
    try:
        version = tracker.register_model(
            run_id=run_id,
            model_name=f"adi-{model_id}-{device}",
            accuracy_gate=gate,
        )
        tracker.transition_to_production(f"adi-{model_id}-{device}", version)
        logger.info(f"Registered adi-{model_id}-{device} v{version}")
        return version
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        return None


@flow(name="adi-ai8x-pipeline", log_prints=True)
def adi_ai8x_flow(
    config_path: str = "config/model_registry.yaml",
    device: str = "MAX78002",
    accuracy_gate: float = 0.85,
    tracking_uri: str = "file:./mlruns",
    models: list = None,
):
    """Full ADI AI8X evaluation + registration Prefect flow."""
    logger = get_run_logger()
    tracker = ExperimentTracker(tracking_uri=tracking_uri)
    registry = load_registry(config_path)

    model_ids = models or list(registry.get("models", {}).keys())
    logger.info(f"Running {len(model_ids)} models on {device}")

    results = []
    for model_id in model_ids:
        result = evaluate_model_task(
            model_id=model_id, device=device,
            config_path=config_path, tracker=tracker,
        )
        results.append(result)

        drift = check_drift_task(
            model_id=model_id, device=device,
            current_accuracy=result.get("top1_accuracy", 0.0),
        )
        if drift.get("drift_detected"):
            logger.warning(f"Accuracy drift detected for {model_id}: {drift}")

        register_model_task(
            run_id=result.get("run_id", ""),
            model_id=model_id, device=device,
            accuracy=result.get("top1_accuracy", 0.0),
            gate=accuracy_gate, tracker=tracker,
        )

    passed = [r for r in results if r.get("top1_accuracy", 0) >= accuracy_gate]
    logger.info(f"Done: {len(passed)}/{len(results)} passed gate {accuracy_gate}")
    return {"total": len(results), "passed": len(passed), "results": results}


if __name__ == "__main__":
    adi_ai8x_flow()
