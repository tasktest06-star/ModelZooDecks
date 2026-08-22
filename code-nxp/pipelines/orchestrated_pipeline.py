"""Prefect 2 orchestrated pipeline for NXP eIQ Model Zoo."""

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


@task(name="load-nxp-registry", retries=2)
def load_registry(config_path: str) -> dict:
    logger = get_run_logger()
    with open(config_path) as f:
        registry = yaml.safe_load(f)
    logger.info(f"Loaded {len(registry.get('models', {}))} NXP models")
    return registry


@task(name="evaluate-nxp-model", retries=1)
def evaluate_model_task(model_id: str, platform: str, config_path: str,
                        tracker: ExperimentTracker) -> dict:
    logger = get_run_logger()
    logger.info(f"Evaluating {model_id} on {platform}")
    with tracker.start_run(model_id=model_id, platform=platform) as run:
        results = {"top1_accuracy": 0.0, "model_id": model_id}
        try:
            from mlops.evaluator import Evaluator
            evaluator = Evaluator(config_path)
            # NXP Evaluator has task-specific methods; dispatch by domain
            model_cfg = {}
            with open(config_path) as _f:
                import yaml as _yaml
                model_cfg = _yaml.safe_load(_f).get("models", {}).get(model_id, {})
            domain = model_cfg.get("domain", "vision")
            task = model_cfg.get("task", "image_classification")
            if domain == "vision" and task == "image_classification":
                raw = evaluator.evaluate_classification(model_id, platform)
            elif domain == "vision" and task == "object_detection":
                raw = evaluator.evaluate_segmentation(model_id, platform)
            elif task == "super_resolution":
                raw = evaluator.evaluate_super_resolution(model_id, platform)
            else:
                raw = {"top1_accuracy": 0.0, "note": f"no evaluator for {task}"}
            results = {
                "top1_accuracy": raw.get("top1_accuracy", raw.get("psnr_db", 0.0)),
                **raw,
            }
        except Exception as e:
            logger.warning(f"Eval failed for {model_id}: {e}")
        tracker.log_eval_metrics(
            {k: v for k, v in results.items() if isinstance(v, (int, float))}
        )
        return {**results, "run_id": run.info.run_id}


@task(name="check-nxp-drift")
def check_drift_task(model_id: str, platform: str, current_accuracy: float) -> dict:
    detector = DriftDetector(reference_path=f"drift_refs/{model_id}_{platform}.json")
    return detector.check_accuracy_drift(current_accuracy)


@task(name="vela-compile-check")
def vela_compile_check_task(model_id: str, registry: dict) -> bool:
    """Check if model needs Vela compilation for imx93."""
    logger = get_run_logger()
    model_cfg = registry.get("models", {}).get(model_id, {})
    for platform_cfg in model_cfg.get("supported_platforms", []):
        if platform_cfg.get("platform") == "imx93" and platform_cfg.get("vela_required"):
            logger.info(f"{model_id} requires Vela compilation for imx93")
            return True
    return False


@task(name="register-nxp-model")
def register_model_task(run_id: str, model_id: str, platform: str,
                        accuracy: float, gate: float,
                        tracker: ExperimentTracker):
    logger = get_run_logger()
    if not run_id or accuracy < gate:
        logger.warning(f"Skipping registration: {model_id} acc={accuracy:.4f} gate={gate}")
        return None
    try:
        version = tracker.register_model(
            run_id=run_id,
            model_name=f"nxp-{model_id}-{platform}",
            accuracy_gate=gate,
        )
        tracker.transition_to_production(f"nxp-{model_id}-{platform}", version)
        logger.info(f"Registered nxp-{model_id}-{platform} v{version}")
        return version
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        return None


@flow(name="nxp-eiq-pipeline", log_prints=True)
def nxp_eiq_flow(
    config_path: str = "config/model_registry.yaml",
    platform: str = "imx8mplus",
    accuracy_gate: float = 0.70,
    tracking_uri: str = "sqlite:///mlflow.db",
    models: list = None,
):
    """Full NXP eIQ evaluation + Vela check + registration Prefect flow."""
    logger = get_run_logger()
    tracker = ExperimentTracker(tracking_uri=tracking_uri)
    registry = load_registry(config_path)

    model_ids = models or list(registry.get("models", {}).keys())
    logger.info(f"Running {len(model_ids)} models on {platform}")

    results = []
    for model_id in model_ids:
        needs_vela = vela_compile_check_task(model_id=model_id, registry=registry)
        if needs_vela and platform == "imx93":
            logger.info(f"{model_id}: Vela compilation required for imx93")

        result = evaluate_model_task(
            model_id=model_id, platform=platform,
            config_path=config_path, tracker=tracker,
        )
        results.append(result)

        drift = check_drift_task(
            model_id=model_id, platform=platform,
            current_accuracy=result.get("top1_accuracy", 0.0),
        )
        if drift.get("drift_detected"):
            logger.warning(f"Drift detected for {model_id}: {drift}")

        register_model_task(
            run_id=result.get("run_id", ""),
            model_id=model_id, platform=platform,
            accuracy=result.get("top1_accuracy", 0.0),
            gate=accuracy_gate, tracker=tracker,
        )

    passed = [r for r in results if r.get("top1_accuracy", 0) >= accuracy_gate]
    logger.info(f"Done: {len(passed)}/{len(results)} passed gate {accuracy_gate}")
    return {"total": len(results), "passed": len(passed), "results": results}


if __name__ == "__main__":
    nxp_eiq_flow()
