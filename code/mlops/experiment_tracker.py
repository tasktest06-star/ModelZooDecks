"""MLflow experiment tracker for TI EdgeAI Model Zoo."""

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow
import mlflow.tracking


class ExperimentTracker:
    """Wraps MLflow for TI EdgeAI edge-model evaluation tracking."""

    EXPERIMENT_NAME = "ti-edgeai-modelzoo"

    def __init__(self, tracking_uri: str = None, experiment_name: str = None):
        self.tracking_uri = tracking_uri or os.getenv(
            "MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"
        )
        self.experiment_name = experiment_name or self.EXPERIMENT_NAME
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

    @contextmanager
    def start_run(self, model_id: str, soc: str, task: str, tags: dict = None):
        """Context manager for a single model evaluation run."""
        run_tags = {"model_id": model_id, "soc": soc, "task": task}
        if tags:
            run_tags.update(tags)
        with mlflow.start_run(tags=run_tags) as run:
            mlflow.log_param("model_id", model_id)
            mlflow.log_param("soc", soc)
            mlflow.log_param("task", task)
            yield run

    def log_eval_metrics(self, metrics: dict, step: int = None):
        """Log evaluation metrics (accuracy, mAP, mIoU, etc.)."""
        mlflow.log_metrics(metrics, step=step)

    def log_compile_params(self, tidl_params: dict):
        """Log TIDL compilation parameters."""
        prefixed = {f"tidl_{k}": v for k, v in tidl_params.items()}
        mlflow.log_params(prefixed)

    def log_latency(self, latency_ms: float, soc: str):
        mlflow.log_metric(f"latency_ms_{soc}", latency_ms)

    def log_artifact(self, local_path: str, artifact_path: str = None):
        mlflow.log_artifact(local_path, artifact_path)

    def log_model_bundle(self, bundle_path: str, model_id: str):
        mlflow.log_artifact(bundle_path, f"bundles/{model_id}")

    def register_model(self, run_id: str, model_name: str, accuracy_gate: float,
                       metric_key: str = "top1_accuracy") -> str:
        """Register model if it passes accuracy gate. Returns registered model version."""
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        metric_val = run.data.metrics.get(metric_key, 0.0)
        if metric_val < accuracy_gate:
            raise ValueError(
                f"Model {model_name} failed gate: {metric_key}={metric_val:.4f} < {accuracy_gate}"
            )
        model_uri = f"runs:/{run_id}/model"
        result = mlflow.register_model(model_uri, model_name)
        client.set_registered_model_tag(model_name, "soc_validated", "true")
        return result.version

    def transition_to_production(self, model_name: str, version: str):
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name, version=version, stage="Production"
        )

    def get_best_run(self, metric: str = "top1_accuracy") -> dict:
        """Return params + metrics of the best run by metric."""
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(self.experiment_name)
        if not experiment:
            return {}
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} DESC"],
            max_results=1,
        )
        if not runs:
            return {}
        best = runs[0]
        return {"run_id": best.info.run_id, **best.data.params, **best.data.metrics}
