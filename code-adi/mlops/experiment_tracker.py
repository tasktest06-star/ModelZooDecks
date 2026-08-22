"""MLflow experiment tracker for ADI AI8X Model Zoo."""

import os
from contextlib import contextmanager
from typing import Optional
import mlflow
import mlflow.tracking


class ExperimentTracker:
    EXPERIMENT_NAME = "adi-ai8x-modelzoo"

    def __init__(self, tracking_uri: str = None, experiment_name: str = None):
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
        self.experiment_name = experiment_name or self.EXPERIMENT_NAME
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

    @contextmanager
    def start_run(self, model_id: str, device: str, tags: dict = None):
        run_tags = {"model_id": model_id, "device": device, "framework": "ai8x"}
        if tags:
            run_tags.update(tags)
        with mlflow.start_run(tags=run_tags) as run:
            mlflow.log_param("model_id", model_id)
            mlflow.log_param("device", device)
            yield run

    def log_training_metrics(self, epoch: int, loss: float, accuracy: float):
        mlflow.log_metrics({"train_loss": loss, "train_accuracy": accuracy}, step=epoch)

    def log_qat_params(self, weight_bits: int, bias_bits: int, act_bits: int,
                       learning_rate: float, batch_size: int):
        mlflow.log_params({
            "weight_bits": weight_bits,
            "bias_bits": bias_bits,
            "act_bits": act_bits,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
        })

    def log_synthesis_results(self, model_id: str, device: str,
                               mac_utilization: float, sram_bytes: int,
                               flash_bytes: int, synthesis_success: bool):
        mlflow.log_metrics({
            "mac_utilization": mac_utilization,
            "sram_bytes": sram_bytes,
            "flash_bytes": flash_bytes,
        })
        mlflow.log_param("synthesis_success", synthesis_success)
        mlflow.set_tag("device_target", device)

    def log_eval_metrics(self, metrics: dict, step: int = None):
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str, artifact_path: str = None):
        mlflow.log_artifact(local_path, artifact_path)

    def register_model(self, run_id: str, model_name: str,
                       accuracy_gate: float, metric_key: str = "top1_accuracy") -> str:
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        val = run.data.metrics.get(metric_key, 0.0)
        if val < accuracy_gate:
            raise ValueError(f"{model_name}: {metric_key}={val:.4f} below gate {accuracy_gate}")
        with mlflow.start_run(run_id=run_id):
            mlflow.log_dict({"model_name": model_name, "metric": val}, "model/metadata.json")
        result = mlflow.register_model(f"runs:/{run_id}/model", model_name)
        client.set_registered_model_tag(model_name, "device_validated", "true")
        return result.version

    def transition_to_production(self, model_name: str, version: str):
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name, version=version, stage="Production"
        )

    def get_best_run(self, metric: str = "top1_accuracy") -> dict:
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
