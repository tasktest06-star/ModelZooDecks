"""Optuna HPO for NXP eIQ TFLite INT8 quantization parameter search."""

import json
from pathlib import Path
from typing import Callable

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


class TFLiteINT8HPORunner:
    """Searches optimal post-training quantization parameters for NXP TFLite INT8."""

    def __init__(self, study_name: str, storage: str = None, n_trials: int = 20):
        self.study_name = study_name
        self.storage = storage
        self.n_trials = n_trials

    def _define_params(self, trial) -> dict:
        return {
            "num_calib_steps": trial.suggest_int("num_calib_steps", 50, 500, step=50),
            "representative_dataset_size": trial.suggest_categorical(
                "representative_dataset_size", [100, 200, 500, 1000]
            ),
            "input_size": trial.suggest_categorical("input_size", [128, 160, 192, 224]),
            "activation_dtype": trial.suggest_categorical(
                "activation_dtype", ["int8", "uint8"]
            ),
        }

    def run(self, objective_fn: Callable[[dict], float],
            direction: str = "maximize") -> dict:
        if not OPTUNA_AVAILABLE:
            return {"error": "optuna not installed", "best_params": {}, "best_value": 0.0}

        def wrapped(trial):
            return objective_fn(self._define_params(trial))

        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            direction=direction,
            load_if_exists=True,
        )
        study.optimize(wrapped, n_trials=self.n_trials)
        return {
            "best_params": study.best_params,
            "best_value": study.best_value,
            "n_trials": len(study.trials),
            "study_name": self.study_name,
        }

    def save_best_params(self, result: dict, output_path: str):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
