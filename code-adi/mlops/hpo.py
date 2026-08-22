"""Optuna HPO for ADI AI8X QAT hyperparameter search."""

import json
from pathlib import Path
from typing import Callable

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


class QATHPORunner:
    """Searches optimal QAT hyperparameters for AI8X models."""

    def __init__(self, study_name: str, storage: str = None, n_trials: int = 20):
        self.study_name = study_name
        self.storage = storage
        self.n_trials = n_trials

    def _define_params(self, trial) -> dict:
        return {
            "weight_bits": trial.suggest_categorical("weight_bits", [4, 8]),
            "bias_bits": trial.suggest_categorical("bias_bits", [8]),
            "act_bits": trial.suggest_categorical("act_bits", [8]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
            "qat_epochs": trial.suggest_int("qat_epochs", 5, 30),
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
