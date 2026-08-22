"""Optuna HPO for TI EdgeAI Model Zoo compilation parameter search."""

import json
from pathlib import Path
from typing import Callable, Optional

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


def _dummy_trial_fn(trial):
    return 0.0


class TIDLHPORunner:
    """Optuna-based HPO for TIDL compilation hyperparameters."""

    PARAM_SPACE = {
        "tensor_bits": (8, 16),
        "num_calib_frames": (50, 500),
        "calib_accuracy_threshold": (0.5, 1.0),
        "max_num_subgraphs": (1, 8),
    }

    def __init__(self, study_name: str, storage: str = None, n_trials: int = 20):
        self.study_name = study_name
        self.storage = storage
        self.n_trials = n_trials

    def _define_params(self, trial) -> dict:
        if not OPTUNA_AVAILABLE:
            return {}
        return {
            "tensor_bits": trial.suggest_categorical("tensor_bits", [8, 16]),
            "num_calib_frames": trial.suggest_int("num_calib_frames", 50, 500, step=50),
            "calib_accuracy_threshold": trial.suggest_float(
                "calib_accuracy_threshold", 0.5, 1.0
            ),
            "max_num_subgraphs": trial.suggest_int("max_num_subgraphs", 1, 8),
        }

    def run(self, objective_fn: Callable[[dict], float],
            direction: str = "maximize") -> dict:
        """Run HPO. objective_fn receives param dict and returns metric float."""
        if not OPTUNA_AVAILABLE:
            return {"error": "optuna not installed", "best_params": {}, "best_value": 0.0}

        def wrapped_objective(trial):
            params = self._define_params(trial)
            return objective_fn(params)

        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            direction=direction,
            load_if_exists=True,
        )
        study.optimize(wrapped_objective, n_trials=self.n_trials)
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
