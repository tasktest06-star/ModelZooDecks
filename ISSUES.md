# Code Review — Known Issues & Fixes

Reviewed 2026-08-22. Applies to `main`, `analog-devices`, and `nxp-modelzoo` branches.

---

## CRITICAL BUGS

### 1. Wrong method name in all three `orchestrated_pipeline.py` files
| | |
|---|---|
| **Files** | `code/pipelines/orchestrated_pipeline.py:46`, `code-adi/pipelines/orchestrated_pipeline.py:38`, `code-nxp/pipelines/orchestrated_pipeline.py` |
| **Type** | bug |
| **Description** | All three orchestrated pipelines call `evaluator.evaluate(model_id=..., ...)` but none of the three `Evaluator` classes have an `evaluate()` method. TI and ADI evaluators expose `evaluator.run(...)`. NXP exposes task-specific methods (`evaluate_classification`, etc.) |
| **Symptom** | `AttributeError: 'Evaluator' object has no attribute 'evaluate'` at runtime |
| **Fix** | Change `evaluator.evaluate(...)` → `evaluator.run(...)` in TI and ADI. For NXP, dispatch to the correct task-specific method based on model domain. The call is wrapped in `try/except Exception` so it degrades gracefully, but results will always be `{"top1_accuracy": 0.0}` silently. |

---

### 2. `Optional` used before import in `eval_pipeline.py`
| | |
|---|---|
| **File** | `code/pipelines/eval_pipeline.py:32,35` |
| **Type** | import-error |
| **Description** | `Optional` is used in function annotations on lines 32–35, but `from typing import Optional` appears on line 147 — after the function definition. Python evaluates annotations lazily in some versions, but the file-level `from typing import Optional` must appear before any use at module scope. Under `from __future__ import annotations` this would work, but that import is absent. In Python 3.9 without that future import this raises `NameError: name 'Optional' is not defined` when the function is first called in non-lazy contexts. |
| **Fix** | Move `from typing import Optional` to the top of the file (line 14, after `import yaml`). |

---

### 3. `register_model` fails — no model artifact logged in the run
| | |
|---|---|
| **Files** | `code/mlops/experiment_tracker.py:65`, `code-adi/mlops/experiment_tracker.py`, `code-nxp/mlops/experiment_tracker.py` |
| **Type** | bug |
| **Description** | `mlflow.register_model(f"runs:/{run_id}/model", model_name)` requires a model artifact to exist at `runs:/{run_id}/model` in the MLflow store. The `start_run` context manager only logs params/metrics — no model is ever logged via `mlflow.pyfunc.log_model()` or similar. MLflow raises `MlflowException: Run ... has no artifacts at path 'model'`. |
| **Fix** | Before calling `mlflow.register_model`, log a minimal artifact. Simplest fix: add `mlflow.log_dict({"registered": True}, "model/metadata.json")` inside `register_model`, or use `mlflow.log_text(model_name, "model/name.txt")`. This satisfies the artifact path requirement without needing a real framework-specific model object. |

---

## MISSING DEPENDENCIES

### 4. `requirements.txt` missing MLOps packages (TI branch only)
| | |
|---|---|
| **File** | `code/requirements.txt` |
| **Type** | missing-dependency |
| **Description** | The TI MLflow commit added `experiment_tracker.py`, `drift_detector.py`, `hpo.py`, `orchestrated_pipeline.py` which import `mlflow`, `prefect`, `optuna`, `evidently` — but `requirements.txt` was not updated. ADI and NXP requirements.txt files were correctly updated by their agents. |
| **Fix** | Append to `code/requirements.txt`: `mlflow>=2.8.0`, `prefect>=2.14.0`, `optuna>=3.4.0`, `evidently>=0.4.0` |

---

## MINOR / STYLE ISSUES

### 5. Unused `compute_stack` parameter in `PipelineStack`
| | |
|---|---|
| **File** | `code/infrastructure/stacks/pipeline_stack.py:21` |
| **Type** | style |
| **Description** | `PipelineStack.__init__` accepts `compute_stack: ComputeStack` as a parameter and imports `ComputeStack`, but never references `compute_stack` in the method body. The pipeline only uses `storage_stack.model_bucket`. The import and parameter add dead weight and a false API promise. |
| **Fix** | Remove the `compute_stack` parameter and the `from stacks.compute_stack import ComputeStack` import. Update the test and `app.py` accordingly. |

### 6. Inconsistent MLflow tracking URI default (TI)
| | |
|---|---|
| **File** | `code/pipelines/orchestrated_pipeline.py:91` |
| **Type** | incorrect-logic |
| **Description** | `ti_edgeai_eval_flow` defaults `tracking_uri="file:./mlruns"` but `ExperimentTracker.__init__` defaults to `sqlite:///mlflow.db`. When the flow is called without arguments it passes `file:./mlruns` to ExperimentTracker. File-based MLflow stores have known issues with concurrent access and are deprecated in newer MLflow versions. |
| **Fix** | Change the flow default to `tracking_uri="sqlite:///mlflow.db"` to match `ExperimentTracker`'s own default. |

### 7. Unused imports in `hpo.py` and `experiment_tracker.py`
| | |
|---|---|
| **Files** | `code/mlops/hpo.py:5` (`Optional`), `code/mlops/experiment_tracker.py:6` (`Any`) |
| **Type** | style |
| **Description** | `Optional` imported but unused in `hpo.py`; `Any` imported but unused in `experiment_tracker.py`. Both trigger `flake8 F401`. |
| **Fix** | Remove unused imports. |

### 8. `DetectionMetrics.update` type hint uses `list[dict]` (Python 3.9+ only)
| | |
|---|---|
| **File** | `code/mlops/evaluator.py:88` |
| **Type** | style |
| **Description** | `def update(self, preds: list[dict], ...)` uses the built-in generic `list[dict]` syntax which requires Python 3.9+. The repo doesn't declare a minimum Python version; if run on 3.8 this raises `TypeError`. Same pattern in `eval_pipeline.py:127` (`list[dict]`). |
| **Fix** | Either add `from __future__ import annotations` at the top of both files, or use `List[Dict]` from `typing`. |

---

## SUMMARY TABLE

| # | Severity | File | Issue |
|---|----------|------|-------|
| 1 | 🔴 Critical | `*/pipelines/orchestrated_pipeline.py` (all 3) | `evaluator.evaluate()` doesn't exist |
| 2 | 🔴 Critical | `code/pipelines/eval_pipeline.py` | `Optional` used before import |
| 3 | 🔴 Critical | `*/mlops/experiment_tracker.py` (all 3) | `register_model` fails — no artifact logged |
| 4 | 🟠 High | `code/requirements.txt` | Missing mlflow/prefect/optuna/evidently |
| 5 | 🟡 Medium | `code/infrastructure/stacks/pipeline_stack.py` | Unused `compute_stack` parameter |
| 6 | 🟡 Medium | `code/pipelines/orchestrated_pipeline.py` | Inconsistent tracking URI default |
| 7 | 🟢 Low | `code/mlops/hpo.py`, `experiment_tracker.py` | Unused imports |
| 8 | 🟢 Low | `code/mlops/evaluator.py`, `eval_pipeline.py` | `list[dict]` requires Python 3.9+ |
