# Code Review Issues — ModelZooDecks

**Review date:** 2026-08-22  
**Branches reviewed:** `main` (TI), `analog-devices` (ADI), `nxp-modelzoo` (NXP)

---

## Severity Legend
- 🔴 **Critical** — will raise an exception or produce silently wrong results at runtime
- 🟠 **High** — causes deployment or infrastructure failures
- 🟡 **Medium** — produces warnings, deprecated behavior, or reduced correctness
- 🟢 **Low** — style, unused code, minor correctness

---

## TI EdgeAI (`main` branch — `code/`)

| # | Severity | File | Issue | Status |
|---|----------|------|-------|--------|
| TI-1 | 🔴 Critical | `pipelines/orchestrated_pipeline.py` | Calls `evaluator.evaluate()` — method does not exist; actual method is `evaluator.run()`. Always falls to except-block, returns `top1_accuracy=0.0` for every model. | Fixed |
| TI-2 | 🔴 Critical | `pipelines/eval_pipeline.py` | `Optional` used in return type annotation before `from typing import Optional` import — raises `NameError` at import time. | Fixed |
| TI-3 | 🔴 Critical | `mlops/experiment_tracker.py` | `mlflow.register_model(f"runs:/{run_id}/model", ...)` raises `MlflowException: No artifact at path 'model'` — no model artifact is ever logged in the run before registration. | Fixed |
| TI-4 | 🟠 High | `requirements.txt` | Missing `mlflow>=2.8.0`, `prefect>=2.14.0`, `optuna>=3.4.0`, `evidently>=0.4.0` — newly added modules will fail to import. | Fixed |
| TI-5 | 🟡 Medium | `infrastructure/stacks/pipeline_stack.py` | `PipelineStack.__init__` accepted `compute_stack` parameter that was never used, causing confusion and potential future misuse. | Fixed |
| TI-6 | 🟡 Medium | `pipelines/orchestrated_pipeline.py` | Default `tracking_uri="file:./mlruns"` deprecated in MLflow 2.x — changed to `sqlite:///mlflow.db`. | Fixed |
| TI-7 | 🟢 Low | `mlops/hpo.py` | Unused `Optional` import in `hpo.py`. | Fixed |
| TI-8 | 🟢 Low | `mlops/experiment_tracker.py` | Unused `Any` import. | Fixed |

---

## ADI AI8X (`analog-devices` branch — `code-adi/`)

| # | Severity | File | Issue | Status |
|---|----------|------|-------|--------|
| ADI-1 | 🔴 Critical | `pipelines/orchestrated_pipeline.py:38` | Calls `evaluator.evaluate(model_id, device, n_samples)` — method does not exist. Actual call is `evaluator.run(model_id, device=device, num_frames=n_samples)`. All evaluations silently return `top1_accuracy=0.0`. | Needs fix |
| ADI-2 | 🔴 Critical | `mlops/experiment_tracker.py:66` | `mlflow.register_model(f"runs:/{run_id}/model", model_name)` raises `MlflowException` — no model artifact logged in run before registration call. | Needs fix |
| ADI-3 | 🟠 High | `infrastructure/stacks/pipeline_stack.py:39-47` | IAM policy ARNs hardcoded as `arn:aws:s3:::adi-modelzoo-weights` but bucket created as `adi-modelzoo-weights-{account}-{region}`. ARNs never match → CodeBuild gets `AccessDenied` on all S3 operations. | Needs fix |
| ADI-4 | 🟠 High | `infrastructure/stacks/pipeline_stack.py:114` | CodeBuild buildspec calls `python code-adi/pipelines/synthesize_pipeline.py` which does not exist. Should call `train_pipeline.py --stage synthesize`. | Needs fix |
| ADI-5 | 🟡 Medium | `mlops/experiment_tracker.py:14` | Default `tracking_uri="file:./mlruns"` deprecated in MLflow 2.x — should use `sqlite:///mlflow.db`. | Needs fix |

---

## NXP eIQ (`nxp-modelzoo` branch — `code-nxp/`)

| # | Severity | File | Issue | Status |
|---|----------|------|-------|--------|
| NXP-1 | 🔴 Critical | `code-nxp/` (repo-level) | MLOps files (`experiment_tracker.py`, `drift_detector.py`, `hpo.py`, `orchestrated_pipeline.py`, `test_experiment_tracker.py`) were written and tested but commit `a413998` never made it onto the branch. Files present in working tree only. | Needs fix |
| NXP-2 | 🔴 Critical | `mlops/experiment_tracker.py` | Same `mlflow.register_model()` artifact-missing bug as TI/ADI. | Needs fix |
| NXP-3 | 🔴 Critical | `pipelines/orchestrated_pipeline.py` | `Evaluator(config_path)` — constructor takes a `dict`, not a file path string. Raises `AttributeError: 'str' object has no attribute 'get'`. | Needs fix |
| NXP-4 | 🔴 Critical | `mlops/evaluator.py` | No unified `evaluate(model_id, platform, n_samples)` dispatcher exists — only task-specific methods. Orchestrated pipeline calls the missing method, raising `AttributeError`. | Needs fix |
| NXP-5 | 🟠 High | `mlops/model_manager.py` | `compile_vela()` calls `subprocess.run(["vela", ...], check=True)` without checking `is_available()` first — raises `FileNotFoundError` if Vela not on PATH instead of a clear error. | Needs fix |
| NXP-6 | 🟠 High | `requirements.txt` | Missing `librosa>=0.10.0` and `soundfile>=0.12.0` — both are required by `compute_mfcc()` and `compute_wav2letter_features()` in `data_pipeline.py`. Also missing `mlflow`, `prefect`, `optuna`, `evidently`. | Needs fix |
| NXP-7 | 🟠 High | `mlops/artifact_manager.py` | `_write_inference_example()` hardcodes `np.zeros([1, input_size[0], input_size[1], 3])` — fails for 1D audio inputs (`[296, 39]` → index OOB) and produces wrong shape for 3D EEG inputs. | Needs fix |
| NXP-8 | 🟡 Medium | `mlops/model_manager.py` | Dual-YAML design flaw: `ModelManager` opens the registry YAML as both pipeline config AND registry. When invoked with `--config config/model_registry.yaml`, accuracy gates are silently disabled. | Needs fix |
| NXP-9 | 🟡 Medium | `mlops/recipe_runner.py` | Docker volume mount `f"-v {abs_model_dir}:/workspace"` uses backslash paths on Windows — invalid Docker syntax. Requires WSL2-style path conversion. | Needs fix |
| NXP-10 | 🟡 Medium | `mlops/vela_compiler.py` | If multiple `*_vela.tflite` files exist in output dir (from prior run), `candidates[0]` returns an arbitrary file. | Needs fix |
| NXP-11 | 🟢 Low | `infrastructure/stacks/__pycache__/` | `.pyc` files committed to git (no `.gitignore` present in `code-nxp/`). | Needs fix |

---

## Cross-cutting Issues (all branches)

| # | Severity | Description |
|---|----------|-------------|
| X-1 | 🔴 Critical | `ExperimentTracker.register_model()` in all three codebases calls `mlflow.register_model()` without first logging a model artifact — will always raise `MlflowException` in production. Fix: log `mlflow.log_dict({"model_id": model_id}, "model/metadata.json")` inside each `start_run()` context. |
| X-2 | 🟡 Medium | Default `tracking_uri="file:./mlruns"` in all three `ExperimentTracker` classes — deprecated in MLflow 2.x. Use `sqlite:///mlflow.db`. |
| X-3 | 🟡 Medium | Prefect graceful-degradation stubs (`flow`/`task` when prefect not installed) do not pass `**kwargs` through — `@flow(name=..., log_prints=True)` calls fail silently, making the stub hard to debug. |

---

## Summary

| Branch | Critical | High | Medium | Low | Total |
|--------|----------|------|--------|-----|-------|
| TI `main` | 3 | 1 | 2 | 2 | **8** |
| ADI `analog-devices` | 2 | 2 | 1 | 0 | **5** |
| NXP `nxp-modelzoo` | 4 | 3 | 3 | 1 | **11** |
| **Total** | **9** | **6** | **6** | **3** | **24** |

TI issues were fixed by the code review agent. ADI and NXP fixes are tracked above.
