# Project Handoff — Edge AI Model Zoo (TI / ADI / NXP)

**Last updated:** 2026-08-23  
**Repo:** https://github.com/tasktest06-star/ModelZooDecks  
**Working directory:** `C:\Users\Administrator\model_zoo\ModelZooDecks`

---

## Repository Structure

```
ModelZooDecks/
├── HANDOFF.md                        ← this file
├── COMPARISON.md                     ← cross-company model comparison (16 sections)
├── ISSUES.md                         ← 24 code review findings + fixes
├── README.md
├── TI_EdgeAI_ModelZoo_Overview.md    ← deep research doc (TI only)
│
├── TI_EdgeAI_ModelZoo_Slides.tex     ← 37-slide TI zoo overview deck
├── TI_EdgeAI_Products_Slides.tex     ← ~40-slide TI hardware + model fit deck
├── ADI_ModelZoo_Slides.tex           ← ADI zoo overview deck
├── ADI_AI8X_Products_Slides.tex      ← ~40-slide ADI hardware + model fit deck
├── NXP_ModelZoo_Slides.tex           ← NXP zoo overview deck
├── NXP_eIQ_Products_Slides.tex       ← ~40-slide NXP hardware + model fit deck
└── ModelZoo_Comparison_Slides.tex    ← 40-slide cross-company comparison deck
```

### Branches

| Branch | Contents |
|--------|----------|
| `main` | All slide decks, COMPARISON.md, ISSUES.md, TI MLOps code (`code/`), TI CDK infra |
| `analog-devices` | ADI MLOps code (`code-adi/`), ADI CDK infra, ADI combination pipelines |
| `nxp-modelzoo` | NXP MLOps code (`code-nxp/`), NXP CDK infra, NXP combination pipelines |
| `model-combination-ti` | TI 3-application multi-model pipeline code |
| `model-combination-adi` | ADI 3-application multi-model pipeline code |
| `model-combination-nxp` | NXP 3-application multi-model pipeline code |

---

## Slide Decks

### 1. `TI_EdgeAI_ModelZoo_Slides.tex` — 37 slides
TI EdgeAI Model Zoo overview. Covers zoo stats, model families, classification/detection/seg/pose/depth tables, MLOps pipeline (6 slides), model registry, data pipeline, CI/CD, deployment/monitoring, code structure.  
Colors: `TIRed{199,0,57}` / `TIBlue{0,71,133}`

### 2. `TI_EdgeAI_Products_Slides.tex` — ~40 slides
TI hardware products + model fit analysis.
- AM62A / AM67A / AM68A / AM69A / TDA4VM SoC family TikZ cards + specs table
- TIDL MMA NPU architecture block diagram
- TIDL software stack flow (Train → Compile → Deploy)
- edgeai-modelzoo repository structure
- Classification / Detection / Segmentation / Pose / Depth hardware fit matrices
- Why each `_lite` model fits TIDL op whitelist
- Latency bars (all 34 models on AM68A), FPS comparison across SoCs
- Memory scatter (weights + activations vs platform budget)
- Application grid by SoC, development ecosystem diagram, MLOps table, scorecard

### 3. `ADI_ModelZoo_Slides.tex`
ADI zoo overview deck (ADI brand colors).

### 4. `ADI_AI8X_Products_Slides.tex` — 678 lines, ~40 slides
ADI hardware products + model fit analysis.
- MAX32690 / MAX78002 / ADSP-SC835 device cards + specs table
- MAX78002 CNN accelerator architecture (64 parallel processors, 5 MB on-chip SRAM, 442 TOPS/W)
- ai8x QAT training → synthesis flow diagram
- QAT vs PTQ accuracy drop bar chart (ADI QAT <0.6% vs PTQ 1.4–3.2%)
- Vision model fit table (FPN, TinierSSD, MobileNetV2 variants)
- FPN layer-by-layer NPU coverage TikZ diagram
- Audio models table (DS-CNN 94.5%, RNNoise, DTLN, GenreNet) + dual audio routing
- CNN accelerator op-set compliance grid (supported vs unsupported)
- 5 MB SRAM stacked bar (all 15 models fit on-chip)
- Energy-per-inference log-scale chart (8–340 µJ vs competitors)
- Battery life estimation line chart (coin cell vs AA)
- Smart Sensor Node FSM pipeline (SLEEP → VWW → KWS → FPN)
- Dev ecosystem (ai8x-training → ai8xize → MaximSDK), MLOps table, scorecard

### 5. `NXP_ModelZoo_Slides.tex`
NXP zoo overview deck.

### 6. `NXP_eIQ_Products_Slides.tex` — 803 lines, ~40 slides
NXP hardware products + model fit analysis.
- MCX N947 / RT1170 / i.MX 93 / i.MX 8M Plus platform spectrum TikZ + specs table
- i.MX 8M Plus 2.3 TOPS NPU architecture block diagram
- eIQ recipe.sh Docker flow (Train → PTQ → Vela → Eval → Deploy)
- Vela compiler pipeline (TFLite INT8 → Ethos-U65 command stream)
- Classification / Detection hardware fit matrices per platform
- NXP-exclusive models: SCI low-light, FaceNet512, WHENet, EEG TCNet — TikZ cards
- Fast-SRGAN architecture (PixelShuffle CPU fallback, 90% on NPU)
- Audio models: microspeech, DS-CNN, wav2letter (full ASR, WER 7.2%)
- Per-channel vs per-tensor PTQ accuracy drop comparison
- Memory budget stacked bar (14 models × weights + activations)
- NPU operator coverage bar chart (all 29 models)
- Latency comparison i.MX 8M Plus vs i.MX 93 Ethos-U65
- Power efficiency vs competitors (inferences/sec/W)
- Driver Monitoring System (DMS) TikZ pipeline
- Dev ecosystem (eIQ Toolkit, Vela, eIQ Portal, Yocto, MCUXpresso), MLOps table, scorecard

### 7. `ModelZoo_Comparison_Slides.tex` — 40 slides
Cross-company comparison deck (Madrid theme, company colors).
- Platform overview TikZ + hardware power spectrum (0.3 mA → 30 W)
- Model inventory stacked xbar (11 task categories × 3 companies)
- Classification accuracy xbar + accuracy-vs-params Pareto scatter
- Detection mAP xbar + latency-vs-mAP scatter
- Audio models (KWS/ASR/denoising) + NXP-exclusive TikZ
- Quantization PTQ vs QAT side-by-side + accuracy-drop ybar
- 3 multi-model pipeline TikZ diagrams (TI ADAS, ADI sensor node FSM, NXP face recog)
- All 9 pipelines scatter (latency vs power log scale)
- Dataset coverage matrix (12 datasets × 3 companies)
- Deployment workflow 6-step column grid
- MLOps stack layered diagram
- Tradeoffs scatter + recommendations + star-rating scorecard + conclusion

---

## Code (`main` branch: `code/`, `analog-devices`: `code-adi/`, `nxp-modelzoo`: `code-nxp/`)

### TI EdgeAI (`code/`)

| File | Purpose |
|------|---------|
| `config/pipeline_config.yaml` | Full MLOps config (SoC target, training, compile, eval gates, monitoring) |
| `config/model_registry.yaml` | 34 production models with per-SoC artifact IDs, metrics, input sizes |
| `mlops/experiment_tracker.py` | MLflow ExperimentTracker with `start_run()` context manager, `register_model()` with artifact gate |
| `mlops/hpo.py` | `TIDLHPORunner` — Optuna HPO for TIDL compile params |
| `mlops/data_pipeline.py` | Task-aware preprocessing: letterbox (detection), normalize (classification/seg) |
| `mlops/evaluator.py` | ONNX/TFLite inference, Top-1/MeanIoU/mAP metrics, `AccuracyGateError` |
| `mlops/artifact_manager.py` | TIDL bundle pack/unpack/verify with preproc+postproc configs |
| `mlops/monitor.py` | Rolling-window latency tracking, FPS, confidence drift detection |
| `mlops/drift_detector.py` | Evidently AI accuracy + confidence drift detection |
| `pipelines/orchestrated_pipeline.py` | Prefect 2 `@flow` / `@task` DAG with graceful degradation |
| `pipelines/train_pipeline.py` | 4-stage orchestrator: train → export → compile → evaluate |
| `pipelines/eval_pipeline.py` | Standalone evaluation with batch eval, gate checking, markdown reports |
| `pipelines/deploy_pipeline.py` | Bundle packaging + SCP push to EVM boards |
| `tests/test_pipeline.py` | 20+ pytest unit tests |
| `.github/workflows/mlops_ci.yml` | 6-job CI: lint → unit tests → eval matrix → registry validate → package → deploy |
| `infrastructure/` | AWS CDK v2: StorageStack + ComputeStack + PipelineStack + MonitoringStack (24 tests) |
| `combinations/adas_scene_understanding.py` | 4-stage ADAS pipeline: YOLOx-S → DeepLabV3+ → FastDepth → YOLOXPose |
| `combinations/people_analytics.py` | People counting + skeleton-based activity classification |
| `combinations/smart_home_security.py` | Motion-gated 2-stage detection pipeline |

### ADI AI8X (`code-adi/`)

| File | Purpose |
|------|---------|
| `mlops/experiment_tracker.py` | MLflow tracker for QAT runs |
| `mlops/hpo.py` | `QATHPORunner` — Optuna HPO (weight_bits, lr, batch_size, qat_start_epoch) |
| `mlops/drift_detector.py` | Accuracy + confidence drift |
| `pipelines/orchestrated_pipeline.py` | Prefect 2 train+synthesize+evaluate DAG |
| `infrastructure/` | AWS CDK v2 with g4dn.xlarge spot for QAT, ai8xize synthesis stage (31 tests) |
| `combinations/smart_sensor_node.py` | Hierarchical VWW → KWS → FPN pipeline, power state FSM |
| `combinations/predictive_maintenance.py` | Vibration + thermal anomaly detection |
| `combinations/smart_audio_intelligence.py` | Dual-path audio routing (speech vs music) |

### NXP eIQ (`code-nxp/`)

| File | Purpose |
|------|---------|
| `mlops/experiment_tracker.py` | MLflow tracker for PTQ + Vela runs |
| `mlops/hpo.py` | `TFLiteINT8HPORunner` — Optuna HPO (calib_steps, dataset_size, per_channel, vela_memory_config) |
| `mlops/drift_detector.py` | Accuracy + confidence + PSNR drift |
| `pipelines/orchestrated_pipeline.py` | Prefect 2 recipe.sh + Vela + eval DAG |
| `infrastructure/` | AWS CDK v2 with Fargate recipe runner, Vela compile stage (29 tests) |
| `combinations/low_light_face_recognition.py` | SCI → FaceDet → FaceNet512 + WHENet pipeline |
| `combinations/driver_monitoring.py` | SCI → FaceDet → WHENet + emotion → drowsiness/distraction alert |
| `combinations/smart_video_analytics.py` | Detection + segmentation + SR enhancement |

---

## Known Issues / Fixes Applied

All 24 findings from `ISSUES.md` have been fixed. Key ones:

| Issue | Fix applied |
|-------|-------------|
| MLflow `register_model` without artifact | Added `mlflow.log_dict(...)` before registration (all 3 branches) |
| Wrong evaluator method (`evaluate()`) | Fixed to call correct dispatcher (all 3 branches) |
| MLflow file store deprecated | Switched to `sqlite:///mlflow.db` (all 3 branches) |
| NXP Evaluator takes dict not path | Added `yaml.safe_load()` before constructing evaluator |
| ADI IAM ARNs missing account/region | Changed to wildcard `adi-modelzoo-weights-*` |
| NXP artifact_manager input shape crash | Fixed `np.zeros([1] + list(input_size))` |
| NXP Vela without availability guard | Added `is_available()` check before `subprocess.run` |
| NXP missing librosa/soundfile | Added to `requirements.txt` |
| Optuna test state leak | Save/restore `OPTUNA_AVAILABLE` around test assignments |
| TI `Optional` before import | Moved `from typing import Optional` to top of file |

---

## How to Compile Slide Decks

```bash
# Requires: texlive-full (or texlive + pgfplots + tikz + booktabs + fontawesome5)
cd C:/Users/Administrator/model_zoo/ModelZooDecks
pdflatex TI_EdgeAI_Products_Slides.tex
pdflatex ADI_AI8X_Products_Slides.tex
pdflatex NXP_eIQ_Products_Slides.tex
pdflatex ModelZoo_Comparison_Slides.tex
pdflatex TI_EdgeAI_ModelZoo_Slides.tex
```

Run twice if TikZ cross-references don't resolve on the first pass.

---

## How to Run MLOps Code

```bash
# TI
cd C:/Users/Administrator/model_zoo/ModelZooDecks/code
pip install -r requirements.txt
pytest tests/ -v
python pipelines/eval_pipeline.py --config config/pipeline_config.yaml --model yolox-s-lite --soc AM68A

# ADI (analog-devices branch)
git checkout analog-devices
cd code-adi
pip install -r requirements.txt
pytest tests/ -v

# NXP (nxp-modelzoo branch)
git checkout nxp-modelzoo
cd code-nxp
pip install -r requirements.txt
pytest tests/ -v
```

---

## Infrastructure (AWS CDK v2)

```bash
# TI (main branch)
cd code/infrastructure
pip install -r requirements.txt
cdk synth
cdk deploy --all

# ADI (analog-devices branch)
cd code-adi/infrastructure
cdk deploy --all   # note: uses g4dn.xlarge spot for QAT

# NXP (nxp-modelzoo branch)
cd code-nxp/infrastructure
cdk deploy --all   # Fargate recipe runner + Vela compile stage
```

---

## Source Model Zoos

| Company | Repo location |
|---------|--------------|
| TI EdgeAI | `C:\Users\Administrator\model_zoo\edgeai-modelzoo\` |
| ADI AI8X | ai8x-training + ai8x-synthesis (GitHub: MaximIntegratedAI) |
| NXP eIQ | eiq-apps-collection + i.MX ML examples (GitHub: NXPmicro) |

---

## Git Commit Timeline (most recent first)

```
af7abbc  Add NXP eIQ product-specific slide deck
4c5495d  Add ADI AI8X product-specific slide deck
315e1a1  Add TI EdgeAI product-specific slide deck
74a2247  Add cross-company comparison slide deck (40 slides)
7b7e13f  Add COMPARISON.md cross-company comparison
73a4c0e  Add ISSUES.md code review findings
6519d72  Fix 8 code issues on main (Optional import, evaluator, MLflow URI...)
6d4184b  Add TI MLOps: MLflow + Prefect + Optuna + Evidently
6a8967e  Add TI AWS CDK v2 infrastructure
f9b62b7  Expand TI registry to 34 models
563898b  Add MLOps pipeline code and slides
```

---

## Next Steps (suggested)

1. **Compile PDFs** — run `pdflatex` on all 7 `.tex` files
2. **ADI/NXP slide decks** — the original overview decks (`ADI_ModelZoo_Slides.tex`, `NXP_ModelZoo_Slides.tex`) could be updated to match the new product-deck style
3. **CDK deployment** — run `cdk deploy` on all three branches against an AWS account
4. **Integration tests** — run `pytest` on all three branches with real model artifacts
5. **CI pipeline** — activate `.github/workflows/mlops_ci.yml` by pushing to a branch with GitHub Actions enabled
