# Session Handoff — TI EdgeAI Model Zoo Project

**Date:** 2026-08-22  
**Repo:** https://github.com/tasktest06-star/ModelZooDecks  
**Source repo:** https://github.com/TexasInstruments/edgeai-modelzoo (cloned to `C:\Users\Administrator\model_zoo\edgeai-modelzoo\`)

---

## What Was Built

### 1. Deep Research Document
**File:** `TI_EdgeAI_ModelZoo_Overview.md` (~26KB)  
13-section comprehensive reference covering:
- Zoo overview, version 11.2.0, 180+ models, 8 vision tasks, 5 SoC targets
- All model families with accuracy tables (ImageNet Top-1, COCO mAP, Cityscapes mIoU)
- Model formats: ONNX (`onnxrt`), TFLite (`tflitert`), TVM-compiled (`tvmrt`)
- INT8 quantization strategy, QAT variants, `.link` file download mechanism
- TIDL runtime, MMA/C7x accelerator dispatch
- Full toolchain ecosystem (edgeai-benchmark, edgeai-tidl-tools, Edge AI Studio, EVM Cloud)
- Scripts directory reference (20+ export scripts in `edgeai-modelzoo/scripts/`)

### 2. LaTeX Beamer Slide Deck
**File:** `TI_EdgeAI_ModelZoo_Slides.tex` (37 slides)  
Theme: Madrid, TI brand colors (`TIRed{200,16,46}`, `TIBlue{0,114,188}`)  
Compile: `pdflatex TI_EdgeAI_ModelZoo_Slides.tex` (requires `texlive-full`)

**Section breakdown:**
| Section | Slides | Key content |
|---------|--------|-------------|
| Overview | 3 | Zoo stats, model count, 8 task categories |
| Supported Hardware | 2 | SoC specs table, TOPS bar chart (pgfplots) |
| Model Categories | 1 | 8-task taxonomy |
| Image Classification | 3 | MobileNet/EfficientNet/Swin families, accuracy scatter (pgfplots) |
| Object Detection | 3 | YOLOX-Lite variants, mAP vs resolution chart |
| Semantic Segmentation | 2 | DeepLabV3-Lite, FPNLite families |
| Advanced Vision Tasks | 3 | Depth estimation, keypoint, face detection, 3D detection |
| Model Formats & Runtimes | 2 | 3 runtime paths TikZ diagram |
| Quantization | 2 | INT8 pipeline, QAT flow, accuracy comparison chart |
| Accuracy Reports | 2 | CSV report structure, INT8 vs FP32 table |
| Datasets | 2 | ImageNet, COCO, Cityscapes, VOC, TIDEx, KITTI |
| Ecosystem & Toolchain | 2 | Tool relationships diagram |
| **Device Compatibility & Tutorials** | **6** | SoC hardware specs, model-device compatibility matrix, tutorial links |
| **MLOps Pipeline** | **7** | Full pipeline flow, registry, preprocessing, CI/CD gates, deploy, monitoring |
| Summary | 2 | Key takeaways, next steps |

### 3. MLOps Python Code
**Directory:** `code/` (16 files, ~3,350 lines)

#### Config files
| File | Purpose |
|------|---------|
| `config/pipeline_config.yaml` | Master config: SoC=AM68A, task=detection, model=yolox-s-lite; training/export/compile/eval/monitoring sections |
| `config/model_registry.yaml` | 14 production models with per-SoC artifact IDs, FP32/INT8 metrics, input sizes |

#### Core modules (`mlops/`)
| Module | Key classes/functions |
|--------|----------------------|
| `model_manager.py` | `ModelManager`: loads registry, `download_model(name, soc)` reads `.link` files, `list_models(task, soc)`, `get_accuracy(name, precision)` |
| `data_pipeline.py` | `preprocess_image(img, task, input_size)` → NCHW float32; `letterbox_resize()` for YOLOX-style padding; `TASK_DEFAULTS` dict; `DataPipeline` with `get_dataloader()` |
| `evaluator.py` | `TopKAccuracy`, `MeanIoU`, `DetectionMetrics` accumulators; `Evaluator.run(model_id, soc, precision, mode)`; `AccuracyGateError` for CI failures |
| `artifact_manager.py` | `ArtifactManager.pack()` → `.tar.gz` bundle (TIDL artifacts + preproc/postproc configs + manifest); `unpack()`, `verify()` |
| `monitor.py` | `InferenceMonitor`: rolling deques for latency/confidence; `track_frame()` context manager; `check_drift()`; `save_log()` → timestamped JSON |

#### Pipeline orchestrators (`pipelines/`)
| Module | Purpose |
|--------|---------|
| `train_pipeline.py` | `TrainPipeline`: `train()` → `export()` → `compile(soc)` → `evaluate()`; CLI `--stage all` |
| `eval_pipeline.py` | `run_evaluation()`: FP32 + INT8 eval, comparison table, accuracy gate check, markdown report |
| `deploy_pipeline.py` | `Deployer.package()` + `push()` (local copy or SCP to EVM); `DeployPipeline.run()` orchestrates end-to-end |

#### CI/CD
| File | Purpose |
|------|---------|
| `tests/test_pipeline.py` | 20+ pytest tests: `TestDataPipeline` (8), `TestInferenceMonitor` (6), `TestArtifactManager` (5), `TestIntegration` (1); no hardware needed |
| `.github/workflows/mlops_ci.yml` | 6 jobs: lint → unit-tests → evaluate (3-task matrix) → package-artifacts → trigger-qat-retrain → nightly |

---

## Key Technical Facts

### TI SoC Lineup
| SoC | TOPS | EVM Board | Notes |
|-----|------|-----------|-------|
| TDA4VM | 8 | J721E EVM | Jacinto 7, automotive |
| AM62A | 1 | SK-AM62A-LP | Entry-level |
| AM67A | 2 | SK-AM67A | Mid-range |
| AM68A | 8 | SK-AM68A | Reference eval SoC (used in config) |
| AM69A | 32 | SK-AM69A | High-end, multi-core |

### Model Naming Conventions
- `-lite` suffix → ReLU replaces Swish, SE blocks removed, depthwise convolutions, fixed input shape
- `onnxrt` / `tflitert` / `tvmrt` → runtime identifier in artifact directory names
- Artifact IDs: `ONR-OD-8220-yolox-s-lite-mmdet-coco-640x640` (format: `ONR-<task>-<id>-<name>-<dataset>-<resolution>`)

### .link File Mechanism
Models are NOT stored in git. Each `modelartifacts/<SoC>/8bits/` directory contains `.link` files whose content is the download URL. `ModelManager.download_model()` reads these to fetch artifacts.

### Accuracy Gates (CI)
| Task | Metric | Max INT8 drop |
|------|--------|---------------|
| Classification | Top-1 | 2.0 pp |
| Object Detection | mAP | 1.0 pp |
| Segmentation | mIoU | 1.5 pp |

---

## Git History
```
563898b  MLOps pipeline slides (7) + code/ directory (16 files)
c88b834  Device compatibility slides + tutorial references (6 slides)
9de31e2  First commit: README + Overview.md + Slides.tex
```

---

## Local Paths
| Resource | Path |
|----------|------|
| ModelZooDecks repo | `C:\Users\Administrator\model_zoo\ModelZooDecks\` |
| edgeai-modelzoo source | `C:\Users\Administrator\model_zoo\edgeai-modelzoo\` |
| Model artifacts (SoC) | `edgeai-modelzoo\modelartifacts\<SoC>\8bits\` |
| Accuracy reports | `edgeai-modelzoo\reports\` |
| Export scripts | `edgeai-modelzoo\scripts\` |
| Git remote | `https://github.com/tasktest06-star/ModelZooDecks.git` |

---

## Quick Verification Commands
```bash
# Compile slides (requires texlive-full)
cd C:\Users\Administrator\model_zoo\ModelZooDecks
pdflatex TI_EdgeAI_ModelZoo_Slides.tex

# Run unit tests (no hardware needed)
cd C:\Users\Administrator\model_zoo\ModelZooDecks\code
python -m pytest tests/ -v

# Preprocessing smoke test
python -c "
from mlops.data_pipeline import preprocess_image
import numpy as np
img = np.zeros((480, 640, 3), dtype='uint8')
for task in ['classification', 'object_detection', 'segmentation']:
    out = preprocess_image(img, task)
    print(f'{task}: {out.shape}')
"

# Run eval pipeline (dry run — no real model needed)
python pipelines/eval_pipeline.py \
    --config config/pipeline_config.yaml \
    --model yolox-s-lite --soc AM68A

# Run deploy pipeline (local bundle only)
python pipelines/deploy_pipeline.py \
    --config config/pipeline_config.yaml \
    --model yolox-s-lite --soc AM68A --target local
```

---

## Possible Next Steps
- Render the PDF slides (`pdflatex`) and review layout
- Add more models to `model_registry.yaml` (e.g., MobileNetV3-Large variants for AM62A)
- Wire `Evaluator` to real `edgeai-benchmark` output CSVs from `edgeai-modelzoo/reports/`
- Add a Jupyter notebook demo (`code/notebooks/quickstart.ipynb`)
- Extend CI matrix to cover all 5 SoCs
- Add `--dry-run` flag to `train_pipeline.py` for CI testing without GPU
