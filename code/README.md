# TI EdgeAI Model Zoo — MLOps Pipeline

Production-ready MLOps pipeline for the [TI EdgeAI Model Zoo](https://github.com/TexasInstruments/edgeai-modelzoo).  
Supports ONNX, TFLite, and TIDL formats across TDA4VM, AM68A, AM69A, AM62A, and AM67A SoCs.

## Prerequisites

- Python 3.8+
- Ubuntu 20.04+ (or WSL2 on Windows)
- Git clone of [edgeai-modelzoo](https://github.com/TexasInstruments/edgeai-modelzoo) for model weights (`.link` files must be resolved)
- TI TIDL Tools *(for on-device compilation)*: https://github.com/TexasInstruments/edgeai-tidl-tools
- EVM board or TI Edge AI Cloud *(for hardware evaluation)*

## Installation

```bash
git clone https://github.com/tasktest06-star/ModelZooDecks
cd ModelZooDecks
pip install -r code/requirements.txt
```

## Quick Start

```bash
cd code

# 1. List all models in the registry
python -c "
from mlops.model_manager import ModelManager
mm = ModelManager('config/pipeline_config.yaml')
print(mm.list_models())
"

# 2. List models for a specific SoC
python -c "
from mlops.model_manager import ModelManager
mm = ModelManager('config/pipeline_config.yaml')
print(mm.list_models(soc='AM62A'))
"

# 3. Run evaluation — single model (dry-run, no hardware required)
python pipelines/eval_pipeline.py \
  --config config/pipeline_config.yaml \
  --model yolox_s_lite \
  --soc AM68A \
  --task object_detection \
  --n-samples 10

# 4. Run all models for a task category
python pipelines/eval_pipeline.py \
  --config config/pipeline_config.yaml \
  --task image_classification \
  --soc AM62A \
  --all-models

# 5. Deploy a bundle to EVM board via SCP
python pipelines/deploy_pipeline.py \
  --config config/pipeline_config.yaml \
  --model yolox_s_lite \
  --soc TDA4VM \
  --target scp \
  --board-host 192.168.1.100 \
  --board-path /home/root/models/

# 6. Run unit tests (no hardware needed)
python -m pytest tests/ -v
```

## Supported Models (34 total)

### Image Classification — ImageNet (15 models)

| Model | Top-1 | GFLOPs | Params | Best SoC |
|-------|-------|--------|--------|----------|
| mobilenet_v2_lite | 72.3% | 0.30 | 3.5M | AM62A+ |
| mobilenet_v2_tf | 71.8% | 0.31 | 3.4M | AM62A+ |
| mobilenet_v3_large_lite | 74.8% | 0.22 | 5.5M | AM62A+ |
| mobilenet_v3_large | 74.4% | 0.22 | 5.5M | AM62A+ |
| mobilenet_v3_small | 67.2% | 0.06 | 2.5M | AM62A |
| inception_v1 (INT8) | 69.8% | 1.5 | 6.6M | AM62A+ |
| inception_v3 | 77.9% | 5.7 | 23.8M | AM68A+ |
| resnet50_lite | 76.8% | 4.1 | 25.6M | AM68A+ |
| resnet101_lite | 78.9% | 7.8 | 44.5M | AM68A+ |
| regnet_x_400mf_lite | 72.6% | 0.40 | 5.2M | AM62A+ |
| regnet_x_800mf_lite | 75.2% | 0.80 | 7.3M | AM62A+ |
| regnet_x_1600mf_lite | 77.9% | 1.6 | 9.2M | AM67A+ |
| resnext50_lite | 77.3% | 4.2 | 25.0M | AM68A+ |
| swin_tiny | 81.2% | 4.5 | 28.3M | AM68A+ |
| fastvit_s12 | 79.3% | 1.4 | 8.8M | AM67A+ |

### Object Detection — COCO (11 models)

| Model | mAP@50 | GFLOPs | Best SoC |
|-------|--------|--------|----------|
| yolox_pico_lite | 22.1% | 0.29 | AM62A |
| yolox_nano_lite | 25.3% | 1.1 | AM62A+ |
| yolox_tiny_lite | 32.6% | 6.5 | AM62A+ |
| yolox_s_lite | 38.9% | 26.9 | AM62A+ |
| yolox_m_lite | 46.0% | 73.8 | AM67A+ |
| yolov7_l_lite | 51.0% | 104 | AM68A+ |
| rtmdet_m_lite | 56.2% | 38.5 | AM67A+ |
| rtmdet_l_lite | 59.6% | 79.3 | AM68A+ |
| ssd_mobilenetv2_lite | 25.5% | 3.9 | AM62A+ |
| ssd_regnetx200mf_lite | 26.0% | 0.6 | AM62A |
| efficientdet_lite0 | 34.0% | 5.2 | AM62A+ |

### Semantic Segmentation — COCO-Seg21 (4 models)

| Model | mIoU | GFLOPs | Best SoC |
|-------|------|--------|----------|
| deeplabv3plus_mobilenetv2 | 64.0% | 9.8 | AM62A+ |
| deeplabv3_mobilenetv3_large | 65.0% | 6.8 | AM62A+ |
| fpn_regnetx800mf | 68.0% | 17.9 | AM67A+ |
| lraspp_mobilenetv3_large | 62.0% | 2.0 | AM62A |

### Keypoint Detection — COCO (3 models)

| Model | AP@50 | GFLOPs | Best SoC |
|-------|-------|--------|----------|
| yoloxpose_tiny_lite | 58.0% | 6.5 | AM62A+ |
| yoloxpose_s_lite | 67.0% | 27.0 | AM62A+ |
| yolox_s_pose_lite | 65.0% | 26.9 | AM62A+ |

### Monocular Depth — NYUDepthV2 (2 models)

| Model | Rel.Err | GFLOPs | Best SoC |
|-------|---------|--------|----------|
| midas_small | 0.142 | 9.2 | AM62A+ |
| fast_depth | 0.158 | 0.74 | AM62A |

## Accuracy Gates

CI fails automatically if measured accuracy drops below these thresholds (set in `config/pipeline_config.yaml`):

| Task | Metric | Default Gate |
|------|--------|--------------|
| image_classification | Top-1 | ≥ 68.0% |
| object_detection | mAP@50 | ≥ 22.0% |
| semantic_segmentation | mIoU | ≥ 58.0% |
| keypoint_detection | AP@50 | ≥ 55.0% |
| monocular_depth | Rel.Error | ≤ 0.18 |

## Pipeline Architecture

```
config/
  pipeline_config.yaml     # SoC target, task, eval gates, monitoring
  model_registry.yaml      # All 34 models with metrics and paths

mlops/
  model_manager.py         # Registry CRUD, SoC filtering, artifact download
  data_pipeline.py         # Task-aware preprocessing (letterbox/normalize/etc.)
  evaluator.py             # ONNX + TFLite inference, Top-1/mAP/mIoU/AP metrics
  artifact_manager.py      # TIDL bundle pack/unpack/verify
  monitor.py               # Latency tracking, FPS, confidence drift alerts

pipelines/
  train_pipeline.py        # Train → export → compile → evaluate (4 stages)
  eval_pipeline.py         # Batch evaluation with gate checking
  deploy_pipeline.py       # Bundle packaging + SCP to EVM board

tests/
  test_pipeline.py         # 20+ pytest unit tests (hardware-free)

.github/workflows/
  mlops_ci.yml             # 6-job CI: lint → tests → eval matrix → deploy
```

## SoC Target Guide

| SoC | TOPS | Recommended models |
|-----|------|--------------------|
| AM62A | 1 | MobileNet, YOLOX-Pico/Nano, SSD-200MF, LRASPP, FastDepth |
| AM67A | 4 | RegNetX-1.6GF, YOLOX-M, RTMDet-M, FPN-RegNetX800 |
| AM68A | 8 | ResNet50/101, YOLOv7, RTMDet-L, Swin-T, FPN |
| AM69A | 32 | All models; multi-camera pipelines |
| TDA4VM | 8 | All models; automotive ADAS pipelines |

## Downloading Model Weights

Model weights are stored as `.link` files in the edgeai-modelzoo repo. Use the TI download script:

```bash
cd /path/to/edgeai-modelzoo
python tools/scripts/download_models.py --model_type vision
```

Or download individual models:
```bash
python tools/scripts/download_models.py \
  --model_path models/vision/detection/coco/edgeai-mmdet/yolox_s_lite_640x640_20220221_model.onnx
```
