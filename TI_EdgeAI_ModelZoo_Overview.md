# Texas Instruments EdgeAI Model Zoo — Comprehensive Overview

**Repository:** [TexasInstruments/edgeai-modelzoo](https://github.com/TexasInstruments/edgeai-modelzoo)  
**Version:** 11.2.0  
**Date of Report:** August 2026

---

## Table of Contents

1. [What is the TI EdgeAI Model Zoo?](#1-what-is-the-ti-edgeai-model-zoo)
2. [Repository Structure](#2-repository-structure)
3. [Supported Hardware — TI SoC Targets](#3-supported-hardware--ti-soc-targets)
4. [Model Categories & Tasks](#4-model-categories--tasks)
   - 4.1 [Image Classification](#41-image-classification)
   - 4.2 [Object Detection](#42-object-detection)
   - 4.3 [Semantic Segmentation](#43-semantic-segmentation)
   - 4.4 [Depth Estimation](#44-depth-estimation)
   - 4.5 [3D Object Detection](#45-3d-object-detection)
   - 4.6 [Keypoint Detection / Human Pose Estimation](#46-keypoint-detection--human-pose-estimation)
   - 4.7 [6D Pose Estimation](#47-6d-pose-estimation)
   - 4.8 [Visual Localization](#48-visual-localization)
5. [Model Formats & Runtimes](#5-model-formats--runtimes)
6. [Pre-compiled Model Artifacts](#6-pre-compiled-model-artifacts)
7. [Quantization & Embedded Optimizations](#7-quantization--embedded-optimizations)
8. [Accuracy Reports](#8-accuracy-reports)
9. [Supported Datasets](#9-supported-datasets)
10. [Development Toolchain Ecosystem](#10-development-toolchain-ecosystem)
11. [Scripts & Export Utilities](#11-scripts--export-utilities)
12. [Model ID Convention](#12-model-id-convention)
13. [Summary Statistics](#13-summary-statistics)

---

## 1. What is the TI EdgeAI Model Zoo?

The **TI EdgeAI Model Zoo** is Texas Instruments' curated collection of pre-trained Deep Neural Network (DNN) models for embedded computer vision, optimized for deployment on TI's portfolio of AI-capable SoCs (System-on-Chips). It serves as a ready starting point for engineers developing intelligent edge applications in automotive, industrial, and robotics domains.

### Core Goals

- Provide **production-ready model baselines** across eight vision tasks
- Deliver models in **embedded-friendly formats** (ONNX, TFLite) ready for TIDL compilation
- Ship **pre-compiled artifacts** for five TI SoC families, eliminating the need for a full compilation pipeline during development
- Include **"Lite" architectural variants** that trade minimal accuracy loss for large inference speedups on resource-constrained hardware
- Support **INT8 quantization** (both Post-Training Quantization and QAT) aligned with TI's Deep Learning accelerators

### Who Is It For?

- Embedded software engineers integrating DNN models into TI SoC-based products
- AI/ML engineers benchmarking vision model performance on TI hardware
- Researchers using TI Edge AI Studio or the `edgeai-benchmark` framework

---

## 2. Repository Structure

```
edgeai-modelzoo/
├── README.md                           # Top-level introduction & quickstart
├── version.py                          # Version: 11.2.0
├── LICENSE.md                          # Per-source model licenses
├── make_release.sh                     # Release automation
├── run_download_modelartifacts.sh      # Downloads pre-compiled artifacts via .link files
├── run_create_models_links.sh          # Creates .link pointer files for models
├── run_create_modelartifacts_links.sh  # Creates .link pointer files for artifacts
│
├── models/                             # Model definitions & .link download pointers
│   ├── vision/
│   │   ├── classification/             # ~80+ classification models
│   │   ├── detection/                  # ~50+ detection models
│   │   ├── segmentation/               # ~30+ segmentation models
│   │   ├── depth_estimation/           # 2 depth models
│   │   ├── detection_3d/               # 4 BEV/LiDAR models
│   │   ├── object_6d_pose/             # 1 pose model
│   │   ├── keypoint/                   # 5 pose estimation models
│   │   ├── visual_localization/        # 1 localization model
│   │   └── high_resolution/            # High-resolution variants
│   ├── configs.yaml                    # Per-model config file mappings
│   └── model_infos.py                  # Python dict: all model IDs + metadata
│
├── modelartifacts/                     # Pre-compiled artifact .link files
│   ├── AM62A/8bits/
│   ├── AM67A/8bits/
│   ├── AM68A/8bits/
│   ├── AM69A/8bits/
│   └── TDA4VM/8bits/
│
├── docs/                               # Sphinx documentation source
│   ├── precompiled_modelartifacts.md
│   └── images/
│
├── reports/                            # Accuracy benchmark results (CSV + MD)
│   ├── accuracy_report_20250307-161945_pc.*
│   └── accuracy_report_20250310-190344_pc.*
│
└── scripts/                            # Export, conversion & utility scripts
```

> **Important:** Model binaries are NOT stored in Git. `.link` files contain hosted download URLs. Run `run_download_modelartifacts.sh` to fetch compiled artifacts.

---

## 3. Supported Hardware — TI SoC Targets

Pre-compiled INT8 artifacts are available for five TI SoC families:

| SoC | Product Line | Primary Application | Notes |
|-----|-------------|--------------------|-|
| **TDA4VM** | Jacinto 7 | ADAS, automotive surround-view | First-generation Jacinto7 automotive SoC |
| **AM62A** | Sitara AM6xA | Entry-level AI, industrial, automotive | Low-power edge AI |
| **AM67A** | Sitara AM6xA | Mid-range AI applications | Balanced performance/power |
| **AM68A** | Sitara AM6xA | High-performance edge AI | Used as accuracy benchmark reference SoC |
| **AM69A** | Sitara AM6xA | Top-tier multi-camera AI | Highest performance in the AM6xA family |

All artifacts are provided in **8-bit quantized (INT8)** format for maximum inference throughput on TI's MMA (Matrix Multiplication Accelerator) and C7x DSP cores.

---

## 4. Model Categories & Tasks

The zoo covers **eight computer vision tasks**:

| Task Code | Full Name | Metric | Dataset |
|-----------|-----------|--------|---------|
| `cl-` | Image Classification | Top-1 Accuracy % | ImageNet-1K |
| `od-` | Object Detection | mAP @ [0.5:0.95] | COCO 2017 |
| `ss-` | Semantic Segmentation | MeanIoU % | ADE20K, Cityscapes, COCO |
| `de-` | Depth Estimation | Delta1 % | NYUDepthV2 |
| `3dod-` | 3D Object Detection | mAP, NDS | PandaSet, KITTI |
| `kd-` | Keypoint Detection | AP [0.5:0.95] | COCO Keypoints |
| `6dpose-` | 6D Pose Estimation | AR, ADD(s) | YCB-Video |
| `visloc-` | Visual Localization | — | CARLA (synthetic) |

---

### 4.1 Image Classification

**Input:** 224×224 RGB (standard); some models use 256×256  
**Dataset:** ImageNet-1K (1000 classes, 50K validation images)  
**Metric:** Top-1 Accuracy %

#### Model Families

| Source Framework | Key Model Families | Format | Accuracy Range |
|-----------------|-------------------|--------|----------------|
| edgeai-torchvision (TI) | MobileNetV1/V2/V3Lite, QAT variants | ONNX | 62–73% |
| torchvision (PyTorch) | MobileNetV2, ShuffleNetV2, ResNet18/50, RegNetX | ONNX | 69–79% |
| edgeai-tv2 (TI, newer) | MobileNetV2-lite, ResNet50/101-lite, ResNeXt50/101, RegNetX | ONNX | 70–82% |
| HuggingFace Transformers (TI fork) | ViT, DeiT, LeViT, Swin-T/S/B/L, ConvNeXt | ONNX | 75–86% |
| TF-TPU / Google | EfficientNet-Lite0/1/2/4, EdgeTPU-S/M/L | TFLite | 75–83% |
| TF1 Model Garden | MobileNetV1/V2, SqueezeNet, DenseNet, Inception, NASNet | TFLite | 61–80% |
| MXNet / GluonCV | MobileNetV2, ResNet50, Xception | ONNX | 70–79% |
| Facebookresearch/pycls | RegNetX-200/400/800MF/1.6GF | ONNX | 68–79% |
| MLPerf | MobileNetV1, ResNet50, MobileNet-EdgeTPU | TFLite | 71–77% |

#### Complexity Spectrum

| Model | GigaMACs | Top-1 |
|-------|----------|-------|
| MobileNetV3Lite-Small | 0.054 | 62.68% |
| MobileNetV2-0.5x | 0.097 | 65.40% |
| MobileNetV1 (MLPerf) | 0.569 | 71.68% |
| MobileNetV2 | 0.300 | 72.13% |
| MobileNetV2-QAT | 0.300 | 71.76% |
| ResNet50 | 4.112 | 76.15% |
| EfficientNet-Lite4 | 2.645 | 81.5% |
| Swin-Small | 8.74 | 83.1% |
| Swin-Large | 34.5 | 86.15% |
| VGG16 | 15.35 | 71.59% |

**Transformer models** (Swin, ConvNeXt, ViT) deliver highest accuracy but require more compute. TI's TIDL now supports transformer operators including multi-head attention.

---

### 4.2 Object Detection

**Dataset:** COCO 2017 (80 classes), WIDER FACE (face detection)  
**Metric:** AP @ [IoU 0.5:0.95] %, AP50 %

#### Detector Families

| Detector Family | Source | Input Size | mAP Range |
|----------------|--------|-----------|-----------|
| YOLOX-nano/tiny/s/m/l/x-lite | edgeai-mmdetection | 416–640px | 24–50.6% |
| YOLOv7-l-lite / orig | edgeai-mmdetection | 640px | ~50% |
| YOLOv9-s-lite / plus | edgeai-mmdetection | 640px | ~46% |
| RTMDet-m/l-lite / orig | edgeai-mmdetection | 640px | ~49% |
| FCOS-r50-lite | edgeai-mmdetection | 512px | ~42% |
| CenterNet-r18 | edgeai-mmdetection | 512px | ~28% |
| EfficientDet-b0/b1-lite | edgeai-mmdetection | 512px | ~33–39% |
| SSD-MobileNetV2+FPN | edgeai-mmdetection | 320–512px | 27–39.8% |
| RegNetX+SSDLite | edgeai-mmdetection | 512–768px | 32–39% |
| SSD-MobileDetDSP | TF1 Model Garden | 320px | 28.9% |
| SSD-ResNet50-FPN | TF2 Model Garden | 640px | 34.3% |
| DETR-ResNet50 | HuggingFace | 768px | 42.0% |
| EfficientDet-lite0-ti-lite | Google AutoML | 512px | 33.61% |

#### Face Detection (WIDER FACE)
| Model | Input | AP50 |
|-------|-------|------|
| YOLOX-tiny-face-lite | 416px | 49.1% |
| YOLOX-s-face-lite | 512px | 63.4% |
| YOLOX-m-face-lite | 640px | 72.3% |

#### "Lite" Modification Details
TI's `-lite` suffix means these architectural changes were applied to optimize for TI hardware:
1. **Activation replacement:** Swish/H-Swish → ReLU (avoids non-linearity hardware overhead)
2. **Depthwise/grouped convolutions** in FPN and detection heads
3. **SE block removal** (Squeeze-and-Excitation) from backbones
4. **Fixed-aspect-ratio input** resizing (no dynamic shapes)
5. **Neck/head conv type alignment** with backbone type

---

### 4.3 Semantic Segmentation

**Datasets:** ADE20K (150/32 classes), Cityscapes (19 classes), COCOSeg21, VOC2012, TI-RoboKit  
**Metric:** MeanIoU %

| Model Family | Source | Dataset | MeanIoU |
|-------------|--------|---------|---------|
| MobileNetV2+DeepLabV3Lite | edgeai-tensorvision | ADE20K32 | 49.95–51.44% |
| MobileNetV2+UNetLite | edgeai-tensorvision | ADE20K32 | 50.40% |
| MobileNetV2+FPNLite | edgeai-tensorvision | COCOSeg21 | 57.68–58.12% |
| RegNetX800MF+FPNLite | edgeai-tensorvision | Cityscapes | 78.90% |
| LR-ASPP MobileNetV3 | torchvision | COCOSeg21 | 57.9% |
| DeepLabV3 ResNet50 | torchvision | COCOSeg21 | 66.4% |
| SegFormer-B0 | HuggingFace | ADE20K | 37.4% |
| SegFormer-B5 | HuggingFace | ADE20K | 51.1% |
| DeepLabV3 MobileNetV2 | TF DeepLab | VOC2012 | 66.94% |
| DeepLabV3 Xception65 | TF DeepLab | Cityscapes | 81.74% |
| DeepLabV3 MobileNetV2 (MLPerf) | MLPerf | ADE20K32 | 54.8% |
| MobileNetV2-QAT+FPNLite | edgeai-tensorvision | Cityscapes | ~76% |
| TI-RoboKit-Seg | edgeai-tensorvision | TI-RoboKit | — |

QAT segmentation models show ~0.5 MeanIoU drop vs FP32 equivalents.

---

### 4.4 Depth Estimation

**Dataset:** NYUDepthV2 (indoor RGB-D scenes)  
**Metric:** Delta1 % (% of predictions within factor of 1.25 of ground truth)

| Model | Input Size | GigaMACs | Delta1 % |
|-------|-----------|---------|---------|
| Fast Depth | 224×224 | 0.38 | 77.1% |
| MiDaS-small v2.1 | 256×256 | 4.63 | 86.67% |

Both models produce dense per-pixel depth maps and are fully accelerated by TIDL.

---

### 4.5 3D Object Detection

**Datasets:** PandaSet (BEV camera-based, 12 classes), KITTI (LiDAR-based, deprecated)  
**Metrics:** mAP, NDS (NuScenes Detection Score)

| Model | Type | Input | mAP | NDS |
|-------|------|-------|-----|-----|
| FastBEV-r18-f1 | Camera BEV, single-frame | 256×704 | 17.47% | 22.95% |
| FastBEV-r34-f4 | Camera BEV, 4-frame temporal | 256×704 | 23.07% | 29.53% |
| BEVFormer-tiny | Transformer BEV | 544×960 | 23.01% | 29.49% |
| PointPillars (KITTI) | LiDAR pillars | 496×432 | 76.36% (Car AP3D) | — |

KITTI-based PointPillars is marked **deprecated**. Camera-based BEV models (FastBEV, BEVFormer) are the active direction for camera-only 3D detection.

---

### 4.6 Keypoint Detection / Human Pose Estimation

**Dataset:** COCO Keypoints (17 human body joints)  
**Metric:** AP @ [IoU 0.5:0.95]  
**Source:** edgeai-mmpose (TI fork of MMPose)

| Model | Input | AP |
|-------|------|----|
| YOLOXPose-tiny-lite | 416×416 | 47.2% |
| YOLOXPose-s-lite | 640×640 | 56.4% |
| YOLOX-s-pose-ti-lite | 640×640 | 51.2% |

All use a **heatmap-free single-stage approach** — joint detection and pose estimation in one forward pass. Fully TIDL-accelerated.

---

### 4.7 6D Pose Estimation

**Dataset:** YCB-Video (21 household objects, RGB-D sequences)  
**Metrics:** AR (Average Recall), ADD(s) (Average Distance of Model Points)

| Model | Input | AR | ADD(s) |
|-------|------|----|--------|
| YOLOX-s-object-pose-ti-lite | 640×480 | 64.73% | 54.12% |

Single-stage end-to-end approach; estimates 3D rotation and translation directly from RGB input. Fully TIDL-accelerated with no post-processing on CPU.

---

### 4.8 Visual Localization

**Dataset:** CARLA (synthetic autonomous driving environment)  
**Model:** TIAD-DKAZE  
**Input:** 768×384  

Used for camera-based ego-position estimation within a pre-built map — relevant for low-cost autonomous navigation without GPS.

---

## 5. Model Formats & Runtimes

Three inference runtimes are supported:

| Runtime Prefix | Runtime Name | Source Format | Backend |
|---------------|-------------|--------------|---------|
| `TFL-` | **tflitert** | `.tflite` | TFLite delegate + TIDL |
| `TVM-` | **tvmrt / tvmdlr** | ONNX → TVM compiled | TVM + TIDL |
| `ONR-` | **onnxrt** | `.onnx` | ONNX Runtime + TIDL |

The actual hardware execution is always through **TIDL** — the runtime prefix simply reflects which model ingestion path was used. TIDL handles operator-level dispatch between the C7x DSP, MMA accelerator, and ARM cores.

---

## 6. Pre-compiled Model Artifacts

Located at `modelartifacts/<SoC>/8bits/` — each entry is a `.link` file pointing to a hosted `.tar.gz` archive.

### Artifact Bundle Contents
Each compiled artifact bundle contains:
- TIDL network files (compiled, hardware-specific)
- Calibration data
- Configuration YAML (`param.yaml`)
- Extract script (`extract.sh`)

### Artifact Naming Convention
```
<RUNTIME>-<TASK>-<ID>-<model_description>
Example: ONR-OD-8220-yolox-s-lite-mmdet-coco-640x640
         TFL-CL-0000-mobileNetV1-mlperf
         TVM-SS-5720-deeplabV3-mobV2-tv-cocoseg21-512x512
```

### Metadata Files
- `artifacts.yaml` — per-artifact metadata (size, session name, task type, shortlisted flag)
- `artifacts.csv` — tabular lookup of all artifacts
- `shortlisted_artifacts.yaml` — curated recommended models for quick start

---

## 7. Quantization & Embedded Optimizations

### INT8 Quantization
- All deployed artifacts use **INT8 (8-bit integer)** weights and activations
- Typical accuracy drop vs FP32: **0.1–2%** for CNN models
- Transformer models show larger drops: e.g., Swin-tiny FP32 82.1% → INT8 77.9%

### QAT (Quantization Aware Training)
Many models include QAT variants (suffix `-qat` or `-qat-p2`):

| Model | FP32 Accuracy | QAT Accuracy | Delta |
|-------|--------------|-------------|-------|
| MobileNetV2 | 72.13% | 71.76% | −0.37% |
| MobileNetV3Lite-Large | 72.22% | 71.61% | −0.61% |
| RegNetX-400MF | 73.81% | 73.38% | −0.43% |
| MobileNetV2+FPNLite Cityscapes | 77.27% | 76.85% | −0.42% |

### "Lite" Model Design Principles
TI's embedded optimization guidelines for model architecture:
1. **Activation functions:** Use ReLU only (avoid Swish, H-Swish, GELU for DSP efficiency)
2. **Convolution types:** Prefer depthwise separable or grouped convolutions in heads/necks
3. **Attention mechanisms:** Avoid or limit SE blocks; transformer attention must use TIDL-supported ops
4. **Input resolution:** Use fixed, compile-time-known resolution (no dynamic shapes)
5. **Upsampling:** Prefer bilinear upsampling over transposed convolutions in decoder heads

---

## 8. Accuracy Reports

Two report snapshots are included under `reports/`:

| Report | Date | Mode |
|--------|------|------|
| `accuracy_report_20250307-161945_pc` | 2025-03-07 | Accuracy benchmark |
| `accuracy_report_20250310-190344_pc` | 2025-03-10 | Performance benchmark |

### Report Schema
| Column | Meaning |
|--------|---------|
| `model_id` | Unique model identifier |
| `runtime_name` | tflitert / tvmrt / onnxrt |
| `task_type` | classification / detection / segmentation etc. |
| `dataset_name` | Evaluation dataset |
| `input_resolution` | Model input dimensions |
| `metric_name` | Top1Accuracy / MeanIoU / COCO_mAP |
| `AM68A_32bits-float-simulation_metric` | FP32 baseline (PC emulation) |
| `AM68A_8bits_metric` | INT8 accuracy on AM68A |
| `metric_reference` | Published accuracy from original training |

### Key Accuracy Examples

| Task | Model | FP32 | INT8 | Reference |
|------|-------|------|------|-----------|
| Classification | MobileNetV1 (MLPerf) | 71.3% | 71.2% | 71.7% |
| Classification | ResNet50 | 77.5% | 77.7% | 76.15% |
| Classification | EfficientNet-Lite4 | 82.6% | 82.4% | 81.5% |
| Classification | Swin-tiny | 82.1% | 77.9% | 80.43% |
| Detection | YOLOX-s-lite (COCO) | 38.6% | 38.2% | 38.3% |
| Detection | YOLOX-nano-lite | 24.9% | 24.6% | 24.8% |
| Keypoint | YOLOX-s-pose | 50.7% | 49.8% | 49.6% |

**Two evaluation modes used in reports:**
- **Accuracy mode:** `detection_threshold=0.05, top_k=500` — maximizes AP score fidelity
- **Performance mode:** `detection_threshold=0.3, top_k=200` — reflects real-time embedded deployment

---

## 9. Supported Datasets

| Dataset | Task(s) | Classes | Size |
|---------|---------|---------|------|
| ImageNet-1K | Classification | 1000 | 1.2M train / 50K val |
| COCO 2017 | Detection, Segmentation, Keypoint | 80 det / 91 seg | 118K train / 5K val |
| WIDER FACE | Face Detection | 1 | 32K images |
| ADE20K | Segmentation | 150 (full) / 32 (MLPerf) | 20K train |
| Cityscapes | Segmentation | 19 | 3K fine-annotated |
| VOC 2012 | Segmentation | 20 + background | 11K images |
| COCOSeg21 | Segmentation | 21 | COCO subset |
| NYUDepthV2 | Depth Estimation | — (continuous depth) | 1449 labeled |
| KITTI | 3D Detection (LiDAR) | Car/Pedestrian/Cyclist | Standard ADAS benchmark |
| PandaSet | 3D Detection (BEV) | 12 object classes | Autonomous driving scenes |
| YCB-Video | 6D Pose Estimation | 21 household objects | RGB-D video sequences |
| TI-RoboKit | Segmentation | Custom | TI robotics kit scenes |
| CARLA | Visual Localization | — | Synthetic urban driving |
| MLPerf subsets | Cls / Det / Seg | Standardized subsets | MLPerf inference benchmark |

---

## 10. Development Toolchain Ecosystem

The Model Zoo is one component of TI's broader EdgeAI development ecosystem:

```
┌─────────────────────────────────────────────────────────────┐
│                TI EdgeAI Development Ecosystem               │
│                                                             │
│  ┌─────────────────┐      ┌──────────────────────────────┐  │
│  │  Model Training  │      │       Model Zoo              │  │
│  │                 │      │  (pre-trained ONNX/TFLite)    │  │
│  │  edgeai-torchvision  │  │  ~180+ models, 8 tasks       │  │
│  │  edgeai-mmdetection  │  └──────────────┬───────────────┘  │
│  │  edgeai-mmpose  │                      │                  │
│  │  edgeai-tensorvision │                 ▼                  │
│  └─────────────────┘      ┌──────────────────────────────┐  │
│                           │   edgeai-tidl-tools           │  │
│                           │  (TIDL compiler: ONNX/TFLite  │  │
│                           │   → hardware-specific binary) │  │
│                           └──────────────┬───────────────┘  │
│                                          │                  │
│                    ┌─────────────────────┼───────────────┐  │
│                    ▼                     ▼               ▼  │
│              ┌──────────┐        ┌──────────────┐ ┌──────┐  │
│              │  TDA4VM  │        │ AM62A/AM67A  │ │AM68A │  │
│              │ (ADAS)   │        │ (industrial) │ │AM69A │  │
│              └──────────┘        └──────────────┘ └──────┘  │
│                                                             │
│  Supporting tools:                                          │
│  • edgeai-benchmark  — accuracy/performance measurement     │
│  • Edge AI Studio    — web GUI model selection tool         │
│  • edgeai-tidlrunner — CLI compilation + benchmarking       │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Role |
|-----------|------|
| **TIDL** | TI Deep Learning library — runtime that dispatches layers to MMA/C7x |
| **edgeai-tidl-tools** | Offline compiler: converts ONNX/TFLite to SoC-specific TIDL binary |
| **edgeai-benchmark** | Python framework for accuracy & FPS measurement (PC emulation + EVM) |
| **edgeai-tidlrunner** | High-level CLI wrapping benchmark + model zoo configs |
| **Edge AI Studio** | Web UI at ti.com/tool/EDGE-AI-STUDIO for model selection & profiling |

---

## 11. Scripts & Export Utilities

All scripts are in `scripts/`:

| Script | Purpose |
|--------|---------|
| `export_model_torchvision.py` | Export PyTorch TorchVision models → ONNX |
| `export_model_timm.py` | Export TIMM models → ONNX |
| `export_model_huggingface.py` | Export HuggingFace transformer models → ONNX |
| `export_model_lenet5.py` | LeNet5 example export |
| `download_models_gluoncv-mxnet.py` | Download GluonCV/MXNet models |
| `tf1_download_detection.sh` | Download TF1 detection models |
| `tf2_download_detection.sh` | Download TF2 detection models |
| `tf2_export_classification.sh` | Export TF2 classification → TFLite |
| `tf2_export_detection.sh` | Export TF2 detection → TFLite |
| `tf_download_segmentation.sh` | Download TF segmentation models |
| `tf1_export_detection.sh` | Export TF1 detection models |
| `tf1_export_segmentation.py` | Export TF1 segmentation |
| `tf1_convert_saved_model.py` | Convert TF1 SavedModel format |
| `tf2_write_signatures.py` | Write TF2 model signatures for export |
| `make_link_files.py` | Generate `.link` pointer files for hosted models |
| `onnx_update_model.sh` | Update ONNX model opset/shape info |
| `onnx_update_models.sh` | Batch ONNX model update |
| `onnx_shape_inference.py` | Run ONNX shape inference pass |
| `update_onnx_model_bs.py` | Update ONNX model batch size |
| `cleanup_model_files.py` | Remove temporary/intermediate files |

---

## 12. Model ID Convention

Model IDs encode task, runtime tier, and sequence:

```
Format: <task_prefix>-<id>
Example: cl-6360, od-8220, ss-8610, kd-7060, de-7300, 3dod-8140

Task prefixes:
  cl    = classification
  od    = object detection
  ss    = semantic segmentation
  de    = depth estimation
  3dod  = 3D object detection
  kd    = keypoint detection
  6dpose = 6D pose estimation
  visloc = visual localization

ID number ranges (approximate):
  0000–2999 → tflitert runtime (TFLite models)
  3000–5999 → tvmrt / tvmdlr runtime (TVM-compiled ONNX)
  6000–9999 → onnxrt runtime (ONNX Runtime models)
```

Artifact file names extend this with runtime prefix:
```
ONR-OD-8220-yolox-s-lite-mmdet-coco-640x640
TFL-CL-0000-mobileNetV1-mlperf
TVM-SS-5720-deeplabV3-mobV2-tv-cocoseg21-512x512
```

---

## 13. Summary Statistics

| Task | Approx. Model Count | Formats Available | SoC Coverage |
|------|--------------------|--------------------|-------------|
| Image Classification | ~80+ | ONNX, TFLite | All 5 SoCs |
| Object Detection | ~50+ | ONNX, TFLite | All 5 SoCs |
| Semantic Segmentation | ~30+ | ONNX, TFLite | All 5 SoCs |
| Depth Estimation | 2 | ONNX | All 5 SoCs |
| 3D Object Detection | 4 | ONNX | AM68A, AM69A |
| Keypoint / Pose | 5 | ONNX | All 5 SoCs |
| 6D Pose Estimation | 1 | ONNX | All 5 SoCs |
| Visual Localization | 1 | ONNX | Select SoCs |
| **Total** | **~180+ models** | ONNX + TFLite | 5 TI SoCs |

### GigaMAC Efficiency Range
- **Most efficient:** MobileNetV3Lite-Small @ 0.054 GMACs → 62.68% Top-1
- **Best efficiency/accuracy tradeoff (detection):** YOLOX-nano @ 1.476 GMACs → 24.8% mAP
- **Highest accuracy (classification):** Swin-Large @ 34.5 GMACs → 86.15% Top-1
- **Embedded sweet spot:** Models <10 GMACs suitable for real-time embedded inference

---

*Report generated from edgeai-modelzoo v11.2.0 — August 2026*
