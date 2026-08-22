# NXP eIQ® Model Zoo — Comprehensive Overview

> **Repository:** https://github.com/NXP/eiq-model-zoo  
> **License:** MIT  
> **Last Verified:** August 2026  

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Target Hardware Platforms](#2-target-hardware-platforms)
3. [eIQ® Toolkit and Software Stack](#3-eiq-toolkit-and-software-stack)
4. [Model Format and Recipe System](#4-model-format-and-recipe-system)
5. [Vision Models — Classification](#5-vision-models--classification)
6. [Vision Models — Object Detection](#6-vision-models--object-detection)
7. [Vision Models — Semantic Segmentation](#7-vision-models--semantic-segmentation)
8. [Vision Models — Instance Segmentation](#8-vision-models--instance-segmentation)
9. [Vision Models — Pose Estimation and Face Recognition](#9-vision-models--pose-estimation-and-face-recognition)
10. [Vision Models — Super Resolution, Low-Light Enhancement, and Depth Estimation](#10-vision-models--super-resolution-low-light-enhancement-and-depth-estimation)
11. [Audio Models](#11-audio-models)
12. [Miscellaneous — EEG/BCI](#12-miscellaneous--eegbci)
13. [MLOps Pipeline](#13-mlops-pipeline)
14. [Comparison: NXP vs TI vs ADI](#14-comparison-nxp-vs-ti-vs-adi)

---

## 1. Introduction

The NXP eIQ® Model Zoo is a curated collection of approximately 35 machine learning models optimized for NXP microcontrollers (MCUs) and microprocessors (MPUs). Unlike conventional model repositories that store pre-converted weight files, the eIQ Model Zoo delivers models as **conversion recipes** — shell scripts and Docker environments that reproduce the complete quantization pipeline from the original upstream source.

### Key Design Principles

- **TFLite-only format**: Every model is delivered as TensorFlow Lite INT8 — no proprietary format lock-in
- **Recipe reproducibility**: `recipe.sh` + Docker ensures identical output across machines and CI pipelines
- **Traceability to upstream**: Recipes download the original FP32 checkpoint, enabling fine-tuning and retraining
- **Broad platform coverage**: Ranges from high-performance MPUs (i.MX 8M Plus, 2.3 TOPS NPU) to constrained MCUs (MCX N947, Cortex-M33)
- **Domain breadth**: 9 vision tasks + 3 audio tasks + EEG/BCI — widest task coverage of the three major embedded AI model zoos

### Repository Structure

```
eiq-model-zoo/
├── README.md
├── Dockerfile
├── products/
│   └── README.md              # per-platform model support table
└── tasks/
    ├── vision/
    │   ├── classification/         (9 models)
    │   ├── object-detection/       (11 models)
    │   ├── semantic-segmentation/  (2 models)
    │   ├── instance-segmentation/  (1 model)
    │   ├── face-recognition/       (1 model)
    │   ├── pose-estimation/        (3 models)
    │   ├── super-resolution/       (1 model)
    │   ├── low-light-enhancement/  (1 model)
    │   └── monocular-depth-estimation/ (1 model)
    ├── audio/
    │   ├── anomaly-detection/      (1 model)
    │   ├── command-recognition/    (2 models)
    │   └── speech-recognition/     (1 model)
    └── misc/
        └── eegTCNet/               (1 model)
```

Each model directory contains: `README.md` (metrics, architecture, citations), `recipe.sh` (conversion pipeline), and optionally `label_map.txt` or calibration scripts.

---

## 2. Target Hardware Platforms

### 2.1 Platform Summary

| Platform | Family | CPU Core | NPU / Accelerator | NPU TOPS | Memory | OS |
|----------|--------|----------|-------------------|----------|--------|-----|
| **i.MX 8M Plus** | MPU | Cortex-A53 × 4 @ 1.8 GHz | NXP NPU | 2.3 TOPS | LPDDR4 up to 4 GB | Linux |
| **i.MX 93** | MPU | Cortex-A55 × 2 + Cortex-M33 | Arm Ethos-U65 | 1.0 TOPS | LPDDR4 up to 2 GB | Linux |
| **i.MX RT1170** | MCU | Cortex-M7 @ 1 GHz + M4 @ 400 MHz | None (CPU only) | — | 2 MB SRAM + 64 MB SDRAM | RTOS/bare-metal |
| **i.MX RT1050** | MCU | Cortex-M7 @ 600 MHz | None | — | 512 KB SRAM + 8 MB SDRAM | RTOS/bare-metal |
| **i.MX RT1060** | MCU | Cortex-M7 @ 600 MHz | None | — | 1 MB SRAM + 8 MB SDRAM | RTOS/bare-metal |
| **MCX N947** | MCU | Cortex-M33 × 2 + NPU | NXP eIQ NPU | ~0.5 TOPS | 3 MB SRAM | RTOS/bare-metal |

### 2.2 i.MX 8M Plus

The i.MX 8M Plus is the flagship inference platform in the portfolio. Its dedicated NXP NPU delivers 2.3 TOPS of INT8 performance with full Linux support via the eIQ TFLite delegate. It supports all 35 models in the zoo and targets industrial, smart home, and edge server applications.

**Key eIQ delegate:** NNAPI hardware abstraction layer → NXP NPU driver  
**Typical use cases:** surveillance cameras, smart factory vision, robot perception

### 2.3 i.MX 93

The i.MX 93 pairs dual Cortex-A55 with an Arm **Ethos-U65** NPU, making it distinctive from the i.MX 8M Plus in two ways: its NPU is an Arm standard IP (enabling cross-vendor toolchain reuse), and it requires the **Vela compiler** for NPU acceleration.

**Vela compilation:**
```bash
# BSP >= LF6.1.36_2.1.0 supports online Vela compilation
vela --accelerator-config ethos-u65-256 model.tflite
# Output: model_vela.tflite (Ethos-U65 custom ops inserted)
```

Models that have NOT been Vela-compiled fall back to Cortex-A55 CPU execution. The eIQ TFLite delegate detects the Ethos-U65 at runtime and routes supported ops to the NPU.

**Typical use cases:** industrial IoT, voice assistants, wearable compute hubs

### 2.4 i.MX RT MCUs

The i.MX RT1170, RT1050, and RT1060 are Cortex-M7 MCUs with no dedicated NPU. Inference runs via CMSIS-NN kernels within the TFLite Micro runtime. Only lightweight models fit: NanoDet-M, Ultraface, MobileNetV1, and FastestDet.

**Memory constraint:** all MCU models must fit within SRAM (typically <2 MB for model + activations)

### 2.5 MCX N947

The MCX N947 combines a Cortex-M33 with NXP's dedicated MCU-class NPU. Six models are supported: Tiny-ResNet, Visual Wake Word, Keyword Spotting (DS-CNN), Anomaly Detection, FaceDet, and Ultraface-Ultraslim.

---

## 3. eIQ® Toolkit and Software Stack

### 3.1 Stack Layers

```
┌─────────────────────────────────────────────────────────┐
│                   Application Code                       │
├─────────────────────────────────────────────────────────┤
│              NXP eIQ® ML Inference API                  │
│   (TFLite C++ API / Python bindings / eIQ Portal)       │
├────────────────┬────────────────┬───────────────────────┤
│  NXP NPU       │  Ethos-U65     │    CPU (CMSIS-NN /    │
│  Delegate      │  Delegate      │    XNNPACK / Neon)    │
│ (i.MX 8M Plus) │ (i.MX 93)      │    (all platforms)    │
├────────────────┴────────────────┴───────────────────────┤
│          TensorFlow Lite Runtime (2.x)                  │
├─────────────────────────────────────────────────────────┤
│           Linux (MPU) / TFLite Micro (MCU)              │
└─────────────────────────────────────────────────────────┘
```

### 3.2 Key Components

- **TFLite runtime**: Standard TF Lite 2.x C++ runtime, cross-compiled for Arm
- **NNAPI delegate**: Hardware abstraction for i.MX 8M Plus NPU
- **Ethos-U delegate**: Arm ML Embedded Evaluation Kit delegate for i.MX 93
- **XNNPACK delegate**: High-performance CPU kernels for A-profile Arm
- **CMSIS-NN**: Cortex-M optimized neural network kernels for MCU targets
- **Vela compiler**: Arm's offline compiler that maps TFLite ops to Ethos-U55/U65 custom ops
- **eIQ Portal**: GUI tool for model visualization, profiling, and deployment

### 3.3 Performance Profiling

```bash
# On-device benchmark
/usr/local/bin/benchmark_model \
  --graph=model.tflite \
  --use_gpu=false \
  --num_threads=4 \
  --num_runs=50
```

---

## 4. Model Format and Recipe System

### 4.1 Why Recipes Instead of Pre-stored Weights

Traditional model zoos store pre-converted weight files, which creates maintenance burden when upstream models update. The eIQ Model Zoo instead stores the **transformation recipe** — a reproducible pipeline that fetches the latest upstream checkpoint and performs the conversion.

Benefits:
- Original retrainable models are easily accessible for domain fine-tuning
- Quantization parameters can be adjusted (e.g., calibration dataset size)
- Vela compilation parameters can be tuned per BSP version
- Smaller repository size (recipes are ~50 lines vs MB weight files)

### 4.2 Recipe Execution

```bash
# Step 1: Build Docker environment once
docker build -t nxp-model-zoo .

# Step 2: Navigate to model directory and run recipe
cd tasks/vision/classification/mobilenetv2
docker run --rm -v "$PWD:/workspace" nxp-model-zoo /workspace/recipe.sh

# Output: mobilenet_v2_1.0_224_quant.tflite (INT8)
```

### 4.3 Typical Recipe Pipeline

A `recipe.sh` performs these steps:

1. **Download**: `wget` or `curl` the original model from TF Hub / PyTorch Hub / original paper release
2. **Convert to TFLite** (if not already TFLite): via `tf.lite.TFLiteConverter` or ONNX → TFLite bridge
3. **INT8 Post-Training Quantization (PTQ)**:
   ```python
   converter.optimizations = [tf.lite.Optimize.DEFAULT]
   converter.representative_dataset = calibration_dataset  # ~100 COCO/ImageNet images
   converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
   converter.inference_input_type = tf.int8
   converter.inference_output_type = tf.int8
   ```
4. **Vela compilation** (i.MX 93 only):
   ```bash
   vela --accelerator-config ethos-u65-256 model_quant.tflite
   ```
5. **Validation**: Run inference on calibration set, compare INT8 vs FP32 accuracy delta

### 4.4 Calibration Data

Most recipes use 100 representative images from the training domain for PTQ calibration. For audio models, 100 audio clips from the validation split are used. The calibration path is configurable in the recipe.

---

## 5. Vision Models — Classification

### 5.1 Model Table

| Model | Dataset | Input Size | Top-1 Acc | MOPS | Params | Platforms |
|-------|---------|-----------|----------|------|--------|----------|
| MobileNetV1 1.0 | ImageNet | 224×224 | 70.9% | 569 | 4.2M | All |
| MobileNetV2 1.0 | ImageNet | 224×224 | 71.8% | 608 | 3.54M | 8MP, 93, MCU |
| MnasNet 1.0 | ImageNet | 224×224 | ~74% | ~312 | 3.9M | 8MP, 93 |
| EfficientNet-Lite0 | ImageNet | 224×224 | ~75% | 407 | 4.7M | 8MP, 93 |
| ResNet50 | ImageNet | 224×224 | 75.9% (FP32) | 8,200 | 25.6M | 8MP, 93 |
| InceptionV4 | ImageNet | 299×299 | 80.0% (FP32) | 24,700 | 42.7M | 8MP, 93 |
| Tiny-ResNet | CIFAR-10 | 32×32 | ~90% | ~26 | ~0.5M | MCX N947 |
| Visual Wake Word | VW COCO2014 | 96×96 | ~86% | ~18 | ~0.3M | MCX N947, 8MP |
| Deepface-emotion | FER2013 | 64×64 | ~60% | ~10 | ~0.5M | 8MP, 93 |

### 5.2 Architecture Notes

**MobileNetV1/V2** are the workhorses of the classification suite. Both use depthwise separable convolutions. MobileNetV2 adds linear bottlenecks and inverted residuals for better accuracy/efficiency tradeoff.

**EfficientNet-Lite** is a mobile-optimized variant of EfficientNet that removes squeeze-and-excitation blocks (which are quantization-hostile) for better INT8 performance.

**Tiny-ResNet** is a custom shallow ResNet designed specifically for 32×32 CIFAR-10 inputs, enabling deployment on MCX N947 with minimal memory.

**Visual Wake Word** uses a binary classifier (person / no-person) at 96×96, targeting always-on sensor-fusion scenarios where a tiny MCU wakes a larger processor.

**Deepface-emotion** classifies 8 emotion categories (angry, disgust, fear, happy, neutral, sad, surprise, contempt) and is used in human-robot interaction and retail analytics.

### 5.3 Preprocessing

All ImageNet-pretrained models use standard ImageNet normalization:
```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
# Or for INT8 range [-128, 127]:
# scale = 1/128.0, zero_point = 0
```

---

## 6. Vision Models — Object Detection

### 6.1 Model Table

| Model | Dataset | Input Size | mAP | OPS | Params | Platforms |
|-------|---------|-----------|-----|-----|--------|----------|
| YOLOv8-m | COCO | 320×320 | 50.2 mAP@50 | 78.9 GOPS | 25.9M | 8MP, 93 |
| YOLOv5s | COCO | 640×640 | ~37 mAP | ~16 GOPS | 7.2M | 8MP, 93 |
| YOLOv4-tiny | COCO | 416×416 | ~22 mAP | ~6.9 GOPS | 5.9M | 8MP, 93 |
| SSDLite MobileNetV2 | COCO | 300×300 | ~22 mAP | ~0.8 GOPS | 4.3M | 8MP, 93 |
| EfficientDet-lite0 | COCO | 320×320 | ~26 mAP | ~0.98 GOPS | 3.9M | 8MP, 93 |
| NanoDet-M | COCO | 320×320 | ~20 mAP | ~0.72 GOPS | 0.95M | 8MP, 93, RT1170, RT1050 |
| CenterNet | COCO | 512×512 | ~30 mAP | ~7.3 GOPS | 32.7M | 8MP, 93 |
| FastestDet | Pascal VOC | 352×352 | ~25 mAP | ~0.25 GOPS | 0.25M | RT1170 |
| FaceDet | WiderFace | 240×320 | — | <0.5 GOPS | ~0.2M | All |
| Ultraface-Slim | WiderFace | 320×240 | — | <0.3 GOPS | ~1.0M | 8MP, 93, RT1170, RT1050 |
| Ultraface-Ultraslim | WiderFace | 128×96 | — | <0.1 GOPS | ~0.2M | MCX N947 |

### 6.2 Architecture Notes

**YOLOv8-m** is the highest-accuracy detector in the zoo, delivering COCO state-of-the-art at 50.2 mAP with 320×320 input. At 78.9 GOPS it requires the i.MX 8M Plus NPU or i.MX 93 Ethos-U65.

**NanoDet-M** (0.95M params, 0.72 GOPS) is the bridge model that runs on both MPUs and the i.MX RT1170 MCU — enabled by its extremely lightweight ShuffleNetV2 backbone and NanoDet head with GFL loss.

**FastestDet** achieves 0.25M params, making it the most constrained detector. Designed for Cortex-M7 targets, it uses a single-stage anchor-free detection head on a custom backbone.

**Ultraface-Ultraslim** at 128×96 input is the smallest face detector, targeting the MCX N947 for always-on face detection.

### 6.3 Post-Processing

Detection models typically include:
- Dequantization of output tensors
- Non-maximum suppression (NMS) — either in the TFLite graph or as post-processing
- Box decoding (anchor-based for SSD/EfficientDet, anchor-free for YOLO/CenterNet/NanoDet)

---

## 7. Vision Models — Semantic Segmentation

### 7.1 Model Table

| Model | Dataset | Input Size | mIoU | OPS | Size (INT8) | Platforms |
|-------|---------|-----------|------|-----|------------|----------|
| DeepLabV3 (MobileNetV2 0.5×) | PASCAL VOC 2012 | 513×513 | 70.19% | 1.76 GOPS | 983 KB | 8MP, 93 |
| Selfie-Segmenter | Proprietary | 256×256 | — | ~0.5 GOPS | ~500 KB | 8MP, 93 |

### 7.2 DeepLabV3

DeepLabV3 with MobileNetV2 0.5× backbone is a strong segmentation baseline. The 0.5× width multiplier reduces computation to 1.76 GOPS while maintaining 70.19% mIoU on 20 PASCAL VOC classes. The INT8 model fits in 983 KB — deployable on i.MX 93 with Vela compilation.

**Output:** 513×513 class probability map → argmax → 513×513 class label map (20 classes + background)

### 7.3 Selfie Segmenter

A lightweight person-background segmenter trained on a proprietary dataset. Intended for video conferencing background replacement on i.MX 93 edge devices. Outputs a binary mask at 256×256.

---

## 8. Vision Models — Instance Segmentation

### 8.1 YOLACT-Edge (MobileNetV2 backbone)

| Attribute | Value |
|-----------|-------|
| Dataset | COCO (80 classes) |
| Input size | 550×550 |
| mAP (box) | 0.21 |
| OPS | 17 GOPS |
| Model size (INT8) | 8.5 MB |
| Platforms | i.MX 8M Plus, i.MX 93 |

YOLACT-Edge is an instance segmentation model derived from YOLACT, optimized for edge deployment. It generates instance masks by predicting mask coefficients and combining them with prototype masks. The MobileNetV2 backbone replaces the original ResNet101, reducing compute from 50+ GOPS to 17 GOPS.

**Output per detection:** bounding box + class label + binary instance mask (resized to bounding box)

---

## 9. Vision Models — Pose Estimation and Face Recognition

### 9.1 Pose Estimation

| Model | Task | Dataset | Input Size | Platforms |
|-------|------|---------|-----------|----------|
| MoveNet | Full-body keypoints (17) | COCO + Active | 192×192 | 8MP, 93 |
| WHENet | Head pose (yaw/pitch/roll) | 300W-LP | 224×224 | 8MP, 93 |
| facial-landmarks-35-adas-0002 | 35 facial landmarks | Proprietary (ADAS) | 60×60 | 8MP, 93 |

**MoveNet** ("Lightning" variant) detects 17 COCO keypoints (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles) at ~30 FPS on i.MX 8M Plus. It uses a bottom-up single-pose detector with a heatmap head.

**WHENet** (Wide-range Head Estimation Network) predicts head orientation angles: yaw (±90°), pitch (±90°), and roll (±90°). Trained on 300W-LP (augmented from 300W with 3D pose synthesis). Useful for driver monitoring, gaze estimation, and social robotics.

**facial-landmarks-35-adas-0002** is an OpenVINO model converted to TFLite, locating 35 facial landmark points. "ADAS" indicates its original purpose: driver face monitoring in Advanced Driver Assistance Systems.

### 9.2 Face Recognition

| Model | Task | Dataset | Input Size | Embedding Dim | Platforms |
|-------|------|---------|-----------|--------------|----------|
| FaceNet512 | Face embedding + verification | LFW | 160×160 | 512 | 8MP, 93 |

**FaceNet512** generates a 512-dimensional L2-normalized face embedding. Similarity between two faces is computed as cosine distance or L2 distance in embedding space. On the LFW benchmark it achieves >99% verification accuracy.

**Usage pattern:**
1. Face detection (e.g., FaceDet or Ultraface-Slim)
2. Face crop + alignment (affine transform to standard 5-point landmarks)
3. FaceNet512 embedding
4. Cosine similarity vs. enrolled embedding database

---

## 10. Vision Models — Super Resolution, Low-Light Enhancement, and Depth Estimation

### 10.1 Super Resolution: Fast-SRGAN

| Attribute | Value |
|-----------|-------|
| Dataset | DIV2k |
| Scale | 4× upscaling |
| OPS | 460 MOPS |
| Params | 168K |
| Model size (INT8) | 240 KB |
| Input | 240×426 LR → 960×1704 HR |
| Platforms | i.MX 8M Plus, i.MX 93 |

Fast-SRGAN is a generative adversarial network for single-image super resolution. At only 168K parameters and 240 KB INT8, it is among the smallest SR models capable of 4× upscaling. The generator uses a residual-in-residual dense block architecture. Despite quantization to INT8, perceptual quality is preserved through adversarial training.

**Practical use:** enhancing low-resolution security camera feeds, upscaling medical imaging thumbnails, improving satellite image resolution at the edge.

### 10.2 Low-Light Image Enhancement: SCI

| Attribute | Value |
|-----------|-------|
| Dataset | LOL (Low-Light Dataset) |
| Input | 1920×1080 (full HD) |
| OPS | 752 MOPS |
| Params | 5.87K (!) |
| Model size (INT8) | 8 KB (!) |
| Platforms | i.MX 8M Plus, i.MX 93 |

**SCI (Self-Calibrated Illumination)** is the most parameter-efficient model in the zoo at 5,870 parameters and 8 KB INT8. It uses a self-calibrated illumination estimation pipeline to enhance low-light images without explicit paired low/normal-light training pairs. Despite full-HD 1920×1080 processing, it requires only 752 MOPS due to its extremely compact architecture.

This is technically the smallest model by parameter count in any of the three model zoos (TI, ADI, NXP).

### 10.3 Monocular Depth Estimation: MiDaS v2.1 Small

| Attribute | Value |
|-----------|-------|
| Datasets | 10 mixed datasets (ReDWeb, MegaDepth, WSVD, etc.) |
| Input | 256×256 |
| OPS | 9.214 GOPS |
| Model size (INT8) | 17 MB |
| Backbone | EfficientNet-Lite |
| Platforms | i.MX 8M Plus, i.MX 93 |

MiDaS v2.1 Small uses a cross-dataset training strategy, mixing 10 datasets covering diverse environments (indoor, outdoor, driving, video). The EfficientNet-Lite backbone provides quantization-friendly operations. Output is a relative inverse depth map that can be converted to metric depth given camera calibration.

**Practical use:** obstacle avoidance, bin-picking robots, agricultural UAVs, accessibility aids for the visually impaired.

---

## 11. Audio Models

### 11.1 Keyword Spotting: DS-CNN

| Attribute | Value |
|-----------|-------|
| Dataset | Speech Commands v2 |
| Input | (1, 49, 10, 1) — 49 MFCC frames × 10 coefficients |
| Accuracy | 91.6% (Top-1 on 12 classes) |
| MOPS | 5 |
| Params | 23,756 |
| Platforms | All (i.MX 8M Plus, 93, MCX N947) |

**DS-CNN** (Depthwise Separable CNN) for keyword spotting uses MFCC features extracted from a 1-second audio window (16 kHz, 25 ms window, 10 ms hop). The MFCC feature map of shape (49, 10) is fed into a stack of depthwise separable conv layers. At 23K params and 5 MOPS it runs on all platforms including MCX N947.

**Preprocessing pipeline:**
```python
# 1-second clip @ 16 kHz
audio = load_audio(16000 samples)
mfcc = compute_mfcc(audio, n_mfcc=10, n_fft=512,
                    hop_length=160, win_length=400)
# MFCC shape: (49, 10)
x = mfcc.reshape(1, 49, 10, 1).astype(np.float32)
x = (x - mean) / std  # normalize
```

### 11.2 Keyword Spotting: MicroSpeech LSTM

| Attribute | Value |
|-----------|-------|
| Dataset | Speech Commands v2 |
| Architecture | LSTM |
| Input | MFCC spectrogram |
| Platforms | i.MX 8M Plus, i.MX 93, MCX N947 |

MicroSpeech LSTM is the TFLite Micro reference model, originally demonstrated at ARM Cortex-M scale. Compared to DS-CNN, it has lower accuracy but demonstrates LSTM-based temporal modeling for keyword detection. The TFLite Micro deployment enables bare-metal MCU inference.

### 11.3 Speech Recognition: Wav2Letter

| Attribute | Value |
|-----------|-------|
| Dataset | LibriSpeech (clean-100h) |
| Input | (1, 296, 39) — 296 frames × 39 MFCC+delta features |
| WER | 7.2% |
| MOPS | 6,982 |
| Platforms | i.MX 8M Plus, i.MX 93 |

**Wav2Letter** is an end-to-end CTC-based speech recognition model. It maps MFCC features directly to character sequences via a 1D convolutional architecture with 17 conv layers and CTC loss. At 7.2% WER on LibriSpeech clean-100h it demonstrates practical on-device ASR.

**This is the only ASR model among the three model zoos (TI, ADI, NXP)** — a unique capability of the NXP zoo.

**Preprocessing:**
```python
# 3 seconds @ 16 kHz
audio = load_audio(48000 samples)
mfcc_13 = compute_mfcc(audio, n_mfcc=13)
delta   = compute_delta(mfcc_13)
delta2  = compute_delta(delta)
x = np.concatenate([mfcc_13, delta, delta2], axis=1)  # (296, 39)
x = x.reshape(1, 296, 39)
```

**Decoding:** CTC beam search with lexicon-constrained or greedy decoding → character sequence → word sequence.

### 11.4 Anomaly Detection: Deep Autoencoder

| Attribute | Value |
|-----------|-------|
| Dataset | ToyADMOS (DCASE 2020 Task 2) |
| Architecture | Dense autoencoder |
| Task | Industrial audio anomaly detection |
| Platforms | All |

The Deep Autoencoder reconstructs normal machine sounds (pump, fan, valve, slider) from log-mel spectrogram features. Anomaly score = reconstruction MSE. A threshold is set per machine type on the validation set; frames exceeding the threshold trigger an anomaly alert.

**Preprocessing:**
```python
mels = compute_log_mel_spectrogram(audio, n_mels=128, n_fft=1024, hop_length=512)
# Flatten to 640-dim input (5 overlapping frames × 128 mel bins)
x = mels[:, i:i+5].T.flatten()
```

---

## 12. Miscellaneous — EEG/BCI

### 12.1 EEG TCNet

| Attribute | Value |
|-----------|-------|
| Dataset | BCI Competition IV-2a |
| Input | (1, 1, 22, 1125) — 1 sample, 1 time window, 22 EEG channels, 1125 time steps |
| Classes | 4 motor imagery classes: left hand, right hand, feet, tongue |
| Accuracy | 77.35% (4-class cross-subject, averaged over 9 subjects) |
| MOPS | 14 |
| Params | 4,096 |
| Platforms | i.MX 8M Plus (CPU only, Ethos-U65 not supported as of LF6.1.36) |

**EEG TCNet** is a **Temporal Convolutional Network** for EEG-based Brain-Computer Interface (BCI) motor imagery classification. The input is 4.5 seconds of 22-channel EEG recorded at 250 Hz from 9 subjects performing imagined hand/foot/tongue movements.

**Architecture:**
1. EEGNet-style block: temporal conv + depthwise spatial conv + ELU + average pooling
2. TCN block: causal dilated 1D convolutions with residual connections
3. Dense output: 4-class softmax

**Training:** 9 per-subject models are provided (Subject01–Subject09). Cross-subject accuracy is 77.35%; per-subject accuracy can exceed 85%.

**Unique position:** This is the only EEG/BCI model across all three model zoos (TI, ADI, NXP), representing an entirely new application domain for embedded edge AI beyond vision and audio.

**Practical applications:**
- Assistive devices for motor-impaired patients
- Neurofeedback and cognitive monitoring
- Human-robot interaction via mental commands
- Gaming and VR control interfaces

---

## 13. MLOps Pipeline

### 13.1 Overview

The NXP eIQ Model Zoo MLOps pipeline (`code-nxp/`) provides a production-quality workflow for managing, evaluating, and deploying TFLite models across all five NXP platforms. Unlike the ADI pipeline (which must handle AI8X + TFLite formats) and the TI pipeline (ONNX + TIDL), the NXP pipeline is unified around a single format: **TFLite INT8**.

### 13.2 Directory Structure

```
code-nxp/
├── config/
│   ├── pipeline_config.yaml      # Platform, task, eval gates, monitoring config
│   └── model_registry.yaml       # All ~35 models with metrics, input sizes, platforms
├── mlops/
│   ├── model_manager.py          # Registry CRUD, artifact resolution, platform filtering
│   ├── data_pipeline.py          # Unified preprocessing: vision/audio/EEG
│   ├── evaluator.py              # TFLite inference, metric computation, accuracy gates
│   ├── recipe_runner.py          # Docker-based recipe execution wrapper
│   ├── vela_compiler.py          # Arm Vela compilation for i.MX 93
│   ├── artifact_manager.py       # Bundle pack/unpack/verify for deployment
│   └── monitor.py                # Inference latency/confidence drift monitoring
├── pipelines/
│   ├── recipe_pipeline.py        # Convert FP32 → TFLite INT8 via recipe.sh
│   ├── eval_pipeline.py          # Batch evaluation with accuracy gates
│   └── deploy_pipeline.py        # Copy artifacts to target board (SCP/local)
├── tests/
│   └── test_pipeline.py          # 30+ pytest unit tests
├── .github/
│   └── workflows/
│       └── mlops_ci.yml          # CI: lint → test → recipe → evaluate → deploy
└── requirements.txt
```

### 13.3 Key Design Decisions

1. **TFLite-only**: All inference uses `tflite_runtime` — no ONNX Runtime, no PyTorch needed at deployment
2. **Recipe wrapping**: `RecipeRunner` provides a Python API over Docker recipe execution
3. **Vela integration**: `VelaCompiler` class handles i.MX 93 Ethos-U65 compilation with per-model config
4. **Platform-aware evaluation**: Each model has a `supported_platforms` list; `Evaluator` selects appropriate delegate
5. **EEG preprocessing**: `preprocess_eeg(data, subject_id)` handles per-subject normalization

### 13.4 Accuracy Gates

| Domain | Task | Metric | Gate Threshold |
|--------|------|--------|---------------|
| Vision | Classification | Top-1 Accuracy | ≥68.0% |
| Vision | Object Detection | mAP@50 | ≥18.0% |
| Vision | Segmentation | mIoU | ≥62.0% |
| Vision | Super Resolution | PSNR | ≥28.0 dB |
| Audio | Keyword Spotting | Top-1 Accuracy | ≥88.0% |
| Audio | Speech Recognition | WER | ≤10.0% |
| Audio | Anomaly Detection | AUC-ROC | ≥0.80 |
| Misc | EEG Classification | Accuracy | ≥70.0% |

---

## 14. Comparison: NXP vs TI vs ADI

| Dimension | NXP eIQ | TI EdgeAI | ADI Model Zoo |
|-----------|---------|-----------|---------------|
| **Model count** | ~35 | 180+ | ~15 |
| **Domains** | Vision + Audio + Misc | Vision only | Vision + Audio + Sensor |
| **Format** | TFLite INT8 only | ONNX + TFLite + TVM + TIDL | AI8X + TFLite + PyTorch |
| **Quantization** | PTQ (recipe) | PTQ + QAT | QAT (AI8X INT8) |
| **Hardware family** | MPU (A-profile) + MCU (M-profile) | MPU/SoC (A-profile only) | MCU (M-profile only) |
| **NPU TOPS range** | 0.5 – 2.3 TOPS | 1 – 32 TOPS | <0.1 TOPS |
| **Deployment workflow** | Docker recipe → TFLite → Vela | TIDL compiler + ONNX tools | ai8xize synthesis + MSDK |
| **Unique tasks** | Super resolution, low-light, EEG/BCI, ASR, depth | ADAS 3D detection, BEV, radar | Motor fault, biosignal, channel folding |
| **Open retrain** | Full (recipe downloads original) | Partial (TIDL compilation barrier) | Full (QAT training framework) |
| **License** | MIT | Apache 2.0 | Various (CC BY-SA, MIT) |

### 14.1 NXP Strengths

1. **Broadest task portfolio**: super resolution, low-light enhancement, instance segmentation, monocular depth, speech recognition, and EEG are unique to NXP
2. **Format standardization**: single TFLite format simplifies toolchain maintenance
3. **Recipe reproducibility**: Docker-based recipes are the most reproducible approach
4. **Platform breadth**: from Cortex-M33 (MCX N947) to Cortex-A53 quad-core with 2.3 TOPS NPU
5. **Ethos-U65 support**: only zoo with Arm standard NPU, enabling cross-vendor toolchain reuse

### 14.2 NXP Weaknesses

1. **Model count**: ~35 vs TI's 180+ — significantly narrower coverage
2. **No proprietary acceleration format**: TFLite INT8 is standard but cannot utilize the full NPU potential (unlike TIDL for TI's specialized architecture)
3. **No QAT support**: PTQ only in recipes, versus AI8X QAT (ADI) which achieves better accuracy at INT8
4. **TOPS ceiling**: 2.3 TOPS (i.MX 8M Plus) vs 32 TOPS (TI TDA4VM) — limited for large models
5. **MCU models limited**: only ~6 models run on MCX N947; TI's support for AM62A+ provides more MPU flexibility

### 14.3 Choosing Between Platforms

| Use case | Recommended platform |
|---------|---------------------|
| ADAS / automotive | TI EdgeAI (TDA4VM/J7 series) |
| Industrial motor + vibration | ADI MAX78002 |
| Edge NLP / ASR on device | NXP i.MX 93 (Ethos-U65) |
| EEG / brain-computer interface | NXP i.MX 8M Plus |
| Smart camera (always-on, <1W) | NXP i.MX 93 or ADI MAX78002 |
| Super resolution pipeline | NXP i.MX 93 |
| Keyword spotting (MCU) | NXP MCX N947 or ADI MAX32690 |
