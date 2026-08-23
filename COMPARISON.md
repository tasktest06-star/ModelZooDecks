# Edge AI Model Zoo — Cross-Company Comparison

**Companies:** Texas Instruments (TI EdgeAI) · Analog Devices (ADI AI8X) · NXP Semiconductors (NXP eIQ)  
**Date:** 2026-08-23  
**Source:** Actual values from `model_registry.yaml` files in each company's codebase.

---

## 1. Platform Overview

| Attribute | TI EdgeAI | ADI AI8X | NXP eIQ |
|-----------|-----------|----------|---------|
| Target hardware | TDA4VM, AM68A, AM69A, AM62A, AM67A | MAX78002, MAX32690, ADSP-SC835 | i.MX 8M Plus, i.MX 93, RT1170/1050/1060, MCX N947 |
| NPU / accelerator | MMA (matrix multiply, TIDL offload) | CNN accelerator 442 TOPS/W (MAX78002) | NPU 2.3 TOPS (i.MX 8M Plus), Ethos-U65 1.0 TOPS (i.MX 93) |
| CPU class | Cortex-A72 / R5F (automotive ASIL) | Cortex-M4F / ARM926 / SigmaDSP | Cortex-A53×4 / Cortex-M7 / Cortex-M33 |
| Primary use case | Automotive ADAS, industrial vision | Ultra-low-power IoT, wearable, industrial sensor | Consumer, industrial, automotive, BCI |
| Model format | ONNX / TFLite → TIDL binary | AI8X format: QAT .pth.tar + ai8xize C synthesis | TFLite INT8 via recipe.sh Docker |
| Quantization method | Post-Training (TIDL calibration) | Quantization-Aware Training (ai8x-training) | Post-Training INT8 (TFLite converter) |
| Vela compiler | No | No | Optional — required for Ethos-U65 on i.MX 93 |
| SDK / toolchain | EdgeAI TIDL Tools + MMDetection | ai8x-training + ai8xize.py | NXP eIQ Toolkit + Vela compiler |
| CI / MLOps | GitHub Actions + AWS CodePipeline | GitHub Actions + AWS CodePipeline | GitHub Actions + AWS CodePipeline |
| Experiment tracking | MLflow + Prefect 2 + Optuna | MLflow + Prefect 2 + Optuna (QAT HPO) | MLflow + Prefect 2 + Optuna (INT8 PTQ HPO) |

---

## 2. Model Inventory Summary

| Category | TI EdgeAI | ADI AI8X | NXP eIQ | Notes |
|----------|-----------|----------|---------|-------|
| **Total models** | **34** | **15** | **29** | — |
| Image classification | 15 | 4 | 9 | TI has broadest portfolio; ADI limited to CIFAR-100/ImageNet |
| Object detection | 11 | 2 | 7 | All three include YOLO variants |
| Segmentation | 4 | 1 | 2 | TI/NXP semantic; ADI binary person seg |
| Pose / keypoint | 3 | 0 | 2 | Body pose (TI/NXP) + head pose (NXP WHENet) |
| Depth estimation | 2 | 0 | 1 | MiDaS family across TI and NXP |
| Audio (KWS / ASR / denoise / genre) | 0 | 4 | 4 | TI is vision-only |
| Anomaly detection | 0 | 2 | 1 | ADI covers motor fault + machine anomaly |
| Super-resolution / low-light enhance | 0 | 0 | 2 | NXP exclusive (Fast-SRGAN + SCI) |
| Face recognition | 0 | 0 | 2 | NXP exclusive (FaceDet + FaceNet512) |
| Visual wake word (VWW) | 0 | 1 | 1 | ADI MAX32690, NXP MCX N947 / i.MX 8M Plus |
| EEG / biosignal (BCI) | 0 | 0 | 1 | NXP exclusive (EEG-TCNet) |
| Instance segmentation | 0 | 0 | 1 | NXP YOLACT-Edge |

---

## 3. Image Classification — Detailed Comparison

| Model | Company | Dataset | # Classes | Top-1 Acc (INT8) | Input | GMACs | Size | Format | HW Target |
|-------|---------|---------|-----------|-----------------|-------|-------|------|--------|-----------|
| efficientnet-lite4 | TI | ImageNet-1K | 1000 | **82.4%** | 224×224 | 2.645 | — | TFLite PTQ | All 5 SoCs |
| swin-tiny | TI | ImageNet-1K | 1000 | 77.9% | 224×224 | 4.50 | — | ONNX PTQ | AM68A/AM69A |
| mobilenetv3lite-large | TI | ImageNet-1K | 1000 | 71.9% | 224×224 | 1.08 | — | ONNX PTQ | All 5 SoCs |
| mobilenetv2 (TI) | TI | ImageNet-1K | 1000 | 71.8% | 224×224 | 0.300 | — | ONNX PTQ | All 5 SoCs |
| inceptionv4 | NXP | ImageNet-1K | 1000 | **78.2%** | 299×299 | 5.49 | 43.2 MB | TFLite PTQ | i.MX 8M Plus, i.MX 93 |
| resnet50 | NXP | ImageNet-1K | 1000 | 75.9% | 224×224 | 2.10 | 25.1 MB | TFLite PTQ | i.MX 8M Plus, i.MX 93 |
| mnasnet | NXP | ImageNet-1K | 1000 | 72.3% | 224×224 | 0.825 | 4.7 MB | TFLite PTQ | i.MX 8M Plus, i.MX 93 |
| efficientnet_lite | NXP | ImageNet-1K | 1000 | 72.2% | 224×224 | 0.407 | 5.0 MB | TFLite PTQ | i.MX 8M Plus, i.MX 93 |
| mobilenetv2 (NXP) | NXP | ImageNet-1K | 1000 | 71.8% | 224×224 | 0.608 | 3.4 MB | TFLite PTQ | i.MX 8M Plus, i.MX 93, RT |
| mobilenetv1 | NXP | ImageNet-1K | 1000 | 70.9% | 224×224 | 0.569 | 4.3 MB | TFLite PTQ | i.MX 8M Plus, i.MX 93, RT |
| deepface_emotion | NXP | FER2013 | 8 | 64.5% | 48×48 | 0.065 | 1.1 MB | TFLite PTQ | i.MX 8M Plus, i.MX 93 |
| visual_wake_word | NXP | VW-COCO2014 | 2 | 85.3% | 96×96 | 0.018 | 324 KB | TFLite PTQ | MCX N947, i.MX 93 |
| tiny_resnet | NXP | CIFAR-10 | 10 | **92.4%** | 32×32 | 0.012 | 156 KB | TFLite PTQ | MCX N947 (MCU) |
| mobilenetv2_075 | ADI | CIFAR-100 | 100 | 64.7% | 32×32 | — | 10.5 MB | AI8X QAT INT8 | MAX78002 |
| mobilenetv2_050 | ADI | CIFAR-100 | 100 | 62.0% | 32×32 | — | 5.12 MB | AI8X QAT INT8 | MAX78002 |
| efficientnetv2 | ADI | ImageNet-1K | 1000 | 61.97% | 112×112 | — | 19.9 MB | AI8X float32 | MAX78002 |
| simplenet | ADI | CIFAR-100 | 100 | 60.3% | 32×32 | — | 8.40 MB | AI8X QAT INT8 | MAX78002 |
| micronet_vww2 | ADI | VWW (COCO) | 2 | 76.8% | 50×50 | — | 267 KB | TFLite INT8 | MAX32690 |

> **Note:** TI INT8 accuracy represents drop from FP32 baseline. ADI models use QAT so INT8 IS the training target. NXP values are post-conversion INT8.

---

## 4. Object Detection — Detailed Comparison

| Model | Company | Dataset | # Classes | mAP / mAP@50 | Input | GMACs | Size | Format | HW Target |
|-------|---------|---------|-----------|-------------|-------|-------|------|--------|-----------|
| yolox-m-lite | TI | COCO | 80 | 44.3 / 63.0% | 640×640 | 36.9 | — | ONNX PTQ | AM68A/AM69A/TDA4VM |
| yolox-s-lite | TI | COCO | 80 | 38.2 / 56.0% | 640×640 | 13.43 | — | ONNX PTQ | All 5 SoCs |
| yolox-nano-lite | TI | COCO | 80 | 24.6 / 41.8% | 416×416 | 1.476 | — | ONNX PTQ | All 5 SoCs |
| yolov8_m | NXP | COCO | 80 | — / **50.2%** | 320×320 | 78.9 | 26.1 MB | TFLite PTQ | i.MX 8M Plus, i.MX 93 |
| yolov5 | NXP | COCO | 80 | — / 37.0% | 320×320 | 4.2 | 7.1 MB | TFLite PTQ | i.MX 8M Plus, i.MX 93 |
| efficientdet_lite0 | NXP | COCO | 90 | — / 25.0% | 320×320 | 0.98 | 4.9 MB | TFLite PTQ | i.MX 8M Plus, i.MX 93 |
| yolov4_tiny | NXP | COCO | 80 | — / 21.7% | 416×416 | 6.9 | 23.1 MB | TFLite PTQ | i.MX 8M Plus, i.MX 93 |
| nanodet_m | NXP | COCO | 80 | — / 20.2% | 320×320 | 0.32 | 0.95 MB | TFLite PTQ | RT1170/RT1050 (MCU) |
| ssdlite_mobilenetv2 | NXP | COCO | 90 | — / 19.0% | 320×320 | 0.80 | 6.7 MB | TFLite PTQ | i.MX 8M Plus, i.MX 93 |
| facedet | NXP | WiderFace | 1 | Easy 83.8% | 128×128 | 0.022 | 250 KB | TFLite PTQ | RT1060/MCX N947 |
| feature_pyramid_net | ADI | Pascal VOC | 21 | **50.5%** mAP | 256×320 | — | 25.2 MB | AI8X QAT INT8 | MAX78002 |
| tinierssd | ADI | BG20K (QR) | 2 | **90.0%** mAP | 120×160 | — | 4.46 MB | AI8X QAT INT8 | MAX78002 |

> **Dataset note:** TI and NXP detect on COCO (80/90 classes, 118K images). ADI FPN on Pascal VOC (21 classes, 11K images). ADI TinierSSD on proprietary QR synthetic dataset. Comparisons across datasets are indicative only.

---

## 5. Segmentation — Detailed Comparison

| Model | Company | Dataset | Classes | mIoU / Acc | Input | GMACs | Size | Type |
|-------|---------|---------|---------|-----------|-------|-------|------|------|
| fpnlite-mobilev2 | TI | COCO Seg 21 | 21 | 57.2% mIoU | 512×512 | 5.44 | — | Semantic |
| deeplabv3lite-mobilev2 | TI | ADE20K-32 | 32 | 49.5% mIoU | 512×512 | 3.61 | — | Semantic |
| deeplabv3 | NXP | Pascal VOC 2012 | 21 | **70.2%** mIoU | 513×513 | 1.76 | 983 KB | Semantic |
| yolact_edge | NXP | COCO | 80 | 21.0% mask mAP | 550×550 | 17.0 | 8.5 MB | Instance |
| unet | ADI | AISegment | 2 | **98.45%** Acc | 48×48 | — | 3.23 MB | Binary person seg |

---

## 6. Audio Models — Detailed Comparison

| Model | Company | Task | Dataset | Metric | Score | Input | Size | Hardware |
|-------|---------|------|---------|--------|-------|-------|------|----------|
| ds_cnn (ADI) | ADI | KWS (12 classes) | GSC-v2 | Top-1 Acc | **94.52%** | 1×490 | 97 KB | MAX32690 / ADSP-SC835 |
| ds_cnn (NXP) | NXP | KWS (12 classes) | GSC-v1 | Top-1 Acc | 91.6% | 49×10×1 | 47 KB | MCX N947 / i.MX 93 |
| microspeech_lstm | NXP | KWS (4 classes) | GSC | Top-1 Acc | 88.1% | — | 28 KB | MCX N947 / i.MX 93 |
| conv1d_audionet | ADI | KWS (20+silence) | GSC-v2 | Top-1 Acc | 86.34% | 128×128 | 1.79 MB | MAX78002 (CNN accel.) |
| wav2letter | NXP | ASR (full vocab) | LibriSpeech | WER | **7.2%** | 296×39 | 14 MB | i.MX 8M Plus / i.MX 93 |
| dtln | ADI | Audio denoise | — | PESQ | **2.950** | 512-frame | 3.78 MB | ADSP-SC835 |
| rnnoise | ADI | Audio denoise | — | PESQ | 2.945 | 160-frame | 107 KB | MAX32690 |
| genrenet_conv2d | ADI | Genre classify (10) | GTZAN | Top-1 Acc | **84.5%** | 128×128 mel | 15.3 MB | ADSP-SC835 |

> PESQ range: 1.0 (bad) – 4.5 (excellent). DTLN and RNNoise deliver near-identical quality at very different hardware cost (DSP vs MCU Cortex-M4).

---

## 7. Anomaly Detection — Detailed Comparison

| Model | Company | Dataset | Metric | Score | Domain | Input | Hardware |
|-------|---------|---------|--------|-------|--------|-------|----------|
| deep_autoencoder_ad | NXP | ToyADMOS | AUC-ROC | **0.820** | Machine audio anomaly | — | i.MX 8M Plus / RT1170 |
| autoencoder_anomaly | ADI | MIMII (fan/pump/slider/valve) | AUC / pAUC | 0.516 / 0.523 | Industrial machine | 196×640 spectrogram | MAX32690 |
| autoencoder_motor | ADI | ADI Motor Limerick (vibration) | MSE | **0.02205** | Motor fault (3-axis vibration) | 256×3 | MAX78002 |

> The NXP autoencoder (ToyADMOS, audio) and ADI autoencoder_anomaly (MIMII, audio) are on different datasets. The ADI motor fault model (vibration) is unique — no equivalent in TI or NXP zoo.

---

## 8. Special / Unique Models

### NXP eIQ exclusives

| Model | Task | Key Metric | Unique Value |
|-------|------|-----------|-------------|
| fast_srgan | 4× super-resolution | PSNR 31.2 dB, SSIM 0.87 | 168K params, 240 KB INT8 — smallest SR model for edge |
| sci_low_light | Low-light image enhancement | PSNR 28.3 dB | Dynamic shape; operates at full 1920×1080; only 5,870 params, 8 KB |
| facenet512 | Face recognition (512-dim) | LFW Acc 99.35% | 512-dim cosine-matchable embedding; only face recognition model across all 3 zoos |
| whenet | Head pose (6-DOF) | MAE yaw 4.2° | Outputs yaw/pitch/roll + 3D; enables driver monitoring + liveness |
| wav2letter | End-to-end ASR | WER 7.2% on LibriSpeech | CTC-based; only full ASR model across 3 zoos |
| eeg_tcnet | EEG motor imagery BCI | 77.35% 4-class | Unique BCI domain; 4K params, 16 KB; 9 subject-specific models |

### ADI AI8X exclusives

| Model | Task | Key Metric | Unique Value |
|-------|------|-----------|-------------|
| tinierssd | QR code keypoint detection | mAP 90.0% | 4-keypoint QR corner detection; only QR model across 3 zoos |
| genrenet_conv2d | 10-class audio genre | Acc 84.5% | Conv2D on mel spectrogram; GTZAN dataset; runs on ADSP-SC835 |
| autoencoder_motor | Motor fault (vibration) | MSE 0.02205 | Only vibration-based predictive maintenance model across 3 zoos |
| unet | Binary person segmentation | Acc 98.45% | Folded 48×48 input; only binary semantic seg across 3 zoos |

### TI EdgeAI standouts (not in ADI or NXP at matching accuracy)

| Model | Task | Key Metric | Unique Value |
|-------|------|-----------|-------------|
| efficientnet-lite4 | ImageNet classification | 82.4% INT8 | Highest classification accuracy across all 3 zoos (INT8) |
| swin-tiny | ImageNet classification | 77.9% INT8 | Only Vision Transformer across 3 zoos; highest accuracy for transformer family |
| yolox-m-lite | COCO detection | mAP@50 63.0% | Highest COCO detection mAP@50 in the TI portfolio |
| midas-small | Monocular depth | δ1 85.9% | NYU DepthV2 trained; δ1 metric alongside NXP MiDaS Rel. Error 12.2% |

---

## 9. Quantization Approach Comparison

| Aspect | TI (TIDL PTQ) | ADI (AI8X QAT) | NXP (TFLite PTQ) |
|--------|--------------|----------------|-----------------|
| **Method** | Post-Training Quantization via TIDL calibration | Quantization-Aware Training (fake-quantization during training) | Post-Training INT8 via TFLite converter |
| **Bit width** | 8-bit (configurable 8 or 16) | 4-bit or 8-bit weights, 8-bit activations | 8-bit weights and activations |
| **Calibration data** | 50–500 representative frames | N/A (trained from scratch with fake-quant) | 100–1000 representative samples |
| **Typical accuracy drop** | 0.2–0.5% (efficient models), up to 4.2% (Swin-Tiny transformer) | Near-lossless — INT8 IS the trained target | 0.3–3% depending on architecture |
| **Observed worst case** | swin-tiny: 4.2% drop (82.1% → 77.9%) | — (no FP32 baseline for comparison) | — (registry shows INT8 values only) |
| **Observed best case** | efficientnet-lite4: 0.2% drop | All ADI models trained with QAT | mobilenetv2: INT8 is direct output |
| **Hardware constraint** | TIDL layer allowlist; some ops fall back to A72 | CNN accelerator has fixed supported op set | TFLite standard ops + Vela delegate subset for Ethos |
| **Deployment output** | TIDL .bin artifacts + metadata | C header files (weights.h, bias.h, sampledata.h) | .tflite INT8 file, optionally *_vela.tflite |
| **Accuracy recovery options** | Layer-by-layer mixed precision (8/16-bit) | Tune QAT schedule, learning rate, epochs, Optuna HPO | Larger calibration set, per-channel quantization |
| **Unique challenge** | Transformer models suffer large PTQ accuracy loss | Fixed CNN op set limits architecture diversity | Vela compiler adds compilation step for i.MX 93 |

---

## 10. Dataset Coverage

| Dataset | TI | ADI | NXP | Task | Size |
|---------|:--:|:---:|:---:|------|------|
| ImageNet-1K | ✓ (15 models) | ✓ (efficientnetv2) | ✓ (7 models) | Classification | 1.28M train, 50K val |
| CIFAR-100 | — | ✓ (3 models) | — | Classification | 50K train, 10K val |
| CIFAR-10 | — | — | ✓ (tiny_resnet) | Classification | 50K train, 10K val |
| FER2013 | — | — | ✓ (deepface_emotion) | Emotion | 35K images, 8 classes |
| VW-COCO2014 / VWW | — | ✓ (micronet_vww2) | ✓ (visual_wake_word) | Wake word | ~250K 96×96 images |
| COCO detection | ✓ (11 det models) | — | ✓ (5 det models) | Detection | 118K images, 80 classes |
| Pascal VOC | — | ✓ (FPN 21-class) | ✓ (deeplabv3) | Detection / Seg | 11K images, 21 classes |
| ADE20K-32 | ✓ (deeplabv3lite) | — | — | Segmentation | 20K images, 32 classes |
| COCO Seg-21 | ✓ (fpnlite) | — | — | Segmentation | 118K images, 21 classes |
| AISegment | — | ✓ (unet) | — | Binary person seg | 34K portrait images |
| WiderFace | — | — | ✓ (facedet) | Face detection | 32K images |
| LFW | — | — | ✓ (facenet512) | Face verification | 13K images, 5749 people |
| 300W-LP | — | — | ✓ (whenet) | Head pose | 61K images (syn. aug.) |
| COCO Keypoints | ✓ (yoloxpose) | — | ✓ (movenet) | Pose estimation | 64K images, 17 KP |
| NYU Depth V2 | ✓ (midas-small) | — | — | Depth estimation | 1449 scenes |
| Mixed-10 datasets | — | — | ✓ (midas_v21_small) | Depth estimation | Multi-source |
| DIV2K | — | — | ✓ (fast_srgan) | Super resolution | 1000 2K images |
| LOL | — | — | ✓ (sci_low_light) | Low-light enhance | 500 paired images |
| Google Speech Cmds v1/v2 | — | ✓ (ds_cnn, conv1d) | ✓ (ds_cnn, microspeech) | KWS | 105K 1-sec clips |
| LibriSpeech | — | — | ✓ (wav2letter) | ASR | 960h clean English |
| GTZAN | — | ✓ (genrenet) | — | Audio genre | 1K tracks, 10 genres, 30s |
| MIMII | — | ✓ (autoencoder_anomaly) | — | Machine anomaly | 4 machine types, 16 SNR |
| ToyADMOS | — | — | ✓ (deep_autoencoder_ad) | Audio anomaly | Toy car/train sounds |
| ADI Motor Limerick | — | ✓ (autoencoder_motor) | — | Vibration / motor fault | Proprietary 3-axis vibration |
| BCI Competition IV-2a | — | — | ✓ (eeg_tcnet) | EEG motor imagery | 9 subjects, 4 motor classes |

---

## 11. Hardware Efficiency Comparison

| Metric | TI EdgeAI | ADI AI8X | NXP eIQ |
|--------|-----------|----------|---------|
| Peak compute | ~8 TOPS (TDA4VM MMA) | 442 TOPS/W efficiency (MAX78002 CNN) | 2.3 TOPS (i.MX 8M Plus NPU) |
| Typical inference latency | 10–50 ms/frame (AM68A TIDL) | 1–20 ms (MAX78002 CNN accel.) | 15–120 ms (i.MX 8M Plus) |
| Power envelope | 5–30 W (automotive SoC) | 0.3–20 mA (MCU / DSP) | 0.5 W (RT1050) – 8 W (i.MX 8M Plus) |
| Active current (typical) | N/A (wall powered) | ~5 mA MAX78002 active | ~2 W i.MX 8M Plus |
| Memory: smallest model | — (TIDL cached) | 267 KB (micronet_vww2 TFLite INT8) | 8 KB (sci_low_light, 5870 params) |
| Memory: largest model | — | 25.2 MB (FPN) | 43.2 MB (inceptionv4) |
| OS requirements | Linux (A72) + RTOS (R5F) | Bare-metal / FreeRTOS | Linux (A53) / bare-metal (M7/M33) |
| Smallest target | AM62A (~3 W, Linux) | MAX32690 (~1 mA idle, bare-metal) | i.MX RT1050 (~0.5 W, bare-metal) |

---

## 12. Model Size vs. Accuracy Tradeoffs

### Classification (ImageNet, where comparable)

| Tier | Model | Company | Top-1 INT8 | GMACs | Size |
|------|-------|---------|-----------|-------|------|
| Smallest / fastest | ds_cnn (MCU, audio only) | NXP | 91.6%* | 0.005 | 47 KB |
| Tiny vision | visual_wake_word | NXP | 85.3% | 0.018 | 324 KB |
| Tiny ImageNet | mobilenetv2 | NXP/TI | 71.8% | 0.3–0.6 | 3–4 MB |
| Balanced | efficientnet_lite | NXP | 72.2% | 0.407 | 5.0 MB |
| Medium | mobilenetv3lite-large | TI | 71.9% | 1.08 | — |
| Large | resnet50 | NXP | 75.9% | 2.10 | 25.1 MB |
| High accuracy | efficientnet-lite4 | TI | **82.4%** | 2.645 | — |
| Transformer | swin-tiny | TI | 77.9% | 4.50 | — |
| Heavy / accurate | inceptionv4 | NXP | 78.2% | 5.49 | 43.2 MB |

*KWS task (12 classes), not ImageNet.

**Key insight:** ~10% accuracy gap between smallest (mobilenetv2 ~70%) and largest (efficientnet-lite4 82.4%). Transformers (swin-tiny) suffer 4.2% INT8 accuracy drop — highest of any model, making PTQ suboptimal for attention-based architectures.

### Detection (COCO mAP@50)

| Tier | Model | Company | mAP@50 | GMACs | Size |
|------|-------|---------|--------|-------|------|
| Ultra-tiny | nanodet_m | NXP | 20.2% | 0.32 | 0.95 MB |
| Tiny | ssdlite_mobilenetv2 | NXP | 19.0% | 0.80 | 6.7 MB |
| Small | yolox-nano-lite | TI | 41.8% | 1.48 | — |
| Medium | yolox-s-lite | TI | 56.0% | 13.43 | — |
| Large | yolov8_m | NXP | **50.2%** | 78.9 | 26.1 MB |
| Domain-specific | tinierssd (QR) | ADI | **90.0%** | — | 4.46 MB |

**Key insight:** 3× mAP gain from nano to medium at 10× compute cost. Domain-specific models (TinierSSD for QR) far exceed general detectors on their domain. ADI FPN at Pascal VOC is not directly comparable to COCO-trained detectors.

### Audio KWS Accuracy vs. Hardware Cost

| Model | Company | Acc | Classes | Size | Hardware | Power |
|-------|---------|-----|---------|------|----------|-------|
| ds_cnn (ADI) | ADI | **94.5%** | 12 | 97 KB | MAX32690 (Cortex-M4F) | ~1 mA |
| ds_cnn (NXP) | NXP | 91.6% | 12 | 47 KB | MCX N947 NPU | ~2 mA |
| microspeech_lstm | NXP | 88.1% | 4 | 28 KB | MCX N947 / RT | ~0.5 mA |
| conv1d_audionet | ADI | 86.3% | 21 | 1.79 MB | MAX78002 CNN accel. | ~5 mA |

**Key insight:** DS-CNN on MAX32690 delivers best KWS accuracy at lowest power. Conv1D AudioNet covers 21 keywords vs 12 but at lower accuracy, trading breadth for coverage.

---

## 13. Multi-Model Pipeline Tradeoffs

Implemented combination applications across 3 branches (`model-combination-ti/adi/nxp`):

| Application | Company | Models Chained | Est. Latency | Hardware | Est. Power | Primary Tradeoff |
|-------------|---------|---------------|-------------|----------|-----------|-----------------|
| ADAS Scene Understanding | TI | yolox-s + deeplabv3 + midas-small + yoloxpose | ~120 ms | AM68A | ~15 W | 4-stage accuracy vs. latency; pose adds ~30 ms |
| 2-Stage Object Recognition | TI | yolox-nano + mobilenetv3-small | ~55 ms | AM62A | ~3 W | Nano detector recall limits pipeline accuracy ceiling |
| People Analytics | TI | yolox-s + yoloxpose + mobilenet_v3_large | ~80 ms | AM68A | ~15 W | Skeleton features (fast) vs. full image crop (accurate) |
| Smart Sensor Node | ADI | micronet_vww2 → conv1d_audionet → FPN | 0.5–15 mA | MAX32690 + MAX78002 | **<5 mA typical** | >10× power saving; VWW 76.8% is gating bottleneck |
| Predictive Maintenance | ADI | rnnoise + ds_cnn + autoencoder_motor + tinierssd | ~30 ms total | MAX78002 + MAX32690 | ~8 mA | 3 independent subsystems; resource contention on shared bus |
| Audio Intelligence | ADI | dtln + genrenet + rnnoise + ds_cnn | ~15 ms (DSP) | ADSP-SC835 + MAX32690 | ~20 mA | Dual-path routing overhead; router accuracy determines which path fires |
| Low-Light Face Recognition | NXP | sci + facedet + facenet512 + whenet | ~115 ms | i.MX 8M Plus | ~6 W | SCI adds ~80 ms but prevents false-reject in low light |
| Driver Monitoring | NXP | facedet + deepface_emotion + whenet + ds_cnn | ~40 ms | i.MX 8M Plus | ~5 W | Emotion proxy for drowsiness (64.5% acc) vs. dedicated eye-tracking |
| Smart Video Analytics | NXP | yolov8_m + mobilenetv2 + deeplabv3 + midas | ~120 ms | i.MX 8M Plus | ~7 W | Full scene understanding at ~8 fps; single-task would reach 25 fps |

---

## 14. Deployment Workflow Comparison

| Step | TI EdgeAI | ADI AI8X | NXP eIQ |
|------|-----------|----------|---------|
| **Model source** | ONNX/TFLite from MMDetection/MMPose training | PyTorch .pth.tar from ai8x-training (QAT) | TFLite INT8 from Docker recipe.sh |
| **Compile / synthesize** | `tidl_compile.py` → `.bin` artifacts | `ai8xize.py` → `weights.h` + C headers | Optional: `vela model.tflite --accelerator-config ethos-u65-256` |
| **Calibration** | 50–500 representative frames (PTQ) | N/A — QAT eliminates calibration | 100–1,000 representative samples |
| **Accuracy gate** | Top-1 / mAP / mIoU threshold per SoC | Accuracy threshold per device | Top-1 / PSNR / WER / AUC threshold |
| **Bundle format** | `.tar.gz` (TIDL bin + config + preproc spec) | `.tar.gz` (C headers + manifest + inference_example.c) | `.tgz` (TFLite + Vela + recipe.sh + metadata.json) |
| **Deploy target** | SCP to EVM board / OTA via TISDK | Flash to MAX78002 / MAX32690 via UART | SCP to i.MX board / container image |
| **CI / CD** | GitHub Actions: 6 jobs (lint → test → eval → package) | GitHub Actions: 6 jobs | GitHub Actions: 6 jobs |
| **Cloud orchestration** | AWS CodePipeline (Source → Evaluate → Deploy) | AWS CodePipeline (QAT Train → Synthesize → Package) | AWS CodePipeline (Recipe → VelaCompile → Package) |
| **Monitoring** | CloudWatch: accuracy + latency alarms | CloudWatch: mAP + KWS + synthesis failure alarms | CloudWatch: accuracy + PSNR + latency p95 alarms |

---

## 15. Key Takeaways and Recommendations

### Choose TI EdgeAI when:
- Deploying in **automotive** environments (TDA4VM/AM68A are ASIL-rated SoCs)
- Need the **widest vision model variety** (34 models, 5 task categories)
- Require **highest classification accuracy** (efficientnet-lite4 at 82.4% INT8)
- Building **4-stage ADAS pipelines** (detect + segment + depth + pose)
- Already invested in the TI software stack (EdgeAI SDK, TIDL)

### Choose ADI AI8X when:
- **Power budget is < 5 mA** (battery-powered IoT sensor, wearable)
- **QAT precision** is required — ADI models are trained with quantization, not converted after
- Deploying **multi-modal sensor fusion** (vision + audio + vibration on separate cores)
- Need **industrial anomaly detection** with dedicated motor fault model
- Require **audio denoising** in the pipeline (DTLN/RNNoise, PESQ ~2.95)

### Choose NXP eIQ when:
- Need the **broadest modality coverage** (vision + audio + face + BCI + SR + enhance)
- Deploying **image enhancement before inference** (SCI low-light → downstream model)
- Need **face identification** (FaceNet512 at 99.35% LFW — unique to NXP)
- Building **end-to-end voice pipelines** (KWS → ASR via Wav2Letter, WER 7.2%)
- Deploying on **standard TFLite toolchain** (lowest deployment complexity)
- Research or niche applications: **BCI motor imagery** (EEG-TCNet, unique to NXP)

---

## 16. Summary Scorecard

| Criterion | TI EdgeAI | ADI AI8X | NXP eIQ |
|-----------|:---------:|:--------:|:-------:|
| Model variety | ⭐⭐⭐⭐⭐ (34) | ⭐⭐⭐ (15) | ⭐⭐⭐⭐ (29) |
| Best classification accuracy (INT8) | ⭐⭐⭐⭐⭐ (82.4%, efficientnet-lite4) | ⭐⭐ (64.7%, CIFAR-100 scope) | ⭐⭐⭐⭐ (78.2%, inceptionv4 ImageNet) |
| Best detection mAP (COCO @50) | ⭐⭐⭐⭐ (63.0%, yolox-m-lite) | ⭐⭐⭐ (50.5% VOC, different dataset) | ⭐⭐⭐⭐ (50.2%, yolov8_m) |
| Quantization quality (accuracy preservation) | ⭐⭐⭐ (PTQ, 0.2–4.2% drop) | ⭐⭐⭐⭐⭐ (QAT, INT8 is the target) | ⭐⭐⭐⭐ (PTQ, typically <1% drop) |
| Lowest power / energy efficiency | ⭐⭐ (5–30 W automotive SoC) | ⭐⭐⭐⭐⭐ (0.3–20 mA MCU/DSP) | ⭐⭐⭐ (0.5 W MCU – 8 W SoC) |
| Audio support | ⭐ (none) | ⭐⭐⭐⭐⭐ (KWS 94.5% + ASR-adjacent + denoise + genre) | ⭐⭐⭐⭐ (KWS 91.6% + ASR 7.2% WER + anomaly) |
| Unique domain coverage | ⭐⭐⭐ (ADAS transformer, ADE20K seg) | ⭐⭐⭐⭐ (motor fault, QR detect, audio genre) | ⭐⭐⭐⭐⭐ (SR, low-light, face ID, BCI, ASR) |
| Smallest deployable model | ⭐⭐⭐ (nano YOLOX 1.48 GMACs) | ⭐⭐⭐⭐⭐ (micronet_vww2, 267 KB) | ⭐⭐⭐⭐⭐ (sci_low_light, 8 KB) |
| Multi-model pipeline depth | ⭐⭐⭐⭐⭐ (4-stage ADAS) | ⭐⭐⭐⭐ (hierarchical VWW → KWS → detect) | ⭐⭐⭐⭐⭐ (enhance → face recog → head pose) |
| Deployment complexity | ⭐⭐⭐ (TIDL toolchain + SoC calibration) | ⭐⭐ (ai8xize C synthesis + flash) | ⭐⭐⭐⭐ (standard TFLite, optional Vela) |
| MLOps tooling | ⭐⭐⭐⭐ (MLflow + Prefect + Optuna TIDL HPO) | ⭐⭐⭐⭐ (MLflow + Optuna QAT HPO) | ⭐⭐⭐⭐ (MLflow + Optuna INT8 PTQ HPO + Vela stats) |
| Dataset diversity | ⭐⭐⭐⭐ (COCO, ADE20K, NYU, COCO-KP) | ⭐⭐⭐⭐ (MIMII, GTZAN, GSC, VOC, ADI proprietary) | ⭐⭐⭐⭐⭐ (ImageNet, COCO, LFW, LibriSpeech, BCI IV, DIV2K, LOL, ToyADMOS) |
