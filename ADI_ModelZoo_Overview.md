# ADI Model Zoo — Comprehensive Overview

**Source:** https://github.com/analogdevicesinc/adi-model-zoo  
**Local clone:** `C:\Users\Administrator\model_zoo\adi-model-zoo\`  
**Last index update:** 2026-08-11  

---

## 1. Introduction

The ADI (Analog Devices Inc.) Model Zoo is a curated collection of **pretrained machine-learning models** targeting Analog Devices embedded hardware. Unlike large cloud-centric zoos, every model here is optimized for ultra-low-power microcontroller and DSP deployment. Models ship with:

- Pretrained weights (stored directly in the repo — no `.link` indirection)
- Sample input/output data
- Preprocessing utilities
- Self-contained `example.py` inference scripts
- `definition.yaml` metadata (schema-validated)
- Per-model `requirements.txt`

---

## 2. Repository Structure

```
adi-model-zoo/
├── model_index.json          # Central registry (all models, per-platform paths, quantization)
├── models.schema.json        # JSON Schema for model_index.json
├── audio_models/
│   ├── audio_denoising/
│   │   ├── dtln/             # DTLN (FP32 TFLite)
│   │   └── rnnoise/          # RNNoise (INT8 TFLite)
│   ├── audio_genre_identification/
│   │   └── genrenet_conv2d/  # GenreNet Conv2D (PyTorch)
│   └── audio_keyword_spotting/
│       ├── conv1d_audionet/  # Conv1D AudioNet (AI8X)
│       └── ds_cnn/           # DS-CNN Small (TFLite)
├── sensor_models/
│   ├── anomaly_detection/
│   │   ├── autoencoder/      # Autoencoder (TFLite, 4 machine variants)
│   │   └── micronet/         # MicroNet AD Small (INT8 TFLite)
│   └── motor_fault_detection/
│       └── autoencoder/      # Autoencoder (AI8X INT8)
├── vision_models/
│   ├── image_classification/
│   │   ├── efficientnet/     # EfficientNetV2 (AI8X)
│   │   ├── mobilenetv2/      # MobileNetV2 × 2 variants (AI8X)
│   │   └── simplenet/        # SimpleNet (AI8X)
│   ├── image_segmentation/
│   │   └── unet/             # U-Net (AI8X)
│   ├── object_detection/
│   │   ├── feature_pyramid_net/  # FPN (AI8X, Pascal VOC)
│   │   └── tinierssd/            # TinierSSD (AI8X, QR code)
│   └── visual_wake_word/
│       └── micronet_vww2_int8/   # MicroNet VWW2 (TFLite INT8)
└── third_party/
    └── ai8x/                 # ADI AI8X PyTorch runtime shim
        ├── ai8x.py           # Quantization layers, fold/unfold, normalize
        ├── ai8x_blocks.py    # Building blocks (ResNet blocks, etc.)
        ├── devices.py        # MAX78000/MAX78002 hardware limits
        └── util_ai8x_inference.py  # AI8XInference runner
```

### Per-Model Internal Structure

```
<model>/
├── data/
│   ├── input/          # Sample input file(s)
│   ├── model/          # Weights (.pth.tar / .tflite / .pt)
│   └── output/         # Reference outputs for verification
├── data_processing/    # Preprocessing / postprocessing scripts
├── definition.yaml     # Schema-validated metadata
├── example.py          # Runnable inference demo
├── README.md           # Model-specific documentation
└── requirements.txt    # Python dependencies
```

---

## 3. Target Hardware Platforms

| Device | Family | Core | Memory | Key Capability |
|--------|--------|------|--------|----------------|
| **MAX78002** | DARWIN CNN | Cortex-M4F + CNN accelerator | 384KB SRAM, 2MB flash | 74-layer CNN, <6mW active, 1024-channel support |
| **MAX32690** | DARWIN MCU | Cortex-M4F | 3MB flash, 1MB SRAM | BLE 5.2, USB, ultra-low power MCU |
| **ADSP-SC835** | Blackfin+ DSP | DSP + Arm | On-chip SRAM + DDR | Audio/voice processing, HiFi4 DSP core |

### Evaluation Boards

| SoC | Eval Board |
|-----|-----------|
| MAX78002 | MAX78002EVKIT |
| MAX32690 | EvKit_V1 |
| ADSP-SC835 | ADSPSC835-EV-SOM |

---

## 4. Model Formats

### AI8X Format (ADI-Proprietary)
The primary format for MAX78000/MAX78002 deployment:
- Models defined in PyTorch using `ai8x.py` constraint-aware layers
- INT8 quantization via **QAT (Quantization-Aware Training)** using `ai8xize.py` synthesis tools
- Output: `.pth.tar` checkpoint + `.py` network definition
- Synthesis step generates per-layer hardware configuration for the CNN accelerator
- Key constraints: fixed weight precision (INT8), limited channel counts per layer, no arbitrary activations

### TFLite
- FP32 and INT8 variants (INT8 = post-training quantization)
- Used on MAX32690 (TFLite Micro runtime) and ADSP-SC835
- INT8 models deployed via MAX-Efficiency TFLite Micro

### PyTorch (.pt)
- FP32 models, primarily for ADSP-SC835 (has sufficient compute for FP32)
- Used by GenreNet Conv2D

---

## 5. Vision Models

### Image Classification

| Model | Dataset | Input | Top-1 Acc | Format | Target |
|-------|---------|-------|-----------|--------|--------|
| EfficientNetV2 | ImageNet | 112×112 | 0.6197 | AI8X | MAX78002 |
| MobileNetV2 (0.75×) | CIFAR-100 | 32×32 | 0.6471 | AI8X INT8 | MAX78002 |
| MobileNetV2 (0.5×) | CIFAR-100 | 32×32 | ~0.62 | AI8X INT8 | MAX78002 |
| SimpleNet | CIFAR-100 | 32×32 | 0.6030 | AI8X INT8 | MAX78002 |

**EfficientNetV2:** Scaled EfficientNet with compound scaling. Adapted to 112×112 input for MAX78002 with `ai87-imagenet-effnet2-q.pth.tar` (19.9MB).

**MobileNetV2:** Inverted residual blocks with depthwise convolutions, width multipliers 0.75 and 0.5. CIFAR-100 variant (100 classes, 32×32). Both ~INT8 via QAT.

**SimpleNet:** Sequential convolutional blocks → global average pooling → Softmax. CIFAR-100, 100 classes. Mixed INT8 quantization.

### Image Segmentation

| Model | Dataset | Input | Accuracy | Format | Target |
|-------|---------|-------|----------|--------|--------|
| U-Net | AISegment | 48×48 | 0.9845 | AI8X INT8 | MAX78002 |

U-Net with contracting + expanding paths and skip connections. Pixel-wise binary segmentation (person vs. background). Input is 48×48×48 (3-channel folded representation).

### Object Detection

| Model | Dataset | Input | mAP | Format | Target |
|-------|---------|-------|-----|--------|--------|
| FeaturePyramidNet | Pascal VOC | 256×320 | 0.50512 | AI8X INT8 | MAX78002 |
| TinierSSD | BG-20k (QR) | 120×160 | 0.89960 | AI8X INT8 | MAX78002 |

**FeaturePyramidNet:** Multi-scale feature pyramid merging deep+shallow features. 21-class VOC detection. Outputs regression (1×10200×4) + classification (1×10200×21). NMS post-processing with `object_detection_utils.py`.

**TinierSSD:** Tinier Single Shot MultiBox Detector specialized for **QR code detection with keypoints**. 4 keypoints per code (corners). Input 120×160, output: loc (1×1548×12), score (1×1548×2).

### Visual Wake Word

| Model | Dataset | Input | Accuracy | Format | Target |
|-------|---------|-------|----------|--------|--------|
| MicroNet VWW2 INT8 | Visual Wake Words | 50×50×1 | 0.768 | TFLite INT8 | MAX32690 |

Binary person/no-person detection. 267KB TFLite INT8, optimized for Cortex-M4F via TFLite Micro. Greyscale 50×50 input.

---

## 6. Audio Models

### Audio Denoising

| Model | Metric | Format | Target | Size |
|-------|--------|--------|--------|------|
| DTLN | PESQ: 2.950 | TFLite FP32 (×2) | General | 1.39MB + 2.39MB |
| RNNoise | PESQ: 2.945 | TFLite INT8 | MAX32690 | 107KB |

**DTLN (Dual-Signal Transformation LSTM Network):** Two-stage LSTM network combining STFT + learned analysis/synthesis basis. <1M parameters. Runs as 2 sequential TFLite models (`model_float_1.tflite` + `model_float_2.tflite`). Stateful — carries 4 state tensors between frames. Input: magnitude (1×1×257) + states.

**RNNoise:** RNN-based noise suppression using GRU layers (24→48→96 units). INT8 quantized. 4 stateful inputs (VAD GRU, noise GRU, denoise GRU states). Output includes confidence score (VAD) + spectral mask (1×1×22).

### Genre Identification

| Model | Dataset | Accuracy | Format | Target | Size |
|-------|---------|----------|--------|--------|------|
| GenreNet Conv2D | GTZAN | 0.8450 | PyTorch | ADSP-SC835 | 15.3MB |

Conv2D model classifying audio into 10 genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock. Input: Mel spectrogram (1×1×128×128). FP32 inference on ADSP-SC835.

### Keyword Spotting

| Model | Dataset | Accuracy | Format | Target | Size |
|-------|---------|----------|--------|--------|------|
| Conv1D AudioNet | Google Speech Commands V2 | 0.8634 | AI8X | MAX78002 | 1.785MB |
| DS-CNN Small | Google Speech Commands V2 | 0.9452 | TFLite | MAX32690, ADSP-SC835 | 97KB |

**Conv1D AudioNet:** 1D convolutional model for 20-keyword spotting. AI8X format (`.pth.tar`), QAT INT8. Input: 128×128 feature map.

**DS-CNN Small:** Depthwise Separable CNN, 12-class keyword classifier. Multi-precision TFLite (INT8/INT16/FP32). Best accuracy (94.52%). Input: 1×490 feature vector.

---

## 7. Sensor Models

### Motor Fault Detection

| Model | Dataset | Metric | Format | Target | Size |
|-------|---------|--------|--------|--------|------|
| Autoencoder | Sample Motor Data (Limerick) | MSE: 0.02205 | AI8X INT8 | MAX78002 | 1.59MB |

Convolutional autoencoder compressing/reconstructing 3-axis motor vibration signals (256×3). Fault = high reconstruction error (MSE threshold). ADI-authored model using proprietary motor data from Limerick facility.

### Anomaly Detection

| Model | Dataset | Metric | Format | Target | Size |
|-------|---------|--------|--------|--------|------|
| Autoencoder ×4 | MIMII | AUC: 0.516, pAUC: 0.523 | TFLite | MAX32690 | 287KB ×4 |
| MicroNet AD Small | DCASE 2020 / MIMII | — | TFLite INT8 | MAX32690 | 240KB |

**Autoencoder (MIMII):** 4 per-machine-type variants (fan, pump, slider, valve). Spectrogram (196×640) reconstruction anomaly detection. Self-supervised: train on normal sounds only.

**MicroNet AD Small:** Lightweight CNN for self-supervised anomaly detection. Trained as machine-ID classifier (DCASE 2020 challenge setup). INT8 TFLite, 32×32×1 log-mel input.

---

## 8. Model Index (model_index.json)

The `model_index.json` file is the canonical registry. Each entry contains:

```json
{
  "uniqueId": "<uuid>",
  "name": "<model-name>",
  "task": "<task-type>",
  "taskType": "Vision | Audio | Sensor",
  "framework": "tensorflow | pytorch | ai8x",
  "license": "Apache-2.0 | MIT",
  "owner": "ADI | <author>",
  "supportedPlatforms": [
    {
      "soc": "MAX78002",
      "board": "MAX78002EVKIT",
      "family": "CNN",
      "core": "CM4+CNN",
      "modelPaths": {
        "int8": "path/to/model_int8.tflite",
        "float32": "path/to/model_float32.tflite"
      },
      "sizeInKb": { "int8": 46.5, "float32": 96.4 },
      "compilation": {
        "optimization": { "values": ["-O0", "-O2"], "default": "-O0" },
        "quantization": { "values": ["int8", "float32"], "default": "int8" }
      }
    }
  ]
}
```

---

## 9. AI8X Runtime (third_party/ai8x/)

The `third_party/ai8x/` module provides the PyTorch shim for MAX78000/MAX78002:

| File | Purpose |
|------|---------|
| `ai8x.py` | `normalize` ([-128, 127] scaling), `fold`/`unfold` (channel folding for HW constraints), quantized layers |
| `ai8x_blocks.py` | `ResidualBottleneck`, `ResidualBottleneckInverse`, FPN block definitions |
| `devices.py` | Hardware limits: max channels (1024), weight bits (8), bias bits (8) |
| `util_ai8x_inference.py` | `AI8XInference` class: loads `.pth.tar`, runs forward pass with proper INT8 scaling |

**Channel folding:** MAX78002 has limited channel depth per layer. `ai8x.fold` interlaces spatial pixels into channel dimension (fold_ratio²× more channels, fold_ratio× smaller spatial), enabling larger receptive fields within hardware limits.

---

## 10. Preprocessing Pipeline

### Vision
| Task | Preprocessing |
|------|--------------|
| Classification | Resize to model input, normalize to [-128, 127] or [0, 1] |
| Segmentation | Resize 48×48, channel fold (ratio=8), INT8 normalize |
| Object Detection | Resize 256×320 (FPN) or 120×160 (TinierSSD), INT8 normalize |
| Visual Wake Word | Greyscale, resize 50×50, INT8 normalize |

### Audio
| Task | Preprocessing |
|------|--------------|
| Denoising (DTLN) | 16kHz, 32ms frame, STFT magnitude, frame-by-frame stateful |
| Denoising (RNNoise) | 16kHz, 10ms frames, 42-feature Bark-band features, INT8 |
| Genre ID | Mel spectrogram 128×128, FP32 |
| KWS | MFCCs or raw spectrogram, 1 sec window |

### Sensor
| Task | Preprocessing |
|------|--------------|
| Motor Fault | 256-sample FFT slices, 3-axis vibration |
| Anomaly Detection | 64-frame log-mel spectrogram (196×640) |

---

## 11. Datasets

| Dataset | Task | Models Using |
|---------|------|-------------|
| ImageNet | Classification | EfficientNetV2 |
| CIFAR-100 | Classification | MobileNetV2, SimpleNet |
| AISegment | Segmentation | U-Net |
| Pascal VOC | Object Detection | FeaturePyramidNet |
| BG-20k (synthetic QR) | Object Detection | TinierSSD |
| Visual Wake Words | VWW | MicroNet VWW2 |
| Google Speech Commands V2 | KWS | DS-CNN, Conv1D AudioNet |
| GTZAN | Genre ID | GenreNet |
| MIMII | Anomaly Detection | Autoencoder, MicroNet AD |
| Sample Motor Data | Motor Fault | Autoencoder (motor) |

---

## 12. Toolchain & Ecosystem

| Tool | Purpose | Where to Get |
|------|---------|-------------|
| **ai8xize.py** | Synthesizes AI8X model → MAX78002 hardware config | github.com/MaximIntegratedAI/ai8x-synthesis |
| **ai8x-training** | QAT training framework for MAX78000/78002 | github.com/MaximIntegratedAI/ai8x-training |
| **TFLite Micro** | INT8 inference on MAX32690 | TensorFlow repo |
| **MSDK (MAX SDK)** | Full SDK for MAX78000/78002/32690 | github.com/analogdevicesinc/msdk |
| **EV-Kit examples** | Ready-to-flash firmware demos | github.com/analogdevicesinc/msdk/Examples |
| **ADI AI Model Zoo** | This repository | github.com/analogdevicesinc/adi-model-zoo |

---

## 13. Running Models

### Setup
```bash
# Per-model install
cd vision_models/object_detection/feature_pyramid_net
pip install -U setuptools
pip install -r requirements.txt

# Run inference
python example.py data/input/input.jpg
```

### Custom data
```bash
python example.py [path_to_file]
```

### AI8X Inference (Python API)
```python
from third_party.ai8x.util_ai8x_inference import AI8XInference

model = AI8XInference("data/model/ai87-pascalvoc-fpndetector-qat8-q.pth.tar")
output = model.run(input_tensor)  # input: numpy array, INT8 normalized
```

### TFLite Inference
```python
import tflite_runtime.interpreter as tflite
interp = tflite.Interpreter("data/model/ds_cnn_s_int8.tflite")
interp.allocate_tensors()
interp.set_tensor(interp.get_input_details()[0]['index'], input_data)
interp.invoke()
result = interp.get_tensor(interp.get_output_details()[0]['index'])
```

---

## 14. Comparison: ADI Model Zoo vs. TI EdgeAI Model Zoo

| Dimension | ADI Model Zoo | TI EdgeAI Model Zoo |
|-----------|--------------|---------------------|
| Model count | ~15 | 180+ |
| Domains | Vision, Audio, Sensor | Vision only |
| Target class | MCU / DSP (µW–mW) | SoC / MPU (W range) |
| Primary format | AI8X (proprietary), TFLite | ONNX, TFLite, TVM |
| Quantization | INT8 QAT (AI8X), TFLite PTQ | INT8 (all), QAT variants |
| Storage | Models in repo | .link file indirection |
| Hardware | MAX78002, MAX32690, ADSP-SC835 | TDA4VM, AM62A–AM69A |
| TOPS range | < 0.1 TOPS (CNN accelerator) | 1–32 TOPS |
| Use case | TinyML, IoT, wearables | ADAS, smart cameras, robotics |
| Inference runtime | AI8X, TFLite Micro | TIDL, onnxrt, tflitert, tvmrt |
| Dataset focus | CIFAR-100, VOC, Speech, MIMII | ImageNet, COCO, Cityscapes |
