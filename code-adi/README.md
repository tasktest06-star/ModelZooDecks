# ADI Model Zoo MLOps Pipeline

End-to-end MLOps pipeline for Analog Devices model zoo models. Supports AI8X (PyTorch QAT), TFLite, and PyTorch runtime targets across MAX78002, MAX32690, and ADSP-SC835 devices.

---

## Prerequisites

### Python environment
```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

### AI8X toolchain (for MAX78002 / MAX78002EVKIT models)
```bash
# Clone ADI training repo (for ai8xize synthesis)
git clone https://github.com/analogdevicesinc/ai8x-training.git
git clone https://github.com/analogdevicesinc/ai8x-synthesis.git

# Install synthesis tool dependencies
cd ai8x-synthesis && pip install -r requirements.txt && cd ..

# Verify ai8xize.py is accessible
python ai8x-synthesis/ai8xize.py --help
```

### TFLite runtime (for MAX32690 / ADSP-SC835 models)
```bash
pip install tflite-runtime
# or: pip install tensorflow (includes tflite)
```

---

## Quick Start

```bash
# Run full evaluation pipeline on all models
python -m pipelines.eval_pipeline

# Evaluate a single model
python -m pipelines.eval_pipeline --model simplenet

# Deploy a model to device
python -m pipelines.deploy_pipeline --model simplenet --device MAX78002EVKIT

# Run tests
pytest tests/ -v
```

---

## Model Registry

All 15 ADI Model Zoo models are registered in `config/model_registry.yaml`.

### Vision — Image Classification (4 models)

| Model | Input | Dataset | Top-1 | Device | Format |
|-------|-------|---------|-------|--------|--------|
| `efficientnetv2` | 112×112 | ImageNet | 61.97% | MAX78002 | AI8X |
| `mobilenetv2_075` | 32×32 | CIFAR-100 | 64.71% | MAX78002 | AI8X |
| `mobilenetv2_050` | 32×32 | CIFAR-100 | 62.0% | MAX78002 | AI8X |
| `simplenet` | 32×32 | CIFAR-100 | 60.30% | MAX78002 | AI8X |

### Vision — Object Detection (2 models)

| Model | Input | Dataset | mAP | Device | Format |
|-------|-------|---------|-----|--------|--------|
| `feature_pyramid_net` | 256×320 | Pascal VOC | 50.51% | MAX78002 | AI8X |
| `tinierssd` | 120×160 | Synthetic QR | 89.96% | MAX78002 | AI8X |

### Vision — Image Segmentation (1 model)

| Model | Input | Dataset | Accuracy | Device | Format |
|-------|-------|---------|----------|--------|--------|
| `unet` | 48×48* | AiSegment | 98.45% | MAX78002 | AI8X |

*channel-folded (fold_ratio=4, 3ch RGB → 48ch)

### Vision — Visual Wake Word (1 model)

| Model | Input | Dataset | Accuracy | Device | Format |
|-------|-------|---------|----------|--------|--------|
| `micronet_vww2` | 50×50 | VWW | 76.8% | MAX32690 | TFLite INT8 |

### Audio — Keyword Spotting (2 models)

| Model | Classes | Dataset | Accuracy | Device | Format |
|-------|---------|---------|----------|--------|--------|
| `conv1d_audionet` | 21 | Google Speech v2 | 86.34% | MAX78002 | AI8X |
| `ds_cnn` | 12 | Google Speech v2 | 94.52% | MAX32690, ADSP-SC835 | TFLite |

### Audio — Denoising (2 models)

| Model | Sample Rate | PESQ | Device | Format |
|-------|-------------|------|--------|--------|
| `dtln` | 16 kHz | 2.950 | ADSP-SC835 | TFLite float32 |
| `rnnoise` | 16 kHz | 2.945 | MAX32690 | TFLite INT8 |

### Audio — Genre Identification (1 model)

| Model | Input | Classes | Accuracy | Device | Format |
|-------|-------|---------|----------|--------|--------|
| `genrenet_conv2d` | 128×128 mel | 10 genres | 84.50% | ADSP-SC835 | PyTorch |

### Sensor — Anomaly Detection (1 model)

| Model | Machine Types | AUC | Device | Format |
|-------|--------------|-----|--------|--------|
| `autoencoder_anomaly` | fan/pump/slider/valve | 0.516 | MAX32690 | TFLite float32 |

### Sensor — Motor Fault Detection (1 model)

| Model | Input | MSE | Device | Format |
|-------|-------|-----|--------|--------|
| `autoencoder_motor` | 256×3 vibration | 0.02205 | MAX78002 | AI8X |

---

## Device Reference

| Device | Board | Family | Best For |
|--------|-------|--------|----------|
| MAX78002 | MAX78002EVKIT | CNN (AI8X) | Vision + audio inference, 87µW idle |
| MAX32690 | EvKit_V1 | Cortex-M4 + BLE | TFLite models, VWW, audio denoising |
| ADSP-SC835 | ADSPSC835-EV-SOM | DSP | Audio processing, genre recognition |

---

## AI8X Workflow (MAX78002 Models)

### 1. Training (QAT)
```bash
# Train a model with quantization-aware training
cd ai8x-training
python train.py --model ai87net-simplenet-wide2x --dataset CIFAR100 \
    --qat-policy policies/qat_policy.yaml --compress policies/schedule.yaml
```

### 2. Quantize weights
```bash
python quantize.py checkpoints/ai85-cifar100-simplenetwide2x.pth.tar \
    --output checkpoints/ai85-cifar100-simplenetwide2x-q.pth.tar \
    --config models/ai85net-simplenet-wide2x.yaml
```

### 3. Synthesize hardware-ready C code (ai8xize)
```bash
cd ai8x-synthesis
python ai8xize.py \
    --test-dir /tmp/cnn_output \
    --prefix simplenet \
    --checkpoint-file path/to/ai85-cifar100-simplenetwide2x-q.pth.tar \
    --config-file networks/cifar100-simplenet-hwc.yaml \
    --sample-input tests/sample_cifar100.npy \
    --overwrite \
    --mexpress \
    --compact-data
```

### Channel Folding (UNet / segmentation)
Channel folding packs spatial pixels into the channel dimension so that 3-channel RGB fits within hardware constraints. For UNet with `fold_ratio=4`:
```python
# 3-channel 192×192 input → 48-channel 48×48 input
import numpy as np
img = np.load("sample_192x192.npy")   # shape (3, 192, 192)
folded = img.reshape(48, 48, 48)       # fold_ratio=4 in both H and W
```
The pipeline's `data_pipeline.py` handles this automatically via `ADIDataPipeline.fold_channels()`.

---

## TFLite Workflow (MAX32690 / ADSP-SC835 Models)

```python
import tflite_runtime.interpreter as tflite
import numpy as np

interpreter = tflite.Interpreter(model_path="ds_cnn_s_int8.tflite")
interpreter.allocate_tensors()

inp = interpreter.get_input_details()
out = interpreter.get_output_details()

interpreter.set_tensor(inp[0]['index'], np.expand_dims(audio_features, 0).astype(np.float32))
interpreter.invoke()
result = interpreter.get_tensor(out[0]['index'])
```

---

## Pipeline Architecture

```
config/
  pipeline_config.yaml     ← device targets, eval gates, artifact settings
  model_registry.yaml      ← all 15 models with metadata

mlops/
  model_manager.py         ← load/register/query models from registry
  data_pipeline.py         ← preprocessing per task (vision, audio, sensor)
  evaluator.py             ← top1, mAP, AUC, MSE, PESQ metrics
  artifact_manager.py      ← pack/unpack deployment bundles
  monitor.py               ← latency tracking, drift alerts

pipelines/
  eval_pipeline.py         ← run evaluation across all or selected models
  deploy_pipeline.py       ← synthesize → bundle → flash to device

tests/
  test_pipeline.py         ← 30+ hardware-free pytest tests
```

---

## Configuration

Edit `config/pipeline_config.yaml` to set:
- `primary_device` — default target device for deployment
- `eval_gates` — minimum accuracy thresholds per task (pipeline fails if not met)
- `artifact_output_dir` — where packaged bundles are written
- `monitor.drift_threshold` — alert threshold for accuracy degradation

---

## Running Tests

```bash
# All tests (no hardware required)
pytest tests/ -v

# Single test class
pytest tests/test_pipeline.py::TestModelManager -v

# With coverage
pytest tests/ --cov=mlops --cov-report=term-missing
```

---

## Downloading Model Weights

Model weights are not included in this repo. Download from the ADI Model Zoo:

```bash
# Clone the ADI model zoo
git clone https://github.com/analogdevicesinc/adi-model-zoo.git

# Point the pipeline at the zoo root
export ADI_ZOO_ROOT=/path/to/adi-model-zoo

# Then model_path entries in model_registry.yaml resolve relative to ADI_ZOO_ROOT
```

Each model's `model_path` in `config/model_registry.yaml` matches the directory layout in the zoo repo exactly.
