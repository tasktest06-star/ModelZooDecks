# NXP eIQ Model Zoo MLOps Pipeline

End-to-end MLOps pipeline for NXP eIQ model zoo models. Supports TFLite INT8 inference across i.MX 8M Plus, i.MX 93 (Ethos-U65 NPU), i.MX RT17xx/10xx MCUs, and MCX N947.

---

## Prerequisites

### Python environment
```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

### Docker (required for recipe execution)
Models are delivered as Docker-based recipes containing a `recipe.sh` script.
```bash
# Verify Docker is running
docker info

# Pull the NXP eIQ base image (used by most recipes)
docker pull ghcr.io/nxp-imx/eiq-model-zoo:latest
```

### Arm Vela compiler (required for i.MX 93 / Ethos-U65 NPU)
```bash
pip install ethos-u-vela

# Verify installation
vela --version
```

### TFLite runtime
```bash
pip install tflite-runtime
```

---

## Quick Start

```bash
# Run evaluation on all models (CPU/XNNPack delegates, no board required)
python -m pipelines.eval_pipeline

# Evaluate a single model
python -m pipelines.eval_pipeline --model mobilenetv2

# Compile a model for i.MX 93 (Vela required)
python -m mlops.vela_compiler --model mobilenetv2 --output /tmp/vela_out

# Deploy a model to an i.MX 8M Plus board
python -m pipelines.deploy_pipeline --model mobilenetv2 --platform imx8mplus \
    --board-ip 192.168.1.100

# Run tests (no hardware required)
pytest tests/ -v
```

---

## Model Registry

All 29 models are registered in `config/model_registry.yaml`, covering ~83% of the zoo.

### Vision — Image Classification (9 models)

| Model | Input | Dataset | Top-1 | Platforms | Vela |
|-------|-------|---------|-------|-----------|------|
| `mobilenetv2` | 224×224 | ImageNet | 71.8% | i.MX 8M+, i.MX 93 | Yes |
| `mobilenetv1` | 224×224 | ImageNet | 70.9% | i.MX 8M+, i.MX 93, RT1170, RT1050 | Yes |
| `resnet50` | 224×224 | ImageNet | 75.9% | i.MX 8M+, i.MX 93 | Yes |
| `efficientnet_lite` | 224×224 | ImageNet | 72.2% | i.MX 8M+, i.MX 93 | Yes |
| `mnasnet` | 224×224 | ImageNet | 72.3% | i.MX 8M+, i.MX 93 | Yes |
| `inceptionv4` | 299×299 | ImageNet | 78.2% | i.MX 8M+, i.MX 93 | Yes |
| `tiny_resnet` | 32×32 | CIFAR-10 | 92.4% | i.MX 8M+, i.MX 93, MCX N947 | Yes |
| `visual_wake_word` | 96×96 | VWW | 85.3% | i.MX 8M+, i.MX 93, MCX N947 | Yes |
| `deepface_emotion` | 48×48 | FER-2013 | 64.5% | i.MX 8M+, i.MX 93 | Yes |

### Vision — Object Detection (7 models)

| Model | Input | Dataset | mAP@50 | Platforms | Vela |
|-------|-------|---------|--------|-----------|------|
| `yolov8_m` | 320×320 | COCO | 50.2% | i.MX 8M+, i.MX 93 | Yes |
| `yolov5` | 320×320 | COCO | 37.0% | i.MX 8M+, i.MX 93 | Yes |
| `yolov4_tiny` | 416×416 | COCO | 21.7% | i.MX 8M+, i.MX 93 | Yes |
| `ssdlite_mobilenetv2` | 320×320 | COCO | 19.0% | i.MX 8M+, i.MX 93 | Yes |
| `nanodet_m` | 320×320 | COCO | 20.2% | i.MX 8M+, i.MX 93, RT1170, RT1050 | Yes |
| `facedet` | 128×128 | WiderFace | 83.8% | i.MX 8M+, i.MX 93, RT1170, RT1060, MCX N947 | Yes |
| `efficientdet_lite0` | 320×320 | COCO | 25.0% | i.MX 8M+, i.MX 93 | Yes |

### Vision — Segmentation (2 models)

| Model | Input | Dataset | mIoU | Platforms | Vela |
|-------|-------|---------|------|-----------|------|
| `deeplabv3` | 513×513 | Pascal VOC | 70.2% | i.MX 8M+ | No |
| `yolact_edge` | 550×550 | COCO | mask mAP 21.0% | i.MX 8M+, i.MX 93 | Yes |

### Vision — Super Resolution / Low-light / Depth (3 models)

| Model | Task | Input | Metric | Platforms |
|-------|------|-------|--------|-----------|
| `fast_srgan` | Super Resolution (4×) | 128×128 | PSNR 31.2 dB | i.MX 8M+, i.MX 93 |
| `sci_low_light` | Low-light Enhancement | 1920×1080 | PSNR 28.3 dB | i.MX 8M+, i.MX 93 |
| `midas_v21_small` | Monocular Depth | 256×256 | rel_err 0.122 | i.MX 8M+ |

### Vision — Face Recognition & Pose Estimation (3 models)

| Model | Task | Input | Metric | Platforms |
|-------|------|-------|--------|-----------|
| `facenet512` | Face Recognition | 160×160 | LFW 99.35% | i.MX 8M+, i.MX 93 |
| `movenet` | Pose Estimation (17 kpts) | 192×192 | PCK@50 72.6% | i.MX 8M+, i.MX 93 |
| `whenet` | Head Pose | 224×224 | MAE yaw 4.2° | i.MX 8M+, i.MX 93 |

### Audio (3 models)

| Model | Task | Classes | Accuracy/WER | Platforms |
|-------|------|---------|-------------|-----------|
| `ds_cnn` | Keyword Spotting | 12 | 91.6% | i.MX 8M+, i.MX 93, MCX N947 |
| `microspeech_lstm` | Keyword Spotting | 4 | 88.1% | i.MX 8M+, i.MX 93, MCX N947 |
| `wav2letter` | Speech Recognition | — | WER 7.2% | i.MX 8M+, i.MX 93 |

### Sensor / Other (2 models)

| Model | Task | Dataset | AUC | Platforms |
|-------|------|---------|-----|-----------|
| `deep_autoencoder_ad` | Anomaly Detection | ToyADMOS | 82.0% | i.MX 8M+, i.MX 93, RT1170, MCX N947 |
| `eeg_tcnet` | EEG BCI | BCI-IV-2a | 77.4% | i.MX 8M+ (CPU only) |

---

## Platform Reference

| Platform | Board | NPU/Accel | Best For |
|----------|-------|-----------|----------|
| i.MX 8M Plus | 8MPLUSLPD4-EVK | 2.3 TOPS NPU (A71CH) | Vision inference, most models |
| i.MX 93 | IMX93EVKBOARD | Ethos-U65 (256 MACs) | Low-power TFLite INT8 |
| i.MX RT1170 | MIMXRT1170-EVK | Cortex-M7 @ 1 GHz | MCU deployment |
| i.MX RT1050 | IMXRT1050-EVKB | Cortex-M7 @ 600 MHz | MCU deployment |
| i.MX RT1060 | IMXRT1060-EVK | Cortex-M7 @ 600 MHz | MCU deployment |
| MCX N947 | FRDM-MCXN947 | Cortex-M33 + NPU | Low-power keywords/detection |

---

## Arm Vela Compilation (i.MX 93)

Models marked `vela_required: true` must be compiled before deployment on i.MX 93.

### CLI
```bash
vela model.tflite \
    --accelerator-config ethos-u65-256 \
    --system-config Ethos_U65_High_End \
    --memory-mode Shared_Sram \
    --output-dir /tmp/vela_out
```

### Python API (via pipeline)
```python
from mlops.vela_compiler import VelaCompiler

compiler = VelaCompiler(
    accelerator_config="ethos-u65-256",
    output_dir="/tmp/vela_compiled"
)
compiled_path = compiler.compile("model_int8.tflite")
```

### Vela output files
- `model_vela.tflite` — compiled model for Ethos-U65
- `model_vela_report.csv` — cycle estimates per operator
- `model_vela.ini` — system config used

---

## Recipe Execution (Docker)

Each model in the zoo ships with a `recipe.sh` that handles dataset download, model conversion, and evaluation inside Docker.

```bash
# Run a recipe directly
cd /path/to/eiq-model-zoo/tasks/vision/classification/mobilenetv2
./recipe.sh

# Via Python pipeline
from mlops.recipe_runner import RecipeRunner

runner = RecipeRunner(zoo_root="/path/to/eiq-model-zoo")
result = runner.run_recipe("mobilenetv2")
```

The `recipe_runner.py` wraps `docker run` with the zoo's standard image and mounts.

---

## Board Deployment

```bash
# Copy compiled model to board via SCP
python -m pipelines.deploy_pipeline \
    --model facedet \
    --platform imx93 \
    --board-ip 192.168.1.100 \
    --board-user root

# Deploy pipeline steps:
# 1. Load model from registry
# 2. Compile with Vela if platform == imx93
# 3. Pack artifact bundle (model + inference_example.py + metadata.json)
# 4. SCP bundle to board /home/root/eiq_deploy/
# 5. Optionally run smoke test via SSH
```

---

## Pipeline Architecture

```
config/
  pipeline_config.yaml     ← platform targets, eval gates, Vela config, monitoring
  model_registry.yaml      ← all 29 models with per-platform delegates

mlops/
  model_manager.py         ← registry CRUD, vela_required check, recipe execution
  data_pipeline.py         ← preprocessing per task (vision/audio/EEG)
  evaluator.py             ← TFLite inference engine, top1/mIoU/PSNR/WER metrics
  recipe_runner.py         ← Docker recipe.sh wrapper
  vela_compiler.py         ← Arm Vela compiler wrapper for Ethos-U65
  artifact_manager.py      ← pack/unpack deployment bundles
  monitor.py               ← latency tracking, drift alerts

pipelines/
  eval_pipeline.py         ← evaluate all or selected models
  deploy_pipeline.py       ← Vela compile → bundle → SCP to board

tests/
  test_pipeline.py         ← 30+ hardware-free pytest tests
```

---

## Configuration

Edit `config/pipeline_config.yaml` to set:
- `default_platform` — target platform for Vela compilation and deployment
- `eval_gates` — per-task minimum metric thresholds (pipeline fails if not met)
- `vela.accelerator_config` — Ethos-U65 config (`ethos-u65-256` for i.MX 93)
- `artifact_output_dir` — where deployment bundles are written
- `board.ip` / `board.user` — target board SSH credentials

---

## Running Tests

```bash
# All tests (no hardware, no Docker required)
pytest tests/ -v

# With coverage
pytest tests/ --cov=mlops --cov-report=term-missing

# Specific test class
pytest tests/test_pipeline.py::TestVelaCompiler -v
```

---

## Downloading Model Weights

Model weights are not included in this repo. Clone the NXP eIQ Model Zoo:

```bash
git clone https://github.com/NXP/eiq-model-zoo.git

# Set zoo root for the pipeline
export NXP_ZOO_ROOT=/path/to/eiq-model-zoo

# model_path entries in model_registry.yaml resolve relative to NXP_ZOO_ROOT
# e.g. "tasks/vision/classification/mobilenetv2" →
#       /path/to/eiq-model-zoo/tasks/vision/classification/mobilenetv2/
```
