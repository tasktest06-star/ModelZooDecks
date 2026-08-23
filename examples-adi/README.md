# ADI AI8X — Practical Edge AI Examples

Practical inference examples for ADI AI8X model zoo models.
Targets: MAX78002 (CNN accel, 442 TOPS/W), MAX32690 (ultra-LP MCU), ADSP-SC835 (audio DSP)

## Structure
- `inference/`     — standalone inference scripts (simulate MAX78002 CNN accel)
- `benchmarking/`  — energy/latency/accuracy measurement
- `applications/`  — complete end-to-end apps
- `sensor_node/`   — hierarchical wake-word + detection pipeline
- `notebooks/`     — Jupyter walkthroughs

## Quick Start
```bash
pip install -r requirements.txt
python inference/classify_max78002.py --image sample.jpg --model mobilenetv2_050
python applications/keyword_spotter.py --source simulate
python sensor_node/smart_sensor_demo.py --simulate
python benchmarking/energy_benchmark.py --device MAX78002
```

## Reference Repositories
- ai8x-training:   https://github.com/MaximIntegratedAI/ai8x-training
- ai8x-synthesis:  https://github.com/MaximIntegratedAI/ai8x-synthesis
- MaximSDK (MSDK): https://github.com/analogdevicesinc/msdk
- ADI AI demos:    https://github.com/MaximIntegratedAI/MAX78000_SDK/tree/master/Examples
- MSDK Examples:   https://github.com/analogdevicesinc/msdk/tree/main/Examples/MAX78002

## Model Zoo Hardware Targets

| Device       | AI Engine          | Power     | Models in Zoo |
|--------------|--------------------|-----------|---------------|
| MAX78002     | CNN Accel 442TOPS/W| 2–15 mA   | 8 models      |
| MAX32690     | SW (M4F 120MHz)    | 0.3–2 mA  | 5 models      |
| ADSP-SC835   | SHARC DSP SIMD     | 30–100 mW | 4 models      |
