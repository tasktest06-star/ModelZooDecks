# NXP eIQ — Practical Edge AI Examples

Practical inference examples for NXP eIQ model zoo models.
Targets: i.MX 8M Plus (2.3 TOPS NPU), i.MX 93 (Ethos-U65 1 TOPS), RT1170 (MCU), MCX N947

## Structure
- `inference/`    — standalone TFLite inference scripts
- `benchmarking/` — latency/FPS/power measurement
- `applications/` — complete end-to-end apps (face recog, DMS, KWS)
- `vela_tools/`   — Vela compiler helpers for i.MX 93 / Ethos-U65
- `notebooks/`    — Jupyter step-by-step walkthroughs

## Quick Start
```bash
pip install -r requirements.txt
python inference/tflite_classify.py --image sample.jpg --model mobilenetv2
python inference/tflite_detect.py --image sample.jpg --model yolov8m
python applications/face_recognition.py --simulate
python vela_tools/compile_for_ethos.py --model models/mobilenetv2.tflite --target imx93
python benchmarking/latency_benchmark.py --platform imx8mplus
```

## Reference Repositories
- eiq-apps-collection:    https://github.com/NXP/eiq-apps-collection
- eiq-tflite-benchmark:   https://github.com/nxp-imx/eiq-tflite-benchmark
- imxrt-ai-examples:      https://github.com/nxp-imx-support/imxrt-ai-examples
- i.MX ML User Guide:     https://www.nxp.com/design/software/development-software/eiq-ml-development-environment
- ARM ML-examples:        https://github.com/ARM-software/ML-examples
- Ethos-U Vela:           https://github.com/ARM-software/ethos-u-vela
