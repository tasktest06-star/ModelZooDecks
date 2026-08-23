# TI EdgeAI — Practical Examples

Practical inference examples for TI EdgeAI model zoo models.
Targets: AM68A, AM67A, AM62A, TDA4VM

## Structure
- `inference/` — standalone inference scripts
- `benchmarking/` — latency/FPS measurement tools
- `camera_demo/` — GStreamer camera pipeline
- `notebooks/` — Jupyter step-by-step walkthroughs
- `evaluation/` — COCO/ImageNet evaluation scripts
- `applications/` — complete end-to-end apps

## Quick Start
```bash
pip install -r requirements.txt
# Run classification on a single image
python inference/classify_image.py --image sample.jpg --model mobilenet_v2_lite --soc AM68A

# Benchmark all models
python benchmarking/benchmark_models.py --soc AM68A --num_runs 100

# Run person detection demo
python applications/person_counter.py --source camera  # or --source video.mp4
```

## Reference Repositories
- edgeai-gst-apps: https://github.com/TexasInstruments/edgeai-gst-apps
- edgeai-benchmark: https://github.com/TexasInstruments/edgeai-benchmark
- edgeai-tidl-tools: https://github.com/TexasInstruments/edgeai-tidl-tools
- Edge AI Studio: https://dev.ti.com/edgeaistudio/
