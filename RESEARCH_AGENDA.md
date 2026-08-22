# Edge AI Research Agenda
## TI EdgeAI Model Zoo · ADI Model Zoo · Broader Edge AI Landscape

**Date:** 2026-08-22  
**Scope:** Further research, extension topics, and publishable gaps across both model zoos and the wider Edge AI field.

---

## Quick-Reference Priority Matrix

| Topic | TI SoCs | ADI MCUs | Effort | Impact |
|-------|---------|----------|--------|--------|
| MobileNetV4 port & benchmark | AM68A ✓ | MAX78002 ✓ | Low | **High** |
| INT4 / W4A8 quantization | TDA4VM ✓ | — | Medium | **High** |
| Hardware-aware NAS (TIDL / AI8X) | All TI ✓ | MAX78002 ✓ | High | **High** |
| Multi-modal fusion (camera+LiDAR / vibration+audio) | TDA4VM ✓ | MAX78002 ✓ | High | **High** |
| MCUNet-V2 on MAX32690 | — | MAX32690 ✓ | Medium | **High** |
| Biomedical models (ECG/PPG/falls) | — | MAX32690 ✓ | Medium | **High** |
| CWRU bearing fault model | — | MAX78002 ✓ | Low | **High** |
| Energy/power profiling (µJ/inference) | AM62A–AM69A ✓ | MAX78002 ✓ | Medium | High |
| Cross-SoC latency parity table | All TI ✓ | — | Medium | High |
| Federated anomaly detection fleet | — | MAX32690 ✓ | High | Medium |
| MC-Dropout uncertainty / safety | AM68A ✓ | MAX78002 ✓ | Medium | High |
| AM69A LLM benchmark (LLaMA 3.2-1B) | AM69A ✓ | — | Low | Medium |
| ROS2 node wrappers for inference | AM69A ✓ | — | Medium | Medium |
| SNN vs. RNNoise denoiser | — | MAX32690 ✓ | High | Research |
| OTA model update via BLE DFU | — | MAX32690 ✓ | Medium | Medium |

---

## Part 1 — TI EdgeAI Model Zoo

### 1.1 Model Coverage Gaps
**What's missing:** The zoo is entirely vision-only. Critical gaps:
- **Instance segmentation** — Mask R-CNN / SOLOv2 (ADAS lane/object isolation, TDA4VM)
- **Video understanding** — SlowFast, VideoMAE for temporal event detection on AM69A
- **Lane detection** — specialized ADAS task; no dedicated model despite BEV 3D existing
- **Optical flow** — RAFT/FlowNet; C7x DSP is well-suited
- **Text detection / OCR** — industrial inspection on AM68A
- **Transformer-based detection** — RT-DETR, DINO-DETR; no "Lite" TIDL-adapted variants yet
- **MobileNetV4** (Google, 2024) — Universal Inverted Bottleneck unifying MHA + depthwise, top mobile benchmark results; absent from zoo

> **Next step:** Port MobileNetV4-S (~3.8MB INT8) using edgeai-torchvision and benchmark on AM68A.

---

### 1.2 Quantization & Efficiency
The zoo is entirely INT8. Research directions:

- **INT4 / W4A8** — GPTQ (Frantar et al., 2022), AWQ (Lin et al., 2023) originally for LLMs now extending to vision CNNs (EfficientQAT 2024). W4A8 halves model size with <1% ImageNet drop. TDA4VM MMA natively supports INT8; INT4 needs software emulation today.
- **Per-channel calibration** — current TIDL uses per-tensor; per-channel recovers 1–2% mAP
- **Structured pruning + QAT co-optimization** — channel pruning on YOLOX-Lite families before INT8 quantization, targeting 1.5× speedup with <1% accuracy drop
- **Knowledge distillation** — FP32 teacher → INT8 TIDL-compatible student; infrastructure exists in edgeai-torchvision but no pipeline automates this

> **Next step:** Apply W4A8 via onnxruntime quantization tools on the zoo's ONNX models; measure accuracy vs. size reduction.

---

### 1.3 Multi-Modal & New Tasks
- **Camera + LiDAR fusion** — BEVFusion-Lite on TDA4VM; 4 BEV 3D models exist but none fuse sensor streams
- **4D radar processing** — AM69A targets next-gen ADAS radar-camera fusion; zero radar models in zoo
- **Event camera models** — neuromorphic sensors + C7x; emerging TI research area
- **Thermal imaging** — industrial inspection on AM68A with FLIR-style inputs

> **Next step:** Design a radar+camera early-fusion neck module for the existing YOLOX-S-Lite backbone on TDA4VM.

---

### 1.4 Deployment Optimization
- **Operator fusion analysis** — no public matrix showing which ONNX op sequences TIDL fuses vs. CPU fallback
- **Dynamic input resolution** — all zoo models use fixed shapes; adaptive resolution based on scene complexity is unexplored
- **Pipeline parallelism** on AM69A — 8 C7x cores; no examples of cross-core model splitting
- **Memory tiling** for large segmentation models on AM62A (1 TOPS, limited SRAM)
- **Zero-copy DMA** between GStreamer capture and TIDL inference

---

### 1.5 Benchmarking Gaps
- **Cross-SoC latency table** — reports cover AM68A only; AM62A / AM67A / TDA4VM comparison missing
- **Energy measurement** — AM62A marketed for low-power but no µJ/inference numbers exist
- **BDD100K / nuScenes benchmarks** — better represent real ADAS than COCO alone
- **MLPerf Edge re-submission** — TI has not submitted official MLPerf results since 2022

---

### 1.6 Neural Architecture Search
No NAS tool constrained to **TIDL-compatible operators** exists. The TIDL op search space is well-defined (conv2d, depthwise, BN, ReLU, add — no dynamic shapes, no unrolled LSTM). Applying Once-for-All or EfficientDet NAS within this space would auto-generate SoC-specific "Lite" architectures instead of hand-designing them.

> **Publishable at:** DAC, DATE — *"TIDL-Aware Neural Architecture Search"*

---

### 1.7 Federated Learning & On-Device Adaptation
AM69A (32 TOPS) has sufficient compute for small-scale on-device fine-tuning:
- Federated fine-tuning of classification head on device-local data (factory-specific defect classes)
- Continual learning with replay buffers in DDR
- LoRA-style adapter fine-tuning on frozen Swin-Tiny backbone

---

### 1.8 Safety & Reliability
- **ISO 26262 ASIL-B/D qualification** — no documented pathway for TIDL inference models
- **Uncertainty quantification** — MC Dropout / Deep Ensembles for detection on TDA4VM (critical for fail-safe ADAS)
- **OOD detection** — flagging out-of-distribution inputs (weather/lighting changes)
- **Adversarial robustness** — no FGSM/PGD attack evaluations in the zoo
- **Runtime drift monitoring** — confidence drift hooks in edgeai-benchmark (the `code/` MLOps monitor in this repo is the seed)

---

### 1.9 Ecosystem Integrations
- **ROS2 node wrappers** — no ROS2 package; inference as a ROS2 node on AM69A is an open gap
- **GStreamer plugins** — TIDL as a reusable GStreamer element (documented but not shipped)
- **OpenVX graph integration** — TDA4VM supports OpenVX but no examples connect Model Zoo models
- **MQTT / OPC-UA telemetry** — inference results to industrial IoT protocols

---

### 1.10 Publishable Research Angles (TI)
| Paper idea | Venue |
|-----------|-------|
| TIDL-Aware NAS: op-constrained architecture search for TI MMA | DAC / DATE |
| Cross-Platform INT8 Degradation Analysis (AM62A vs. AM69A calibration sensitivity) | CVPR Efficient DL Workshop |
| Multi-SoC Federated Calibration (EVMs as federated PTQ nodes) | MLSys |
| Energy-Latency Pareto Curves for ADAS SoCs (first published energy study for full TI lineup) | IEEE Embedded Systems Letters |
| Lite Architecture Design Rules (empirical study of ReLU↔Swish, SE removal, depthwise substitution) | ECCV / ICCV workshops |

---

## Part 2 — ADI Model Zoo

### 2.1 Model Coverage Gaps
The zoo covers 3 domains but misses entire ADI product verticals:
- **ECG/PPG arrhythmia** — AD8233 ECG front-end + MAX32690; PhysioNet PTB-XL, ~50KB TFLite INT8
- **IMU gesture recognition** — ADXL series accelerometers → MAX32690 (10-class gesture, ~30KB)
- **Gas / chemical sensing** — ADPD series + CNN for VOC classification
- **Radar gesture** — FMCW radar (ADXL radar dev kit) + MAX78002 CNN accelerator
- **Vibro-acoustic bearing fault** — CWRU dataset; same autoencoder architecture as existing motor model, just retrained on a public dataset → immediately achievable

> **Next step:** Add CWRU bearing fault model to `code-adi/config/model_registry.yaml` and train using the existing `mlops/` pipeline.

---

### 2.2 Tiny Model Architecture Research
- Systematic NAS within AI8X hardware constraints (weight bits=8, max channels per layer, specific activations) — no public implementation
- **MCUNet-V2** (Lin et al., MIT, 2022) — co-designs NAS + memory scheduler for 256KB SRAM; directly applicable to MAX32690
- Sub-100KB keyword spotters beyond DS-CNN: conformer-tiny, MHAtt-RNN both fit MAX32690's 3MB flash
- TinyBERT distillation for 21-class KWS on AI8X — not attempted

> **Publishable:** Benchmark all 15 zoo models against CMSIS-NN equivalents on the same MCU → direct comparison study for DATE/tinyML Summit.

---

### 2.3 AI8X Hardware Utilization
Underexplored MAX78002 features:
- **Streaming pipeline inference** — DMA-chained inference without CPU involvement for continuous sensor frames
- **Multi-model time-slicing** — VWW at low power → wake → FPN for detection; sharing CNN SRAM between models
- **Layer fusion profiling** — no published analysis of which ops ai8xize actually fuses vs. CPU fallback
- **Sub-6mW active mode** — no model demonstrates the datasheet claim with measured duty cycle

> **Next step:** Write a benchmark paper measuring energy-per-inference for all 7 vision models on MAX78002EVKIT using MSDK PowerMonitor.

---

### 2.4 Sensor Fusion
Nothing in the zoo combines modalities. MAX78002 has capacity for a fusion model:
- Vibration (3-axis ADXL355) + acoustic emission (MEMS mic) + temperature (ADT7420) → single fault classification head
- Must satisfy AI8X constraint: all inputs map into the same channel-folded tensor format
- Directly relevant to ADI's CN0549 condition monitoring reference design

---

### 2.5 On-Device Learning & Personalization
- **Few-shot KWS enrollment** — user-specific wake word via ProtoNets on MAX32690 (1MB SRAM feasible)
- **Auto-calibrating anomaly threshold** — Welford online algorithm on running reconstruction errors during first N minutes of normal operation; would improve pAUC from 0.52 → estimated 0.65+
- **FedAvg over anomaly detector fleet** — one client per machine type (fan/pump/slider/valve); Flower framework + TFLite Micro

---

### 2.6 Power & Latency Profiling
No measured latency or energy numbers in `model_index.json`. Research deliverable:
- Measure µJ/inference for every model on its target EV-Kit using MAX32625PICO power monitor
- Plot Pareto frontier: accuracy vs. µJ across all 15 models
- DS-CNN is a known MLPerf Tiny model — direct comparison to published MLPerf numbers is possible

---

### 2.7 Neuromorphic & Event-based
MAX78002's weight-stationary CNN dataflow resembles temporal spike accumulation. Research question: can SNN inference be mapped to AI8X by encoding spike trains as binary activation tensors? The channel-fold mechanism is structurally similar to rate-coded spike binning.

> **Next step:** Train an SNN denoiser with snnTorch on the RNNoise task and compare PESQ — quantifies the accuracy gap of SNNs today.

---

### 2.8 MLOps Maturity
- **Hardware-in-the-loop CI** — GitHub Actions self-hosted runner + MAX78002EVKIT via USB-JTAG; flash-and-infer as a CI step. The `code-adi/` pipeline in this repo is the foundation.
- **OTA model updates via BLE DFU** — MAX32690 has BLE 5.2 bootloader supporting DFU; no model deployment pipeline uses this yet.

---

### 2.9 Biomedical Applications
ADI makes the AFE chips used in wearables (AD8233 ECG, MAX86xxx PPG/SpO2, ADXL362 accelerometer). Zero biomedical models in the zoo. Nearest baselines:
- MobileNetV1-based ECG arrhythmia classifier: ~50KB, 5-class, PhysioNet PTB-XL
- TinyML falls detection on ADXL data: ~30KB CNN
- Both fit MAX32690 flash with INT8 TFLite Micro

> **Next step:** Add a "health" domain in the registry with 3 models: ECG (AD8233), PPG SpO2, falls (ADXL362).

---

### 2.10 Publishable Research Angles (ADI)
| Paper idea | Venue |
|-----------|-------|
| AI8X vs. CMSIS-NN vs. X-CUBE-AI vs. TFLite Micro: identical models, head-to-head | DATE / tinyML Summit |
| Channel-fold as a general spatial compression technique: accuracy/efficiency trade-off curve | IEEE SENSORS / ISCAS |
| QAT sensitivity analysis across all 15 zoo models | MLSys / NeurIPS Efficient DL Workshop |
| Energy-per-inference Pareto frontier for MAX78002 vision models | IEEE JSSC / ESSCIRC |

---

## Part 3 — Broader Edge AI Research Landscape

### 3.1 Efficient Inference Architectures
**Production-ready today:**
- **MobileNetV4** (Google, 2024) — Universal Inverted Bottleneck, best mobile benchmark at <5ms; runs on TDA4VM C7x and ADSP-SC835
- **FastViT** (Apple, 2023) — reparameterization to collapse training complexity at inference
- **EfficientFormer-L1** (Snap, 2022) — ~3.4MB, ViT-class accuracy on mobile hardware

**Research-only on MCUs:** MobileVLM (multimodal), EfficientViT (MIT, 2023) — require >100MB; MAX78002 is out of scope.

> **Next step:** Port MobileNetV4-S to MAX78002 via ai8x-training; benchmark vs. existing MobileNetV2-0.75.

---

### 3.2 Quantization Frontiers
- **W4A8** (4-bit weights, 8-bit activations) — AWQ (Lin et al., 2023), EfficientQAT (2024); halves model size with <1% ImageNet drop
- **INT2** — remains research; severe training instability below INT4
- **TDA4VM MMA** natively handles INT8; INT4 requires software emulation today
- **MAX78002** is hardwired to INT8 weights — INT4 not directly exploitable

> **Next step:** Apply W4A8 on TI ONNX models via onnxruntime quantization tools; measure accuracy vs. compression.

---

### 3.3 Edge-Native Transformers
| Model | Size INT8 | Top-1 | TDA4VM | MAX78002 |
|-------|-----------|-------|--------|----------|
| MobileViT-XXS | ~5.7MB | 70.4% | ✓ ~12ms | ✗ (no attention HW) |
| EfficientFormer-L1 | ~3.4MB | 79.2% | ✓ | ✗ |
| TinyViT-5M | ~5MB | 79.1% | ✓ | ✗ |

MAX78002 cannot run MHSA — attention layers fall back to Cortex-M4F at 100× slower speed. This gap is a research opportunity: approximating attention with efficient convolutions that AI8X can accelerate.

---

### 3.4 TinyML & MCU Deployment
- **MCUNet-V2** (Lin et al., MIT, 2022) — NAS + memory scheduler for 256KB SRAM; directly maps to MAX32690 via TFLite Micro
- **CMSIS-NN v6** (Arm, 2024) — adds INT4 and dynamic kernel support
- **Memory-optimal scheduling** — patch-based inference avoiding full activation buffers (Gu et al., 2023)

> **Next step:** Run MCUNet-V2 on MAX32690 using the existing `code-adi/` TFLite pipeline; compare with MicroNet VWW2.

---

### 3.5 Multimodal Edge AI
- **Audio-visual wake word** — KWS + VWW simultaneously on MAX32690; both models already in ADI zoo
- **TI mmWave + AM68A camera** — radar-camera fusion with shared feature extraction (TI white paper, 2023)
- **Earable computing** — IMU + mic + PPG fusion; TinyML Foundation benchmark 2024

> **Next step:** Design a dual-model inference demo on MAX32690 running DS-CNN (KWS) + MicroNet VWW2 (person detection) concurrently.

---

### 3.6 Federated Learning at the Edge
- **Flower framework** — FedAvg/FedProx with TFLite Micro client; production-ready
- **FedMCCS** (2023) — targets sub-1MB models on MCUs
- **SplitFed** — splits model at a layer boundary between MCU and server to avoid on-device backprop
- **Federated anomaly detection** — fleet of MAX32690 industrial sensors, each client = one machine type (fan/pump/slider/valve)

> **Next step:** Prototype FedAvg over ADI anomaly detection autoencoder using Flower simulation mode; 4 clients = 4 machine types.

---

### 3.7 NAS for Edge Hardware
- **Once-for-All** (Cai et al., MIT, 2020) — trains one supernet, queries subnets per device constraint
- **ProxylessNAS**, **DARTS** — mature but not constrained to specific accelerator ISAs
- **AI8X-aware NAS** — no public implementation; open research gap for MAX78002's channel/layer limits

> **Next step:** Use Once-for-All to generate a subnet fitting MAX78002's 1024-channel, 74-layer limits; synthesize via ai8xize.

---

### 3.8 Edge LLMs & Small Language Models
| Model | Params | Precision | Device | Throughput |
|-------|--------|-----------|--------|------------|
| LLaMA 3.2-1B | 1.2B | INT4 | AM69A (32 TOPS) | ~5 tok/s (est.) |
| Phi-3-mini | 3.8B | INT4 | AM69A | ~2 tok/s (est.) |
| Gemma-2-2B | 2B | INT8 | TDA4VM | ~2 tok/s (est.) |
| MiniCPM-1.2B | 1.2B | INT4 | AM68A | ~1 tok/s (est.) |

MAX78002/MAX32690: out of scope (require >32MB SRAM minimum).

**Speculative decoding** (Leviathan et al., 2023) with a ~60M draft model on-chip reduces AM68A latency 2-3×.

> **Next step:** Add AM69A LLM benchmark slide to the TI deck showing LLaMA 3.2-1B throughput vs. TOPS.

---

### 3.9 Neuromorphic Computing
| Platform | TOPS equivalent | Energy efficiency | Ecosystem |
|----------|----------------|-------------------|-----------|
| Intel Loihi 2 (2022) | ~0.01 | ~1000× vs. GPU | Research only |
| IBM NorthPole (2023) | ~22 TOPS/W | Non-von-Neumann | Research only |
| MAX78002 CNN | ~0.04 TOPS | ~6mW active | Production |

SNN training (SpikingJelly, snnTorch) lags ANN accuracy by ~3–5% for vision. Loihi 2 is 4–6× more energy-efficient than MAX78002 for sparse spike workloads — but has no commercial synthesis tools.

> **Next step:** Train SNN-based denoiser with snnTorch on RNNoise task; compare PESQ and energy vs. RNNoise INT8 TFLite baseline.

---

### 3.10 Energy Harvesting & Ultra-Low-Power AI
- **Ambiq Apollo4** — 10µW sleep, 6µW/MHz active, CMSIS-NN based; closest competitor to MAX32690
- **Intermittent computing** — Hibernus++ (2022); harvesting-aware checkpointing for bursty solar/RF power
- **Sub-mW CNN** — 0.4V inference on TSMC 28nm demonstrated (MIT JSSC, 2024)
- MAX32690 at 96MHz draws ~5mW active — near harvesting limits for solar/RF

> **Next step:** Profile MAX32690 current draw during DS-CNN inference using MSDK energy profiler; compare against Ambiq Apollo4 reference numbers.

---

### 3.11 Reliability & Safety
- **ISO 26262 ASIL-B** — TI FuSa SDK covers TDA4VM; no model certification pathway documented
- **MC-Dropout uncertainty** — 5 forward passes, averaged confidence; +15% latency overhead on MAX78002
- **Adversarial patch robustness** — INT8 models are more fragile to patch attacks than FP32 (Eykholt et al., adapted for INT8, 2023)
- **TinyML safety taxonomy** — Banbury et al., MLSys 2021; provides framework for safety classification

> **Next step:** Add MC-Dropout to ADI FPN detector (5 passes, averaged confidence); measure latency overhead on MAX78002 as groundwork for safety-critical deployment.

---

### 3.12 Emerging Applications
| Application | Device | Dataset/Approach | Status |
|------------|--------|-----------------|--------|
| Predictive maintenance (multi-sensor fusion) | MAX78002 | CWRU bearing + MIMII | Ready to prototype |
| Smart agriculture (soil/leaf sensing) | MAX32690 | Custom sensor datasets | Research |
| Wearable health AI (ECG/AFib) | MAX32690 | PhysioNet PTB-XL | Near-term |
| Autonomous nano-drone obstacle avoidance | GWT GAP9 (comparable to MAX78002) | PULP-Frontnet | Publishable comparison |
| Building energy management (occupancy + HVAC) | AM62A | Open Buildings dataset | Production ready |

---

## Cross-Cutting Research Themes

1. **Hardware-Aware NAS** is the single highest-leverage research area — it affects both zoos and all of Edge AI. No public TIDL-aware or AI8X-aware NAS exists.

2. **Energy benchmarking** is a gap in both zoos. Neither TI nor ADI publishes µJ/inference numbers. This is low-hanging publishable work.

3. **Multi-domain / multi-modal** models are absent from both zoos despite the hardware supporting them. A sensor-fusion model (ADI) or radar-camera model (TI) would be the most differentiating additions.

4. **Safety & reliability** is underinvested relative to the target markets (ADAS for TI, industrial for ADI). Uncertainty quantification and adversarial robustness are near-term additions.

5. **Biomedical** is ADI's clearest competitive advantage — they make the AFE chips. A wearable health domain in the ADI zoo with ECG/PPG/falls models would directly leverage the full ADI system stack.

---

## Recommended 90-Day Roadmap

### Quick wins (< 2 weeks each)
- [ ] Add CWRU bearing fault model to ADI registry (retrain existing autoencoder on public dataset)
- [ ] Add AM69A LLM benchmark slide to TI deck (LLaMA 3.2-1B throughput table)
- [ ] Profile MAX32690 DS-CNN energy using MSDK PowerMonitor and document µJ/inference
- [ ] Port MobileNetV4-S INT8 to TI edgeai-benchmark and compare with MobileNetV2 zoo entries

### Medium-term (2–6 weeks each)
- [ ] MCUNet-V2 classification demo on MAX32690 via `code-adi/` TFLite pipeline
- [ ] Dual-model inference (DS-CNN + MicroNet VWW2) on single MAX32690 — KWS + person detection
- [ ] Cross-SoC latency table (AM62A / AM67A / AM68A / AM69A / TDA4VM) using edgeai-benchmark
- [ ] W4A8 quantization experiment on TI ONNX models using onnxruntime

### Research-grade (6+ weeks)
- [ ] Hardware-aware NAS within TIDL operator constraints (publishable at DAC/DATE)
- [ ] AI8X-constrained NAS using Once-for-All supernet (publishable at tinyML Summit)
- [ ] Federated anomaly detection with Flower: 4 MAX32690 clients (fan/pump/slider/valve)
- [ ] Biomedical health domain: ECG arrhythmia + falls detection models for ADI zoo
- [ ] MC-Dropout safety extension for ADI FPN detector on MAX78002
