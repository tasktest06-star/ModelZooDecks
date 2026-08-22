# NXP eIQ Model Combination Applications

## Overview
Three multi-model applications using NXP eIQ Model Zoo TFLite INT8 models, targeting i.MX 8M Plus.

## Application 1: Low-Light Face Recognition
**Target:** i.MX 8M Plus NPU
**Pipeline:** SCI Low-Light → FaceDet → FaceNet512 + WHENet

| Stage | Model | Input | Output | Latency (est.) |
|-------|-------|-------|--------|----------------|
| Enhance | sci_low_light | Full frame (dynamic) | Enhanced frame | ~80ms |
| Detect | facedet | Enhanced frame | Face bboxes | ~15ms |
| Recognize | facenet512 | 160×160 face crop | 512-dim embedding | ~12ms/face |
| Pose | whenet | 160×160 face crop | yaw/pitch/roll | ~8ms/face |

**Data flow:**
```
dark_frame → [SCI] → enhanced → [FaceDet] → face_bboxes
  → crop each face → [FaceNet512] → embedding → cosine match → identity
                   → [WHENet]    → head pose (yaw, pitch, roll)
```
**Application:** ATM security, building access control, night surveillance

## Application 2: Driver Monitoring System
**Target:** i.MX 8M Plus NPU
**Pipeline:** FaceDet → DeepFace Emotion + WHENet (head pose) + DS-CNN (voice)

| Stage | Model | Function |
|-------|-------|----------|
| FaceDet | facedet | Detect driver face region |
| Emotion | deepface_emotion | 7-class emotion → drowsiness proxy |
| Head pose | whenet | Head pose → road-gaze detection |
| KWS | ds_cnn | Voice commands ("alert", "navigate") |

**Alert logic:**
- Neutral/sad emotion + 15 consecutive frames → drowsiness WARNING
- Head yaw > 30° → distraction WARNING
- Both active simultaneously → CRITICAL alert
- Keyword "help" → CRITICAL

## Application 3: Smart Video Analytics
**Target:** i.MX 8M Plus NPU
**Pipeline:** (SCI enhance optional) → YOLOv8-M → MobileNetV2 (crops) + DeepLabV3 (seg) + MiDaS (depth)

| Branch | Model | Output |
|--------|-------|--------|
| Detection | yolov8_m | Object bboxes + classes |
| Classification | mobilenetv2 | Fine-grain class per crop |
| Segmentation | deeplabv3 | Per-pixel semantic labels |
| Depth | midas_v21_small | Relative depth map |

**Estimated total latency (i.MX 8M Plus):** ~120ms/frame (~8 fps)

## Running the Pipelines
```bash
cd code-nxp
python combinations/low_light_face_recognition.py
python combinations/driver_monitoring.py
python combinations/smart_video_analytics.py
```

## Testing
```bash
cd code-nxp
python -m pytest tests/test_combinations.py -v
```
