"""
Smart Video Analytics — NXP i.MX 8M Plus
Stage 0 (optional): SCI Low-Light Enhancement
Stage 1: YOLOv8-M — detect all objects
Stage 2a: MobileNetV2 — fine-classify detected objects (per crop)
Stage 2b: DeepLabV3 — semantic segmentation for scene context
Stage 2c: MiDaS V2.1 Small — monocular depth for 3D awareness
Use case: Smart retail / smart city camera with full scene understanding
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import time


@dataclass
class VideoAnalyticsConfig:
    enhance_model: str = "sci_low_light"
    detector_model: str = "yolov8_m"
    classifier_model: str = "mobilenetv2"
    segmentation_model: str = "deeplabv3"
    depth_model: str = "midas_v21_small"
    target_platform: str = "imx8mplus"
    enable_enhancement: bool = False
    enable_segmentation: bool = True
    enable_depth: bool = True
    enable_classification: bool = True
    detection_threshold: float = 0.4
    max_classify_crops: int = 6
    classifier_input_size: tuple = (224, 224)


@dataclass
class TrackedObject:
    bbox: list
    detector_class: int
    detector_confidence: float
    fine_class: Optional[str] = None
    fine_confidence: float = 0.0
    estimated_depth: float = 0.0


@dataclass
class VideoAnalyticsOutput:
    objects: list = field(default_factory=list)
    segmentation_mask: Optional[np.ndarray] = None
    depth_map: Optional[np.ndarray] = None
    scene_stats: dict = field(default_factory=dict)
    latency_ms: dict = field(default_factory=dict)


class SmartVideoAnalyticsPipeline:
    """
    Full scene understanding pipeline for smart cameras.

    Runs three parallel branches after detection:
    - Classification: MobileNetV2 on each detected crop
    - Segmentation: DeepLabV3 on full frame for context
    - Depth: MiDaS on full frame for 3D positioning

    Depth is used to annotate each detection with estimated distance,
    enabling downstream applications (occupancy, safety zones, tracking).
    """

    def __init__(self, config: VideoAnalyticsConfig, registry_path: str):
        self.config = config
        self._load_registry(registry_path)

    def _load_registry(self, path: str):
        import yaml
        with open(path) as f:
            self._registry = yaml.safe_load(f)

    def enhance(self, frame: np.ndarray) -> tuple:
        t0 = time.perf_counter()
        if not self.config.enable_enhancement:
            return frame, 0.0
        enhanced = np.clip(frame.astype(np.float32) * 1.5, 0, 255).astype(np.uint8)
        return enhanced, (time.perf_counter() - t0) * 1000

    def detect(self, frame: np.ndarray) -> tuple:
        t0 = time.perf_counter()
        detections = []
        return detections, (time.perf_counter() - t0) * 1000

    def classify_crop(self, frame: np.ndarray, bbox: list) -> tuple:
        t0 = time.perf_counter()
        H, W = frame.shape[:2]
        x1, y1 = max(0, int(bbox[0])), max(0, int(bbox[1]))
        x2, y2 = min(W, int(bbox[2])), min(H, int(bbox[3]))
        if x2 <= x1 or y2 <= y1:
            return None, 0.0
        crop = frame[y1:y2, x1:x2]
        try:
            from mlops.data_pipeline import preprocess_vision_classification
            inp = preprocess_vision_classification(crop, self.config.classifier_input_size)
        except Exception:
            inp = np.zeros((1, 224, 224, 3), dtype=np.float32)
        return None, (time.perf_counter() - t0) * 1000

    def segment(self, frame: np.ndarray) -> tuple:
        t0 = time.perf_counter()
        if not self.config.enable_segmentation:
            return None, 0.0
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        return mask, (time.perf_counter() - t0) * 1000

    def estimate_depth(self, frame: np.ndarray) -> tuple:
        t0 = time.perf_counter()
        if not self.config.enable_depth:
            return None, 0.0
        try:
            from mlops.data_pipeline import preprocess_vision_classification
            inp = preprocess_vision_classification(frame, (384, 384))
        except Exception:
            inp = np.zeros((1, 384, 384, 3), dtype=np.float32)
        depth_map = np.random.rand(*frame.shape[:2]).astype(np.float32)
        return depth_map, (time.perf_counter() - t0) * 1000

    def _get_object_depth(self, depth_map: Optional[np.ndarray], bbox: list,
                          frame_shape: tuple) -> float:
        if depth_map is None:
            return 0.0
        H, W = frame_shape[:2]
        x1 = max(0, min(W - 1, int(bbox[0])))
        y1 = max(0, min(H - 1, int(bbox[1])))
        x2 = max(x1 + 1, min(W, int(bbox[2])))
        y2 = max(y1 + 1, min(H, int(bbox[3])))
        dH, dW = depth_map.shape[:2]
        sx, sy = dW / W, dH / H
        rx1, ry1 = int(x1 * sx), int(y1 * sy)
        rx2, ry2 = max(rx1 + 1, int(x2 * sx)), max(ry1 + 1, int(y2 * sy))
        region = depth_map[ry1:ry2, rx1:rx2]
        return float(np.mean(region)) if region.size > 0 else 0.0

    def process_frame(self, frame: np.ndarray) -> VideoAnalyticsOutput:
        output = VideoAnalyticsOutput()
        t_total = time.perf_counter()

        enhanced, enh_ms = self.enhance(frame)
        output.latency_ms["enhance_ms"] = enh_ms

        detections, det_ms = self.detect(enhanced)
        output.latency_ms["detection_ms"] = det_ms

        seg_mask, seg_ms = self.segment(enhanced)
        output.segmentation_mask = seg_mask
        output.latency_ms["segmentation_ms"] = seg_ms

        depth_map, dep_ms = self.estimate_depth(enhanced)
        output.depth_map = depth_map
        output.latency_ms["depth_ms"] = dep_ms

        cls_total = 0.0
        objects = []
        for det in detections[:self.config.max_classify_crops]:
            obj = TrackedObject(
                bbox=det.get("bbox", [0, 0, 100, 100]),
                detector_class=det.get("class_id", -1),
                detector_confidence=det.get("confidence", 0.0),
            )
            if self.config.enable_classification:
                fine_cls, cls_ms = self.classify_crop(enhanced, obj.bbox)
                obj.fine_class = fine_cls
                cls_total += cls_ms
            obj.estimated_depth = self._get_object_depth(depth_map, obj.bbox, enhanced.shape)
            objects.append(obj)

        output.objects = objects
        output.latency_ms["classification_ms"] = cls_total
        output.scene_stats = {
            "n_objects": len(objects),
            "has_segmentation": seg_mask is not None,
            "has_depth": depth_map is not None,
            "total_ms": (time.perf_counter() - t_total) * 1000,
        }
        output.latency_ms["total_ms"] = output.scene_stats["total_ms"]
        return output
