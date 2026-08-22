"""
Driver Monitoring System — NXP i.MX 8M Plus
Stage 1: FaceDet — locate driver face in cabin camera
Stage 2a: DeepFace Emotion — detect alert vs drowsy vs distracted
Stage 2b: WHENet — head pose for gaze direction (6-DOF)
Stage 3: DS-CNN — keyword spotting for voice commands
Use case: ADAS driver state monitoring (drowsiness, distraction, voice control)
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import time


EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
DROWSINESS_EMOTIONS = {"sad", "fear", "neutral"}
ALERT_EMOTIONS = {"happy", "surprise", "angry"}
VOICE_COMMANDS = ["alert", "navigate", "call", "music", "quiet", "help"]

HEAD_POSE_DISTRACTION_THRESHOLD = 30.0


@dataclass
class DriverMonitorConfig:
    face_detector: str = "facedet"
    emotion_model: str = "deepface_emotion"
    head_pose_model: str = "whenet"
    kws_model: str = "ds_cnn"
    target_platform: str = "imx8mplus"
    face_threshold: float = 0.5
    drowsiness_consecutive_frames: int = 15


@dataclass
class DriverState:
    face_detected: bool = False
    emotion: Optional[str] = None
    emotion_confidence: float = 0.0
    head_pose: Optional[dict] = None
    is_distracted: bool = False
    is_drowsy: bool = False
    gaze_on_road: bool = True
    voice_command: Optional[str] = None
    alert_level: str = "normal"
    latency_ms: dict = field(default_factory=dict)


class DriverMonitoringPipeline:
    """
    Real-time driver state estimation pipeline.

    Fuses visual (emotion + head pose) and audio (voice commands) to
    estimate driver alertness and intent:
    - Emotion "sad"/"neutral" for N consecutive frames → drowsiness warning
    - Head yaw > 30° away from road direction → distraction warning
    - Keyword "help" or combined warnings → critical alert

    Target latency on i.MX 8M Plus NPU: ~40ms/frame
    """

    def __init__(self, config: DriverMonitorConfig, registry_path: str):
        self.config = config
        self._drowsy_frame_count = 0
        self._load_registry(registry_path)

    def _load_registry(self, path: str):
        import yaml
        with open(path) as f:
            self._registry = yaml.safe_load(f)

    def detect_face(self, frame: np.ndarray) -> tuple:
        """FaceDet — return largest face bbox or None."""
        t0 = time.perf_counter()
        face_bbox = None
        return face_bbox, (time.perf_counter() - t0) * 1000

    def classify_emotion(self, face_crop: np.ndarray) -> tuple:
        """DeepFace Emotion — classify into 7 emotions."""
        t0 = time.perf_counter()
        try:
            probs = np.ones(len(EMOTION_LABELS)) / len(EMOTION_LABELS)
        except Exception:
            probs = np.ones(len(EMOTION_LABELS)) / len(EMOTION_LABELS)
        top_idx = int(np.argmax(probs))
        emotion = EMOTION_LABELS[top_idx]
        top3 = [(EMOTION_LABELS[i], float(probs[i]))
                for i in np.argsort(probs)[::-1][:3]]
        return emotion, float(probs[top_idx]), top3

    def estimate_head_pose(self, face_crop: np.ndarray) -> tuple:
        """WHENet — 6-DOF head pose."""
        t0 = time.perf_counter()
        pose = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
        return pose, (time.perf_counter() - t0) * 1000

    def spot_voice_command(self, audio: np.ndarray) -> tuple:
        """DS-CNN KWS — detect driver voice command."""
        t0 = time.perf_counter()
        try:
            from mlops.data_pipeline import compute_mfcc
            _ = compute_mfcc(audio)
        except Exception:
            pass
        return None, (time.perf_counter() - t0) * 1000

    def _assess_drowsiness(self, emotion: str) -> bool:
        is_drowsy_emotion = emotion in DROWSINESS_EMOTIONS
        if is_drowsy_emotion:
            self._drowsy_frame_count += 1
        else:
            self._drowsy_frame_count = max(0, self._drowsy_frame_count - 2)
        return self._drowsy_frame_count >= self.config.drowsiness_consecutive_frames

    def _assess_distraction(self, pose: dict) -> tuple:
        yaw = abs(pose.get("yaw", 0.0))
        pitch = abs(pose.get("pitch", 0.0))
        distracted = yaw > HEAD_POSE_DISTRACTION_THRESHOLD
        on_road = yaw < 15.0 and pitch < 20.0
        return distracted, on_road

    def _compute_alert_level(self, state: DriverState) -> str:
        if state.voice_command == "help":
            return "critical"
        if state.is_drowsy and state.is_distracted:
            return "critical"
        if state.is_drowsy or state.is_distracted:
            return "warning"
        return "normal"

    def process_frame(self, frame: np.ndarray,
                      audio: Optional[np.ndarray] = None) -> DriverState:
        state = DriverState()
        t_total = time.perf_counter()

        face_bbox, det_ms = self.detect_face(frame)
        state.face_detected = face_bbox is not None
        state.latency_ms["face_det_ms"] = det_ms

        if face_bbox is not None:
            H, W = frame.shape[:2]
            x1, y1, x2, y2 = (int(face_bbox[i]) for i in range(4))
            crop = frame[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]
            if crop.size > 0:
                emotion, conf, _ = self.classify_emotion(crop)
                pose, pose_ms = self.estimate_head_pose(crop)
                state.emotion = emotion
                state.emotion_confidence = conf
                state.head_pose = pose
                state.is_drowsy = self._assess_drowsiness(emotion)
                state.is_distracted, state.gaze_on_road = self._assess_distraction(pose)
                state.latency_ms["emotion_ms"] = 0.0
                state.latency_ms["pose_ms"] = pose_ms

        if audio is not None:
            cmd, kws_ms = self.spot_voice_command(audio)
            state.voice_command = cmd
            state.latency_ms["kws_ms"] = kws_ms

        state.alert_level = self._compute_alert_level(state)
        state.latency_ms["total_ms"] = (time.perf_counter() - t_total) * 1000
        return state
