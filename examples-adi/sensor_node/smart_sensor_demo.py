"""
Smart Sensor Node Demo — ADI Hierarchical Activation Pipeline.

Three-state power FSM:
  SLEEP  (MAX32690 only, 0.3 mA)
    -> [MicroNet VWW detects person]
    -> LISTENING (MAX32690 + MAX78002 wake, 5 mA)
      -> [Conv1D KWS confirms keyword]
      -> ACTIVE (Full FPN detection, 15 mA)
        -> [Report + return to SLEEP]

Achieves >10x power saving vs always-on detection.

References:
  ai8x-training:  https://github.com/MaximIntegratedAI/ai8x-training
  MAX78002 CNN:   442 TOPS/W, 5MB on-chip SRAM, 64 parallel CNN processors
"""

import argparse
import time
import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from collections import deque
from typing import Optional


class State(Enum):
    SLEEPING  = "SLEEPING"
    LISTENING = "LISTENING"
    ACTIVE    = "ACTIVE"


@dataclass
class PowerProfile:
    state: State
    current_ma: float
    duration_ms: float

    @property
    def energy_uj(self) -> float:
        return self.current_ma * 3.3 * (self.duration_ms / 1000) * 1000


@dataclass
class SensorNodeStats:
    total_frames: int = 0
    sleep_frames: int = 0
    listen_frames: int = 0
    active_frames: int = 0
    detections: int = 0
    false_wakes: int = 0
    power_log: list = field(default_factory=list)

    @property
    def total_energy_uj(self) -> float:
        return sum(p.energy_uj for p in self.power_log)

    @property
    def avg_current_ma(self) -> float:
        if not self.power_log:
            return 0.0
        total_t = sum(p.duration_ms for p in self.power_log)
        total_e = self.total_energy_uj
        return (total_e / 3.3 / 1000) / (total_t / 1000) * 1000 if total_t > 0 else 0.0

    def always_on_energy_uj(self) -> float:
        total_t = sum(p.duration_ms for p in self.power_log)
        return 15.0 * 3.3 * (total_t / 1000) * 1000

    @property
    def power_saving_ratio(self) -> float:
        aon = self.always_on_energy_uj()
        return aon / max(self.total_energy_uj, 1e-9)


VWW_MODEL = {
    "name": "micronet_vww", "device": "MAX32690",
    "accuracy": 76.8, "latency_ms": 5.0, "power_ma": 1.5, "energy_uj": 5,
    "threshold": 0.65,
}
KWS_MODEL = {
    "name": "conv1d_audionet", "device": "MAX78002",
    "accuracy": 86.3, "latency_ms": 10.0, "power_ma": 5.0, "energy_uj": 90,
    "threshold": 0.70,
    "keywords": ["stop", "go", "yes", "no", "alarm", "call", "help"],
}
DET_MODEL = {
    "name": "feature_pyramid_net", "device": "MAX78002",
    "mAP": 50.5, "latency_ms": 20.0, "power_ma": 15.0, "energy_uj": 340,
    "classes": ["person", "vehicle", "animal", "package"],
}


class SmartSensorNode:
    """MAX78002 + MAX32690 hierarchical sensor node."""

    SLEEP_CURRENT_MA  = 0.3
    LISTEN_CURRENT_MA = 5.0
    ACTIVE_CURRENT_MA = 15.0
    TIMEOUT_FRAMES    = 30

    def __init__(self, simulate: bool = True):
        self.state       = State.SLEEPING
        self.timeout_ctr = 0
        self.stats       = SensorNodeStats()
        self.simulate    = simulate
        self._state_hist = deque(maxlen=200)

    def _run_vww(self, frame: np.ndarray) -> tuple:
        person_prob = float(np.random.beta(1.5, 5)) if self.simulate else 0.0
        return person_prob, VWW_MODEL["latency_ms"], VWW_MODEL["energy_uj"]

    def _run_kws(self, audio: Optional[np.ndarray]) -> tuple:
        keyword_prob = float(np.random.beta(2, 5)) if self.simulate else 0.0
        keyword = np.random.choice(KWS_MODEL["keywords"]) if keyword_prob > 0.5 else None
        return keyword_prob, keyword, KWS_MODEL["latency_ms"], KWS_MODEL["energy_uj"]

    def _run_detection(self, frame: np.ndarray) -> tuple:
        n_det = int(np.random.poisson(1.5)) if self.simulate else 0
        detections = [
            {
                "class": str(np.random.choice(DET_MODEL["classes"])),
                "confidence": float(np.random.uniform(0.5, 0.95)),
                "bbox": [int(np.random.randint(0, 100)), int(np.random.randint(0, 100)),
                         int(np.random.randint(100, 200)), int(np.random.randint(100, 200))],
            }
            for _ in range(n_det)
        ]
        return detections, DET_MODEL["latency_ms"], DET_MODEL["energy_uj"]

    def _log_power(self, state: State, duration_ms: float):
        current = {
            State.SLEEPING:  self.SLEEP_CURRENT_MA,
            State.LISTENING: self.LISTEN_CURRENT_MA,
            State.ACTIVE:    self.ACTIVE_CURRENT_MA,
        }[state]
        self.stats.power_log.append(PowerProfile(state, current, duration_ms))

    def process(self, frame: np.ndarray, audio: Optional[np.ndarray] = None) -> dict:
        self.stats.total_frames += 1
        self._state_hist.append(self.state.value)
        result = {
            "state": self.state.value,
            "frame": self.stats.total_frames,
            "detections": [],
            "triggered": False,
        }

        if self.state == State.SLEEPING:
            self.stats.sleep_frames += 1
            vww_conf, lat, _ = self._run_vww(frame)
            self._log_power(State.SLEEPING, lat + 1000 / 30)

            if vww_conf >= VWW_MODEL["threshold"]:
                self.state = State.LISTENING
                self.timeout_ctr = 0
                result["triggered"] = True
                result["vww_confidence"] = vww_conf

        elif self.state == State.LISTENING:
            self.stats.listen_frames += 1
            self.timeout_ctr += 1
            kws_conf, keyword, lat, _ = self._run_kws(audio)
            self._log_power(State.LISTENING, lat)

            if kws_conf >= KWS_MODEL["threshold"] and keyword:
                self.state = State.ACTIVE
                self.timeout_ctr = 0
                result["keyword"] = keyword
                result["kws_confidence"] = kws_conf
            elif self.timeout_ctr >= self.TIMEOUT_FRAMES:
                self.state = State.SLEEPING
                self.stats.false_wakes += 1

        elif self.state == State.ACTIVE:
            self.stats.active_frames += 1
            dets, lat, _ = self._run_detection(frame)
            self._log_power(State.ACTIVE, lat)

            result["detections"] = dets
            self.stats.detections += len(dets)
            self.state = State.SLEEPING

        return result

    def print_report(self):
        s = self.stats
        tf = max(s.total_frames, 1)
        print(f"\n{'='*65}")
        print(f"Smart Sensor Node Report — {s.total_frames} frames processed")
        print(f"{'='*65}")
        print(f"  State distribution:")
        print(f"    SLEEPING  : {s.sleep_frames:5d} frames ({s.sleep_frames/tf*100:.1f}%)")
        print(f"    LISTENING : {s.listen_frames:5d} frames ({s.listen_frames/tf*100:.1f}%)")
        print(f"    ACTIVE    : {s.active_frames:5d} frames ({s.active_frames/tf*100:.1f}%)")
        print(f"  Total detections  : {s.detections}")
        print(f"  False wakes       : {s.false_wakes}")
        print(f"\n  Power Analysis:")
        print(f"    Total energy    : {s.total_energy_uj:.0f} uJ = {s.total_energy_uj/3600:.4f} mAh")
        print(f"    Always-on energy: {s.always_on_energy_uj():.0f} uJ")
        print(f"    Power saving    : {s.power_saving_ratio:.1f}x")
        print(f"    Avg current     : {s.avg_current_ma:.2f} mA")
        sessions = 230_000 / max(s.total_energy_uj / 3600, 1e-9)
        print(f"\n  CR2032 (230mAh): ~{sessions:.0f} sessions like this")
        print(f"{'='*65}")


def main():
    parser = argparse.ArgumentParser(description="ADI Smart Sensor Node Demo")
    parser.add_argument("--simulate", action="store_true", default=True)
    parser.add_argument("--frames",   type=int, default=200)
    parser.add_argument("--verbose",  action="store_true")
    args = parser.parse_args()

    print("ADI Smart Sensor Node — Hierarchical Activation Pipeline")
    print("Devices: MAX32690 (VWW) + MAX78002 (KWS + FPN)")
    print(f"Running {args.frames} frames...\n")

    node = SmartSensorNode(simulate=args.simulate)

    for i in range(args.frames):
        frame = np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8)
        audio = np.random.randn(16000).astype(np.float32) * 0.05
        result = node.process(frame, audio)

        if args.verbose or result.get("triggered") or result.get("keyword") or result["detections"]:
            sym = {"SLEEPING": "Zzz", "LISTENING": "EAR", "ACTIVE": "ACT"}.get(result["state"], "?")
            print(f"Frame {i+1:4d} [{sym}] {result['state']:<10}", end="")
            if result.get("keyword"):
                print(f"  keyword='{result['keyword']}'", end="")
            if result["detections"]:
                cls_list = [d["class"] for d in result["detections"]]
                print(f"  detected={cls_list}", end="")
            print()

    node.print_report()


if __name__ == "__main__":
    main()
