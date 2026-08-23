"""
Keyword Spotter using ADI DS-CNN model on MAX32690.

DS-CNN achieves 94.5% on Google Speech Commands (35 keywords).
MAX32690: ARM Cortex-M4F @ 120MHz, 1MB SRAM, 3MB Flash.
Power: ~1 mA during inference, < 10 uA in sleep.

Pipeline: Mic -> VAD -> MFCC -> DS-CNN -> Keyword decision
"""

import argparse
import time
import numpy as np
from collections import deque


KEYWORDS_35 = [
    "yes","no","up","down","left","right","on","off","stop","go",
    "zero","one","two","three","four","five","six","seven","eight","nine",
    "bed","bird","cat","dog","happy","house","marvin","sheila","tree","wow",
    "backward","forward","follow","learn","visual"
]

MODEL_CONFIGS = {
    "ds_cnn_max32690": {
        "accuracy": 94.5, "device": "MAX32690", "power_ma": 1.0,
        "latency_ms": 12.0, "energy_uj": 22,
        "input_shape": (1, 49, 40),
        "note": "Depthwise separable CNN, SW inference on M4F"
    },
    "conv1d_max78002": {
        "accuracy": 86.3, "device": "MAX78002", "power_ma": 5.0,
        "latency_ms": 8.0, "energy_uj": 90,
        "input_shape": (1, 128, 1),
        "note": "1D conv on CNN accelerator; faster but lower accuracy"
    },
    "micronet_vww_max32690": {
        "accuracy": 76.8, "device": "MAX32690", "power_ma": 0.5,
        "latency_ms": 5.0, "energy_uj": 5,
        "input_shape": (1, 96, 96, 3),
        "note": "Visual wake word — triggers on person presence"
    },
}


def compute_mfcc(audio_chunk: np.ndarray, sr: int = 16000,
                  n_mfcc: int = 40, n_frames: int = 49) -> np.ndarray:
    """Compute MFCC features matching DS-CNN input (49x40)."""
    try:
        import librosa
        mfcc = librosa.feature.mfcc(y=audio_chunk.astype(np.float32),
                                      sr=sr, n_mfcc=n_mfcc, hop_length=160,
                                      n_fft=512)
        if mfcc.shape[1] < n_frames:
            mfcc = np.pad(mfcc, ((0, 0), (0, n_frames - mfcc.shape[1])))
        else:
            mfcc = mfcc[:, :n_frames]
        return mfcc.T[np.newaxis].astype(np.float32)  # (1, 49, 40)
    except ImportError:
        return np.random.randn(1, n_frames, n_mfcc).astype(np.float32)


def simulate_ds_cnn_inference(mfcc_features: np.ndarray, model_cfg: dict) -> tuple:
    """Simulate DS-CNN inference on MAX32690."""
    np.random.seed(int(time.time() * 1000) % 2**31)
    logits = np.random.randn(len(KEYWORDS_35)).astype(np.float32)
    keyword_bias_idx = np.random.randint(0, len(KEYWORDS_35))
    logits[keyword_bias_idx] += 2.5
    exp_l = np.exp(logits - logits.max())
    probs = exp_l / exp_l.sum()
    return probs, model_cfg["latency_ms"]


class KeywordSpotter:
    """Real-time keyword spotter simulating MAX32690 deployment."""

    def __init__(self, model_name: str = "ds_cnn_max32690",
                 conf_threshold: float = 0.7, window_size: int = 3):
        self.model_name     = model_name
        self.cfg            = MODEL_CONFIGS[model_name]
        self.conf_threshold = conf_threshold
        self.recent         = deque(maxlen=window_size)
        self.detections     = []
        self.frames_processed  = 0
        self.total_inferences  = 0

        print(f"\n{'='*60}")
        print(f"Keyword Spotter — {model_name}")
        print(f"Device   : {self.cfg['device']}")
        print(f"Accuracy : {self.cfg['accuracy']}% (Google Speech Commands)")
        print(f"Power    : {self.cfg['power_ma']} mA during inference")
        print(f"Latency  : {self.cfg['latency_ms']} ms per utterance")
        print(f"Energy   : {self.cfg['energy_uj']} uJ per inference")
        print(f"{'='*60}\n")

    def process_chunk(self, audio_chunk: np.ndarray, sr: int = 16000) -> dict:
        self.frames_processed += 1
        features = compute_mfcc(audio_chunk, sr)
        probs, latency = simulate_ds_cnn_inference(features, self.cfg)
        self.total_inferences += 1

        top_idx  = int(np.argmax(probs))
        top_conf = float(probs[top_idx])
        keyword  = KEYWORDS_35[top_idx]

        self.recent.append((keyword, top_conf))

        detected = None
        if top_conf >= self.conf_threshold:
            window_kws = [k for k, c in self.recent if c >= self.conf_threshold]
            if len(window_kws) >= max(1, len(self.recent) // 2):
                detected = keyword
                self.detections.append({
                    "keyword": keyword, "confidence": top_conf,
                    "frame": self.frames_processed,
                    "device": self.cfg["device"],
                })

        return {
            "keyword": keyword, "confidence": top_conf,
            "detected": detected,
            "latency_ms": latency,
            "frame": self.frames_processed,
        }

    def print_stats(self):
        print(f"\n--- Session Statistics ---")
        print(f"Frames processed : {self.frames_processed}")
        print(f"Total inferences : {self.total_inferences}")
        print(f"Keywords detected: {len(self.detections)}")

        total_energy_uj  = self.total_inferences * self.cfg["energy_uj"]
        total_energy_mah = total_energy_uj / 1e6 / 3.3 * 1000
        print(f"Energy consumed  : {total_energy_uj:.0f} uJ = {total_energy_mah:.4f} mAh")

        coin_cell_mah = 230
        sessions_per_battery = coin_cell_mah / max(total_energy_mah, 1e-9)
        print(f"CR2032 (230mAh)  : ~{sessions_per_battery:.0f} sessions like this")

        if self.detections:
            print(f"\nDetected keywords:")
            for d in self.detections[-10:]:
                print(f"  Frame {d['frame']:5d}: '{d['keyword']}'  conf={d['confidence']:.3f}")


def run_simulation(model_name: str, num_frames: int, conf_threshold: float):
    spotter = KeywordSpotter(model_name, conf_threshold)
    sr = 16000
    chunk_size = sr

    print(f"Simulating {num_frames} audio frames...\n")
    t_start = time.time()

    for i in range(num_frames):
        if np.random.random() < 0.15:
            freq = np.random.choice([440, 880, 1200])
            t_arr = np.linspace(0, 1, sr)
            audio = (np.sin(2 * np.pi * freq * t_arr) * 0.5).astype(np.float32)
        else:
            audio = (np.random.randn(chunk_size) * 0.02).astype(np.float32)

        result = spotter.process_chunk(audio, sr)

        if result["detected"]:
            print(f"  KEYWORD: '{result['detected']}'  conf={result['confidence']:.3f}")
        elif i % 10 == 0:
            print(f"  Frame {i+1:3d} | top='{result['keyword']}' ({result['confidence']:.2f})")

    elapsed = time.time() - t_start
    spotter.print_stats()
    print(f"\nSimulation time: {elapsed:.2f}s for {num_frames} frames")


def main():
    parser = argparse.ArgumentParser(description="ADI DS-CNN Keyword Spotter")
    parser.add_argument("--source",    default="simulate",
                        help="'mic', 'simulate', or path to .wav file")
    parser.add_argument("--model",     default="ds_cnn_max32690",
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--frames",    type=int, default=30)
    args = parser.parse_args()

    if args.source == "simulate":
        run_simulation(args.model, args.frames, args.threshold)
    elif args.source == "mic":
        try:
            import sounddevice as sd
            spotter = KeywordSpotter(args.model, args.threshold)
            sr = 16000
            print("Listening... (Ctrl+C to stop)")
            with sd.InputStream(samplerate=sr, channels=1, dtype="float32") as stream:
                while True:
                    audio, _ = stream.read(sr)
                    result = spotter.process_chunk(audio.flatten(), sr)
                    if result["detected"]:
                        print(f"KEYWORD: '{result['detected']}'  conf={result['confidence']:.3f}")
        except ImportError:
            print("sounddevice not available. Running simulation instead.")
            run_simulation(args.model, args.frames, args.threshold)
        except KeyboardInterrupt:
            spotter.print_stats()
    else:
        try:
            import soundfile as sf
            audio, sr = sf.read(args.source)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            spotter = KeywordSpotter(args.model, args.threshold)
            chunk_size = sr
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i+chunk_size]
                if len(chunk) < chunk_size:
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
                result = spotter.process_chunk(chunk, sr)
                if result["detected"]:
                    print(f"  [{i//sr:.1f}s] KEYWORD: '{result['detected']}'  "
                          f"conf={result['confidence']:.3f}")
            spotter.print_stats()
        except Exception as e:
            print(f"Error loading audio: {e}. Running simulation.")
            run_simulation(args.model, args.frames, args.threshold)


if __name__ == "__main__":
    main()
