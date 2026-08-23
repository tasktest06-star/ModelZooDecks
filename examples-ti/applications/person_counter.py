"""
Person Counter Application — TI EdgeAI YOLOX models.
Pipeline: Camera frame -> YOLOX detect -> IoU tracking -> count -> alert.
Target SoC: AM67A (yolox_s_lite) or AM68A (yolox_m_lite).
"""

import argparse
import time
import numpy as np
import cv2
from collections import deque
from dataclasses import dataclass
from typing import List


@dataclass
class PersonTrack:
    track_id: int
    bbox: list
    centroid: tuple
    age: int
    lost: int


class PersonCounter:
    MODEL_CONFIGS = {
        "yolox_pico_lite": {"input": (320,320), "mAP": 20.1, "target_soc": "AM62A"},
        "yolox_nano_lite": {"input": (416,416), "mAP": 22.4, "target_soc": "AM62A"},
        "yolox_s_lite":    {"input": (640,640), "mAP": 38.4, "target_soc": "AM67A"},
        "yolox_m_lite":    {"input": (640,640), "mAP": 44.2, "target_soc": "AM68A"},
    }

    def __init__(self, model_name="yolox_s_lite", model_path=None,
                 soc="AM67A", conf_thresh=0.4, max_count_alert=5):
        self.model_name  = model_name
        self.model_path  = model_path or f"models/{model_name}.onnx"
        self.soc         = soc
        self.conf_thresh = conf_thresh
        self.alert_limit = max_count_alert
        self.tracks: List[PersonTrack] = []
        self._next_id      = 0
        self.count_history = deque(maxlen=30)
        self.total_entered = 0
        self.frame_count   = 0
        self.latencies     = deque(maxlen=100)
        self._load_model()

    def _load_model(self):
        try:
            import onnxruntime as ort
            self.sess = ort.InferenceSession(
                self.model_path, providers=["CPUExecutionProvider"])
            self.inp_name  = self.sess.get_inputs()[0].name
            self.model_loaded = True
            print(f"Loaded {self.model_name} — target SoC: {self.soc}")
        except Exception as e:
            print(f"Model not found ({e}). Running in SIMULATION mode.")
            self.model_loaded = False

    def _letterbox(self, img, size):
        h, w = img.shape[:2]
        s = min(size[0]/w, size[1]/h)
        nw, nh = int(w*s), int(h*s)
        resized = cv2.resize(img, (nw, nh))
        canvas = np.full((size[1], size[0], 3), 114, dtype=np.uint8)
        dw, dh = (size[0]-nw)//2, (size[1]-nh)//2
        canvas[dh:dh+nh, dw:dw+nw] = resized
        return canvas, s, dw, dh

    def _detect(self, frame):
        if not self.model_loaded:
            n = np.random.randint(0, 4)
            h, w = frame.shape[:2]
            return [{"bbox":[np.random.randint(0,w//2), np.random.randint(0,h//2),
                              np.random.randint(w//2,w), np.random.randint(h//2,h)],
                     "score": np.random.uniform(0.5, 0.95)} for _ in range(n)]

        inp_size = self.MODEL_CONFIGS[self.model_name]["input"]
        lb, scale, dw, dh = self._letterbox(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), inp_size)
        tensor = lb.astype(np.float32).transpose(2,0,1)[np.newaxis]

        t0 = time.perf_counter()
        raw = self.sess.run(None, {self.inp_name: tensor})[0]
        self.latencies.append((time.perf_counter()-t0)*1000)

        persons = []
        if raw.ndim == 3:
            raw = raw[0]
        for row in raw:
            score = row[4] * row[5]
            if score < self.conf_thresh:
                continue
            cx, cy, bw, bh = row[:4]
            persons.append({"bbox":[
                (cx-bw/2-dw)/scale, (cy-bh/2-dh)/scale,
                (cx+bw/2-dw)/scale, (cy+bh/2-dh)/scale],
                "score": float(score)})
        return persons

    def _iou(self, a, b):
        ix1 = max(a[0],b[0]); iy1 = max(a[1],b[1])
        ix2 = min(a[2],b[2]); iy2 = min(a[3],b[3])
        inter = max(0,ix2-ix1)*max(0,iy2-iy1)
        ua = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
        return inter/ua if ua > 0 else 0

    def _update_tracks(self, detections):
        matched = set()
        for det in detections:
            best_iou, best_t = 0, None
            for t in self.tracks:
                if t.lost > 0:
                    continue
                iou = self._iou(det["bbox"], t.bbox)
                if iou > best_iou:
                    best_iou, best_t = iou, t
            if best_iou > 0.3 and best_t:
                best_t.bbox = det["bbox"]
                cx = (det["bbox"][0]+det["bbox"][2])/2
                cy = (det["bbox"][1]+det["bbox"][3])/2
                best_t.centroid = (cx, cy)
                best_t.age += 1; best_t.lost = 0
                matched.add(best_t.track_id)
            else:
                cx = (det["bbox"][0]+det["bbox"][2])/2
                cy = (det["bbox"][1]+det["bbox"][3])/2
                self.tracks.append(PersonTrack(self._next_id, det["bbox"], (cx,cy), 1, 0))
                self._next_id += 1
                self.total_entered += 1

        for t in self.tracks:
            if t.track_id not in matched:
                t.lost += 1
        self.tracks = [t for t in self.tracks if t.lost <= 10]

    def process_frame(self, frame):
        self.frame_count += 1
        self._update_tracks(self._detect(frame))
        active = [t for t in self.tracks if t.lost == 0]
        self.count_history.append(len(active))
        smoothed = int(np.mean(list(self.count_history)[-5:]))
        avg_lat = np.mean(list(self.latencies)) if self.latencies else 0
        return {
            "frame": self.frame_count,
            "current_persons": len(active),
            "smoothed_count": smoothed,
            "total_entered": self.total_entered,
            "alert": smoothed >= self.alert_limit,
            "avg_latency_ms": round(avg_lat, 1),
            "fps": round(1000/avg_lat, 1) if avg_lat > 0 else 0,
        }

    def draw_overlay(self, frame, result):
        out = frame.copy()
        for t in [t for t in self.tracks if t.lost == 0]:
            x1,y1,x2,y2 = [int(v) for v in t.bbox]
            cv2.rectangle(out, (x1,y1), (x2,y2), (0,200,0), 2)
            cv2.putText(out, f"ID:{t.track_id}", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,200,0), 1)
        color = (0,0,220) if result["alert"] else (220,220,0)
        cv2.putText(out, f"Count: {result['smoothed_count']}",
                    (15,35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(out, f"Total: {result['total_entered']} | FPS: {result['fps']:.0f}",
                    (15,65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
        cv2.putText(out, f"{self.model_name} | {self.soc}",
                    (15, out.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150,150,150), 1)
        if result["alert"]:
            cv2.putText(out, f"ALERT: >{self.alert_limit} persons",
                        (15,100), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,220), 2)
        return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",     default="0")
    parser.add_argument("--model",      default="yolox_s_lite",
                        choices=list(PersonCounter.MODEL_CONFIGS.keys()))
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--soc",        default="AM67A")
    parser.add_argument("--conf",       type=float, default=0.4)
    parser.add_argument("--alert",      type=int, default=5)
    parser.add_argument("--no_display", action="store_true")
    args = parser.parse_args()

    counter = PersonCounter(args.model, args.model_path, args.soc, args.conf, args.alert)

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    use_synthetic = not cap.isOpened()
    if use_synthetic:
        print(f"Cannot open {source}. Using synthetic frames.")

    result = {}
    try:
        while True:
            if use_synthetic:
                frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
                time.sleep(0.033)
            else:
                ret, frame = cap.read()
                if not ret:
                    break

            result = counter.process_frame(frame)

            if result["frame"] % 10 == 0:
                print(f"Frame {result['frame']:5d} | Persons: {result['smoothed_count']} "
                      f"| Total: {result['total_entered']} | FPS: {result['fps']:.0f}"
                      + (" | ALERT" if result["alert"] else ""))

            if not args.no_display:
                cv2.imshow("TI EdgeAI Person Counter", counter.draw_overlay(frame, result))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if result:
            print(f"\nDone: {result['total_entered']} persons over {result['frame']} frames")


if __name__ == "__main__":
    main()
