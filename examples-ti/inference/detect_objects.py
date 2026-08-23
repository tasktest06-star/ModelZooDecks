"""Object detection using TI EdgeAI YOLOX models via ONNX runtime."""

import argparse
import time
import numpy as np
import cv2
from pathlib import Path
import json


YOLOX_MODELS = {
    "yolox_pico_lite": {"input": (320, 320), "format": "onnx", "mAP": 20.1, "soc": "AM62A"},
    "yolox_nano_lite": {"input": (416, 416), "format": "onnx", "mAP": 22.4, "soc": "AM62A"},
    "yolox_tiny_lite": {"input": (416, 416), "format": "onnx", "mAP": 29.2, "soc": "AM67A"},
    "yolox_s_lite":    {"input": (640, 640), "format": "onnx", "mAP": 38.4, "soc": "AM67A"},
    "yolox_m_lite":    {"input": (640, 640), "format": "onnx", "mAP": 44.2, "soc": "AM68A"},
    "rtmdet_m_lite":   {"input": (640, 640), "format": "onnx", "mAP": 56.0, "soc": "AM69A"},
}

COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote",
    "keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book",
    "clock","vase","scissors","teddy bear","hair drier","toothbrush"
]

COLORS = np.random.randint(0, 255, size=(len(COCO_CLASSES), 3), dtype=np.uint8)


def letterbox(img: np.ndarray, target_size: tuple) -> tuple:
    h, w = img.shape[:2]
    th, tw = target_size
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh))
    padded = np.full((th, tw, 3), 114, dtype=np.uint8)
    dw, dh = (tw - nw) // 2, (th - nh) // 2
    padded[dh:dh+nh, dw:dw+nw] = resized
    return padded, scale, dw, dh


def preprocess_yolox(img_bgr: np.ndarray, input_size: tuple) -> tuple:
    padded, scale, dw, dh = letterbox(img_bgr, input_size)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32)
    tensor = rgb.transpose(2, 0, 1)[np.newaxis]  # NCHW
    return tensor, scale, dw, dh


def decode_yolox(output: np.ndarray, input_size: tuple,
                 conf_thresh: float = 0.3, nms_thresh: float = 0.45) -> list:
    if output.ndim == 2:
        output = output[np.newaxis]

    grids, strides = [], []
    for stride in [8, 16, 32]:
        h, w = input_size[0] // stride, input_size[1] // stride
        g = np.stack(np.meshgrid(np.arange(w), np.arange(h)), axis=-1).reshape(-1, 2)
        grids.append(g)
        strides.append(np.full((h * w, 1), stride))
    grids   = np.concatenate(grids, axis=0)
    strides = np.concatenate(strides, axis=0)

    pred = output[0].copy()
    pred[:, :2] = (pred[:, :2] + grids) * strides
    pred[:, 2:4] = np.exp(pred[:, 2:4]) * strides

    box_conf  = pred[:, 4:5]
    cls_conf  = pred[:, 5:]
    scores    = box_conf * cls_conf
    class_ids = np.argmax(scores, axis=1)
    max_scores = scores[np.arange(len(scores)), class_ids]

    mask = max_scores > conf_thresh
    boxes      = pred[mask, :4]
    max_scores = max_scores[mask]
    class_ids  = class_ids[mask]

    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    detections = []
    for cls in np.unique(class_ids):
        idx = np.where(class_ids == cls)[0]
        cls_boxes  = boxes_xyxy[idx]
        cls_scores = max_scores[idx]
        keep = cv2.dnn.NMSBoxes(
            cls_boxes.tolist(), cls_scores.tolist(), conf_thresh, nms_thresh)
        for k in (keep.flatten() if len(keep) > 0 else []):
            detections.append({
                "class_id":   int(cls),
                "class_name": COCO_CLASSES[int(cls)] if int(cls) < len(COCO_CLASSES) else "unknown",
                "score":      float(cls_scores[k]),
                "bbox":       cls_boxes[k].tolist(),
            })
    return detections


def draw_detections(img: np.ndarray, detections: list,
                    scale: float, dw: int, dh: int) -> np.ndarray:
    out = img.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        x1 = int((x1 - dw) / scale)
        y1 = int((y1 - dh) / scale)
        x2 = int((x2 - dw) / scale)
        y2 = int((y2 - dh) / scale)
        cls = det["class_id"] % len(COLORS)
        color = tuple(int(c) for c in COLORS[cls])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{det['class_name']} {det['score']:.2f}"
        cv2.putText(out, label, (x1, max(y1-5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return out


def detect(image_path: str, model_name: str, model_path: str,
           soc: str, conf_thresh: float = 0.3, save_viz: str = None) -> dict:
    if model_name not in YOLOX_MODELS:
        raise ValueError(f"Unknown model: {model_name}")

    meta = YOLOX_MODELS[model_name]
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot load: {image_path}")

    input_tensor, scale, dw, dh = preprocess_yolox(img_bgr, meta["input"])

    import onnxruntime as ort
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name

    t0 = time.perf_counter()
    output = sess.run(None, {inp_name: input_tensor})[0]
    latency_ms = (time.perf_counter() - t0) * 1000

    detections = decode_yolox(output, meta["input"], conf_thresh)

    print(f"\nModel: {model_name} | SoC: {soc} | COCO mAP: {meta['mAP']}%")
    print(f"Found {len(detections)} objects | CPU latency: {latency_ms:.1f}ms")
    for d in detections:
        print(f"  {d['class_name']:<20} score={d['score']:.3f}  bbox={[round(v) for v in d['bbox']]}")

    if save_viz:
        viz = draw_detections(img_bgr, detections, scale, dw, dh)
        cv2.imwrite(save_viz, viz)
        print(f"Visualization saved: {save_viz}")

    return {
        "model": model_name, "soc": soc,
        "detections": detections, "latency_cpu_ms": latency_ms,
        "note": f"On {soc} NPU expect ~{latency_ms/8:.0f}ms",
    }


def main():
    parser = argparse.ArgumentParser(description="TI EdgeAI YOLOX Detection Example")
    parser.add_argument("--image",      required=True)
    parser.add_argument("--model",      default="yolox_s_lite", choices=list(YOLOX_MODELS.keys()))
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--soc",        default="AM68A")
    parser.add_argument("--conf",       type=float, default=0.3)
    parser.add_argument("--save_viz",   default=None)
    parser.add_argument("--output",     default=None)
    args = parser.parse_args()

    if args.model_path is None:
        args.model_path = f"models/{args.model}.onnx"

    result = detect(args.image, args.model, args.model_path,
                    args.soc, args.conf, args.save_viz)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
