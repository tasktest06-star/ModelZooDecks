"""
TFLite INT8 object detection inference for NXP eIQ models.
Handles SSD-family models with per-channel INT8 quantization (NXP recipe.sh style).
Targets: i.MX 8M Plus NPU, i.MX 93 Ethos-U65 (Vela-compiled), RT1170 TFLite Micro
"""

import argparse
import time
import json
import numpy as np


DETECTION_MODELS = {
    "ssdlite_mobiledet_coco": {
        "input": (320, 320), "map_coco": 25.9,
        "anchors": "ssd_mobile", "num_classes": 90,
        "latency_imx8": 28, "latency_imx93": 85,
        "target": "RT1170 / i.MX 93",
        "note": "Best size/accuracy for RT1170; all layers NPU-compatible"
    },
    "ssdlite_mobilenetv2_coco": {
        "input": (300, 300), "map_coco": 22.1,
        "anchors": "ssd_mobile", "num_classes": 90,
        "latency_imx8": 22, "latency_imx93": 65,
        "target": "RT1170",
        "note": "Lightest SSD; fits RT1170 2MB OCRAM"
    },
    "yolov8n": {
        "input": (640, 640), "map_coco": 37.3,
        "anchors": "yolov8", "num_classes": 80,
        "latency_imx8": 95, "latency_imx93": 220,
        "target": "i.MX 8M Plus",
        "note": "NXP-optimised YOLOv8n; Ultralytics → TFLite INT8 via recipe.sh"
    },
    "yolov8m": {
        "input": (640, 640), "map_coco": 50.2,
        "anchors": "yolov8", "num_classes": 80,
        "latency_imx8": 280, "latency_imx93": 900,
        "target": "i.MX 8M Plus",
        "note": "Medium variant; i.MX 8M Plus only at real-time"
    },
    "centernet_mobilenetv2": {
        "input": (512, 512), "map_coco": 29.5,
        "anchors": "centernet", "num_classes": 90,
        "latency_imx8": 55, "latency_imx93": 170,
        "target": "i.MX 93",
        "note": "CenterNet anchor-free; 92% ops on Ethos-U65"
    },
    "nanodet_plus": {
        "input": (416, 416), "map_coco": 30.4,
        "anchors": "nanodet", "num_classes": 80,
        "latency_imx8": 35, "latency_imx93": 100,
        "target": "i.MX 93 / RT1170",
        "note": "NanoDet-Plus: lightweight, Ethos-U65 friendly"
    },
}

COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",
    "book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
]


def letterbox(image: np.ndarray, target_size: tuple) -> tuple:
    h, w = image.shape[:2]
    th, tw = target_size
    scale = min(tw / w, th / h)
    new_w, new_h = int(w * scale), int(h * scale)

    try:
        import cv2
        resized = cv2.resize(image, (new_w, new_h))
    except ImportError:
        resized = image

    pad_top  = (th - new_h) // 2
    pad_left = (tw - new_w) // 2
    canvas = np.full((th, tw, 3), 114, dtype=np.uint8)
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    return canvas, scale, pad_top, pad_left


def preprocess(image_path: str, input_size: tuple) -> tuple:
    try:
        import cv2
        img = cv2.imread(image_path)
        if img is None:
            img = np.random.randint(0, 255, (*input_size, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception:
        img = np.random.randint(0, 255, (*input_size, 3), dtype=np.uint8)

    lb, scale, pad_top, pad_left = letterbox(img, input_size)
    tensor = lb.astype(np.float32) / 255.0
    tensor = tensor[np.newaxis]  # (1, H, W, 3)
    return tensor, scale, pad_top, pad_left, img.shape


def run_tflite(model_path: str, input_data: np.ndarray) -> list:
    try:
        import tflite_runtime.interpreter as tflite
        Interpreter = tflite.Interpreter
    except ImportError:
        try:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
        except ImportError:
            return [np.zeros((100, 4)), np.zeros(100), np.zeros(100).astype(np.int32), np.array([50])]

    interp = Interpreter(model_path=model_path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]

    data = input_data
    if inp["dtype"] == np.int8:
        scale, zp = inp["quantization"]
        data = (input_data / scale + zp).astype(np.int8)
    elif inp["dtype"] == np.uint8:
        data = (input_data * 255).astype(np.uint8)

    interp.set_tensor(inp["index"], data)
    interp.invoke()
    return [interp.get_tensor(d["index"]) for d in interp.get_output_details()]


def decode_ssd(outputs: list, scale: float, pad_top: int, pad_left: int,
               orig_shape: tuple, input_size: tuple,
               conf_thresh: float = 0.3) -> list:
    """Decode SSD / SSDLite 4-output format: boxes, classes, scores, num_detections."""
    if len(outputs) < 4:
        return []

    boxes   = outputs[0][0]  # (N, 4) normalized [y1,x1,y2,x2]
    classes = outputs[2][0].astype(int)
    scores  = outputs[1][0]
    count   = int(outputs[3][0])

    ih, iw = input_size
    oh, ow  = orig_shape[:2]
    detections = []

    for i in range(count):
        if scores[i] < conf_thresh:
            continue
        y1, x1, y2, x2 = boxes[i]
        # undo letterbox
        x1 = (x1 * iw - pad_left) / scale
        x2 = (x2 * iw - pad_left) / scale
        y1 = (y1 * ih - pad_top)  / scale
        y2 = (y2 * ih - pad_top)  / scale
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(ow, x2), min(oh, y2)

        cls_id = classes[i]
        cls_name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f"cls_{cls_id}"
        detections.append({
            "class": cls_name, "class_id": cls_id,
            "confidence": float(scores[i]),
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
        })

    return detections


def decode_yolov8(outputs: list, scale: float, pad_top: int, pad_left: int,
                  orig_shape: tuple, input_size: tuple,
                  conf_thresh: float = 0.3, iou_thresh: float = 0.45) -> list:
    """Decode YOLOv8 output tensor (1, 84, num_anchors) style."""
    if not outputs:
        return []

    pred = outputs[0]
    if pred.ndim == 3:
        pred = pred[0]  # (84, num_anchors) or (num_anchors, 84)
    if pred.shape[0] == 84:
        pred = pred.T   # -> (num_anchors, 84)

    boxes   = pred[:, :4]
    scores  = pred[:, 4:].max(axis=1)
    classes = pred[:, 4:].argmax(axis=1)

    keep = scores > conf_thresh
    boxes, scores, classes = boxes[keep], scores[keep], classes[keep]

    iw, ih = input_size[1], input_size[0]
    oh, ow  = orig_shape[:2]

    # xywh → xyxy
    x1 = (boxes[:, 0] - boxes[:, 2] / 2 - pad_left) / scale
    y1 = (boxes[:, 1] - boxes[:, 3] / 2 - pad_top)  / scale
    x2 = (boxes[:, 0] + boxes[:, 2] / 2 - pad_left) / scale
    y2 = (boxes[:, 1] + boxes[:, 3] / 2 - pad_top)  / scale

    detections = []
    for i in range(len(scores)):
        cls_id = int(classes[i])
        cls_name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f"cls_{cls_id}"
        detections.append({
            "class": cls_name, "class_id": cls_id,
            "confidence": float(scores[i]),
            "bbox": [float(max(0, x1[i])), float(max(0, y1[i])),
                     float(min(ow, x2[i])), float(min(oh, y2[i]))],
        })
    return detections


def detect(image_path: str, model_name: str, model_path: str,
           platform: str = "imx8mplus", conf_thresh: float = 0.3) -> dict:
    if model_name not in DETECTION_MODELS:
        raise ValueError(f"Unknown model: {model_name}. Choices: {list(DETECTION_MODELS.keys())}")

    meta = DETECTION_MODELS[model_name]
    lat_key = "latency_imx93" if "93" in platform or "ethos" in platform else "latency_imx8"
    print(f"\n{'='*65}")
    print(f"Model    : {model_name}")
    print(f"mAP COCO : {meta['map_coco']}%  |  Target: {meta['target']}")
    print(f"Input    : {meta['input']}  |  Classes: {meta['num_classes']}")
    print(f"Est. NPU latency ({platform}): {meta[lat_key]} ms")
    print(f"Note     : {meta['note']}")
    print(f"{'='*65}")

    tensor, scale, pad_top, pad_left, orig_shape = preprocess(image_path, meta["input"])

    t0 = time.perf_counter()
    outputs = run_tflite(model_path, tensor)
    cpu_ms  = (time.perf_counter() - t0) * 1000

    anchor_type = meta["anchors"]
    if anchor_type in ("ssd_mobile", "centernet", "nanodet"):
        detections = decode_ssd(outputs, scale, pad_top, pad_left,
                                orig_shape, meta["input"], conf_thresh)
    else:
        detections = decode_yolov8(outputs, scale, pad_top, pad_left,
                                   orig_shape, meta["input"], conf_thresh)

    print(f"\nDetected {len(detections)} objects  (CPU: {cpu_ms:.1f}ms)")
    for d in detections[:10]:
        x1, y1, x2, y2 = d["bbox"]
        print(f"  {d['class']:<20} {d['confidence']*100:.1f}%  "
              f"[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")

    return {"model": model_name, "platform": platform,
            "detections": detections, "cpu_latency_ms": cpu_ms,
            "npu_latency_ms": meta[lat_key]}


def main():
    parser = argparse.ArgumentParser(description="NXP eIQ TFLite Object Detection")
    parser.add_argument("--image",      default="sample.jpg")
    parser.add_argument("--model",      default="ssdlite_mobiledet_coco",
                        choices=list(DETECTION_MODELS.keys()))
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--platform",   default="imx8mplus",
                        choices=["imx8mplus", "imx93", "rt1170"])
    parser.add_argument("--conf",       type=float, default=0.3)
    parser.add_argument("--output",     default=None)
    args = parser.parse_args()

    if args.model_path is None:
        suffix = "_vela" if args.platform == "imx93" else ""
        args.model_path = f"models/{args.model}{suffix}.tflite"

    result = detect(args.image, args.model, args.model_path, args.platform, args.conf)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
