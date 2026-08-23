"""Shared preprocessing for TI EdgeAI models."""
import numpy as np

IMAGENET_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
IMAGENET_STD  = np.array([58.395,  57.12,  57.375], dtype=np.float32)


def letterbox(img: np.ndarray, target_hw: tuple, pad_val: int = 114) -> tuple:
    h, w = img.shape[:2]
    th, tw = target_hw
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    canvas = np.full((th, tw, 3), pad_val, dtype=np.uint8)
    dw, dh = (tw - nw) // 2, (th - nh) // 2
    canvas[dh:dh+nh, dw:dw+nw] = img[:nh, :nw]
    return canvas, scale, dw, dh


def normalize_imagenet(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return img[np.newaxis]  # NHWC batch


def preprocess_classification(img: np.ndarray, size: tuple = (224, 224)) -> np.ndarray:
    h, w = size
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    sh = min(img.shape[0], h)
    sw = min(img.shape[1], w)
    canvas[:sh, :sw] = img[:sh, :sw]
    return normalize_imagenet(canvas)


def preprocess_detection(img: np.ndarray, size: tuple = (640, 640)) -> tuple:
    padded, scale, dw, dh = letterbox(img, size)
    tensor = padded.astype(np.float32) / 255.0
    return tensor[np.newaxis].transpose(0, 3, 1, 2), scale, dw, dh  # NCHW


def preprocess_segmentation(img: np.ndarray, size: tuple = (512, 512)) -> np.ndarray:
    canvas = np.zeros((*size, 3), dtype=np.uint8)
    sh, sw = min(img.shape[0], size[0]), min(img.shape[1], size[1])
    canvas[:sh, :sw] = img[:sh, :sw]
    tensor = (canvas.astype(np.float32) - 128.0) / 128.0
    return tensor[np.newaxis].transpose(0, 3, 1, 2)


def preprocess_pose(img: np.ndarray, size: tuple = (640, 640)) -> tuple:
    return preprocess_detection(img, size)


def preprocess_depth(img: np.ndarray, size: tuple = (224, 224)) -> np.ndarray:
    return preprocess_classification(img, size)
