"""
data_pipeline.py — Task-specific data loading and preprocessing for TI EdgeAI models.

Implements per-task preprocessing that matches edgeai-benchmark conventions:
  - Classification: Resize + CenterCrop + Normalize (ImageNet stats)
  - Detection:      LetterBox resize, BGR→RGB, /255.0 (YOLOX/COCO style)
  - Segmentation:   Resize + Normalize + ignore_index=255
  - Depth:          Resize + Normalize
  - Keypoint:       LetterBox resize (same as detection)
"""

import os
import numpy as np
from pathlib import Path
from typing import Iterator, Optional
from dataclasses import dataclass, field

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
    import torchvision.transforms as T
    from PIL import Image
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ── Preprocessing parameter tables ─────────────────────────────────────────────

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

TASK_DEFAULTS = {
    "classification": {
        "input_size": (224, 224),
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "letterbox": False,
        "normalize_0_255": False,
    },
    "object_detection": {
        "input_size": (640, 640),
        "mean": [0.0, 0.0, 0.0],
        "std": [1.0, 1.0, 1.0],
        "letterbox": True,
        "normalize_0_255": True,       # divide by 255
    },
    "segmentation": {
        "input_size": (512, 512),
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "letterbox": False,
        "normalize_0_255": False,
        "ignore_index": 255,
    },
    "depth_estimation": {
        "input_size": (256, 256),
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "letterbox": False,
        "normalize_0_255": False,
    },
    "keypoint": {
        "input_size": (640, 640),
        "mean": [0.0, 0.0, 0.0],
        "std": [1.0, 1.0, 1.0],
        "letterbox": True,
        "normalize_0_255": True,
    },
}


# ── Preprocessing utilities ─────────────────────────────────────────────────────

def letterbox_resize(img: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """
    Resize image preserving aspect ratio with grey padding (YOLOX-style).
    img: HWC uint8 numpy array.
    Returns: HWC uint8 numpy array of shape (target_size[0], target_size[1], 3).
    """
    h, w = img.shape[:2]
    th, tw = target_size
    scale = min(th / h, tw / w)
    nh, nw = int(h * scale), int(w * scale)

    try:
        from PIL import Image as PILImage
        pil = PILImage.fromarray(img).resize((nw, nh), PILImage.BILINEAR)
        resized = np.array(pil)
    except ImportError:
        # Fallback: nearest-neighbour via slicing (no external deps)
        resized = img[
            np.round(np.linspace(0, h - 1, nh)).astype(int), :, :
        ][:, np.round(np.linspace(0, w - 1, nw)).astype(int), :]

    canvas = np.full((th, tw, 3), 114, dtype=np.uint8)
    pad_y = (th - nh) // 2
    pad_x = (tw - nw) // 2
    canvas[pad_y: pad_y + nh, pad_x: pad_x + nw] = resized
    return canvas


def preprocess_image(
    img: np.ndarray,
    task: str,
    input_size: Optional[tuple[int, int]] = None,
) -> np.ndarray:
    """
    Apply task-specific preprocessing. Returns float32 NCHW tensor (numpy).
    img: HWC uint8 BGR or RGB numpy array.
    """
    params = TASK_DEFAULTS.get(task, TASK_DEFAULTS["classification"])
    size = input_size or params["input_size"]

    if params["letterbox"]:
        img = letterbox_resize(img, size)
    else:
        try:
            from PIL import Image as PILImage
            img = np.array(PILImage.fromarray(img).resize(
                (size[1], size[0]), PILImage.BILINEAR))
        except ImportError:
            pass

    # BGR → RGB for detection (OpenCV loads BGR)
    if task in ("object_detection", "keypoint"):
        img = img[:, :, ::-1]

    img = img.astype(np.float32)

    if params["normalize_0_255"]:
        img /= 255.0
    else:
        mean = np.array(params["mean"], dtype=np.float32)
        std  = np.array(params["std"],  dtype=np.float32)
        img = (img / 255.0 - mean) / std

    # HWC → NCHW
    img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]
    return img


# ── Dataset classes ─────────────────────────────────────────────────────────────

@dataclass
class SampleBatch:
    images: "np.ndarray"             # shape (N, C, H, W)
    targets: list                    # annotations (dicts or tensors)
    image_ids: list[int] = field(default_factory=list)
    paths: list[str] = field(default_factory=list)


class _ImageFolderDataset:
    """Minimal flat-folder image dataset (no torch dependency)."""

    def __init__(self, root: str, task: str, input_size: tuple[int, int],
                 max_samples: int = -1):
        self.root = Path(root)
        self.task = task
        self.input_size = input_size
        self.paths = sorted(self.root.rglob("*.jpg")) + \
                     sorted(self.root.rglob("*.jpeg")) + \
                     sorted(self.root.rglob("*.png"))
        if max_samples > 0:
            self.paths = self.paths[:max_samples]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple:
        path = self.paths[idx]
        try:
            from PIL import Image as PILImage
            img = np.array(PILImage.open(path).convert("RGB"))
        except Exception:
            img = np.zeros((*self.input_size, 3), dtype=np.uint8)
        tensor = preprocess_image(img, self.task, self.input_size)
        return tensor, {}, str(path)


# ── Public API ──────────────────────────────────────────────────────────────────

class DataPipeline:
    """
    Task-aware data pipeline for TI EdgeAI model evaluation.

    Usage:
        pipeline = DataPipeline(task="detection", input_size=(640, 640),
                                dataset="coco")
        loader = pipeline.get_dataloader(split="val", batch_size=1)
        for imgs, targets in loader:
            preds = model.infer(imgs)
            metrics.update(preds, targets)
    """

    def __init__(
        self,
        task: str,
        input_size: Optional[tuple[int, int]] = None,
        dataset: str = "coco",
        dataset_root: str = "./datasets",
        num_frames: int = -1,
    ):
        if task not in TASK_DEFAULTS:
            raise ValueError(f"Unknown task '{task}'. "
                             f"Choose from: {list(TASK_DEFAULTS)}")
        self.task = task
        self.params = TASK_DEFAULTS[task]
        self.input_size = input_size or self.params["input_size"]
        self.dataset = dataset
        self.dataset_root = Path(dataset_root)
        self.num_frames = num_frames

    def get_dataloader(
        self,
        split: str = "val",
        batch_size: int = 1,
    ) -> Iterator[SampleBatch]:
        """
        Return an iterator over SampleBatch objects.

        For full COCO/ImageNet support, install the respective dataset
        libraries (pycocotools, torchvision). This fallback iterates over
        raw images in dataset_root/<split>/.
        """
        data_dir = self.dataset_root / self.dataset / split
        if not data_dir.exists():
            data_dir = self.dataset_root / split
        if not data_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {data_dir}. "
                "Set dataset_root in pipeline_config.yaml."
            )

        ds = _ImageFolderDataset(
            str(data_dir), self.task, self.input_size,
            max_samples=self.num_frames
        )
        print(f"[DataPipeline] {self.task}/{self.dataset}/{split}: "
              f"{len(ds)} images, input={self.input_size}")

        batch_imgs, batch_targets, batch_ids, batch_paths = [], [], [], []
        for i in range(len(ds)):
            tensor, target, path = ds[i]
            batch_imgs.append(tensor)
            batch_targets.append(target)
            batch_ids.append(i)
            batch_paths.append(path)

            if len(batch_imgs) == batch_size:
                yield SampleBatch(
                    images=np.concatenate(batch_imgs, axis=0),
                    targets=batch_targets,
                    image_ids=batch_ids,
                    paths=batch_paths,
                )
                batch_imgs, batch_targets, batch_ids, batch_paths = [], [], [], []

        if batch_imgs:
            yield SampleBatch(
                images=np.concatenate(batch_imgs, axis=0),
                targets=batch_targets,
                image_ids=batch_ids,
                paths=batch_paths,
            )

    def preprocess_single(self, img: np.ndarray) -> np.ndarray:
        """Preprocess a single HWC uint8 image. Returns NCHW float32."""
        return preprocess_image(img, self.task, self.input_size)

    @property
    def input_shape(self) -> tuple:
        return (1, 3, *self.input_size)

    def summary(self) -> dict:
        return {
            "task": self.task,
            "dataset": self.dataset,
            "input_size": self.input_size,
            "preprocessing": self.params,
        }
