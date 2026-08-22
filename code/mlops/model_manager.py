"""
model_manager.py — Download, register, and retrieve TI EdgeAI Model Zoo artifacts.

Wraps edgeai-modelzoo .link file resolution and local caching.
"""

import os
import json
import shutil
import hashlib
import tarfile
import urllib.request
from pathlib import Path
from typing import Optional

import yaml


class ModelManager:
    """
    Central registry and download manager for edgeai-modelzoo v11.2.0 artifacts.

    Usage:
        manager = ModelManager("config/pipeline_config.yaml")
        manager.download_model("yolox-s-lite", soc="AM68A")
        artifact_dir = manager.get_artifact("yolox-s-lite", soc="AM68A")
    """

    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        registry_path = Path(config_path).parent / "model_registry.yaml"
        with open(registry_path) as f:
            self.registry: dict = yaml.safe_load(f)["models"]

        self.cache_dir = Path(self.config["modelzoo"]["local_cache"])
        self.artifact_dir = Path(self.config["modelzoo"]["artifact_dir"])
        self.base_url = self.config["modelzoo"]["base_url"]

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

        self._metadata_store: dict = {}
        self._load_metadata()

    # ── Public API ──────────────────────────────────────────────────────────────

    def list_models(
        self,
        task: Optional[str] = None,
        soc: Optional[str] = None,
        status: str = "production",
    ) -> list[dict]:
        """Return models filtered by task, SoC, and status."""
        results = []
        for name, meta in self.registry.items():
            if status and meta.get("status") != status:
                continue
            if task and meta.get("task") != task:
                continue
            if soc and soc not in meta.get("artifact_ids", {}):
                continue
            results.append({"name": name, **meta})
        return results

    def get_artifact(self, model_name: str, soc: str) -> Path:
        """
        Return path to the compiled artifact directory for model_name on soc.
        Downloads and extracts if not already cached.
        """
        meta = self._get_meta(model_name)
        artifact_id = self._resolve_artifact_id(meta, soc)
        dest = self.artifact_dir / soc / artifact_id

        if dest.exists():
            return dest

        self.download_model(model_name, soc=soc)
        return dest

    def download_model(self, model_name: str, soc: str) -> Path:
        """Download and extract the pre-compiled artifact for model_name on soc."""
        meta = self._get_meta(model_name)
        artifact_id = self._resolve_artifact_id(meta, soc)
        dest = self.artifact_dir / soc / artifact_id

        if dest.exists():
            print(f"[ModelManager] Already cached: {artifact_id} ({soc})")
            return dest

        url = f"{self.base_url}modelartifacts/{soc}/8bits/{artifact_id}.tar.gz"
        archive = self.cache_dir / f"{artifact_id}.tar.gz"

        print(f"[ModelManager] Downloading {artifact_id} for {soc} ...")
        self._download_file(url, archive)

        print(f"[ModelManager] Extracting to {dest} ...")
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(dest)

        self._save_metadata(model_name, soc, meta)
        print(f"[ModelManager] Ready: {dest}")
        return dest

    def register(self, model_name: str, metadata: dict) -> None:
        """Add or update a model entry in the in-memory registry."""
        self.registry[model_name] = metadata
        self._save_metadata(model_name, metadata.get("soc", "any"), metadata)
        print(f"[ModelManager] Registered: {model_name}")

    def get_metadata(self, model_name: str) -> dict:
        """Return registry metadata for a model."""
        return self._get_meta(model_name)

    def get_model_path(self, model_name: str) -> Optional[Path]:
        """
        Return path to the raw ONNX/TFLite model file (not compiled artifact).
        Searches the local model cache.
        """
        meta = self._get_meta(model_name)
        ext = ".onnx" if meta["format"] == "onnx" else ".tflite"
        candidates = list(self.cache_dir.rglob(f"*{meta['model_id']}*{ext}"))
        return candidates[0] if candidates else None

    def get_accuracy(self, model_name: str, precision: str = "int8") -> Optional[float]:
        """Return the known accuracy metric for a model at a given precision."""
        meta = self._get_meta(model_name)
        metrics = meta.get("metrics", {})
        task = meta.get("task", "")
        key_map = {
            "classification": ("top1_fp32", "top1_int8"),
            "object_detection": ("mAP_fp32", "mAP_int8"),
            "segmentation": ("miou_fp32", "miou_int8"),
            "keypoint": ("AP_fp32", "AP_int8"),
            "depth_estimation": ("delta1_fp32", "delta1_int8"),
        }
        keys = key_map.get(task, ("fp32", "int8"))
        return metrics.get(keys[1] if precision == "int8" else keys[0])

    # ── Internal helpers ────────────────────────────────────────────────────────

    def _get_meta(self, model_name: str) -> dict:
        if model_name not in self.registry:
            available = list(self.registry.keys())
            raise KeyError(
                f"Model '{model_name}' not in registry. "
                f"Available: {available}"
            )
        return self.registry[model_name]

    def _resolve_artifact_id(self, meta: dict, soc: str) -> str:
        artifact_ids = meta.get("artifact_ids", {})
        if soc not in artifact_ids:
            supported = list(artifact_ids.keys())
            raise ValueError(
                f"Model not available for SoC '{soc}'. "
                f"Supported SoCs: {supported}"
            )
        return artifact_ids[soc]

    @staticmethod
    def _download_file(url: str, dest: Path) -> None:
        if dest.exists():
            return
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception as exc:
            raise RuntimeError(f"Failed to download {url}: {exc}") from exc

    def _save_metadata(self, model_name: str, soc: str, meta: dict) -> None:
        key = f"{model_name}:{soc}"
        self._metadata_store[key] = meta
        store_file = self.artifact_dir / "metadata.json"
        with open(store_file, "w") as f:
            json.dump(self._metadata_store, f, indent=2)

    def _load_metadata(self) -> None:
        store_file = self.artifact_dir / "metadata.json"
        if store_file.exists():
            with open(store_file) as f:
                self._metadata_store = json.load(f)
