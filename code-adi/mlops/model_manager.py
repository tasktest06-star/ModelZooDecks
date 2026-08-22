"""
model_manager.py — ADI Model Zoo registry loader and artifact resolver.

Unlike the TI Model Zoo (which uses .link files), ADI stores all model weights
directly in the repository. This manager reads model_index.json and the
model_registry.yaml to resolve local paths and metadata.

Usage:
    manager = ModelManager("config/pipeline_config.yaml")
    path = manager.get_model_path("feature_pyramid_net", precision="int8")
    meta = manager.get_metadata("feature_pyramid_net")
    models = manager.list_models(task="object_detection", device="MAX78002")
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional

import yaml


class ModelManager:
    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.repo_root = Path(
            self.config.get("adi_modelzoo", {}).get("repo_root", "./adi-model-zoo")
        )
        self.cache_dir = Path(
            self.config.get("adi_modelzoo", {}).get("local_cache", "./model_cache")
        )

        registry_path = Path(config_path).parent / "model_registry.yaml"
        with open(registry_path) as f:
            self._registry = yaml.safe_load(f)["models"]

        index_path = self.repo_root / "model_index.json"
        if index_path.exists():
            with open(index_path) as f:
                index_data = json.load(f)
            self._index = {m["name"]: m for m in index_data.get("models", [])}
        else:
            self._index = {}

    def get_metadata(self, model_name: str) -> dict:
        if model_name not in self._registry:
            raise KeyError(f"Model '{model_name}' not found in registry. "
                           f"Available: {list(self._registry.keys())}")
        return self._registry[model_name]

    def get_model_path(self, model_name: str, precision: str = "int8") -> Path:
        """
        Return local path to the weight file.

        For TFLite models that have multi-precision variants, `precision` selects
        the right file key (int8 / int16 / float32). For AI8X models the single
        weight_file is returned regardless of precision.
        """
        meta = self.get_metadata(model_name)
        model_dir = self.repo_root / meta["model_path"]

        wf_key = f"weight_file_{precision}"
        if wf_key in meta:
            weight_file = meta[wf_key]
        elif isinstance(meta.get("weight_file"), list):
            return [model_dir / f for f in meta["weight_file"]]
        else:
            weight_file = meta["weight_file"]

        path = model_dir / weight_file
        if not path.exists():
            print(f"[ModelManager] WARN: weight file not found: {path}")
        return path

    def get_net_def(self, model_name: str) -> Optional[Path]:
        meta = self.get_metadata(model_name)
        net_def = meta.get("net_def")
        if net_def is None:
            return None
        path = self.repo_root / meta["model_path"] / ".." / net_def
        return path.resolve()

    def list_models(
        self,
        task: Optional[str] = None,
        domain: Optional[str] = None,
        device: Optional[str] = None,
        status: Optional[str] = None,
    ) -> list:
        results = []
        for name, meta in self._registry.items():
            if task and meta.get("task") != task:
                continue
            if domain and meta.get("domain") != domain:
                continue
            if status and meta.get("status") != status:
                continue
            if device:
                devices = [d["device"] for d in meta.get("supported_devices", [])]
                if device not in devices:
                    continue
            results.append({"name": name, **meta})
        return results

    def get_accuracy(self, model_name: str) -> dict:
        meta = self.get_metadata(model_name)
        return meta.get("metrics", {})

    def get_supported_devices(self, model_name: str) -> list:
        meta = self.get_metadata(model_name)
        return meta.get("supported_devices", [])

    def get_example_input(self, model_name: str) -> Optional[Path]:
        """Return path to sample input data (for smoke testing)."""
        meta = self.get_metadata(model_name)
        model_dir = Path(meta["model_path"])
        input_dir = self.repo_root / model_dir.parent / "data" / "input"
        if input_dir.exists():
            files = list(input_dir.iterdir())
            if files:
                return files[0]
        return None

    def register(self, name: str, metadata: dict) -> None:
        self._registry[name] = metadata

    def summary(self) -> None:
        print(f"\n{'='*60}")
        print(f" ADI Model Zoo Registry — {len(self._registry)} models")
        print(f"{'='*60}")
        domains = {}
        for name, meta in self._registry.items():
            d = meta.get("domain", "unknown")
            domains.setdefault(d, []).append(name)
        for domain, names in sorted(domains.items()):
            print(f"\n  [{domain.upper()}]")
            for n in names:
                m = self._registry[n]
                devices = [d["device"] for d in m.get("supported_devices", [])]
                print(f"    {n:30s} {m.get('task',''):25s} {', '.join(devices)}")
        print()
