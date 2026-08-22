"""
artifact_manager.py — Package and manage ADI Model Zoo deployment artifacts.

Creates versioned .tar.gz bundles containing:
  - Model weights (.pth.tar / .tflite / .pt)
  - Network definition (.py, for AI8X models)
  - Synthesized hardware config (if ai8xize has been run)
  - Preprocessing config
  - metadata.json (model info, device target, metrics)
  - manifest.json (file list + checksums)
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _default_preproc(domain: str, task: str) -> dict:
    preproc_map = {
        ("vision", "image_classification"): {
            "input_size": [32, 32], "normalize": "ai8x", "fold_ratio": 1,
        },
        ("vision", "image_segmentation"): {
            "input_size": [48, 48], "normalize": "ai8x", "fold_ratio": 4,
        },
        ("vision", "object_detection"): {
            "input_size": [256, 320], "normalize": "ai8x", "fold_ratio": 1,
        },
        ("vision", "visual_wake_word"): {
            "input_size": [50, 50], "normalize": "int8_unsigned", "grayscale": True,
        },
        ("audio", "audio_denoising"): {
            "sample_rate": 16000, "frame_size": 512, "hop_size": 256, "n_fft": 512,
        },
        ("audio", "keyword_spotting"): {
            "sample_rate": 16000, "n_mels": 40, "window_ms": 40, "hop_ms": 20,
        },
        ("audio", "audio_genre_identification"): {
            "sample_rate": 22050, "n_mels": 128, "n_fft": 2048, "hop_length": 512,
        },
        ("sensor", "anomaly_detection"): {
            "sample_rate": 16000, "n_mels": 64, "n_fft": 1024, "hop_length": 512,
        },
        ("sensor", "motor_fault_detection"): {
            "window_size": 256, "n_axes": 3, "normalize": "int8",
        },
    }
    return preproc_map.get((domain, task), {"normalize": "none"})


def _default_postproc(domain: str, task: str) -> dict:
    postproc_map = {
        ("vision", "image_classification"): {
            "type": "softmax", "top_k": 5,
        },
        ("vision", "image_segmentation"): {
            "type": "argmax", "num_classes": 2,
        },
        ("vision", "object_detection"): {
            "type": "nms", "num_classes": 21, "min_score": 0.3,
            "max_overlap": 0.3, "top_k": 50,
        },
        ("vision", "visual_wake_word"): {
            "type": "binary_threshold", "threshold": 0.5,
        },
        ("audio", "audio_denoising"): {
            "type": "istft", "frame_size": 512, "hop_size": 256,
        },
        ("audio", "keyword_spotting"): {
            "type": "argmax", "num_classes": 12,
        },
        ("audio", "audio_genre_identification"): {
            "type": "argmax", "num_classes": 10,
            "class_names": ["blues", "classical", "country", "disco",
                            "hiphop", "jazz", "metal", "pop", "reggae", "rock"],
        },
        ("sensor", "anomaly_detection"): {
            "type": "reconstruction_error", "threshold": 0.05,
        },
        ("sensor", "motor_fault_detection"): {
            "type": "reconstruction_error", "threshold": 0.1,
        },
    }
    return postproc_map.get((domain, task), {"type": "raw"})


class ArtifactManager:
    def __init__(self, config: dict):
        self.config = config
        deploy_cfg = config.get("deployment", {})
        self.bundle_dir = Path(deploy_cfg.get("bundle_output", "./deploy_bundles"))
        self.bundle_dir.mkdir(parents=True, exist_ok=True)
        self.device = config.get("device", "MAX78002")

    def pack(
        self,
        model_name: str,
        device: str,
        model_files: list,
        version: str,
        domain: str,
        task: str,
        net_def: Optional[str] = None,
        synthesized_dir: Optional[str] = None,
        metrics: Optional[dict] = None,
    ) -> Path:
        """
        Create a deployment bundle .tar.gz.

        model_files: list of paths to weight files (.pth.tar / .tflite / .pt)
        net_def:     path to network definition .py (AI8X only)
        synthesized_dir: path to ai8xize output directory (optional)
        """
        ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        bundle_name = f"{model_name}_{device}_{version}_{ts}.tar.gz"
        bundle_path = self.bundle_dir / bundle_name

        staging = self.bundle_dir / f"_stage_{model_name}_{ts}"
        staging.mkdir(parents=True)

        try:
            weights_dir = staging / "weights"
            weights_dir.mkdir()
            for mf in model_files:
                src = Path(mf)
                if src.exists():
                    shutil.copy2(src, weights_dir / src.name)
                else:
                    print(f"  [WARN] Weight file not found: {src}")

            if net_def:
                nd = Path(net_def)
                if nd.exists():
                    shutil.copy2(nd, staging / nd.name)

            if synthesized_dir:
                synth = Path(synthesized_dir)
                if synth.exists():
                    synth_dest = staging / "synthesized"
                    shutil.copytree(synth, synth_dest)

            preproc = _default_preproc(domain, task)
            with open(staging / "preproc_config.yaml", "w") as f:
                yaml.dump({"domain": domain, "task": task, **preproc}, f)

            postproc = _default_postproc(domain, task)
            with open(staging / "postproc_config.yaml", "w") as f:
                yaml.dump({"domain": domain, "task": task, **postproc}, f)

            meta = {
                "model": model_name,
                "version": version,
                "device": device,
                "domain": domain,
                "task": task,
                "packed_at": datetime.utcnow().isoformat(),
                "metrics": metrics or {},
            }
            with open(staging / "metadata.json", "w") as f:
                json.dump(meta, f, indent=2)

            manifest = {}
            for p in staging.rglob("*"):
                if p.is_file():
                    rel = str(p.relative_to(staging))
                    manifest[rel] = _sha256(p)
            with open(staging / "manifest.json", "w") as f:
                json.dump(manifest, f, indent=2)

            with tarfile.open(bundle_path, "w:gz") as tar:
                tar.add(staging, arcname=f"{model_name}_{version}")

            print(f"[ArtifactManager] Bundle created: {bundle_path}")
        finally:
            shutil.rmtree(staging, ignore_errors=True)

        return bundle_path

    def unpack(self, bundle_path: str, dest: str) -> Path:
        dest_path = Path(dest)
        dest_path.mkdir(parents=True, exist_ok=True)
        with tarfile.open(bundle_path, "r:gz") as tar:
            tar.extractall(dest_path)
        print(f"[ArtifactManager] Unpacked to: {dest_path}")
        return dest_path

    def verify(self, bundle_path: str) -> bool:
        with tarfile.open(bundle_path, "r:gz") as tar:
            names = tar.getnames()
        has_weights = any("weights" in n for n in names)
        has_meta = any("metadata.json" in n for n in names)
        ok = has_weights and has_meta
        print(f"[ArtifactManager] Verify {'PASS' if ok else 'FAIL'}: {Path(bundle_path).name}")
        return ok

    def list_bundles(self) -> list:
        bundles = []
        for p in sorted(self.bundle_dir.glob("*.tar.gz")):
            bundles.append({"path": str(p), "size_mb": round(p.stat().st_size / 1e6, 2)})
        return bundles
