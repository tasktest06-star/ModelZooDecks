"""
artifact_manager.py — Package, store, and retrieve TIDL compiled artifact bundles.

A deployment bundle contains everything needed to run inference on a TI SoC:
  compiled TIDL network files, param.yaml, preprocessing config,
  post-processing params, and version metadata.
"""

import json
import shutil
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml


BUNDLE_MANIFEST_FILE = "metadata.json"


class ArtifactManager:
    """
    Packs TIDL artifacts + config into versioned deployment bundles.

    Usage:
        am = ArtifactManager(config)
        bundle_path = am.pack(
            model_id="yolox-s-lite", soc="AM68A",
            artifact_dir="./compiled/AM68A/ONR-OD-8220-...",
            version="11.2.0")
        am.unpack(bundle_path, dest="./deploy")
        am.list_bundles()
    """

    def __init__(self, config: dict):
        self.bundle_dir = Path(config.get("deployment", {})
                               .get("bundle_output", "./deploy_bundles"))
        self.bundle_dir.mkdir(parents=True, exist_ok=True)

    def pack(
        self,
        model_id: str,
        soc: str,
        artifact_dir: str,
        version: str,
        task: str = "",
        metrics: Optional[dict] = None,
        preproc_config: Optional[dict] = None,
        postproc_config: Optional[dict] = None,
    ) -> Path:
        """
        Create a .tar.gz deployment bundle from a compiled TIDL artifact directory.
        Returns path to the bundle archive.
        """
        artifact_dir = Path(artifact_dir)
        if not artifact_dir.exists():
            raise FileNotFoundError(f"Artifact directory not found: {artifact_dir}")

        ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        bundle_name = f"{model_id.replace('/', '-')}_{soc}_{version}_{ts}"
        staging = self.bundle_dir / bundle_name
        staging.mkdir(parents=True, exist_ok=True)

        # Copy TIDL artifact files
        shutil.copytree(artifact_dir, staging / "tidl_artifacts",
                        dirs_exist_ok=True)

        # Write preprocessing config
        preproc = preproc_config or _default_preproc(task)
        with open(staging / "preproc_config.yaml", "w") as f:
            yaml.dump(preproc, f, default_flow_style=False)

        # Write post-processing config
        postproc = postproc_config or _default_postproc(task)
        with open(staging / "postproc_config.yaml", "w") as f:
            yaml.dump(postproc, f, default_flow_style=False)

        # Write bundle metadata
        metadata = {
            "model_id": model_id,
            "soc": soc,
            "version": version,
            "task": task,
            "packed_at": datetime.utcnow().isoformat(),
            "metrics": metrics or {},
            "artifact_dir": str(artifact_dir),
        }
        with open(staging / BUNDLE_MANIFEST_FILE, "w") as f:
            json.dump(metadata, f, indent=2)

        # Create archive
        archive = self.bundle_dir / f"{bundle_name}.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(staging, arcname=bundle_name)
        shutil.rmtree(staging)

        print(f"[ArtifactManager] Bundle packed: {archive}")
        return archive

    def unpack(self, bundle_path: str, dest: str) -> Path:
        """Extract a bundle archive to dest/. Returns the extracted directory."""
        archive = Path(bundle_path)
        if not archive.exists():
            raise FileNotFoundError(f"Bundle not found: {archive}")
        dest_path = Path(dest)
        dest_path.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(dest_path)
        extracted_dirs = [d for d in dest_path.iterdir() if d.is_dir()]
        result = extracted_dirs[0] if extracted_dirs else dest_path
        print(f"[ArtifactManager] Unpacked to: {result}")
        return result

    def list_bundles(self) -> list[dict]:
        """List all packed bundles and their metadata."""
        bundles = []
        for archive in sorted(self.bundle_dir.glob("*.tar.gz")):
            bundles.append({
                "name": archive.stem.replace(".tar", ""),
                "path": str(archive),
                "size_mb": round(archive.stat().st_size / 1024 / 1024, 2),
            })
        return bundles

    def get_latest(self, model_id: str, soc: str) -> Optional[Path]:
        """Return the most recent bundle for model_id + soc."""
        pattern = f"{model_id.replace('/', '-')}_{soc}_*.tar.gz"
        candidates = sorted(self.bundle_dir.glob(pattern))
        return candidates[-1] if candidates else None

    def verify(self, bundle_path: str) -> bool:
        """Basic integrity check: ensure archive is readable and manifest exists."""
        try:
            with tarfile.open(bundle_path, "r:gz") as tar:
                names = tar.getnames()
            has_manifest = any(BUNDLE_MANIFEST_FILE in n for n in names)
            has_tidl = any("tidl_artifacts" in n for n in names)
            if not has_manifest:
                print(f"[ArtifactManager] WARN: No {BUNDLE_MANIFEST_FILE} in bundle")
            if not has_tidl:
                print("[ArtifactManager] WARN: No tidl_artifacts in bundle")
            return has_manifest and has_tidl
        except Exception as e:
            print(f"[ArtifactManager] Bundle verification failed: {e}")
            return False


# ── Default config helpers ───────────────────────────────────────────────────────

def _default_preproc(task: str) -> dict:
    task_preproc = {
        "classification": {
            "input_size": [224, 224], "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225], "letterbox": False,
            "normalize_0_255": False,
        },
        "object_detection": {
            "input_size": [640, 640], "mean": [0.0, 0.0, 0.0],
            "std": [1.0, 1.0, 1.0], "letterbox": True,
            "normalize_0_255": True,
        },
        "segmentation": {
            "input_size": [512, 512], "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225], "letterbox": False,
            "normalize_0_255": False, "ignore_index": 255,
        },
    }
    return task_preproc.get(task, task_preproc["classification"])


def _default_postproc(task: str) -> dict:
    task_postproc = {
        "classification": {"top_k": 5, "softmax": True},
        "object_detection": {
            "num_classes": 80, "conf_threshold": 0.3,
            "nms_threshold": 0.45, "top_k": 200,
            "class_labels": "coco80",
        },
        "segmentation": {
            "num_classes": 32, "ignore_index": 255,
            "class_labels": "ade20k32",
        },
        "keypoint": {
            "num_keypoints": 17, "conf_threshold": 0.3,
            "skeleton": "coco17",
        },
    }
    return task_postproc.get(task, {})
