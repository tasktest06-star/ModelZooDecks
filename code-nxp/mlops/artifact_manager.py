"""Artifact packaging and deployment bundle management for NXP eIQ models."""

import json
import os
import tarfile
from pathlib import Path
from typing import Optional


class ArtifactManager:
    def __init__(self, config: dict):
        self.config = config
        artifact_cfg = config.get("artifact", {})
        self.output_dir = Path(artifact_cfg.get("output_dir", "deployed"))
        self.bundle_format = artifact_cfg.get("bundle_format", "tgz")
        self.include_recipe = artifact_cfg.get("include_recipe", True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_bundle(
        self,
        model_id: str,
        tflite_path: str,
        metadata: dict,
        recipe_path: Optional[str] = None,
        vela_path: Optional[str] = None,
    ) -> Path:
        tflite_path = Path(tflite_path)
        bundle_name = f"{model_id}_bundle"
        bundle_dir = self.output_dir / bundle_name
        bundle_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "model_id": model_id,
            "task": metadata.get("task"),
            "domain": metadata.get("domain"),
            "format": "tflite_int8",
            "input_size": metadata.get("input_size"),
            "num_classes": metadata.get("num_classes"),
            "supported_platforms": [
                p["platform"] for p in metadata.get("supported_platforms", [])
            ],
            "metrics": metadata.get("metrics", {}),
            "files": {},
        }

        # Copy TFLite model
        import shutil
        dst = bundle_dir / tflite_path.name
        shutil.copy2(tflite_path, dst)
        manifest["files"]["tflite"] = tflite_path.name

        # Copy Vela compiled model
        if vela_path:
            vela_path = Path(vela_path)
            if vela_path.exists():
                vela_dst = bundle_dir / vela_path.name
                shutil.copy2(vela_path, vela_dst)
                manifest["files"]["vela"] = vela_path.name

        # Copy recipe
        if self.include_recipe and recipe_path:
            recipe_path = Path(recipe_path)
            if recipe_path.exists():
                shutil.copy2(recipe_path, bundle_dir / "recipe.sh")
                manifest["files"]["recipe"] = "recipe.sh"

        # Write manifest
        manifest_path = bundle_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        manifest["files"]["manifest"] = "manifest.json"

        # Write inference example
        self._write_inference_example(bundle_dir, model_id, metadata)
        manifest["files"]["inference_example"] = "inference_example.py"

        # Update manifest with all files
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        # Pack archive
        archive_path = self.output_dir / f"{bundle_name}.{self.bundle_format}"
        if self.bundle_format == "tgz":
            with tarfile.open(archive_path, "w:gz") as tar:
                tar.add(bundle_dir, arcname=bundle_name)
        else:
            import shutil as sh
            sh.make_archive(str(self.output_dir / bundle_name), "zip", bundle_dir)
            archive_path = self.output_dir / f"{bundle_name}.zip"

        shutil.rmtree(bundle_dir)
        print(f"[artifact] Bundle created: {archive_path}")
        return archive_path

    def verify_bundle(self, bundle_path: str) -> bool:
        bundle_path = Path(bundle_path)
        if not bundle_path.exists():
            return False
        try:
            with tarfile.open(bundle_path, "r:gz") as tar:
                names = tar.getnames()
                has_tflite = any(n.endswith(".tflite") for n in names)
                has_manifest = any("manifest.json" in n for n in names)
                return has_tflite and has_manifest
        except Exception as e:
            print(f"[artifact] Bundle verify failed: {e}")
            return False

    def extract_bundle(self, bundle_path: str, output_dir: Optional[str] = None) -> Path:
        bundle_path = Path(bundle_path)
        out = Path(output_dir) if output_dir else bundle_path.parent / bundle_path.stem
        out.mkdir(parents=True, exist_ok=True)
        with tarfile.open(bundle_path, "r:gz") as tar:
            tar.extractall(out)
        print(f"[artifact] Extracted to: {out}")
        return out

    def list_bundles(self) -> list:
        return sorted(
            str(p) for p in self.output_dir.glob("*.tgz")
        ) + sorted(str(p) for p in self.output_dir.glob("*.zip"))

    def _write_inference_example(self, bundle_dir: Path, model_id: str, metadata: dict) -> None:
        task = metadata.get("task", "image_classification")
        input_size = metadata.get("input_size", [224, 224])
        example = f'''"""Inference example for {model_id}."""
import numpy as np

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite


def run_{model_id.replace("-", "_")}(model_path: str, input_data: np.ndarray):
    interp = tflite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    inp_det = interp.get_input_details()
    out_det = interp.get_output_details()
    interp.set_tensor(inp_det[0]["index"], input_data)
    interp.invoke()
    return [interp.get_tensor(d["index"]) for d in out_det]


if __name__ == "__main__":
    import glob
    model_file = glob.glob("*.tflite")[0]
    # {task} — input shape {input_size}
    dummy = np.zeros([1] + {list(input_size)}, dtype=np.float32)
    outputs = run_{model_id.replace("-", "_")}(model_file, dummy)
    print("Output shapes:", [o.shape for o in outputs])
'''
        (bundle_dir / "inference_example.py").write_text(example)
