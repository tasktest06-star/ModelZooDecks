"""Docker-based recipe runner for NXP eIQ Model Zoo."""

import os
import subprocess
from pathlib import Path


class RecipeRunner:
    """Wraps the NXP eIQ recipe.sh / Docker workflow in a Python API."""

    def __init__(self, docker_image="nxp-model-zoo", zoo_root="."):
        self.docker_image = docker_image
        self.zoo_root = Path(zoo_root).resolve()

    def build_image(self, dockerfile_dir=None):
        """Build the nxp-model-zoo Docker image."""
        ctx = str(dockerfile_dir or self.zoo_root)
        cmd = ["docker", "build", "-t", self.docker_image, ctx]
        print(f"[recipe] Building Docker image: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"Docker build failed (exit {result.returncode})")

    def run(self, model_path, timeout=600):
        """Run recipe.sh for a model directory.

        Args:
            model_path: path relative to zoo_root (e.g. tasks/vision/classification/mobilenetv2)
            timeout:    seconds before aborting
        Returns:
            Path to the generated .tflite file (first match in model_path)
        """
        abs_model_dir = (self.zoo_root / model_path).resolve()
        recipe = abs_model_dir / "recipe.sh"
        if not recipe.exists():
            raise FileNotFoundError(f"No recipe.sh found at: {recipe}")

        cmd = [
            "docker", "run", "--rm",
            "-v", f"{abs_model_dir}:/workspace",
            self.docker_image,
            "/workspace/recipe.sh",
        ]
        print(f"[recipe] Running recipe for: {model_path}")
        result = subprocess.run(cmd, check=False, timeout=timeout)
        if result.returncode != 0:
            raise RuntimeError(
                f"Recipe failed (exit {result.returncode}) for {model_path}"
            )

        tflite_files = list(abs_model_dir.glob("*.tflite"))
        if not tflite_files:
            raise FileNotFoundError(
                f"Recipe completed but no .tflite found in {abs_model_dir}"
            )
        return tflite_files[0]

    def run_all(self, model_paths, continue_on_error=True):
        """Run recipes for multiple model paths.

        Returns:
            dict mapping model_path → (tflite_path | Exception)
        """
        results = {}
        for mp in model_paths:
            try:
                results[mp] = self.run(mp)
                print(f"[recipe] OK: {mp}")
            except Exception as exc:
                print(f"[recipe] FAILED: {mp} — {exc}")
                results[mp] = exc
                if not continue_on_error:
                    raise
        return results
