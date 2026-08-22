"""Model registry management for NXP eIQ Model Zoo."""

import os
import subprocess
from pathlib import Path
from typing import Optional

import yaml


class ModelManager:
    """Load and query the NXP eIQ model registry."""

    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        registry_path = Path(config_path).parent / "model_registry.yaml"
        with open(registry_path) as f:
            data = yaml.safe_load(f)
        self.registry = data.get("models", {})
        self.zoo_root = Path(
            self.config.get("zoo_root", ".")
        ).expanduser().resolve()

    # ── lookup ──────────────────────────────────────────────────────────────

    def get_metadata(self, model_id: str) -> dict:
        if model_id not in self.registry:
            raise KeyError(f"Model '{model_id}' not in registry")
        return self.registry[model_id]

    def get_model_path(self, model_id: str, platform: Optional[str] = None) -> Path:
        meta = self.get_metadata(model_id)
        return self.zoo_root / meta["model_path"] / meta["weight_file"]

    def get_recipe_path(self, model_id: str) -> Path:
        meta = self.get_metadata(model_id)
        return self.zoo_root / meta["model_path"] / "recipe.sh"

    def vela_required(self, model_id: str, platform: str) -> bool:
        meta = self.get_metadata(model_id)
        for plat in meta.get("supported_platforms", []):
            if plat["platform"] == platform:
                return bool(plat.get("vela_required", False))
        return False

    def list_models(
        self,
        task: Optional[str] = None,
        domain: Optional[str] = None,
        platform: Optional[str] = None,
        status: Optional[str] = None,
    ) -> list:
        results = []
        for model_id, meta in self.registry.items():
            if task and meta.get("task") != task:
                continue
            if domain and meta.get("domain") != domain:
                continue
            if status and meta.get("status") != status:
                continue
            if platform:
                platforms = [p["platform"] for p in meta.get("supported_platforms", [])]
                if platform not in platforms:
                    continue
            results.append(model_id)
        return sorted(results)

    def get_accuracy(self, model_id: str) -> dict:
        return self.get_metadata(model_id).get("metrics", {})

    def check_accuracy_gate(self, model_id: str, measured: dict) -> bool:
        meta = self.get_metadata(model_id)
        task_key = f"{meta['domain']}_{meta['task']}"
        gates = self.config.get("evaluation", {}).get("gates", {})
        gate = gates.get(task_key)
        if gate is None:
            return True
        metric = gate["metric"]
        threshold = gate["threshold"]
        lower_is_better = gate.get("lower_is_better", False)
        value = measured.get(metric)
        if value is None:
            return True
        if lower_is_better:
            return value <= threshold
        return value >= threshold

    # ── recipe execution ────────────────────────────────────────────────────

    def run_recipe(self, model_id: str, docker_image: str = "nxp-model-zoo") -> int:
        recipe = self.get_recipe_path(model_id)
        if not recipe.exists():
            raise FileNotFoundError(f"recipe.sh not found: {recipe}")
        model_dir = str(recipe.parent)
        cmd = [
            "docker", "run", "--rm",
            "-v", f"{model_dir}:/workspace",
            docker_image,
            "/workspace/recipe.sh",
        ]
        print(f"[recipe] Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=False)
        return result.returncode

    def compile_vela(self, model_id: str, output_dir: Optional[str] = None) -> Path:
        tflite_path = self.get_model_path(model_id)
        if not tflite_path.exists():
            raise FileNotFoundError(f"TFLite model not found: {tflite_path}")
        out_dir = Path(output_dir) if output_dir else tflite_path.parent / "vela_output"
        out_dir.mkdir(parents=True, exist_ok=True)
        vela_cfg = self.config.get("vela", {})
        cmd = [
            "vela",
            str(tflite_path),
            "--output-dir", str(out_dir),
            "--accelerator-config", vela_cfg.get("accelerator_config", "ethos-u65-256"),
            "--optimise", vela_cfg.get("optimise", "Performance"),
            "--system-config", vela_cfg.get("system_config", "Ethos_U65_High_End"),
        ]
        print(f"[vela] Compiling: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        stem = tflite_path.stem
        return out_dir / f"{stem}_vela.tflite"

    def summary(self) -> None:
        totals: dict = {}
        for meta in self.registry.values():
            domain = meta.get("domain", "unknown")
            totals[domain] = totals.get(domain, 0) + 1
        print(f"NXP eIQ Model Registry — {len(self.registry)} models")
        for domain, count in sorted(totals.items()):
            print(f"  {domain}: {count}")
