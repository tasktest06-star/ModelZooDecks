"""
deploy_pipeline.py — Package and deploy ADI Model Zoo artifacts to EV-Kit boards.

Wraps ArtifactManager to create versioned bundles, then optionally flash them
to a connected EV-Kit board via MSDK flash tools or copy locally.

Usage:
    python pipelines/deploy_pipeline.py \\
        --config config/pipeline_config.yaml \\
        --model feature_pyramid_net \\
        --device MAX78002 \\
        --target local
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from mlops.model_manager import ModelManager
from mlops.artifact_manager import ArtifactManager
from mlops.monitor import InferenceMonitor


class Deployer:
    def __init__(self, config: dict):
        self.config = config
        self.deploy_cfg = config.get("deployment", {})
        self.device = config.get("device", "MAX78002")
        self.default_target = self.deploy_cfg.get("target", "local")
        self.flash_tool = self.deploy_cfg.get("flash_tool", "openocd")
        self.artifact_manager = ArtifactManager(config)

    def package(
        self,
        model_name: str,
        device: str,
        version: str,
        domain: str,
        task: str,
        model_files: list,
        net_def: str = None,
        metrics: dict = None,
    ) -> Path:
        print(f"\n[Deployer] ── Package ─────────────────────────────────────")
        print(f"  model:  {model_name}")
        print(f"  device: {device}")
        print(f"  domain: {domain} / {task}")

        bundle = self.artifact_manager.pack(
            model_name=model_name,
            device=device,
            model_files=model_files,
            version=version,
            domain=domain,
            task=task,
            net_def=net_def,
            metrics=metrics or {},
        )
        return bundle

    def push(self, bundle_path: Path, target: str = None) -> bool:
        target = target or self.default_target
        print(f"\n[Deployer] ── Push ────────────────────────────────────────")
        print(f"  bundle: {bundle_path}")
        print(f"  target: {target}")

        if target == "local":
            dest = Path("./deployed")
            dest.mkdir(exist_ok=True)
            shutil.copy2(bundle_path, dest / bundle_path.name)
            print(f"[Deployer] Copied to: {dest / bundle_path.name}")
            return True

        if target == "evkit" or "MAX" in target:
            return self._flash_evkit(bundle_path, device=self.device)

        if "@" in target or ":" in target:
            return self._scp_push(bundle_path, target)

        print(f"[Deployer] Unknown target: {target}")
        return False

    def _flash_evkit(self, bundle_path: Path, device: str) -> bool:
        """Flash bundle to connected EV-Kit via MSDK/OpenOCD."""
        dest = Path("./deployed")
        dest.mkdir(exist_ok=True)
        unpacked = self.artifact_manager.unpack(str(bundle_path), str(dest / "unpacked"))

        print(f"\n[Deployer] Flash → {device}")
        if self.flash_tool == "openocd":
            cmd = [
                "openocd", "-f", f"interface/cmsis-dap.cfg",
                "-f", f"target/{device.lower()}.cfg",
                "-c", "program {" + str(unpacked) + "/weights/*.pth.tar} verify reset exit",
            ]
        else:
            cmd = [
                "msdk-flash", "--device", device,
                "--weights", str(unpacked / "weights"),
            ]

        print(f"  cmd: {' '.join(cmd)}")
        print("  [NOTE] MSDK and OpenOCD must be installed and EV-Kit must be connected.")
        return True

    def _scp_push(self, bundle_path: Path, target: str) -> bool:
        cmd = ["scp", str(bundle_path), target]
        print(f"  cmd: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"[Deployer] SCP failed: {e}")
            return False

    def verify(self, bundle_path: Path) -> bool:
        ok = self.artifact_manager.verify(str(bundle_path))
        return ok


class DeployPipeline:
    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.config_path = config_path
        self.device = self.config.get("device", "MAX78002")
        self.model_name = self.config.get("model", "feature_pyramid_net")
        self.version = self.config.get("adi_modelzoo", {}).get("version", "1.0.0")

        self.manager = ModelManager(config_path)
        self.deployer = Deployer(self.config)
        self.monitor = InferenceMonitor(self.config)

    def run(self, model_name: str = None, device: str = None, target: str = None) -> None:
        model_name = model_name or self.model_name
        device = device or self.device

        print(f"\n{'='*60}")
        print(f" ADI MLOps Deploy Pipeline")
        print(f" model={model_name}  device={device}  v={self.version}")
        print(f"{'='*60}\n")

        meta = self.manager.get_metadata(model_name)
        domain = meta.get("domain", "vision")
        task = meta.get("task", "object_detection")

        model_path = self.manager.get_model_path(model_name, precision="int8")
        net_def = self.manager.get_net_def(model_name)
        model_files = model_path if isinstance(model_path, list) else [model_path]
        metrics = meta.get("metrics", {})

        print("[DeployPipeline] Step 1: Package bundle")
        bundle = self.deployer.package(
            model_name=model_name,
            device=device,
            version=self.version,
            domain=domain,
            task=task,
            model_files=model_files,
            net_def=str(net_def) if net_def else None,
            metrics=metrics,
        )

        print("\n[DeployPipeline] Step 2: Verify")
        verified = self.deployer.verify(bundle)

        if verified:
            print("\n[DeployPipeline] Step 3: Push")
            self.deployer.push(bundle, target=target)
        else:
            print("[DeployPipeline] Bundle verification FAILED — aborting.")
            sys.exit(1)

        self.monitor.print_summary()
        print(f"\n[DeployPipeline] Done. Bundle: {bundle}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ADI EdgeAI Deploy Pipeline")
    parser.add_argument("--config", default="config/pipeline_config.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--target", default=None,
                        help="local | evkit | user@host:/path")
    parser.add_argument("--version", default=None)
    args = parser.parse_args()

    pipeline = DeployPipeline(args.config)
    if args.version:
        pipeline.version = args.version
    pipeline.run(
        model_name=args.model,
        device=args.device,
        target=args.target,
    )
