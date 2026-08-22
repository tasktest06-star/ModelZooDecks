"""
deploy_pipeline.py — Package compiled TIDL artifacts and push to target EVM board.

Wraps ArtifactManager to create versioned deployment bundles and optionally
push them to a remote EVM board over SCP/SSH or to a local directory.

Usage:
    python pipelines/deploy_pipeline.py \\
        --config config/pipeline_config.yaml \\
        --model yolox-s-lite \\
        --soc AM68A \\
        --version 11.2.0
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from mlops.model_manager import ModelManager
from mlops.artifact_manager import ArtifactManager
from mlops.monitor import InferenceMonitor


class Deployer:
    """
    Packages a TIDL artifact + configs into a versioned bundle and
    pushes it to a local directory or remote EVM board.

    Usage:
        deployer = Deployer(config)
        bundle = deployer.package("yolox-s-lite", soc="AM68A", version="11.2.0")
        deployer.push(bundle, target="local")
        deployer.verify(bundle)
    """

    def __init__(self, config: dict):
        self.config = config
        self.deploy_cfg = config.get("deployment", {})
        self.default_target = self.deploy_cfg.get("target", "local")
        self.bundle_dir = Path(self.deploy_cfg.get("bundle_output", "./deploy_bundles"))
        self.artifact_dir = Path(config.get("modelzoo", {}).get("artifact_dir", "./artifacts"))

        self.artifact_manager = ArtifactManager(config)
        self.manager = ModelManager.__new__(ModelManager)  # lazy init

    def package(
        self,
        model_name: str,
        soc: str,
        version: str,
        metrics: Optional[dict] = None,
    ) -> Path:
        """
        Create a deployment bundle for model_name on soc.

        The artifact directory is expected at:
            artifacts/<soc>/<artifact_id>/

        Returns path to the .tar.gz bundle.
        """
        print(f"\n[Deployer] ── Package ───────────────────────────────────")
        print(f"  model:   {model_name}")
        print(f"  soc:     {soc}")
        print(f"  version: {version}")

        # Resolve artifact directory
        possible_dirs = list((self.artifact_dir / soc).glob(f"*{model_name.replace('-', '*')}*"))
        if possible_dirs:
            artifact_dir = possible_dirs[0]
        else:
            artifact_dir = self.artifact_dir / soc / f"artifact_{model_name}"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            print(f"  [WARN] Artifact dir not found, using placeholder: {artifact_dir}")

        # Infer task from model name
        task_hints = {
            "yolox": "object_detection", "ssd": "object_detection",
            "deeplabv3": "segmentation", "fpnlite": "segmentation",
            "mobilenet": "classification", "efficientnet": "classification",
            "resnet": "classification", "swin": "classification",
            "midas": "depth_estimation", "pose": "keypoint",
        }
        task = next((t for k, t in task_hints.items() if k in model_name.lower()),
                    "object_detection")

        bundle = self.artifact_manager.pack(
            model_id=model_name,
            soc=soc,
            artifact_dir=str(artifact_dir),
            version=version,
            task=task,
            metrics=metrics or {},
        )
        print(f"[Deployer] Bundle created: {bundle}\n")
        return bundle

    def push(
        self,
        bundle_path: Path,
        target: Optional[str] = None,
    ) -> bool:
        """
        Push bundle to target.

        target="local"         → copy to ./deployed/
        target="user@host:/path" → scp to remote EVM board
        """
        target = target or self.default_target
        print(f"[Deployer] ── Push ─────────────────────────────────────")
        print(f"  bundle: {bundle_path}")
        print(f"  target: {target}")

        if target == "local":
            dest = Path("./deployed")
            dest.mkdir(exist_ok=True)
            import shutil
            shutil.copy2(bundle_path, dest / bundle_path.name)
            print(f"[Deployer] Pushed to local: {dest / bundle_path.name}")
            return True

        # Remote EVM board via SCP
        if ":" in target or "@" in target:
            cmd = ["scp", str(bundle_path), target]
            print(f"  cmd: {' '.join(cmd)}")
            try:
                subprocess.run(cmd, check=True)
                print("[Deployer] SCP transfer complete.")
                return True
            except subprocess.CalledProcessError as e:
                print(f"[Deployer] SCP failed: {e}")
                return False
            except FileNotFoundError:
                print("[Deployer] WARN: scp not found. Install OpenSSH.")
                return False

        print(f"[Deployer] Unknown target format: {target}")
        return False

    def verify(self, bundle_path: Path) -> bool:
        """Verify bundle integrity."""
        ok = self.artifact_manager.verify(str(bundle_path))
        status = "PASS" if ok else "FAIL"
        print(f"[Deployer] Bundle verify: {status} — {bundle_path.name}")
        return ok

    def list_deployed(self) -> list[dict]:
        """List all available deployment bundles."""
        return self.artifact_manager.list_bundles()


class DeployPipeline:
    """
    Full deployment pipeline: download artifact → package bundle → push → verify → monitor.
    """

    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.config_path = config_path
        self.soc = self.config.get("soc", "AM68A")
        self.model_name = self.config.get("model", "yolox-s-lite")
        self.version = self.config.get("modelzoo", {}).get("version", "11.2.0")

        self.manager = ModelManager(config_path)
        self.deployer = Deployer(self.config)
        self.monitor = InferenceMonitor(self.config)

    def run(
        self,
        model_name: Optional[str] = None,
        soc: Optional[str] = None,
        push_target: Optional[str] = None,
    ) -> None:
        model_name = model_name or self.model_name
        soc = soc or self.soc

        print(f"\n{'='*60}")
        print(f" TI EdgeAI MLOps Deploy Pipeline")
        print(f" model={model_name}  SoC={soc}  v={self.version}")
        print(f"{'='*60}\n")

        # 1. Download pre-compiled artifact
        print("[DeployPipeline] Step 1: Download artifact")
        try:
            artifact_dir = self.manager.download_model(model_name, soc=soc)
        except Exception as e:
            print(f"[DeployPipeline] WARN: Download failed ({e}). "
                  "Proceeding with existing cache.")
            artifact_dir = Path("./artifacts") / soc / model_name

        # 2. Get accuracy metrics from registry
        try:
            meta = self.manager.get_metadata(model_name)
            metrics = meta.get("metrics", {})
        except KeyError:
            metrics = {}

        # 3. Package into deployment bundle
        print("\n[DeployPipeline] Step 2: Package bundle")
        bundle = self.deployer.package(
            model_name=model_name, soc=soc,
            version=self.version, metrics=metrics,
        )

        # 4. Verify bundle
        print("\n[DeployPipeline] Step 3: Verify")
        verified = self.deployer.verify(bundle)

        # 5. Push
        if verified:
            print("\n[DeployPipeline] Step 4: Push")
            target = push_target or self.config.get("deployment", {}).get("target", "local")
            self.deployer.push(bundle, target=target)
        else:
            print("[DeployPipeline] Bundle verification FAILED — aborting push.")
            sys.exit(1)

        # 6. Monitor summary (if any existing log)
        self.monitor.print_summary()

        print(f"\n[DeployPipeline] Done. Bundle: {bundle}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TI EdgeAI Deploy Pipeline")
    parser.add_argument("--config",  default="config/pipeline_config.yaml")
    parser.add_argument("--model",   default=None)
    parser.add_argument("--soc",     default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--target",  default=None,
                        help="Push target: 'local' or 'user@host:/path'")
    args = parser.parse_args()

    pipeline = DeployPipeline(args.config)
    if args.version:
        pipeline.version = args.version

    pipeline.run(
        model_name=args.model,
        soc=args.soc,
        push_target=args.target,
    )
