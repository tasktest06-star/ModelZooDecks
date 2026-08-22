"""Deployment pipeline: bundle packaging and optional board copy for NXP eIQ models."""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from mlops.model_manager import ModelManager
from mlops.artifact_manager import ArtifactManager


def deploy_model(
    config_path: str,
    model_id: str,
    platform: str = "imx8mplus",
    board_host: Optional[str] = None,
    board_path: str = "/home/root/models/",
) -> dict:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    manager = ModelManager(config_path)
    artifact_mgr = ArtifactManager(config)
    meta = manager.get_metadata(model_id)

    print(f"\n{'='*60}")
    print(f"Deploy Pipeline: {model_id} → {platform}")
    print(f"{'='*60}")

    # Locate TFLite model
    tflite_path = manager.get_model_path(model_id, platform)
    if not tflite_path.exists():
        print(f"[deploy] TFLite not found: {tflite_path}")
        print(f"[deploy] Run recipe first: python pipelines/recipe_pipeline.py --model {model_id}")
        return {"status": "model_not_found", "model_id": model_id}

    # Vela compilation if required
    vela_path = None
    if manager.vela_required(model_id, platform):
        print(f"[deploy] Running Vela compilation for {platform}")
        try:
            vela_path = manager.compile_vela(model_id)
            print(f"[deploy] Vela output: {vela_path}")
        except Exception as e:
            print(f"[deploy] Vela compilation skipped (Vela not installed?): {e}")

    # Get recipe path
    recipe_path = manager.get_recipe_path(model_id)
    recipe_str = str(recipe_path) if recipe_path.exists() else None

    # Create bundle
    bundle_path = artifact_mgr.create_bundle(
        model_id=model_id,
        tflite_path=str(tflite_path),
        metadata=meta,
        recipe_path=recipe_str,
        vela_path=str(vela_path) if vela_path else None,
    )

    # Verify bundle
    if not artifact_mgr.verify_bundle(str(bundle_path)):
        print(f"[deploy] Bundle verification FAILED: {bundle_path}")
        return {"status": "bundle_verify_failed", "bundle": str(bundle_path)}

    print(f"[deploy] Bundle verified: {bundle_path}")

    # Copy to board via SCP
    if board_host:
        scp_cmd = [
            "scp",
            str(bundle_path),
            f"root@{board_host}:{board_path}",
        ]
        print(f"[deploy] Copying to board: {' '.join(scp_cmd)}")
        rc = subprocess.run(scp_cmd, check=False).returncode
        if rc != 0:
            print(f"[deploy] SCP failed (exit {rc})")
            return {"status": "scp_failed", "bundle": str(bundle_path)}
        print(f"[deploy] Copied to {board_host}:{board_path}")

    return {
        "status": "success",
        "model_id": model_id,
        "platform": platform,
        "bundle": str(bundle_path),
        "vela_compiled": vela_path is not None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NXP eIQ Deploy Pipeline")
    parser.add_argument("--config", default="config/pipeline_config.yaml")
    parser.add_argument("--model", required=True)
    parser.add_argument("--platform", default="imx8mplus")
    parser.add_argument("--board-host", help="Board IP/hostname for SCP (optional)")
    parser.add_argument("--board-path", default="/home/root/models/")
    args = parser.parse_args()

    result = deploy_model(
        args.config,
        args.model,
        platform=args.platform,
        board_host=args.board_host,
        board_path=args.board_path,
    )
    print(f"\n[deploy] Result: {result}")
    sys.exit(0 if result.get("status") == "success" else 1)
