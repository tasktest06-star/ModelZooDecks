"""Orchestrate Docker-based recipe execution for one or all NXP eIQ models."""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from mlops.model_manager import ModelManager


def run_recipe_pipeline(
    config_path: str,
    model_id: str,
    docker_image: str = "nxp-model-zoo",
    compile_vela: bool = False,
    platform: str = "imx8mplus",
) -> int:
    manager = ModelManager(config_path)

    # Validate model exists
    meta = manager.get_metadata(model_id)
    print(f"\n{'='*60}")
    print(f"Recipe Pipeline: {model_id}")
    print(f"  Task   : {meta['task']}")
    print(f"  Domain : {meta['domain']}")
    print(f"  Format : {meta['format']}")
    print(f"{'='*60}")

    # Run recipe
    rc = manager.run_recipe(model_id, docker_image=docker_image)
    if rc != 0:
        print(f"[pipeline] Recipe failed for {model_id} (exit {rc})")
        return rc

    # Optionally compile with Vela
    if compile_vela or manager.vela_required(model_id, platform):
        print(f"[pipeline] Vela compilation required for {platform}")
        try:
            vela_out = manager.compile_vela(model_id)
            print(f"[pipeline] Vela output: {vela_out}")
        except Exception as e:
            print(f"[pipeline] Vela compilation failed: {e}")
            return 1

    print(f"[pipeline] Recipe pipeline complete for {model_id}")
    return 0


def run_all_recipes(
    config_path: str,
    docker_image: str = "nxp-model-zoo",
    domain: str = None,
    task: str = None,
) -> dict:
    manager = ModelManager(config_path)
    model_ids = manager.list_models(domain=domain, task=task)
    results = {}
    for model_id in model_ids:
        print(f"\n[pipeline] Processing: {model_id}")
        rc = run_recipe_pipeline(config_path, model_id, docker_image=docker_image)
        results[model_id] = "success" if rc == 0 else "failed"
    success = sum(1 for v in results.values() if v == "success")
    print(f"\n[pipeline] Summary: {success}/{len(results)} recipes succeeded")
    for mid, status in results.items():
        print(f"  {'✓' if status == 'success' else '✗'} {mid}: {status}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NXP eIQ Recipe Pipeline")
    parser.add_argument("--config", default="config/pipeline_config.yaml")
    parser.add_argument("--model", help="Single model ID (omit for all)")
    parser.add_argument("--domain", help="Filter by domain")
    parser.add_argument("--task", help="Filter by task")
    parser.add_argument("--docker-image", default="nxp-model-zoo")
    parser.add_argument("--vela", action="store_true", help="Run Vela compilation")
    parser.add_argument("--platform", default="imx8mplus")
    args = parser.parse_args()

    if args.model:
        rc = run_recipe_pipeline(
            args.config,
            args.model,
            docker_image=args.docker_image,
            compile_vela=args.vela,
            platform=args.platform,
        )
        sys.exit(rc)
    else:
        run_all_recipes(
            args.config,
            docker_image=args.docker_image,
            domain=args.domain,
            task=args.task,
        )
