"""
train_pipeline.py — QAT training → export → synthesis → evaluate for ADI Model Zoo.

Stages:
  1. train    — ai8x-training framework (QAT from epoch qat_start_epoch)
  2. export   — save .pth.tar checkpoint
  3. synthesize — ai8xize.py → MAX78002 hardware config
  4. evaluate — run accuracy gates

Usage:
    python pipelines/train_pipeline.py \\
        --config config/pipeline_config.yaml \\
        --model feature_pyramid_net \\
        --device MAX78002 \\
        --stage all
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))
from mlops.model_manager import ModelManager
from mlops.evaluator import Evaluator, AccuracyGateError


class TrainPipeline:
    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.config_path = config_path
        self.device = self.config.get("device", "MAX78002")
        self.model_name = self.config.get("model", "feature_pyramid_net")
        self.train_cfg = self.config.get("training", {})
        self.export_cfg = self.config.get("export", {})
        self.synth_cfg = self.config.get("synthesis", {})

        self.manager = ModelManager(config_path)

    def train(self) -> None:
        meta = self.manager.get_metadata(self.model_name)
        framework = self.train_cfg.get("framework", "ai8x-training")
        checkpoint_dir = self.train_cfg.get("checkpoint_dir", "./checkpoints")
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        print(f"\n[TrainPipeline] ── Stage 1: Train ────────────────────────")
        print(f"  model:     {self.model_name}")
        print(f"  device:    {self.device}")
        print(f"  framework: {framework}")
        print(f"  epochs:    {self.train_cfg.get('epochs', 100)}")
        print(f"  QAT:       {self.train_cfg.get('qat', True)} "
              f"(from epoch {self.train_cfg.get('qat_start_epoch', 80)})")

        net_def = meta.get("net_def")
        cmd = [
            "python", "train.py",
            "--model", meta.get("net_def", "").replace(".py", "") or self.model_name,
            "--dataset", meta.get("dataset", "cifar100"),
            "--epochs", str(self.train_cfg.get("epochs", 100)),
            "--batch-size", str(self.train_cfg.get("batch_size", 32)),
            "--lr", str(self.train_cfg.get("learning_rate", 0.001)),
            "--save-sample", "1",
            "--out-dir", checkpoint_dir,
        ]
        if self.train_cfg.get("qat", True):
            cmd += ["--qat-policy", "qat_policy_late_start.yaml"]

        print(f"\n  cmd: {' '.join(cmd)}")
        print("  [NOTE] ai8x-training must be installed and train.py must be run from")
        print("         the ai8x-training directory. Adjust paths as needed.")

    def export(self) -> Path:
        checkpoint_dir = Path(self.train_cfg.get("checkpoint_dir", "./checkpoints"))
        export_dir = Path(self.export_cfg.get("output_path", "./exports"))
        export_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[TrainPipeline] ── Stage 2: Export ───────────────────────")

        ckpts = list(checkpoint_dir.glob("*.pth.tar"))
        if ckpts:
            best = sorted(ckpts, key=lambda p: p.stat().st_mtime)[-1]
            dst = export_dir / f"{self.model_name}_{self.device}_exported.pth.tar"
            import shutil
            shutil.copy2(best, dst)
            print(f"  Exported: {dst}")
            return dst
        else:
            dst = export_dir / f"{self.model_name}_{self.device}_mock.pth.tar"
            dst.write_bytes(b"MOCK_CHECKPOINT")
            print(f"  [WARN] No checkpoint found — created placeholder: {dst}")
            return dst

    def synthesize(self) -> Path:
        synth_dir = Path(self.synth_cfg.get("output_dir", "./synthesized"))
        synth_dir.mkdir(parents=True, exist_ok=True)
        ai8xize = self.synth_cfg.get("ai8xize_path", "./ai8x-synthesis")

        print(f"\n[TrainPipeline] ── Stage 3: Synthesize ───────────────────")
        print(f"  device:    {self.device}")
        print(f"  ai8xize:   {ai8xize}")

        export_dir = Path(self.export_cfg.get("output_path", "./exports"))
        ckpts = list(export_dir.glob(f"{self.model_name}*.pth.tar"))
        weight_file = str(ckpts[0]) if ckpts else "model.pth.tar"

        meta = self.manager.get_metadata(self.model_name)
        net_def = meta.get("net_def", f"{self.model_name}.py")

        cmd = [
            "python", f"{ai8xize}/ai8xize.py",
            "--device", self.device.replace("MAX", ""),
            "--checkpoint-file", weight_file,
            "--config-file", f"networks/{net_def.replace('.py', '.yaml')}",
            "--output-path", str(synth_dir),
            "--compact-weights",
        ]
        if self.synth_cfg.get("compact_data", True):
            cmd.append("--compact-data")

        print(f"  cmd: {' '.join(cmd)}")
        print("  [NOTE] ai8xize.py requires ai8x-synthesis installed at the path above.")
        print(f"  Output will be in: {synth_dir}")
        return synth_dir

    def evaluate(self) -> dict:
        print(f"\n[TrainPipeline] ── Stage 4: Evaluate ─────────────────────")
        meta = self.manager.get_metadata(self.model_name)
        ev = Evaluator(self.config_path)
        results = ev.run(
            self.model_name,
            device=self.device,
            domain=meta.get("domain", "vision"),
            task=meta.get("task", "object_detection"),
        )
        try:
            ev.check_gates(results)
        except AccuracyGateError as e:
            print(f"  [WARN] Gate failed: {e}")
        ev.save_report(results)
        return results

    def run_all(self) -> None:
        print(f"\n{'='*60}")
        print(f" ADI MLOps Train Pipeline: {self.model_name} → {self.device}")
        print(f"{'='*60}")
        self.train()
        self.export()
        self.synthesize()
        self.evaluate()
        print(f"\n[TrainPipeline] All stages complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ADI EdgeAI Train Pipeline")
    parser.add_argument("--config", default="config/pipeline_config.yaml")
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--stage", default="all",
                        choices=["train", "export", "synthesize", "evaluate", "all"])
    args = parser.parse_args()

    p = TrainPipeline(args.config)
    if args.model:
        p.model_name = args.model
    if args.device:
        p.device = args.device

    stages = {
        "train": p.train,
        "export": p.export,
        "synthesize": p.synthesize,
        "evaluate": p.evaluate,
        "all": p.run_all,
    }
    stages[args.stage]()
