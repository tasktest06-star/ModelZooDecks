"""
train_pipeline.py — End-to-end Train → Export → Compile → Evaluate pipeline
for TI EdgeAI Model Zoo models.

Stages:
  1. train()   — invoke edgeai-torchvision / edgeai-mmdetection training
  2. export()  — export checkpoint to ONNX (torch.onnx.export) or TFLite
  3. compile() — compile ONNX/TFLite to TIDL INT8 binary via edgeai-tidl-tools
  4. evaluate()— run edgeai-benchmark accuracy + FPS evaluation

Usage:
    python pipelines/train_pipeline.py \\
        --config config/pipeline_config.yaml \\
        --soc AM68A \\
        --task detection \\
        --model yolox-s-lite
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
    """
    Orchestrates the full training pipeline for a single model on a target SoC.

    All external tool invocations use subprocess so that the underlying
    frameworks (edgeai-torchvision, edgeai-tidl-tools, edgeai-benchmark)
    can be installed separately without hard Python dependencies here.
    """

    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.config_path = config_path

        self.soc = self.config.get("soc", "AM68A")
        self.task = self.config.get("task", "detection")
        self.model_name = self.config.get("model", "yolox-s-lite")

        self.train_cfg = self.config.get("training", {})
        self.export_cfg = self.config.get("export", {})
        self.compile_cfg = self.config.get("compilation", {})

        self.checkpoint_dir = Path(self.train_cfg.get("checkpoint_dir", "./checkpoints"))
        self.export_dir = Path(self.export_cfg.get("output_path", "./exports"))
        self.compile_dir = Path(self.compile_cfg.get("output_dir", "./compiled"))

        for d in [self.checkpoint_dir, self.export_dir, self.compile_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.manager = ModelManager(config_path)
        self.evaluator = Evaluator(config_path)

    # ── Public stage methods ────────────────────────────────────────────────────

    def train(self) -> Path:
        """
        Launch training via the appropriate edgeai framework.

        Supports:
          - edgeai-torchvision (classification, segmentation)
          - edgeai-mmdetection (object detection)
          - edgeai-mmpose (keypoint)

        Returns path to the best checkpoint (.pth).
        """
        framework = self.train_cfg.get("framework", "edgeai-mmdetection")
        epochs = self.train_cfg.get("epochs", 100)
        batch_size = self.train_cfg.get("batch_size", 16)
        lr = self.train_cfg.get("learning_rate", 0.001)
        quantization = self.train_cfg.get("quantization", False)
        resume = self.train_cfg.get("resume_from")

        print(f"\n[TrainPipeline] ── Stage 1: Train ──────────────────────")
        print(f"  framework:    {framework}")
        print(f"  model:        {self.model_name}")
        print(f"  task:         {self.task}")
        print(f"  epochs:       {epochs}")
        print(f"  QAT:          {quantization}")

        checkpoint = self.checkpoint_dir / f"{self.model_name}_best.pth"

        if framework == "edgeai-torchvision":
            cmd = self._build_torchvision_cmd(
                epochs, batch_size, lr, quantization, resume, checkpoint)
        elif framework == "edgeai-mmdetection":
            cmd = self._build_mmdet_cmd(
                epochs, batch_size, lr, quantization, resume, checkpoint)
        elif framework == "edgeai-mmpose":
            cmd = self._build_mmpose_cmd(
                epochs, batch_size, lr, quantization, resume, checkpoint)
        else:
            raise ValueError(f"Unknown framework: {framework}")

        print(f"  cmd: {' '.join(cmd)}")
        self._run(cmd, stage="train")
        print(f"[TrainPipeline] Checkpoint: {checkpoint}\n")
        return checkpoint

    def export(self, checkpoint: Optional[Path] = None) -> Path:
        """
        Export checkpoint to ONNX (or TFLite) using export scripts.

        Returns path to the exported model file.
        """
        fmt = self.export_cfg.get("format", "onnx")
        opset = self.export_cfg.get("opset", 11)
        input_shape = self.export_cfg.get("input_shape", [1, 3, 640, 640])
        checkpoint = checkpoint or self.checkpoint_dir / f"{self.model_name}_best.pth"

        output_file = self.export_dir / f"{self.model_name}.{fmt}"

        print(f"[TrainPipeline] ── Stage 2: Export ─────────────────────")
        print(f"  format:       {fmt}")
        print(f"  opset:        {opset}")
        print(f"  input_shape:  {input_shape}")
        print(f"  checkpoint:   {checkpoint}")

        if fmt == "onnx":
            task = self.task
            if task == "classification":
                script = "scripts/export_model_torchvision.py"
            else:
                script = "scripts/export_model_torchvision.py"

            cmd = [
                sys.executable, script,
                "--checkpoint", str(checkpoint),
                "--output", str(output_file),
                "--opset", str(opset),
                "--input-shape", *[str(s) for s in input_shape],
            ]
        else:
            cmd = [
                "bash", "scripts/tf2_export_classification.sh",
                str(checkpoint), str(output_file),
            ]

        print(f"  cmd: {' '.join(cmd)}")
        self._run(cmd, stage="export")
        print(f"[TrainPipeline] Exported: {output_file}\n")
        return output_file

    def compile(
        self,
        model_file: Optional[Path] = None,
        soc: Optional[str] = None,
    ) -> Path:
        """
        Compile ONNX/TFLite model to TIDL INT8 binary via edgeai-tidl-tools.

        Returns path to compiled artifact directory.
        """
        soc = soc or self.soc
        fmt = self.export_cfg.get("format", "onnx")
        model_file = model_file or self.export_dir / f"{self.model_name}.{fmt}"
        tidl_path = self.compile_cfg.get("tidl_tools_path", "./edgeai-tidl-tools")
        calib_frames = self.compile_cfg.get("calibration_frames", 500)
        tensor_bits = self.compile_cfg.get("tensor_bits", 8)
        accuracy_level = self.compile_cfg.get("accuracy_level", 1)
        out_dir = self.compile_dir / soc / self.model_name

        print(f"[TrainPipeline] ── Stage 3: Compile ────────────────────")
        print(f"  SoC:          {soc}")
        print(f"  model:        {model_file}")
        print(f"  tidl_tools:   {tidl_path}")
        print(f"  INT{tensor_bits} calibration: {calib_frames} frames")

        cmd = [
            sys.executable, f"{tidl_path}/scripts/compile_model.py",
            "--model", str(model_file),
            "--soc", soc,
            "--output-dir", str(out_dir),
            "--calibration-frames", str(calib_frames),
            "--tensor-bits", str(tensor_bits),
            "--accuracy-level", str(accuracy_level),
        ]

        print(f"  cmd: {' '.join(cmd)}")
        out_dir.mkdir(parents=True, exist_ok=True)
        self._run(cmd, stage="compile")
        print(f"[TrainPipeline] TIDL artifact: {out_dir}\n")
        return out_dir

    def evaluate(
        self,
        model_id: Optional[str] = None,
        soc: Optional[str] = None,
        mode: str = "accuracy",
    ) -> dict:
        """
        Run evaluation using edgeai-benchmark and check accuracy gates.

        Returns the result dict.
        """
        model_id = model_id or self.manager.get_metadata(self.model_name).get(
            "model_id", self.model_name
        )
        soc = soc or self.soc

        print(f"[TrainPipeline] ── Stage 4: Evaluate ───────────────────")
        result = self.evaluator.run(
            model_id=model_id, soc=soc, mode=mode
        )
        print(f"  metrics: {result['metrics']}")
        self.evaluator.save_report()
        print(f"[TrainPipeline] Evaluation complete.\n")
        return result

    def run_all(self) -> None:
        """Run all four stages end-to-end."""
        print(f"\n{'='*60}")
        print(f" TI EdgeAI MLOps Train Pipeline")
        print(f" model={self.model_name}  task={self.task}  SoC={self.soc}")
        print(f"{'='*60}\n")
        checkpoint = self.train()
        model_file = self.export(checkpoint)
        self.compile(model_file)
        result = self.evaluate()
        try:
            self.evaluator.check_gates()
            print("[TrainPipeline] All accuracy gates PASSED.")
        except AccuracyGateError as e:
            print(f"[TrainPipeline] Gate FAILED — consider QAT retraining.\n{e}")
        print(f"\n[TrainPipeline] Pipeline complete for {self.model_name} on {self.soc}.")

    # ── Command builders ────────────────────────────────────────────────────────

    def _build_torchvision_cmd(self, epochs, batch_size, lr,
                                quantization, resume, checkpoint):
        cmd = [
            sys.executable, "train.py",
            "--model", self.model_name,
            "--epochs", str(epochs),
            "--batch-size", str(batch_size),
            "--lr", str(lr),
            "--output-dir", str(self.checkpoint_dir),
        ]
        if quantization:
            cmd += [
                "--quantization",
                "--bitwidth-weights", str(self.train_cfg.get("qat_bitwidth_weights", 8)),
                "--bitwidth-activations",
                str(self.train_cfg.get("qat_bitwidth_activations", 8)),
            ]
        if resume:
            cmd += ["--resume", str(resume)]
        return cmd

    def _build_mmdet_cmd(self, epochs, batch_size, lr,
                          quantization, resume, checkpoint):
        cmd = [
            sys.executable, "tools/train.py",
            f"configs/{self.model_name}.py",
            "--cfg-options",
            f"runner.max_epochs={epochs}",
            f"data.samples_per_gpu={batch_size}",
            f"optimizer.lr={lr}",
            "--work-dir", str(self.checkpoint_dir),
        ]
        if quantization:
            cmd += ["--cfg-options", "quantization.bitwidth_weights=8"]
        if resume:
            cmd += ["--resume-from", str(resume)]
        return cmd

    def _build_mmpose_cmd(self, epochs, batch_size, lr,
                           quantization, resume, checkpoint):
        cmd = [
            sys.executable, "tools/train.py",
            f"configs/body/{self.model_name}.py",
            "--work-dir", str(self.checkpoint_dir),
        ]
        if resume:
            cmd += ["--resume-from", str(resume)]
        return cmd

    @staticmethod
    def _run(cmd: list, stage: str) -> None:
        try:
            result = subprocess.run(cmd, check=True, text=True,
                                    capture_output=False)
        except FileNotFoundError:
            print(f"[TrainPipeline] WARN: {stage} command not found "
                  f"(install the corresponding edgeai framework to run).")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"[TrainPipeline] {stage} failed (exit {e.returncode})") from e


# Allow Optional type hint without importing Optional at top
from typing import Optional


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TI EdgeAI Train Pipeline")
    parser.add_argument("--config", default="config/pipeline_config.yaml")
    parser.add_argument("--soc",   default=None)
    parser.add_argument("--task",  default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--stage", default="all",
                        choices=["all", "train", "export", "compile", "evaluate"])
    args = parser.parse_args()

    pipeline = TrainPipeline(args.config)
    if args.soc:   pipeline.soc = args.soc
    if args.task:  pipeline.task = args.task
    if args.model: pipeline.model_name = args.model

    if args.stage == "all":
        pipeline.run_all()
    elif args.stage == "train":
        pipeline.train()
    elif args.stage == "export":
        pipeline.export()
    elif args.stage == "compile":
        pipeline.compile()
    elif args.stage == "evaluate":
        pipeline.evaluate()
