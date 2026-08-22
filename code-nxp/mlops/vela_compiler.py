"""Arm Vela compiler wrapper for i.MX 93 Ethos-U65 NPU."""

import subprocess
from pathlib import Path


class VelaCompiler:
    """Compiles TFLite models for Ethos-U65 NPU on i.MX 93.

    Prerequisites:
        pip install ethos-u-vela
        # or: apt-get install python3-ethosu-vela (NXP BSP environment)
    """

    PRESET_CONFIGS = {
        "ethos-u65-256": {
            "accelerator_config": "ethos-u65-256",
            "system_config": "Ethos_U65_High_End",
            "memory_mode": "Shared_Sram",
        },
        "ethos-u65-512": {
            "accelerator_config": "ethos-u65-512",
            "system_config": "Ethos_U65_High_End",
            "memory_mode": "Shared_Sram",
        },
    }

    def __init__(self, accelerator_config="ethos-u65-256", optimise="Performance",
                 system_config=None, memory_mode=None):
        preset = self.PRESET_CONFIGS.get(accelerator_config, {})
        self.accelerator_config = accelerator_config
        self.optimise = optimise
        self.system_config = system_config or preset.get("system_config", "Ethos_U65_High_End")
        self.memory_mode = memory_mode or preset.get("memory_mode", "Shared_Sram")

    def compile(self, tflite_path, output_dir=None):
        """Compile a TFLite model for Ethos-U65 NPU.

        Args:
            tflite_path: path to the INT8 TFLite model
            output_dir:  where to write *_vela.tflite (default: same dir)
        Returns:
            Path to the Vela-compiled model (*_vela.tflite)
        """
        src = Path(tflite_path).resolve()
        if not src.exists():
            raise FileNotFoundError(f"TFLite model not found: {src}")

        out_dir = Path(output_dir) if output_dir else src.parent / "vela_output"
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "vela",
            str(src),
            "--output-dir", str(out_dir),
            "--accelerator-config", self.accelerator_config,
            "--optimise", self.optimise,
            "--system-config", self.system_config,
            "--memory-mode", self.memory_mode,
        ]
        print(f"[vela] Compiling {src.name} → {out_dir}/")
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Vela failed for {src.name}:\n{result.stderr}"
            )

        vela_path = out_dir / f"{src.stem}_vela.tflite"
        if not vela_path.exists():
            # Try alternate naming: ethos_u65 suffix
            candidates = list(out_dir.glob("*_vela.tflite"))
            if candidates:
                vela_path = candidates[0]
            else:
                raise FileNotFoundError(
                    f"Vela completed but no *_vela.tflite found in {out_dir}"
                )

        size_kb = vela_path.stat().st_size / 1024
        print(f"[vela] Output: {vela_path.name} ({size_kb:.1f} KB)")
        return vela_path

    def is_available(self):
        """Check if the vela command-line tool is on PATH."""
        result = subprocess.run(
            ["vela", "--version"], check=False,
            capture_output=True, text=True
        )
        return result.returncode == 0

    def version(self):
        result = subprocess.run(
            ["vela", "--version"], check=True,
            capture_output=True, text=True
        )
        return result.stdout.strip()
