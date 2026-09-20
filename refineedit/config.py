"""Public inference parameters; no machine-specific paths."""

from dataclasses import dataclass
import math
from pathlib import Path

MODEL_FILENAME = "GRN_T2I_2B_FSA_251600.pth"
VAE_FILENAME = "HBQ_image_video_tokenizer_64dim_M4_20260626.ckpt"
TEXT_FILENAME = "models_t5_umt5-xxl-enc-bf16.pth"


@dataclass(frozen=True)
class RefineEditConfig:
    switch_step: int = 12
    tau_spatial: float = 0.015
    tau_power: float = 0.12
    bit_lock_steps: int = 4
    adaptive_spatial_freezing: bool = True
    freeze_multiplier: float = 2.0
    steps: int = 50
    guidance_scale: float = 3.0
    temperature: float = 1.1
    seed: int = 42
    mask_interval: int = 3

    def __post_init__(self):
        for name in ("switch_step", "bit_lock_steps", "steps", "seed", "mask_interval"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.steps < 2:
            raise ValueError("steps must be at least 2")
        if not 0 <= self.switch_step < self.steps:
            raise ValueError("switch_step must satisfy 0 <= switch_step < steps")
        if self.bit_lock_steps < 1 or self.mask_interval < 1:
            raise ValueError("bit_lock_steps and mask_interval must be positive")
        if not 0 <= self.seed < 2**63 - 1:
            raise ValueError("seed must be in [0, 2**63 - 1)")
        for name in ("tau_spatial", "tau_power", "freeze_multiplier", "guidance_scale", "temperature"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.tau_spatial > 1 or self.tau_power > 1:
            raise ValueError("Probability thresholds must lie in [0, 1]")
        if self.temperature == 0:
            raise ValueError("temperature must be positive")


def checkpoint_paths(weights_dir="weights", model_path=None, vae_path=None,
                     text_encoder_path=None):
    root = Path(weights_dir).expanduser()
    return (
        Path(model_path).expanduser() if model_path else root / MODEL_FILENAME,
        Path(vae_path).expanduser() if vae_path else root / VAE_FILENAME,
        Path(text_encoder_path).expanduser() if text_encoder_path else root / "umt5-xxl",
    )


def check_weights(paths):
    model, vae, text = paths
    required = (model, vae, text / TEXT_FILENAME, text / "umt5-xxl" / "tokenizer_config.json")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing checkpoint/tokenizer files:\n  " + "\n  ".join(missing)
            + "\nRun python download_weights.py --output-dir weights, or set --weights-dir."
        )
