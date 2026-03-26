"""Configuration for VLM-MPPI."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    """Embodied-R1 model configuration."""

    model_id: str = "IffYuan/Embodied-R1-3B-v1"
    torch_dtype: str = "auto"
    device_map: str = "auto"
    max_new_tokens: int = 4096       # official default (model needs room for <think> block)
    repetition_penalty: float = 1.05 # official default; prevents token-level repetition
    flash_attn2: bool = False        # enable Flash Attention 2 (requires nvcc + flash-attn package)
    torch_compile: bool = False      # torch.compile for ~20-40% speedup (no extra deps, slow first run)
    local_files_only: bool = False   # set True to skip HuggingFace network checks


@dataclass
class CameraConfig:
    """Pinhole camera intrinsics."""

    fx: float = 615.0
    fy: float = 615.0
    cx: float = 320.0
    cy: float = 240.0
    width: int = 640
    height: int = 480


@dataclass
class Config:
    """Top-level configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    output_dir: Path = Path("outputs")

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
