"""Configuration for VLM-MPPI."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    """Embodied-R1 model configuration."""

    model_id: str = "IffYuan/Embodied-R1-3B-v1"
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    max_new_tokens: int = 512            # actual outputs are 70-130 tokens; 512 is a safe cap
    repetition_penalty: float = 1.05
    load_in_4bit: bool = True            # NF4 quantization: ~2-3× faster decoding, needs bitsandbytes
    flash_attn2: bool = True             # FlashAttention-2: fastest attention, requires flash-attn package
    use_sdpa: bool = False              # PyTorch built-in scaled dot-product attention (no extra deps)
    torch_compile: bool = True           # ~20-40% speedup after first (slow ~2min) compilation run
    local_files_only: bool = False       # skip HuggingFace network checks after first download
    max_image_pixels: int | None = None  # override processor default (None = use model default)


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
