"""Default configuration for VLM-MPPI."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class VLMConfig:
    """Configuration for the VLM planner layer."""

    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    torch_dtype: str = "float16"     # "float16", "bfloat16", or "float32"
    device_map: str = "auto"
    max_new_tokens: int = 512
    temperature: float = 0.1
    do_sample: bool = False

    # vLLM server mode
    vllm_base_url: str = "http://localhost:8000/v1"
    use_vllm: bool = False


@dataclass
class MPPIConfig:
    """Configuration for the MPPI controller interface."""

    control_frequency_hz: float = 50.0
    vlm_replan_interval_s: float = 1.0
    default_safety_margin_m: float = 0.05
    collision_threshold_m: float = 0.02


@dataclass
class Config:
    """Top-level configuration."""

    vlm: VLMConfig = field(default_factory=VLMConfig)
    mppi: MPPIConfig = field(default_factory=MPPIConfig)
    prompt_dir: Path = Path(__file__).parent.parent / "prompts"
