"""Default configuration for VLM-MPPI."""

from dataclasses import dataclass, field
from pathlib import Path


# ── Qwen VLM model options ────────────────────────────────────────────────────
#
# All variants load transparently via AutoModelForImageTextToText.
#
# Qwen2.5-VL  (previous gen, transformers >= 4.45)
#   "Qwen/Qwen2.5-VL-3B-Instruct"   ~8 GB VRAM
#   "Qwen/Qwen2.5-VL-7B-Instruct"   ~16 GB VRAM
#   "Qwen/Qwen2.5-VL-32B-Instruct"  ~70 GB VRAM
#   "Qwen/Qwen2.5-VL-72B-Instruct"  ~150 GB VRAM
#
# Qwen3-VL    (dedicated VLM, transformers >= 4.57)  ← default
#   "Qwen/Qwen3-VL-2B-Instruct"          ~6 GB VRAM
#   "Qwen/Qwen3-VL-4B-Instruct"          ~10 GB VRAM
#   "Qwen/Qwen3-VL-8B-Instruct"          ~18 GB VRAM  ← default
#   "Qwen/Qwen3-VL-32B-Instruct"         ~70 GB VRAM
#   "Qwen/Qwen3-VL-30B-A3B-Instruct"     MoE, ~30 GB active
#   "Qwen/Qwen3-VL-235B-A22B-Instruct"   MoE, large cluster
#
# Qwen3.5     (latest unified text+vision, transformers >= 4.57)
#   "Qwen/Qwen3.5-4B-Instruct"           ~10 GB VRAM
#   "Qwen/Qwen3.5-9B-Instruct"           ~20 GB VRAM
#   "Qwen/Qwen3.5-27B-Instruct"          ~55 GB VRAM
#   "Qwen/Qwen3.5-35B-A3B-Instruct"      MoE, ~35 GB active
#   "Qwen/Qwen3.5-397B-A17B-Instruct"    MoE, large cluster
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class VLMConfig:
    """Configuration for the VLM planner layer."""

    model_id: str = "Qwen/Qwen3-VL-8B-Instruct"
    torch_dtype: str = "bfloat16"    # "float16", "bfloat16", or "float32"
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
