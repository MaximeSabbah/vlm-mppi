# VLM-MPPI: Vision-Language Model Planning for Safe Human-Robot Interaction

A hierarchical control framework combining a Vision-Language Model (VLM) for high-level semantic planning with [COSMIK-MPPI](https://exquisite-parfait-ffa925.netlify.app) for safe, real-time collision avoidance in human-shared workspaces.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│               VLM Planner (~1-5 Hz)                     │
│  Qwen2.5-VL-7B-Instruct (Apache 2.0)                   │
│  Input:  RGB image + language instruction + keypoints   │
│  Output: structured task plan (target poses, semantic   │
│          constraints, safety parameters)                │
└───────────────────────┬─────────────────────────────────┘
                        │ T_goal, constraints, margins
                        ▼
┌─────────────────────────────────────────────────────────┐
│            COSMIK-MPPI Controller (50 Hz)               │
│  Input:  T_goal + RT-COSMIK human capsules              │
│  Output: joint torques with CaT collision avoidance     │
└─────────────────────────────────────────────────────────┘
```

## Licensing

Every dependency in this project uses a **commercially permissive license** (Apache 2.0, BSD, MIT), making it safe for proprietary products and IP filings.

| Component | License |
|-----------|---------|
| Qwen2.5-VL-7B-Instruct | Apache 2.0 |
| Qwen2.5-VL-32B-Instruct | Apache 2.0 |
| Qwen3-VL (all sizes) | Apache 2.0 |
| PyTorch | BSD-3 |
| Transformers (HuggingFace) | Apache 2.0 |
| vLLM | Apache 2.0 |
| Pinocchio | BSD-2 |
| Crocoddyl | BSD-3 |

## Hardware Requirements

**Minimum (Qwen2.5-VL-7B, 4-bit quantized):**
- GPU: 1× NVIDIA with ≥ 10 GB VRAM (RTX 3080, RTX 4070, etc.)
- RAM: 16 GB
- Disk: ~8 GB for quantized weights

**Recommended (Qwen2.5-VL-7B, float16):**
- GPU: 1× NVIDIA with ≥ 16 GB VRAM (RTX 4090, A5000)
- RAM: 32 GB
- Disk: ~15 GB for model weights

**For Qwen2.5-VL-32B (upgrade path):**
- GPU: 1× 48 GB (A6000) or 2× 24 GB
- With 4-bit quantization: 1× 24 GB GPU

## Installation

### Option A: Python venv (recommended)

```bash
# Clone the repository
git clone https://github.com/<your-org>/vlm-mppi.git
cd vlm-mppi

# Create and activate virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Install with vLLM for fast inference
pip install -r requirements-vllm.txt
```

### Option B: Nix (reproducible builds)

```bash
# Enter the development shell
nix develop

# Or build and run directly
nix run .#test-vlm
```

See [docs/nix-setup.md](docs/nix-setup.md) for details on the Nix flake.

### Verify installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "from transformers import Qwen2_5_VLForConditionalGeneration; print('Qwen2.5-VL OK')"
```

## Quick Start

### 1. Test the VLM on a scene image

```bash
# Place any scene photo as scene.jpg (or pass a path)
python examples/01_test_vlm_basic.py --image scene.jpg
```

### 2. Keypoint-based planning (IKER-style)

```bash
python examples/02_vlm_keypoint_planner.py --image scene.jpg
```

### 3. Fast inference with vLLM server

```bash
# Terminal 1: start the server
python scripts/start_vllm_server.py

# Terminal 2: query it
python examples/03_vlm_client_vllm.py --image scene.jpg --instruction "pick up the red cup"
```

### 4. Full VLM-MPPI loop (requires COSMIK-MPPI)

```bash
python examples/04_vlm_mppi_loop.py --instruction "place the cup near the human's left hand"
```

## Project Structure

```
vlm-mppi/
├── README.md
├── LICENSE                          # Apache 2.0
├── requirements.txt                 # Core dependencies
├── requirements-vllm.txt            # Optional: vLLM for fast serving
├── flake.nix                        # Nix flake for reproducible env
├── pyproject.toml                   # Project metadata
├── examples/
│   ├── 01_test_vlm_basic.py         # Minimal VLM test
│   ├── 02_vlm_keypoint_planner.py   # IKER-style keypoint planning
│   ├── 03_vlm_client_vllm.py        # Query vLLM server
│   ├── 04_vlm_mppi_loop.py          # Full hierarchical loop (sketch)
│   └── 05_demo_handover.py          # Handover demo with cost reflection
├── scripts/
│   └── start_vllm_server.py         # Launch vLLM OpenAI-compat server
├── prompts/
│   ├── single_step.txt              # Prompt for single-step tasks
│   └── multi_step.txt               # Prompt for iterative replanning
├── vlm_mppi/
│   ├── __init__.py
│   ├── vlm_planner.py               # VLM wrapper class
│   ├── keypoint_utils.py            # Keypoint projection & overlay
│   ├── mppi_interface.py            # Bridge between VLM output and MPPI
│   ├── cost_reflection.py           # Eureka-inspired execution feedback
│   └── config.py                    # Model paths, default parameters
├── tests/
│   └── test_vlm_output_parsing.py   # Unit tests for JSON parsing
└── docs/
    ├── architecture.md              # Design doc with demo proposal
    ├── nix-setup.md                 # Nix installation guide
    └── licensing.md                 # Full licensing analysis
```

## Related Work

- [COSMIK-MPPI](https://exquisite-parfait-ffa925.netlify.app) — Collision avoidance with CaT for MPPI in human environments
- [Eureka](https://eureka-research.github.io) (ICLR 2024) — LLM-powered evolutionary reward design (inspires our cost reflection)
- [VLMPC](https://arxiv.org/abs/2407.09829) (RSS 2024) — VLM integrated into MPC for manipulation
- [VoxPoser](https://voxposer.github.io/) (CoRL 2023) — LLM + VLM composing 3D value maps for planning
- [IKER](https://iker-robot.github.io/) — VLM-generated iterative keypoint rewards (inspires our keypoint interface)
- [SayCan](https://say-can.github.io/) — LLM grounding in robotic affordances

## Citation

If you use this work, please cite:

```bibtex
@misc{vlm-mppi2026,
  title={VLM-MPPI: Vision-Language Model Planning for Safe Human-Robot Interaction},
  author={TODO},
  year={2026},
  url={https://github.com/<your-org>/vlm-mppi}
}
```
