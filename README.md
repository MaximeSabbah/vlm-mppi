# VLM-MPPI

Embodied-R1 spatial reasoning for model-predictive robot control.

This repo provides a clean Python interface to [Embodied-R1](https://github.com/pickxiguapi/Embodied-R1) (Yuan et al., ICLR 2026), a 3B vision-language model that outputs precise spatial coordinates for robotic manipulation. The goal is to use these coordinates as task-space targets for MPC/MPPI controllers, bypassing learned action decoders entirely.

## Architecture

```
Camera RGB-D + language instruction
         │
         ▼
┌──────────────────────────────┐
│  Embodied-R1 (3B, frozen)    │  ~1-3 Hz
│                              │
│  Outputs per ability:        │
│  • OFG → grasp point (u,v)  │
│  • RRG → place target (u,v) │
│  • VTG → trajectory sketch  │
│  • REG → object location    │
└────────────┬─────────────────┘
             │ pixel coordinates
             ▼
┌──────────────────────────────┐
│  2D → 3D projection         │  depth + camera intrinsics
└────────────┬─────────────────┘
             │ SE(3) targets
             ▼
┌──────────────────────────────┐
│  MPC / MPPI (your stack)     │  50-1000 Hz
│  (not implemented yet)       │
└──────────────────────────────┘
```

## Setup

Requires Python 3.10+ and a CUDA-capable GPU with ≥8 GB VRAM.

```bash
git clone https://github.com/MaximeSabbah/vlm-mppi.git
cd vlm-mppi

# Option A: uv (recommended, fast)
pip install uv
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"

# Option B: plain pip
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Verify:
```bash
python -c "from vlm_mppi.model import EmbodiedR1; print('OK')"
pytest  # runs parsing tests (no GPU needed)
```

## Usage

### CLI
```bash
# Run Embodied-R1 on any image
python -m vlm_mppi.cli scene.png "pick up the red cup" --save results.png

# Choose specific abilities
python -m vlm_mppi.cli scene.png "put the plate in the sink" --abilities OFG RRG VTG
```

### Python API
```python
from vlm_mppi.model import EmbodiedR1, Ability

model = EmbodiedR1.load()

# Single ability
result = model.point("scene.png", "pick up the mug by the handle", Ability.OFG)
print(result.reasoning)   # chain-of-thought explanation
print(result.points_px)   # [(312.5, 201.3)]

# All abilities at once
results = model.point_all("scene.png", "put the red block on the yellow block")
for ability, r in results.items():
    print(f"{ability.value}: {r.points_px}")
```

### Visualization
```python
from vlm_mppi.viz import draw_results
draw_results("scene.png", results, save_path="output.png")
```

### 2D → 3D projection (when you have depth)
```python
from vlm_mppi.projection import project_to_3d
from vlm_mppi.config import CameraConfig

points_3d = project_to_3d(
    result.points_px,
    depth_image,               # (H, W) numpy array in meters
    CameraConfig(fx=615, fy=615, cx=320, cy=240),
    T_cam_to_base=np.eye(4),  # camera-to-robot transform
)
```

## Project structure

```
vlm-mppi/
├── vlm_mppi/
│   ├── __init__.py
│   ├── model.py          # EmbodiedR1 wrapper + output parsing
│   ├── viz.py            # matplotlib visualization
│   ├── projection.py     # 2D → 3D back-projection
│   ├── config.py         # dataclass configs
│   └── cli.py            # command-line interface
├── examples/
│   └── 01_test_embodied_r1.py
├── tests/
│   └── test_parsing.py   # unit tests (no GPU)
├── data/sample_images/   # put test images here
├── outputs/              # generated visualizations
└── pyproject.toml
```

## Hardware

| Setup | GPU | Notes |
|-------|-----|-------|
| Minimum | 8 GB VRAM (RTX 3070) | float16, ~2s per query |
| Comfortable | 16 GB (RTX 4090, A5000) | fast inference |
| CPU only | none | ~30s per query, functional for testing |

## Pointing abilities

| Ability | Question answered | Output |
|---------|-------------------|--------|
| **OFG** | Where to grasp? | Functional affordance point (handle, rim, edge) |
| **RRG** | Where to place? | Target region in free space |
| **REG** | Where is the object? | Point on the referred object |
| **VTG** | How to move? | Sequence of waypoints (trajectory sketch) |

## References

- [Embodied-R1](https://arxiv.org/abs/2508.13998) — Yuan et al., ICLR 2026
- [Eureka](https://eureka-research.github.io) — LLM-powered reward design (inspires cost function generation)
