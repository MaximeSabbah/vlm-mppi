# Architecture: VLM-MPPI

## Dual-Loop Design

```
Camera Image ──→ VLM ──→ JSON Plan ──→ SE(3) Goal ──→ MPPI ──→ Torques
                  ↑                        │              ↑
          Keypoints (3D)            Cost weights    Human Capsules
          from perception                          from RT-COSMIK
                  ↑                        │
                  └──── Cost Reflection ◄──┘
                     (execution feedback)
```

### VLM Loop (~1-5 Hz)
- Captures an RGB image from the workspace camera
- Overlays current 3D keypoints (from FoundationPose / RT-COSMIK)
- Queries the VLM with instruction + image + keypoints + execution history
- Parses structured JSON output
- Converts target keypoints into SE(3) goal for MPPI
- Optionally adjusts MPPI cost weights and safety parameters via **cost reflection**

### MPPI Loop (50 Hz)
- Runs COSMIK-MPPI with current robot state + human capsules
- CaT (Constraints-as-Terminations) ensures collision safety
- Produces joint torques sent to the robot
- Logs per-component cost statistics for cost reflection

---

## Inspiration from Related Work

### From IKER (Patel et al., 2025)
- **Keypoint-based task specification**: VLMs reason about named 3D keypoints
  rather than SE(3) poses, encoding both position and orientation.
- **Iterative replanning with execution history**: each VLM call includes
  previous observations and plans, enabling closed-loop adaptation.
- **Code generation for spatial reasoning**: the VLM outputs computation over
  keypoint positions, enabling precise arithmetic.

### From Eureka (Ma et al., ICLR 2024)
- **Environment-as-context**: Eureka feeds raw environment source code to the
  LLM. We provide the MPPI cost structure (ℓ_ee, ℓ_x, ℓ_u, ℓ_coll) as
  context so the VLM knows which terms it can modulate.
- **Reward reflection → Cost reflection**: Eureka tracks per-component reward
  values during RL training and feeds them as text for targeted editing.
  We track per-component MPPI cost values during execution and feed them
  back to the VLM. Since MPPI gives immediate feedback (no training), this
  loop is orders of magnitude faster than Eureka's RL-based loop.
- **Free-form generation outperforms templates**: Eureka's unconstrained code
  generation beats templated approaches on 83% of tasks. We similarly let
  the VLM generate arbitrary keypoint computations.

### From VLMPC (Zhao et al., RSS 2024)
- **VLM in the MPC loop**: integrates VLM perception into MPC planning.
  We replace video prediction with physics-based MPPI rollouts.

### From VoxPoser (Huang et al., CoRL 2023)
- **3D value maps**: affordance/avoidance maps as cost functions for planning.
  The VLM could generate semantic avoidance regions supplementing geometric
  collision capsules.

### From SayCan / Inner Monologue (Google, 2022)
- **Affordance grounding**: feasibility naturally emerges from CaT — infeasible
  goals lead to low-survival rollouts, no learned value function needed.
- **Closed-loop feedback**: execution results feed back to the planner.

---

## Key Design Decisions

1. **Keypoints, not poses** (from IKER): VLMs handle Cartesian reasoning much
   better than rotation representations.

2. **VLM is advisory, MPPI is authoritative**: The VLM suggests goals, MPPI
   has final say on safety. CaT terminates unsafe rollouts automatically.

3. **Cost reflection for iterative improvement** (from Eureka): execution
   statistics feed back to the VLM. Unlike Eureka (hours of RL per iteration),
   our feedback is immediate.

4. **Stateless VLM queries**: full history included in each call, robust to
   inference failures.

---

## Comparison with Related Work

|                     | VLMPC      | IKER     | VoxPoser  | Eureka     | **Ours**        |
|---------------------|------------|----------|-----------|------------|-----------------|
| Low-level           | MPC(video) | RL (PPO) | Planner   | RL (PPO)   | MPPI (dynamics) |
| Safety guarantees   | None       | None     | None      | None       | CaT + capsules  |
| Human-aware         | No         | No       | No        | No         | Yes (RT-COSMIK) |
| VLM model           | Proprietary| GPT-4o   | GPT-4     | GPT-4      | Qwen2.5-VL (Apache 2.0) |
| Feedback loop       | Video pred | RL reward| Open-loop | Reward refl| Cost reflection |
| Training per task   | Video model| 5 min    | None      | Hours      | **None**        |
| Control freq        | ~10 Hz     | 10 Hz    | Open-loop | N/A        | 50 Hz           |
| IP-safe             | No         | No       | No        | No         | **Yes**         |

---

## Demo: Voice-Guided Handover with Human Avoidance

### Equipment
- Franka Panda arm (7-DOF, torque-controlled)
- 2× RGB cameras for RT-COSMIK (human tracking)
- 1× RealSense RGB-D for scene understanding + VLM input
- 3 colored objects on table (red cup, blue box, green bottle)
- 1 human participant in the shared workspace
- PC with GPU (≥ RTX 4090) running VLM + MPPI

### Scenario
```
Human: "Give me the red cup"

1. VLM sees scene → identifies red cup → outputs grasp target
2. MPPI reaches for cup while avoiding human arm (RT-COSMIK + CaT)
3. Human moves arm to block path → robot smoothly avoids
4. Robot grasps cup, moves toward human hand (VLM detects hand position)
5. VLM observes result → if delivered, done; if not, replan
```

### What This Demonstrates
1. Language-conditioned manipulation (VLM interprets "red cup")
2. Semantic goal generation (VLM outputs WHERE to go)
3. Human-aware safety (RT-COSMIK + CaT prevent collision)
4. Dynamic replanning (VLM + MPPI adapt when human moves)
5. Fully open-source, IP-safe stack

### Why This Is Novel
No existing system combines all five. IKER has no human awareness. VLMPC has
no safety guarantees. Eureka requires hours of training. SayCan uses pre-trained
skills. VLM-MPPI achieves zero-shot, language-conditioned, safe manipulation.
