# Architecture: VLM-MPPI

## Dual-Loop Design

The system runs two control loops at different frequencies:

### VLM Loop (~1-5 Hz)
- Captures an RGB image from the workspace camera
- Overlays current 3D keypoints (from FoundationPose / RT-COSMIK)
- Queries the VLM with instruction + image + keypoints + history
- Parses structured JSON output
- Converts target keypoints into SE(3) goal for MPPI
- Optionally adjusts MPPI parameters (safety margins, speed limits)

### MPPI Loop (50 Hz)
- Runs COSMIK-MPPI with current robot state + human capsules
- CaT (Constraints-as-Terminations) ensures collision safety
- Produces joint torques sent to the robot
- Continues executing the last VLM goal until a new one arrives

## Data Flow

```
Camera Image ──→ VLM ──→ JSON Plan ──→ SE(3) Goal ──→ MPPI ──→ Torques
                  ↑                                     ↑
          Keypoints (3D)                         Human Capsules
          from perception                        from RT-COSMIK
```

## Key Design Decisions

1. **Keypoints, not poses**: Following IKER, the VLM reasons about named
   3D keypoints rather than SE(3) poses. VLMs handle Cartesian reasoning
   much better than rotation representations.

2. **JSON output, not code generation**: IKER generates Python code.
   We use JSON for safety — no arbitrary code execution on the robot.
   The JSON-to-SE(3) conversion is deterministic and auditable.

3. **VLM is advisory, MPPI is authoritative**: The VLM suggests goals,
   but MPPI has final say on safety. If the VLM suggests a goal that
   would cause collision, CaT terminates those rollouts automatically.

4. **Stateless VLM queries**: Each VLM call includes the full execution
   history (images + previous plans), so the VLM doesn't need memory.
   This makes the system robust to VLM inference failures.

## Comparison with Related Work

| | VLMPC (RSS'24) | IKER | VoxPoser | **Ours** |
|---|---|---|---|---|
| Low-level | MPC (video prediction) | RL (PPO) | Motion planner | MPPI (dynamics) |
| Safety | None | None | None | CaT + capsules |
| Human-aware | No | No | No | Yes (RT-COSMIK) |
| VLM model | Proprietary | GPT-4o | GPT-4 | Qwen2.5-VL (Apache 2.0) |
| Training needed | Video model | 5 min/task | None | None |
| Control freq | ~10 Hz | 10 Hz | Open-loop | 50 Hz |
| IP-safe | No | No | No | Yes |
