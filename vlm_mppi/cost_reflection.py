"""
Cost Reflection: Eureka-inspired execution feedback for VLM replanning.

Eureka tracks per-component reward values during RL training and feeds them
back to the LLM as text for targeted reward editing. We adapt this idea for
MPPI: track per-component cost values during execution and feed them to the
VLM so it can make informed replanning decisions.

Key difference from Eureka: MPPI gives immediate feedback (no RL training loop),
so our reflection cycle is orders of magnitude faster (~1s vs hours).
"""

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CostSnapshot:
    """A single snapshot of MPPI cost components at one timestep."""

    timestamp: float = 0.0
    goal_distance_m: float = 0.0
    min_collision_distance_m: float = 0.0
    torque_smoothness: float = 0.0
    cartesian_speed_m_s: float = 0.0
    task_progress: float = 0.0  # 0.0 = not started, 1.0 = done


@dataclass
class CostReflection:
    """
    Aggregated execution statistics formatted as text for VLM feedback.

    Inspired by Eureka's reward reflection (Section 3.3 of the paper):
    "reward reflection tracks the scalar values of all reward components
    and the task fitness function at intermediate policy checkpoints."

    We track analogous quantities for MPPI execution.
    """

    snapshots: list[CostSnapshot] = field(default_factory=list)
    task_success: bool = False
    failure_reason: str = ""

    def record(self, snapshot: CostSnapshot):
        """Add an execution snapshot."""
        self.snapshots.append(snapshot)

    def to_text(self) -> str:
        """
        Format the reflection as text for the VLM prompt.

        This is the key interface: convert numerical execution data
        into natural language that the VLM can reason about.
        """
        if not self.snapshots:
            return "No execution data available yet."

        n = len(self.snapshots)
        goal_dists = [s.goal_distance_m for s in self.snapshots]
        collision_dists = [s.min_collision_distance_m for s in self.snapshots]
        speeds = [s.cartesian_speed_m_s for s in self.snapshots]
        progress = [s.task_progress for s in self.snapshots]

        # Sample ~10 evenly spaced values for compact representation
        indices = np.linspace(0, n - 1, min(n, 10), dtype=int)

        def fmt(values, idx):
            return "[" + ", ".join(f"{values[i]:.3f}" for i in idx) + "]"

        lines = [
            "=== MPPI Execution Feedback ===",
            f"Duration: {n} timesteps",
            f"Task success: {self.task_success}",
        ]

        if self.failure_reason:
            lines.append(f"Failure reason: {self.failure_reason}")

        lines.extend([
            "",
            f"goal_distance_m: {fmt(goal_dists, indices)}, "
            f"Final: {goal_dists[-1]:.3f}, Min: {min(goal_dists):.3f}",
            "",
            f"min_collision_distance_m: {fmt(collision_dists, indices)}, "
            f"Min: {min(collision_dists):.3f}",
            "",
            f"cartesian_speed_m_s: {fmt(speeds, indices)}, "
            f"Mean: {np.mean(speeds):.3f}",
            "",
            f"task_progress: {fmt(progress, indices)}, "
            f"Final: {progress[-1]:.3f}",
        ])

        # Add interpretation hints (like Eureka's reflection tips)
        lines.extend([
            "",
            "Analysis tips:",
            "- If goal_distance is not decreasing, the target may be unreachable or blocked.",
            "- If min_collision_distance < 0.02m, the robot nearly collided — increase safety_margin_m.",
            "- If task_progress stalled, consider breaking the task into smaller steps.",
            "- If cartesian_speed is near zero but goal_distance is large, the robot may be stuck.",
        ])

        return "\n".join(lines)

    def summary(self) -> dict:
        """Return a compact dict summary for logging."""
        if not self.snapshots:
            return {"status": "no_data"}

        goal_dists = [s.goal_distance_m for s in self.snapshots]
        collision_dists = [s.min_collision_distance_m for s in self.snapshots]

        return {
            "task_success": self.task_success,
            "failure_reason": self.failure_reason,
            "final_goal_dist": goal_dists[-1],
            "min_goal_dist": min(goal_dists),
            "min_collision_dist": min(collision_dists),
            "n_timesteps": len(self.snapshots),
        }


def build_reflection_from_mppi_log(
    goal_distances: list[float],
    collision_distances: list[float],
    speeds: list[float],
    success: bool = False,
    failure_reason: str = "",
) -> CostReflection:
    """
    Convenience function to build a CostReflection from raw MPPI logs.

    In practice, you'd hook this into your COSMIK-MPPI controller's
    logging output after each execution phase.
    """
    reflection = CostReflection(task_success=success, failure_reason=failure_reason)

    n = len(goal_distances)
    for i in range(n):
        snapshot = CostSnapshot(
            timestamp=i * 0.02,  # 50 Hz
            goal_distance_m=goal_distances[i],
            min_collision_distance_m=collision_distances[i] if i < len(collision_distances) else 0.0,
            cartesian_speed_m_s=speeds[i] if i < len(speeds) else 0.0,
            task_progress=1.0 - (goal_distances[i] / max(goal_distances[0], 1e-6)),
        )
        reflection.record(snapshot)

    return reflection
