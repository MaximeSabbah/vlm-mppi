"""
Bridge between VLM planner output and COSMIK-MPPI controller.

This module converts structured VLM plans (target keypoints, constraints)
into the goal format expected by MPPI (SE3 poses, cost parameters).

NOTE: The actual COSMIK-MPPI import is left as a placeholder.
      Replace the stub class with your real controller.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


def vlm_plan_to_se3_goal(plan: dict) -> np.ndarray:
    """
    Convert VLM keypoint targets to an SE(3) goal pose for MPPI.

    Strategy:
      - Position = centroid of all target keypoints
      - Orientation = identity quaternion (TODO: infer from keypoint geometry)

    Args:
        plan: VLM output dict with "target_keypoints": {"label": [x,y,z], ...}

    Returns:
        (7,) array: [x, y, z, qw, qx, qy, qz]
    """
    targets = plan.get("target_keypoints", {})
    if not targets:
        raise ValueError("VLM plan has no target_keypoints")

    positions = np.array(list(targets.values()), dtype=np.float64)
    centroid = positions.mean(axis=0)

    # TODO: compute orientation from keypoint arrangement
    # For now, use identity quaternion
    quat = np.array([1.0, 0.0, 0.0, 0.0])

    goal = np.concatenate([centroid, quat])
    logger.info("SE(3) goal: pos=%s, quat=%s", centroid, quat)
    return goal


def extract_safety_margin(plan: dict, default: float = 0.05) -> float:
    """Extract the safety margin from a VLM plan."""
    return float(plan.get("safety_margin_m", default))


# ─── Stub for COSMIK-MPPI controller ────────────────────────────
# Replace this with your actual COSMIK-MPPI import when integrating.


class MPPIControllerStub:
    """
    Placeholder for the real COSMIK-MPPI controller.

    Replace this class with your actual controller, e.g.:
        from cosmik_mppi import CosmiKMPPI

    The interface contract is:
        - set_goal(goal_se3: np.ndarray)     -> set T_goal
        - set_collision_threshold(d: float)  -> set d_th for collision capsules
        - compute_control(state, human_capsules) -> torques
        - apply_torques(torques)             -> send to robot
    """

    def __init__(self):
        self._goal = None
        self._threshold = 0.05
        logger.info("MPPIControllerStub initialized (replace with real COSMIK-MPPI)")

    def set_goal(self, goal_se3: np.ndarray):
        self._goal = goal_se3
        logger.info("MPPI goal set: %s", goal_se3[:3])

    def set_collision_threshold(self, d: float):
        self._threshold = d
        logger.info("Collision threshold: %.3f m", d)

    def compute_control(self, state: np.ndarray, human_capsules=None) -> np.ndarray:
        """Compute torques. Returns zeros in stub mode."""
        logger.debug("compute_control called (stub)")
        return np.zeros(7)  # 7-DOF Panda

    def apply_torques(self, torques: np.ndarray):
        """Send torques to robot. No-op in stub mode."""
        pass
