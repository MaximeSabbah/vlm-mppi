#!/usr/bin/env python3
"""
Example 04: Full VLM-MPPI hierarchical control loop.

This is a SKETCH showing how the VLM planner and COSMIK-MPPI
would interact in a real deployment. The MPPI controller is
stubbed — replace with your actual COSMIK-MPPI instance.

Architecture:
    VLM (slow, ~1 Hz)  →  generates T_goal + constraints
    MPPI (fast, 50 Hz) →  executes with CaT collision avoidance

Usage:
    python examples/04_vlm_mppi_loop.py --instruction "place the cup near the human"
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from vlm_mppi.vlm_planner import VLMPlanner
from vlm_mppi.mppi_interface import (
    MPPIControllerStub,
    vlm_plan_to_se3_goal,
    extract_safety_margin,
)
from vlm_mppi.config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("vlm_mppi_loop")


class DemoPerception:
    """
    Stub perception pipeline.
    Replace with RT-COSMIK + RealSense cameras.
    """

    def __init__(self, image_path: str):
        self._image = Image.open(image_path).convert("RGB")

    def get_rgb_image(self) -> Image.Image:
        return self._image

    def get_object_keypoints(self) -> dict[str, list[float]]:
        """Return current 3D keypoints. Replace with FoundationPose / RT-COSMIK."""
        return {
            "1": [0.40, 0.10, 0.05],
            "2": [0.40, -0.10, 0.05],
            "3": [0.50, 0.10, 0.05],
            "4": [0.50, -0.10, 0.05],
            "5": [0.60, 0.00, 0.10],
        }

    def get_robot_state(self) -> np.ndarray:
        """Return [q, dq] robot state. Replace with real robot driver."""
        return np.zeros(14)  # 7 pos + 7 vel for Panda

    def get_human_capsules(self):
        """Return human collision capsules from RT-COSMIK. Replace with real pipeline."""
        return None


def run_loop(instruction: str, image_path: str, max_iterations: int = 5):
    """Run the hierarchical VLM-MPPI loop."""

    cfg = Config()

    # ── Initialize components ────────────────────────────────
    logger.info("Initializing VLM planner...")
    planner = VLMPlanner(cfg.vlm)

    logger.info("Initializing MPPI controller (stub)...")
    mppi = MPPIControllerStub()  # TODO: replace with real COSMIK-MPPI

    perception = DemoPerception(image_path)
    execution_history = []

    # ── Main loop ────────────────────────────────────────────
    for iteration in range(1, max_iterations + 1):
        logger.info("=" * 60)
        logger.info("ITERATION %d / %d", iteration, max_iterations)
        logger.info("=" * 60)

        # 1. VLM planning step (slow, ~1 Hz)
        image = perception.get_rgb_image()
        keypoints = perception.get_object_keypoints()

        logger.info("Querying VLM planner...")
        t0 = time.time()
        try:
            plan = planner.plan(
                image=image,
                instruction=instruction,
                keypoints_3d=keypoints,
                execution_history=execution_history,
            )
        except ValueError as e:
            logger.error("VLM planning failed: %s", e)
            continue

        vlm_time = time.time() - t0
        logger.info("VLM response in %.2f s: %s", vlm_time, plan.get("task_description", "?"))

        # Check if done
        if plan.get("done", False):
            logger.info("VLM says task is DONE!")
            break

        # 2. Convert to MPPI goal
        goal_se3 = vlm_plan_to_se3_goal(plan)
        margin = extract_safety_margin(plan, cfg.mppi.default_safety_margin_m)

        mppi.set_goal(goal_se3)
        mppi.set_collision_threshold(margin)

        # 3. MPPI execution step (fast, 50 Hz) — simulate a few seconds
        n_mppi_steps = int(2.0 * cfg.mppi.control_frequency_hz)  # 2 seconds
        logger.info("Running MPPI for %d steps (%.1f s)...", n_mppi_steps, 2.0)

        for step in range(n_mppi_steps):
            state = perception.get_robot_state()
            capsules = perception.get_human_capsules()
            torques = mppi.compute_control(state, capsules)
            mppi.apply_torques(torques)
            # In real deployment: time.sleep(1.0 / cfg.mppi.control_frequency_hz)

        # 4. Record history for next VLM iteration
        execution_history.append(plan)
        logger.info("Step completed. History length: %d", len(execution_history))

    logger.info("Loop finished after %d iterations.", len(execution_history))


def main():
    parser = argparse.ArgumentParser(description="VLM-MPPI control loop")
    parser.add_argument("--instruction", type=str, default="Push the box left, then place the cup.")
    parser.add_argument("--image", type=str, default="scene.jpg")
    parser.add_argument("--max-iter", type=int, default=5)
    args = parser.parse_args()

    if not Path(args.image).exists():
        logger.warning("Image %s not found. Using a blank 640x480 image for demo.", args.image)
        Image.new("RGB", (640, 480), (200, 200, 200)).save(args.image)

    run_loop(args.instruction, args.image, args.max_iter)


if __name__ == "__main__":
    main()
