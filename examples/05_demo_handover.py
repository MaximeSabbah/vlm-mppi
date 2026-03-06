#!/usr/bin/env python3
"""
Example 05: Voice-guided handover demo with cost reflection.

This demonstrates the target demo scenario:
  1. Human says "give me the red cup"
  2. VLM identifies the cup, plans grasp + handover
  3. MPPI executes with human avoidance
  4. Cost reflection feeds execution stats back to VLM for replanning

This example simulates the full loop with stubs for hardware.
Replace stubs with real RT-COSMIK + COSMIK-MPPI + RealSense when integrating.

Usage:
    python examples/05_demo_handover.py --image scene.jpg
    python examples/05_demo_handover.py --image scene.jpg --instruction "give me the blue box"
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from vlm_mppi.vlm_planner import VLMPlanner
from vlm_mppi.mppi_interface import MPPIControllerStub, vlm_plan_to_se3_goal, extract_safety_margin
from vlm_mppi.cost_reflection import CostReflection, CostSnapshot, build_reflection_from_mppi_log
from vlm_mppi.config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("handover_demo")


# ── Simulated scene ──────────────────────────────────────────────

SCENE_OBJECTS = {
    "red_cup":      {"keypoints": {"1": [0.45, 0.15, 0.05], "2": [0.45, 0.15, 0.12]},
                     "color": "red", "type": "cup", "graspable": True},
    "blue_box":     {"keypoints": {"3": [0.50, -0.10, 0.03], "4": [0.55, -0.10, 0.03]},
                     "color": "blue", "type": "box", "graspable": True},
    "green_bottle": {"keypoints": {"5": [0.35, 0.00, 0.04], "6": [0.35, 0.00, 0.18]},
                     "color": "green", "type": "bottle", "graspable": True},
}

# Simulated human hand position (from RT-COSMIK)
HUMAN_RIGHT_HAND = [0.55, 0.25, 0.20]


def get_all_keypoints() -> dict[str, list[float]]:
    """Flatten all object keypoints + human hand into one dict."""
    kp = {}
    for obj_name, obj_data in SCENE_OBJECTS.items():
        for label, pos in obj_data["keypoints"].items():
            kp[label] = pos
    # Add human hand as a reference keypoint
    kp["H1"] = HUMAN_RIGHT_HAND
    return kp


def build_handover_prompt(instruction: str, keypoints: dict, reflection_text: str = "") -> str:
    """Build the VLM prompt for the handover task."""

    kp_lines = []
    for label, pos in keypoints.items():
        kp_lines.append(f"  Keypoint {label}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
    kp_desc = "\n".join(kp_lines)

    # Object descriptions for grounding
    obj_desc = []
    for obj_name, obj_data in SCENE_OBJECTS.items():
        labels = list(obj_data["keypoints"].keys())
        obj_desc.append(f"  {obj_name} ({obj_data['color']} {obj_data['type']}): keypoints {labels}")
    obj_desc.append(f"  human_right_hand: keypoint [H1]")
    obj_text = "\n".join(obj_desc)

    prompt = f"""You are a robotic manipulation planner for a Franka Panda arm in a shared
workspace with a human. A human is present and tracked in real-time.

Objects in the scene:
{obj_text}

Current keypoint positions (meters, X=forward, Y=left, Z=up):
{kp_desc}

The robot can grasp small objects and hand them to the human.
To hand an object to the human, move the object's keypoints near keypoint H1 (human hand)
with a small offset (e.g., 5cm closer to the robot for safety).

Task: "{instruction}"
"""

    if reflection_text:
        prompt += f"""
Previous execution feedback:
{reflection_text}

Use this feedback to improve your plan. If the previous attempt failed, adjust the strategy.
"""

    prompt += """
Output ONLY valid JSON (no markdown, no extra text):
{
    "task_description": "what to do in this step",
    "phase": "reach" or "grasp" or "handover" or "done",
    "interaction_object": "object name",
    "grasp_required": true or false,
    "target_keypoints": {
        "1": [x, y, z]
    },
    "safety_margin_m": 0.08,
    "done": false
}

Rules:
- For "reach" phase: move end-effector near the object.
- For "grasp" phase: close gripper on object.
- For "handover" phase: move grasped object near H1 (human hand), offset by [−0.05, 0, 0] for safety.
- Increase safety_margin_m to 0.10 when near the human.
- Set done=true only after successful handover.
"""
    return prompt


def simulate_mppi_execution(goal: np.ndarray, n_steps: int = 100) -> CostReflection:
    """
    Simulate MPPI execution and produce cost reflection.
    Replace with real COSMIK-MPPI execution + logging.
    """
    # Simulate decreasing distance to goal with some noise
    initial_dist = np.linalg.norm(goal[:3] - np.array([0.3, 0.0, 0.3]))  # from home pose
    goal_dists = []
    collision_dists = []
    speeds = []

    for i in range(n_steps):
        t = i / n_steps
        dist = initial_dist * (1 - t) * (1 + 0.05 * np.random.randn())
        dist = max(dist, 0.01)
        goal_dists.append(dist)
        collision_dists.append(0.08 + 0.03 * np.random.randn())  # ~8cm from human
        speeds.append(0.15 * (1 - 0.5 * t))  # slowing down near target

    success = goal_dists[-1] < 0.05
    return build_reflection_from_mppi_log(
        goal_dists, collision_dists, speeds,
        success=success,
        failure_reason="" if success else "Did not converge to target within threshold.",
    )


def run_demo(instruction: str, image_path: str):
    """Run the handover demo with cost reflection loop."""

    cfg = Config()
    planner = VLMPlanner(cfg.vlm)
    mppi = MPPIControllerStub()

    image = Image.open(image_path).convert("RGB") if Path(image_path).exists() else Image.new("RGB", (640, 480), (200, 200, 200))

    keypoints = get_all_keypoints()
    execution_history = []
    reflection_text = ""

    phases = ["reach", "grasp", "handover", "done"]
    max_iterations = 5

    for iteration in range(1, max_iterations + 1):
        logger.info("=" * 50)
        logger.info("ITERATION %d — Instruction: '%s'", iteration, instruction)
        logger.info("=" * 50)

        # 1. Build prompt with cost reflection from previous execution
        prompt = build_handover_prompt(instruction, keypoints, reflection_text)

        # 2. Query VLM
        logger.info("Querying VLM...")
        t0 = time.time()
        raw = planner.query(image, prompt)
        vlm_time = time.time() - t0
        logger.info("VLM responded in %.2fs", vlm_time)

        # 3. Parse
        try:
            plan = planner._parse_json_response(raw)
        except (json.JSONDecodeError, ValueError):
            logger.warning("Failed to parse VLM output, retrying...")
            continue

        logger.info("Phase: %s | Object: %s | Description: %s",
                     plan.get("phase", "?"),
                     plan.get("interaction_object", "?"),
                     plan.get("task_description", "?"))

        if plan.get("done", False):
            logger.info("Task DONE — handover complete!")
            break

        # 4. Convert to MPPI goal and execute
        if plan.get("target_keypoints"):
            goal = vlm_plan_to_se3_goal(plan)
            margin = extract_safety_margin(plan, 0.08)
            mppi.set_goal(goal)
            mppi.set_collision_threshold(margin)

            logger.info("Executing MPPI (goal: [%.3f, %.3f, %.3f], margin: %.3f)",
                        goal[0], goal[1], goal[2], margin)

            # 5. Simulate execution and get cost reflection
            reflection = simulate_mppi_execution(goal)
            reflection_text = reflection.to_text()

            logger.info("Execution: success=%s, final_dist=%.3f, min_collision=%.3f",
                        reflection.task_success,
                        reflection.summary()["final_goal_dist"],
                        reflection.summary()["min_collision_dist"])

        execution_history.append(plan)

    if not any(p.get("done", False) for p in execution_history):
        logger.warning("Task not completed within %d iterations.", max_iterations)


def main():
    parser = argparse.ArgumentParser(description="Handover demo with cost reflection")
    parser.add_argument("--image", type=str, default="scene.jpg")
    parser.add_argument("--instruction", type=str, default="Give me the red cup")
    args = parser.parse_args()

    if not Path(args.image).exists():
        logger.info("No image found at %s, creating blank placeholder.", args.image)
        Image.new("RGB", (640, 480), (200, 200, 200)).save(args.image)

    run_demo(args.instruction, args.image)


if __name__ == "__main__":
    main()
