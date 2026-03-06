#!/usr/bin/env python3
"""
Example 02: IKER-style keypoint planner.

Given a scene image and 3D keypoints, ask the VLM to output
target keypoint positions — the direct interface to MPPI goals.

Usage:
    python examples/02_vlm_keypoint_planner.py --image scene.jpg
    python examples/02_vlm_keypoint_planner.py --image scene.jpg --instruction "push the box left"
"""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from vlm_mppi.vlm_planner import VLMPlanner
from vlm_mppi.mppi_interface import vlm_plan_to_se3_goal, extract_safety_margin
from vlm_mppi.config import VLMConfig


# ── Simulated keypoints (replace with real perception pipeline) ──

DEMO_KEYPOINTS = {
    # Object: a box (keypoints 1-4)
    "1": [0.40, 0.10, 0.05],
    "2": [0.40, -0.10, 0.05],
    "3": [0.50, 0.10, 0.05],
    "4": [0.50, -0.10, 0.05],
    # Target area marker (keypoint 5)
    "5": [0.60, 0.00, 0.10],
    # Table corner references (keypoints 6-7)
    "6": [0.20, 0.30, 0.00],
    "7": [0.70, -0.30, 0.00],
}


def main():
    parser = argparse.ArgumentParser(description="Keypoint-based VLM planner")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument(
        "--instruction", type=str,
        default="Push the box to the left to make space, then place the cup on the table.",
    )
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    print(f"Loaded image: {args.image}")
    print(f"Instruction: {args.instruction}")
    print(f"Keypoints: {len(DEMO_KEYPOINTS)} points")

    # Initialize VLM
    config = VLMConfig(model_id=args.model)
    planner = VLMPlanner(config)

    # Query for structured plan
    print("\nQuerying VLM for keypoint targets...")
    try:
        plan = planner.plan(
            image=image,
            instruction=args.instruction,
            keypoints_3d=DEMO_KEYPOINTS,
        )
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print("\n=== VLM Plan ===")
    print(json.dumps(plan, indent=2))

    # Convert to MPPI goal
    if not plan.get("done", False):
        goal_se3 = vlm_plan_to_se3_goal(plan)
        margin = extract_safety_margin(plan)
        print(f"\n=== MPPI Goal ===")
        print(f"  Position:       [{goal_se3[0]:.3f}, {goal_se3[1]:.3f}, {goal_se3[2]:.3f}]")
        print(f"  Quaternion:     [{goal_se3[3]:.3f}, {goal_se3[4]:.3f}, {goal_se3[5]:.3f}, {goal_se3[6]:.3f}]")
        print(f"  Safety margin:  {margin:.3f} m")
        print(f"\n  → Feed this T_goal into COSMIK-MPPI's set_goal()")
    else:
        print("\nVLM says the task is done!")


if __name__ == "__main__":
    main()
