#!/usr/bin/env python3
"""
Example 01: Basic VLM test.

Feed a scene image to Qwen2.5-VL-7B and get a structured JSON plan.
This is the simplest possible test — no keypoints, no MPPI.

Usage:
    python examples/01_test_vlm_basic.py --image scene.jpg
    python examples/01_test_vlm_basic.py --image scene.jpg --model Qwen/Qwen2.5-VL-32B-Instruct
"""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vlm_mppi.vlm_planner import VLMPlanner
from vlm_mppi.config import VLMConfig


def main():
    parser = argparse.ArgumentParser(description="Test VLM on a scene image")
    parser.add_argument("--image", type=str, required=True, help="Path to scene image")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--instruction", type=str, default="Describe the scene and list all objects you see.")
    args = parser.parse_args()

    # Load image
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: image not found at {image_path}")
        sys.exit(1)

    image = Image.open(image_path).convert("RGB")
    print(f"Loaded image: {image_path} ({image.size[0]}x{image.size[1]})")

    # Initialize VLM
    config = VLMConfig(model_id=args.model)
    planner = VLMPlanner(config)

    # Simple query
    prompt = f"""Look at this scene from a robot workspace. Answer in JSON only, no markdown.

Task: "{args.instruction}"

Output format:
{{
    "objects_detected": ["list", "of", "objects"],
    "scene_description": "brief description",
    "task_feasible": true or false,
    "suggested_steps": ["step 1", "step 2"]
}}"""

    print("\nQuerying VLM...")
    raw_response = planner.query(image, prompt)

    print("\n=== Raw Response ===")
    print(raw_response)

    # Try to parse JSON
    try:
        parsed = planner._parse_json_response(raw_response)
        print("\n=== Parsed JSON ===")
        print(json.dumps(parsed, indent=2))
    except (json.JSONDecodeError, ValueError) as e:
        print(f"\nCould not parse as JSON: {e}")
        print("This is normal for first tests — adjust the prompt if needed.")


if __name__ == "__main__":
    main()
