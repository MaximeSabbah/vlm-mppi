#!/usr/bin/env python3
"""
Example 03: Query a running vLLM server for fast inference.

This is the production-friendly approach: vLLM serves the model
with continuous batching and PagedAttention for ~3-5x faster inference.

Prerequisites:
    1. Start the server:  python scripts/start_vllm_server.py
    2. Then run this:     python examples/03_vlm_client_vllm.py --image scene.jpg

Usage:
    python examples/03_vlm_client_vllm.py --image scene.jpg --instruction "pick up the cup"
"""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from vlm_mppi.vlm_planner import VLMPlanner
from vlm_mppi.config import VLMConfig


DEMO_KEYPOINTS = {
    "1": [0.40, 0.10, 0.05],
    "2": [0.40, -0.10, 0.05],
    "3": [0.50, 0.10, 0.05],
    "4": [0.50, -0.10, 0.05],
}


def main():
    parser = argparse.ArgumentParser(description="Query vLLM server")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--instruction", type=str, default="Pick up the red object.")
    parser.add_argument("--server", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")

    config = VLMConfig(
        model_id=args.model,
        use_vllm=True,
        vllm_base_url=args.server,
    )
    planner = VLMPlanner(config)

    print(f"Querying vLLM at {args.server} ...")
    try:
        plan = planner.plan(
            image=image,
            instruction=args.instruction,
            keypoints_3d=DEMO_KEYPOINTS,
        )
        print("\n=== Plan ===")
        print(json.dumps(plan, indent=2))
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the vLLM server is running: python scripts/start_vllm_server.py")
        sys.exit(1)


if __name__ == "__main__":
    main()
