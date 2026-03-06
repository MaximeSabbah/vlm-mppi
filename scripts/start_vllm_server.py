#!/usr/bin/env python3
"""
Start a vLLM server for fast VLM inference.

This launches an OpenAI-compatible API server that supports
vision-language models with continuous batching and PagedAttention.

Usage:
    python scripts/start_vllm_server.py
    python scripts/start_vllm_server.py --model Qwen/Qwen2.5-VL-32B-Instruct --port 8001

Then query from another terminal:
    python examples/03_vlm_client_vllm.py --image scene.jpg
"""

import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Launch vLLM server")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=4096,
                        help="Max context length. Lower = less VRAM.")
    args = parser.parse_args()

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model,
        "--host", args.host,
        "--port", str(args.port),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--max-model-len", str(args.max_model_len),
        "--trust-remote-code",
    ]

    print(f"Starting vLLM server:")
    print(f"  Model: {args.model}")
    print(f"  URL:   http://{args.host}:{args.port}/v1")
    print(f"  GPU memory utilization: {args.gpu_memory_utilization}")
    print()

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print("Error: vLLM not installed. Run: pip install -r requirements-vllm.txt")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
