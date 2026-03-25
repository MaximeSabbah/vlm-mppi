"""Example: Run all Embodied-R1 pointing abilities on a scene image.

Usage:
    python examples/01_test_embodied_r1.py
    python examples/01_test_embodied_r1.py --image my_scene.png --instruction "pick up the mug"
"""

from pathlib import Path

from vlm_mppi.model import Ability, EmbodiedR1
from vlm_mppi.viz import draw_results, print_results


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, default=Path("data/sample_images/scene.png"))
    parser.add_argument("--instruction", type=str, default="pick up the red block")
    args = parser.parse_args()

    if not args.image.exists():
        print(f"Image not found: {args.image}")
        print("Place a test image in data/sample_images/ or pass --image <path>")
        return

    # Load model (downloads ~6 GB on first run)
    model = EmbodiedR1.load()

    # Run all abilities
    results = model.point_all(args.image, args.instruction)

    # Display results
    print_results(results)
    draw_results(args.image, results, save_path=f"outputs/{args.image.stem}_results.png")


if __name__ == "__main__":
    main()
