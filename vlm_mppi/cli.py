"""CLI entry point: run Embodied-R1 on an image from the terminal."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from vlm_mppi.config import ModelConfig
from vlm_mppi.model import Ability, EmbodiedR1
from vlm_mppi.viz import draw_results, print_results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Test Embodied-R1 spatial reasoning on a scene image."
    )
    parser.add_argument("image", type=Path, help="Path to an RGB image")
    parser.add_argument("instruction", type=str, help="Task instruction")
    parser.add_argument(
        "--abilities",
        nargs="+",
        choices=["OFG", "RRG", "VTG", "REG"],
        default=["OFG", "RRG", "VTG"],
    )
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--offline", action="store_true",
                        help="Skip HuggingFace network checks (use local cache)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s",
                        datefmt="%H:%M:%S")

    if not args.image.exists():
        print(f"Error: image not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    abilities = [Ability(a) for a in args.abilities]
    model = EmbodiedR1.load(ModelConfig(local_files_only=args.offline))
    results = model.point_all(args.image, args.instruction, abilities)

    print_results(results)

    if args.verbose:
        for ability, r in results.items():
            print(f"\n[{ability.value} raw output]\n{r.raw_output}")

    save_path = args.save or Path("outputs") / f"{args.image.stem}_results.png"
    draw_results(args.image, results, save_path=save_path, show=not args.no_show)


if __name__ == "__main__":
    main()
