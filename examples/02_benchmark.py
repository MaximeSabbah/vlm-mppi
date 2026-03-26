"""Benchmark Embodied-R1 on the official example images from Yuan et al., ICLR 2026.

Each image targets one specific ability (as in the paper's evaluation):
  - HANDAL dataset    → OFG  (object affordance / functional grasp point)
  - RoboRefit dataset → REG  (referring expression grounding)
  - Kitchen scene     → RRG  (target placement region)
  - Block stacking    → VTG  (visual trace / 8-waypoint trajectory)

Running this gives you:
  1. Visual output per ability (saved to outputs/benchmark/)
  2. Inference time per ability
  3. Raw model output for inspection

Usage:
    python examples/02_benchmark.py
    python examples/02_benchmark.py --verbose    # also print raw model output
    python examples/02_benchmark.py --runs 3     # repeat each query N times for stable timing
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from vlm_mppi.config import ModelConfig
from vlm_mppi.model import Ability, EmbodiedR1
from vlm_mppi.viz import draw_results, print_results

# Official test cases from inference_example.py (pickxiguapi/Embodied-R1)
BENCHMARK = [
    {
        "ability":     Ability.OFG,
        "image":       Path("data/official_examples/handal_090002.png"),
        "instruction": "loosening stuck bolts",
        "dataset":     "HANDAL",
        "description": "Find the functional grasp region on a tool",
    },
    {
        "ability":     Ability.REG,
        "image":       Path("data/official_examples/roborefit_18992.png"),
        "instruction": "bring me the camel model",
        "dataset":     "RoboRefit",
        "description": "Locate the referred object in a cluttered scene",
    },
    {
        "ability":     Ability.RRG,
        "image":       Path("data/official_examples/put pepper in pan.png"),
        "instruction": "put pepper in pan",
        "dataset":     "Custom (paper)",
        "description": "Identify the target placement region (inside the pan)",
    },
    {
        "ability":     Ability.VTG,
        "image":       Path("data/official_examples/put the red block on top of the yellow block.png"),
        "instruction": "put the red block on top of the yellow block",
        "dataset":     "Custom (paper)",
        "description": "Generate 8-waypoint trajectory from block to stacking target",
    },
]


def run_single(model: EmbodiedR1, case: dict, verbose: bool) -> dict:
    ab    = case["ability"]
    image = case["image"]
    instr = case["instruction"]

    print(f"\n{'━'*60}")
    print(f"  [{ab.value}]  {case['dataset']}")
    print(f"  Instruction : {instr}")
    print(f"  Task        : {case['description']}")
    print(f"{'━'*60}")

    t0 = time.perf_counter()
    result = model.point(image, instr, ab)
    elapsed = time.perf_counter() - t0

    print(f"  Inference   : {elapsed:.2f}s  ({1/elapsed:.3f} Hz)")
    print(f"  Points found: {result.n_points}")

    if result.reasoning:
        snippet = result.reasoning[:300] + ("…" if len(result.reasoning) > 300 else "")
        print(f"\n  Reasoning:\n    {snippet}")

    for i, (x, y) in enumerate(result.points_px):
        print(f"  Point {i}: ({x:.0f}, {y:.0f}) px")

    if verbose:
        print(f"\n  [raw output]\n{result.raw_output}")

    return {"ability": ab.value, "dataset": case["dataset"],
            "instruction": instr, "n_points": result.n_points,
            "time_s": elapsed, "result": result, "image": image}


def print_summary(rows: list[dict]) -> None:
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Ability':<6}  {'Dataset':<15}  {'Points':>6}  {'Time (s)':>9}  {'Hz':>7}")
    print(f"  {'-'*55}")
    for r in rows:
        print(f"  {r['ability']:<6}  {r['dataset']:<15}  {r['n_points']:>6}  {r['time_s']:>9.2f}  {1/r['time_s']:>7.3f}")
    times = [r["time_s"] for r in rows]
    print(f"  {'-'*55}")
    print(f"  {'Mean':<6}  {'':15}  {'':6}  {statistics.mean(times):>9.2f}  {1/statistics.mean(times):>7.3f}")
    print(f"  {'Min':<6}  {'':15}  {'':6}  {min(times):>9.2f}  {1/min(times):>7.3f}")
    print(f"  {'Max':<6}  {'':15}  {'':6}  {max(times):>9.2f}  {1/max(times):>7.3f}")
    print(f"{'='*60}")

    print(f"""
  Control integration guidance
  ─────────────────────────────
  Single-step grounding (OFG/REG/RRG): ~{statistics.mean(r["time_s"] for r in rows if r["ability"] != "VTG"):.1f}s
    → fire once per subtask, MPC executes continuously underneath

  Trajectory (VTG, 8 waypoints):       ~{next(r["time_s"] for r in rows if r["ability"] == "VTG"):.1f}s
    → replanning budget per motion primitive

  At 15s average subtask execution, VLM overhead ≈ {100*statistics.mean(times)/(statistics.mean(times)+15):.0f}%
""")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true", help="Print full raw model output")
    parser.add_argument("--runs", type=int, default=1, help="Repeat each case N times and average")
    parser.add_argument("--save-dir", type=Path, default=Path("outputs/benchmark"))
    args = parser.parse_args()

    args.save_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model…")
    model = EmbodiedR1.load(ModelConfig(local_files_only=True))

    all_rows: list[dict] = []

    for case in BENCHMARK:
        run_times = []
        last_row = None

        for run in range(args.runs):
            if args.runs > 1:
                print(f"  run {run+1}/{args.runs}")
            row = run_single(model, case, verbose=args.verbose and run == 0)
            run_times.append(row["time_s"])
            last_row = row

        if args.runs > 1:
            last_row["time_s"] = statistics.mean(run_times)
            print(f"  → mean over {args.runs} runs: {last_row['time_s']:.2f}s")

        # Save visualization
        save_path = args.save_dir / f"{case['ability'].value}_{case['image'].stem}.png"
        draw_results(
            case["image"],
            {case["ability"]: last_row["result"]},
            save_path=save_path,
            show=False,
        )
        print(f"  Saved: {save_path}")
        all_rows.append(last_row)

    print_summary(all_rows)


if __name__ == "__main__":
    main()
