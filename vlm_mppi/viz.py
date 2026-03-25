"""Visualization helpers for Embodied-R1 pointing results."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np

from vlm_mppi.model import Ability, PointingResult

COLORS: dict[Ability, str] = {
    Ability.OFG: "#e74c3c",  # red
    Ability.RRG: "#3498db",  # blue
    Ability.REG: "#2ecc71",  # green
    Ability.VTG: "#f39c12",  # orange
}

LABELS: dict[Ability, str] = {
    Ability.OFG: "Grasp (OFG)",
    Ability.RRG: "Place (RRG)",
    Ability.REG: "Object (REG)",
    Ability.VTG: "Trace (VTG)",
}


def draw_results(
    image_path: str | Path,
    results: dict[Ability, PointingResult],
    save_path: Optional[str | Path] = None,
    show: bool = True,
    figsize: tuple[int, int] = (12, 9),
) -> plt.Figure:
    """Overlay pointing results on an image.

    Args:
        image_path: path to the original RGB image.
        results: mapping from ability → PointingResult.
        save_path: if set, save the figure to this path.
        show: whether to call plt.show().
        figsize: matplotlib figure size.

    Returns:
        The matplotlib Figure object.
    """
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(img)

    for ability, result in results.items():
        if not result.has_points:
            continue

        pts = np.array(result.points_px)
        color = COLORS[ability]
        label = LABELS[ability]

        if ability == Ability.VTG and len(pts) > 1:
            ax.plot(
                pts[:, 0], pts[:, 1], "-o",
                color=color, markersize=4, linewidth=2, label=label,
            )
            ax.plot(pts[0, 0], pts[0, 1], "s", color="#2ecc71", markersize=10)
            ax.plot(pts[-1, 0], pts[-1, 1], "*", color="#e74c3c", markersize=14)
        else:
            ax.scatter(
                pts[:, 0], pts[:, 1],
                c=color, s=120, zorder=5, edgecolors="white", linewidths=2, label=label,
            )

    ax.legend(loc="upper right", fontsize=11, framealpha=0.8)
    ax.axis("off")
    fig.tight_layout()

    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig


def print_results(results: dict[Ability, PointingResult]) -> None:
    """Pretty-print pointing results to stdout."""
    for ability, r in results.items():
        print(f"\n{'─' * 50}")
        print(f"  {LABELS[ability]}  ({r.n_points} point{'s' if r.n_points != 1 else ''})")
        print(f"{'─' * 50}")
        if r.reasoning:
            # Truncate long reasoning for display
            text = r.reasoning[:300] + ("..." if len(r.reasoning) > 300 else "")
            print(f"  Reasoning: {text}")
        for i, (u, v) in enumerate(r.points_px):
            print(f"  Point {i}: ({u:.1f}, {v:.1f}) px")
