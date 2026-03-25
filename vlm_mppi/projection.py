"""2D → 3D projection for Embodied-R1 pointing outputs."""

from __future__ import annotations

import numpy as np

from vlm_mppi.config import CameraConfig


def project_to_3d(
    points_px: list[tuple[float, float]],
    depth: np.ndarray,
    camera: CameraConfig,
    T_cam_to_base: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Back-project 2D pixel points to 3D using a depth map.

    Args:
        points_px: (u, v) pixel coordinates from Embodied-R1.
        depth: (H, W) depth image in meters.
        camera: pinhole camera intrinsics.
        T_cam_to_base: 4×4 transform camera → robot base (identity if None).

    Returns:
        List of 3D points as (3,) arrays in robot base frame.
        Points with invalid depth are silently skipped.
    """
    if T_cam_to_base is None:
        T_cam_to_base = np.eye(4)

    H, W = depth.shape[:2]
    points_3d = []

    for u, v in points_px:
        ui, vi = int(round(u)), int(round(v))
        ui = np.clip(ui, 0, W - 1)
        vi = np.clip(vi, 0, H - 1)

        z = float(depth[vi, ui])
        if z <= 0 or np.isnan(z):
            continue

        x = (u - camera.cx) * z / camera.fx
        y = (v - camera.cy) * z / camera.fy

        p_cam = np.array([x, y, z, 1.0])
        p_base = T_cam_to_base @ p_cam
        points_3d.append(p_base[:3])

    return points_3d
