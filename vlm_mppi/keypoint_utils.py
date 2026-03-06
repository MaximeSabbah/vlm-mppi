"""Keypoint utilities: 3D-to-2D projection and image overlay."""

import cv2
import numpy as np
from PIL import Image


def project_3d_to_2d(
    points_3d: np.ndarray,
    camera_matrix: np.ndarray,
    rvec: np.ndarray = None,
    tvec: np.ndarray = None,
    dist_coeffs: np.ndarray = None,
) -> np.ndarray:
    """
    Project Nx3 world-frame points onto the image plane.

    Args:
        points_3d: (N, 3) array of 3D points.
        camera_matrix: 3x3 intrinsic matrix.
        rvec: Rodrigues rotation vector (3,). Defaults to identity.
        tvec: Translation vector (3,). Defaults to zero.
        dist_coeffs: Distortion coefficients. Defaults to zero.

    Returns:
        (N, 2) array of pixel coordinates.
    """
    if rvec is None:
        rvec = np.zeros(3, dtype=np.float64)
    if tvec is None:
        tvec = np.zeros(3, dtype=np.float64)
    if dist_coeffs is None:
        dist_coeffs = np.zeros(5, dtype=np.float64)

    pts = points_3d.reshape(-1, 1, 3).astype(np.float64)
    pts_2d, _ = cv2.projectPoints(pts, rvec, tvec, camera_matrix, dist_coeffs)
    return pts_2d.reshape(-1, 2)


def overlay_keypoints(
    image: np.ndarray | Image.Image,
    keypoints_3d: dict[str, list[float]],
    camera_matrix: np.ndarray,
    rvec: np.ndarray = None,
    tvec: np.ndarray = None,
    dist_coeffs: np.ndarray = None,
    radius: int = 10,
    color: tuple[int, int, int] = (0, 255, 0),
    font_scale: float = 0.7,
) -> Image.Image:
    """
    Draw numbered keypoint markers on a scene image.

    Args:
        image: BGR numpy array or PIL Image.
        keypoints_3d: Dict {"label": [x, y, z], ...}.
        camera_matrix: 3x3 intrinsic matrix.
        radius: Circle radius in pixels.
        color: BGR color tuple.

    Returns:
        PIL Image with keypoints drawn.
    """
    if isinstance(image, Image.Image):
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        img = image.copy()

    labels = list(keypoints_3d.keys())
    pts_3d = np.array([keypoints_3d[l] for l in labels])
    pts_2d = project_3d_to_2d(pts_3d, camera_matrix, rvec, tvec, dist_coeffs)

    for label, (px, py) in zip(labels, pts_2d):
        x, y = int(px), int(py)
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            cv2.circle(img, (x, y), radius, color, -1)
            cv2.circle(img, (x, y), radius, (0, 0, 0), 2)  # outline
            cv2.putText(
                img, str(label), (x + radius + 4, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2,
            )

    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def keypoints_to_centroid(keypoints: dict[str, list[float]]) -> np.ndarray:
    """Compute the 3D centroid of a set of keypoints."""
    positions = np.array(list(keypoints.values()))
    return positions.mean(axis=0)
