"""Small, dependency-light pinhole camera geometry helpers."""

from __future__ import annotations

import numpy as np


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Build a homogeneous transform from a 3x3 rotation and 3-vector."""
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64).reshape(-1)
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("expected a 3x3 rotation and a 3-vector translation")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    """Invert a rigid homogeneous transform (rotation must be orthonormal)."""
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4) or not np.allclose(transform[3], [0, 0, 0, 1]):
        raise ValueError("transform must be a 4x4 homogeneous matrix")
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = transform[:3, :3].T
    inverse[:3, 3] = -inverse[:3, :3] @ transform[:3, 3]
    return inverse


def camera_intrinsics(width: int, height: int, fov_y_deg: float) -> np.ndarray:
    """Compute a square-pixel pinhole calibration matrix from vertical FOV."""
    if width <= 0 or height <= 0 or not 0 < fov_y_deg < 180:
        raise ValueError("positive image dimensions and a FOV in (0, 180) are required")
    focal = 0.5 * height / np.tan(np.deg2rad(fov_y_deg) / 2)
    return np.array([[focal, 0, (width - 1) / 2],
                     [0, focal, (height - 1) / 2],
                     [0, 0, 1]], dtype=np.float64)


def project_points(points_camera: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Project camera-frame XYZ points; points behind the camera become NaN."""
    points = np.asarray(points_camera, dtype=np.float64).reshape(-1, 3)
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    if intrinsics.shape != (3, 3):
        raise ValueError("intrinsics must be a 3x3 matrix")
    pixels = (intrinsics @ points.T).T
    valid = points[:, 2] > 0
    result = np.full((len(points), 2), np.nan, dtype=np.float64)
    result[valid] = pixels[valid, :2] / pixels[valid, 2, None]
    return result
