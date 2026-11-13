"""Curvature-based keypoint selection for closed 2D contours.

The threshold, adaptive-percentile, multi-level, and distance-filtering workflow
is adapted from PCGSplineCurvatureDetector in Isaac Sim's 6D Pose Keypoint
Detector (NVIDIA, Apache-2.0). This standalone extraction removes simulator
dependencies and changes contour representation and candidate suppression.
See NOTICE and LICENSE-APACHE-2.0.
"""

from __future__ import annotations

import cv2
import numpy as np


def largest_contour(mask: np.ndarray) -> np.ndarray:
    """Return the largest external contour as an (N, 2) float array."""
    if mask.ndim != 2:
        raise ValueError("mask must be a single-channel image")
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.empty((0, 2), dtype=np.float64)
    return max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)


def turning_angles(contour: np.ndarray, window: int = 4) -> np.ndarray:
    """Measure local direction change in degrees at each point of a closed contour."""
    points = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
    if window < 1:
        raise ValueError("window must be positive")
    if len(points) < 2 * window + 1:
        return np.zeros(len(points), dtype=np.float64)
    before = points - np.roll(points, window, axis=0)
    after = np.roll(points, -window, axis=0) - points
    denom = np.linalg.norm(before, axis=1) * np.linalg.norm(after, axis=1)
    cosine = np.divide(np.einsum("ij,ij->i", before, after), denom,
                       out=np.ones(len(points)), where=denom > 1e-12)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def select_keypoints(
    contour: np.ndarray, threshold_deg: float = 28.0, min_distance: float = 12.0,
    window: int = 4, max_points: int = 0, detection_mode: str = "threshold",
    adaptive_percentile: float = 85.0, high_threshold_deg: float = 45.0,
    medium_threshold_deg: float = 25.0, low_threshold_deg: float = 10.0,
) -> np.ndarray:
    """Pick curvature maxima using fixed, adaptive, or multi-level thresholds.

    Returns an (M, 3) array of x, y, and turning angle, strongest first.
    Multi-level mode accepts candidates at or above the low threshold; their
    exact level can be derived from each returned turning angle.
    """
    points = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
    if len(points) == 0:
        return np.empty((0, 3), dtype=np.float64)
    angles = turning_angles(points, window)
    local_max = (angles >= np.roll(angles, 1)) & (angles >= np.roll(angles, -1))
    if detection_mode == "threshold":
        threshold = threshold_deg
    elif detection_mode == "adaptive":
        if not 0 <= adaptive_percentile <= 100:
            raise ValueError("adaptive_percentile must be in [0, 100]")
        threshold = float(np.percentile(angles, adaptive_percentile))
    elif detection_mode == "multi_level":
        if not high_threshold_deg >= medium_threshold_deg >= low_threshold_deg:
            raise ValueError("multi-level thresholds must be ordered high >= medium >= low")
        threshold = low_threshold_deg
    else:
        raise ValueError("detection_mode must be 'threshold', 'adaptive', or 'multi_level'")
    candidates = np.flatnonzero(local_max & (angles >= threshold))
    candidates = candidates[np.argsort(angles[candidates])[::-1]]
    chosen: list[int] = []
    for idx in candidates:
        if all(np.linalg.norm(points[idx] - points[kept]) >= min_distance for kept in chosen):
            chosen.append(int(idx))
            if max_points > 0 and len(chosen) >= max_points:
                break
    if not chosen:
        return np.empty((0, 3), dtype=np.float64)
    return np.column_stack((points[chosen], angles[chosen]))
