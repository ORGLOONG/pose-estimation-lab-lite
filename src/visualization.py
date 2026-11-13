"""Generate a synthetic contour and a pinhole-projection illustration."""

from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np

from camera_geometry import camera_intrinsics, project_points
from contour_keypoints import largest_contour, select_keypoints


def make_examples(output_dir: str | Path) -> tuple[Path, Path]:
    """Create deterministic, synthetic input/output PNGs without external assets."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image = np.full((480, 640, 3), 248, dtype=np.uint8)
    mask = np.zeros((480, 640), dtype=np.uint8)
    cv2.circle(mask, (240, 240), 130, 255, -1)
    # Four shallow notches create stable, inspectable curvature features.
    for center in ((240, 110), (370, 240), (240, 370), (110, 240)):
        cv2.circle(mask, center, 15, 0, -1)
    image[mask > 0] = (90, 155, 220)
    contour = largest_contour(mask)
    keypoints = select_keypoints(contour, threshold_deg=24, min_distance=60, window=4, max_points=16)
    input_path, output_path = output_dir / "sample_input.png", output_dir / "sample_output.png"
    cv2.imwrite(str(input_path), image)

    result = image.copy()
    cv2.drawContours(result, [contour.astype(np.int32).reshape(-1, 1, 2)], -1, (50, 95, 130), 2)
    for x, y, _ in keypoints:
        cv2.circle(result, (round(x), round(y)), 6, (20, 35, 225), -1, cv2.LINE_AA)
    cv2.putText(result, f"curvature keypoints: {len(keypoints)}", (24, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (25, 25, 25), 2, cv2.LINE_AA)

    # Project the corners of a synthetic cube into a second, inset view.
    cube = np.array([[-0.5, -0.5, 0], [0.5, -0.5, 0], [0.5, 0.5, 0], [-0.5, 0.5, 0],
                     [-0.5, -0.5, 1], [0.5, -0.5, 1], [0.5, 0.5, 1], [-0.5, 0.5, 1]])
    cube[:, 2] += 3.0
    K = camera_intrinsics(640, 480, 55)
    pixels = project_points(cube, K) * 0.42 + np.array([390, 320])
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    for a, b in edges:
        p1, p2 = pixels[a].round().astype(int), pixels[b].round().astype(int)
        cv2.line(result, tuple(p1), tuple(p2), (20, 145, 65), 2, cv2.LINE_AA)
    for x, y in pixels:
        cv2.circle(result, (round(x), round(y)), 3, (210, 75, 35), -1, cv2.LINE_AA)
    cv2.putText(result, "synthetic cube projection", (390, 370),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (25, 25, 25), 1, cv2.LINE_AA)
    cv2.imwrite(str(output_path), result)
    return input_path, output_path


if __name__ == "__main__":
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "examples"
    for path in make_examples(base):
        print(path)
