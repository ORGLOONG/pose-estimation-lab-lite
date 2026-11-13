import unittest

import cv2
import numpy as np

from src.contour_keypoints import largest_contour, select_keypoints, turning_angles


class KeypointTests(unittest.TestCase):
    def test_circle_selects_no_sharp_corners(self):
        mask = np.zeros((240, 240), dtype=np.uint8)
        cv2.circle(mask, (120, 120), 70, 255, -1)
        points = largest_contour(mask)
        keypoints = select_keypoints(points, threshold_deg=40, min_distance=10, window=6)
        self.assertGreaterEqual(len(points), 300)
        self.assertEqual(len(keypoints), 0)

    def test_square_finds_four_corners_in_expected_regions(self):
        mask = np.zeros((240, 240), dtype=np.uint8)
        cv2.rectangle(mask, (50, 50), (190, 190), 255, -1)
        points = largest_contour(mask)
        angles = turning_angles(points, window=5)
        self.assertEqual(len(angles), len(points))
        keypoints = select_keypoints(points, threshold_deg=45, min_distance=35, window=5, max_points=4)
        self.assertEqual(len(keypoints), 4)
        expected = np.array([[50, 50], [190, 50], [190, 190], [50, 190]])
        for corner in expected:
            self.assertLess(np.min(np.linalg.norm(keypoints[:, :2] - corner, axis=1)), 5)

    def test_adaptive_and_multi_level_modes_select_square_features(self):
        mask = np.zeros((240, 240), dtype=np.uint8)
        cv2.rectangle(mask, (50, 50), (190, 190), 255, -1)
        points = largest_contour(mask)
        common = dict(min_distance=35, window=5, max_points=4)
        adaptive = select_keypoints(points, detection_mode="adaptive", adaptive_percentile=90, **common)
        multi = select_keypoints(points, detection_mode="multi_level", **common)
        self.assertEqual(len(adaptive), 4)
        self.assertEqual(len(multi), 4)

    def test_unknown_detection_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            select_keypoints(np.array([[0, 0], [1, 0], [1, 1]]), detection_mode="unknown")


if __name__ == "__main__":
    unittest.main()
