"""Regression tests for contour selection, camera geometry and demo output."""
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from contour_keypoints import largest_contour, turning_angles, select_keypoints
from camera_geometry import make_transform, invert_transform, camera_intrinsics, project_points
from visualization import make_examples


class ContourContracts(unittest.TestCase):
    def test_empty_mask(self):
        self.assertEqual(largest_contour(np.zeros((32, 32), np.uint8)).shape, (0, 2))

    def test_largest_component(self):
        mask = np.zeros((100, 120), np.uint8)
        cv2.rectangle(mask, (10, 10), (60, 60), 255, -1)
        cv2.rectangle(mask, (90, 80), (95, 85), 255, -1)
        contour = largest_contour(mask)
        np.testing.assert_array_equal(contour.min(axis=0), [10, 10])
        np.testing.assert_array_equal(contour.max(axis=0), [60, 60])

    def test_known_right_angles(self):
        square = np.array([[0, 0], [20, 0], [20, 20], [0, 20]])
        np.testing.assert_allclose(turning_angles(square, window=1), 90)

    def test_translation_invariance(self):
        polygon = np.array([[0, 0], [30, 0], [40, 10], [20, 40], [0, 20]])
        np.testing.assert_allclose(turning_angles(polygon, 1), turning_angles(polygon + [123, -42], 1))

    def test_selection_modes_and_spacing(self):
        square = np.array([[0, 0], [20, 0], [20, 20], [0, 20]])
        for mode in ("threshold", "adaptive", "multi_level"):
            with self.subTest(mode=mode):
                points = select_keypoints(square, window=1, min_distance=25, max_points=2, detection_mode=mode)
                self.assertEqual(points.shape, (2, 3))
                self.assertGreaterEqual(np.linalg.norm(points[0, :2] - points[1, :2]), 25)
                np.testing.assert_allclose(points[:, 2], 90)

    def test_invalid_inputs(self):
        square = np.array([[0, 0], [20, 0], [20, 20], [0, 20]])
        with self.assertRaises(ValueError):
            largest_contour(np.zeros((8, 8, 3), np.uint8))
        with self.assertRaises(ValueError):
            turning_angles(square, 0)
        with self.assertRaises(ValueError):
            select_keypoints(square, detection_mode="unknown")
        with self.assertRaises(ValueError):
            select_keypoints(square, detection_mode="adaptive", adaptive_percentile=101)


class GeometryContracts(unittest.TestCase):
    def test_rigid_inverse_and_roundtrip(self):
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        transform = make_transform(rotation, [2, -3, 5])
        inverse = invert_transform(transform)
        np.testing.assert_allclose(inverse @ transform, np.eye(4), atol=1e-12)
        point = np.array([1, 2, 3, 1])
        np.testing.assert_allclose(inverse @ (transform @ point), point, atol=1e-12)

    def test_known_intrinsics(self):
        expected = [[240, 0, 319.5], [0, 240, 239.5], [0, 0, 1]]
        np.testing.assert_allclose(camera_intrinsics(640, 480, 90), expected)

    def test_known_projection_and_invalid_depth(self):
        intrinsics = np.array([[100, 0, 320], [0, 100, 240], [0, 0, 1]])
        points = [[0, 0, 2], [1, -1, 2], [0, 0, 0], [0, 0, -1]]
        pixels = project_points(points, intrinsics)
        np.testing.assert_allclose(pixels[:2], [[320, 240], [370, 190]])
        self.assertTrue(np.isnan(pixels[2:]).all())

    def test_invalid_camera_parameters(self):
        for args in ((0, 480, 60), (640, 480, 180), (640, 480, 0)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                camera_intrinsics(*args)


class DemoContracts(unittest.TestCase):
    def test_demo_is_repeatable_and_changes_image(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            paths_a, paths_b = make_examples(first), make_examples(second)
            images = []
            for path_a, path_b in zip(paths_a, paths_b):
                a, b = cv2.imread(str(path_a)), cv2.imread(str(path_b))
                self.assertIsNotNone(a)
                self.assertIsNotNone(b)
                self.assertEqual(a.shape, (480, 640, 3))
                np.testing.assert_array_equal(a, b)
                images.append(a)
            self.assertFalse(np.array_equal(images[0], images[1]))


if __name__ == "__main__":
    unittest.main(verbosity=2)

