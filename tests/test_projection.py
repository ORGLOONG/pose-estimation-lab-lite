import unittest

import numpy as np

from src.camera_geometry import camera_intrinsics, invert_transform, make_transform, project_points


class ProjectionTests(unittest.TestCase):
    def test_rigid_transform_inverse_round_trip(self):
        angle = np.deg2rad(30)
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                             [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        transform = make_transform(rotation, [1.2, -0.4, 2.0])
        point = np.array([0.2, 0.3, -0.1, 1.0])
        np.testing.assert_allclose(invert_transform(transform) @ transform @ point, point, atol=1e-12)

    def test_known_point_projection_and_behind_camera(self):
        K = camera_intrinsics(640, 480, 90)
        point = np.array([[0.0, 0.0, 2.0], [1.0, -0.5, 2.0], [0.0, 0.0, -1.0]])
        pixels = project_points(point, K)
        np.testing.assert_allclose(pixels[:2], [[319.5, 239.5], [439.5, 179.5]], atol=1e-10)
        self.assertTrue(np.isnan(pixels[2]).all())


if __name__ == "__main__":
    unittest.main()
