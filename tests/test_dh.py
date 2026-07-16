import unittest

import numpy as np

from src.kinematics.dh import (
    dh_transform,
    figure_joint_positions,
)
from src.model.default_pose import figure


class TestDHTransform(unittest.TestCase):

    def test_identity_transform(self):
        transform = dh_transform(0.0, 0.0, 0.0, 0.0)

        np.testing.assert_allclose(
            transform,
            np.eye(4),
            atol=1e-9,
        )

    def test_translation_along_z(self):
        transform = dh_transform(0.0, 10.0, 0.0, 0.0)

        np.testing.assert_allclose(
            transform[:3, 3],
            [0.0, 0.0, 10.0],
            atol=1e-9,
        )


class TestDefaultPose(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.joints = figure_joint_positions(figure)

    def test_head_position(self):
        head = self.joints["head"]

        np.testing.assert_allclose(
            head[-1],
            [0.0, 0.0, 10.0],
            atol=1e-9,
        )

    def test_left_arm_t_pose(self):
        left_arm = self.joints["left_arm"]

        np.testing.assert_allclose(
            left_arm[-1],
            [90.0, 0.0, 0.0],
            atol=1e-9,
        )

    def test_right_arm_t_pose(self):
        right_arm = self.joints["right_arm"]

        np.testing.assert_allclose(
            right_arm[-1],
            [-90.0, 0.0, 0.0],
            atol=1e-9,
        )

    def test_left_ankle_position(self):
        left_leg = self.joints["left_leg"]

        np.testing.assert_allclose(
            left_leg[3],
            [17.5, 0.0, -180.0],
            atol=1e-9,
        )

    def test_left_toe_position(self):
        left_leg = self.joints["left_leg"]

        np.testing.assert_allclose(
            left_leg[-2],
            [17.5, -28.0, -180.0],
            atol=1e-9,
        )

    def test_hips_are_symmetric(self):
        torso = figure.torso

        np.testing.assert_allclose(
            torso.left_hip,
            -torso.right_hip * [1.0, -1.0, -1.0],
            atol=1e-9,
        )


if __name__ == "__main__":
    unittest.main()