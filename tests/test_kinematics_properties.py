import unittest

import numpy as np

from src.kinematics.dh import (
    chain_transforms,
    figure_joint_positions,
)
from src.model.default_pose import figure as default_figure
from src.model.pose import create_pose


class TestKinematicProperties(unittest.TestCase):

    def test_default_left_right_arm_symmetry(self):
        joints = figure_joint_positions(default_figure)

        left = joints["left_arm"]
        right = joints["right_arm"]

        np.testing.assert_allclose(left[:, 0], -right[:, 0], atol=1e-9)
        np.testing.assert_allclose(left[:, 1:], right[:, 1:], atol=1e-9)

    def test_default_left_right_leg_symmetry(self):
        joints = figure_joint_positions(default_figure)

        left = joints["left_leg"]
        right = joints["right_leg"]

        np.testing.assert_allclose(left[:, 0], -right[:, 0], atol=1e-9)
        np.testing.assert_allclose(left[:, 1:], right[:, 1:], atol=1e-9)

    def test_arm_link_lengths_are_preserved(self):
        rotations = np.zeros((5, 3), dtype=float)
        rotations[1] = [0.4, -0.3, 0.2]
        rotations[2] = [0.0, 0.8, 0.0]

        posed = create_pose(
            default_figure,
            {"left_arm": rotations},
        )

        joints = figure_joint_positions(posed)["left_arm"]
        lengths = np.linalg.norm(np.diff(joints, axis=0), axis=1)

        np.testing.assert_allclose(
            lengths,
            [0.0, 30.0, 30.0, 10.0, 0.0],
            atol=1e-9,
        )

    def test_leg_link_lengths_are_preserved(self):
        rotations = np.zeros((6, 3), dtype=float)
        rotations[1] = [0.2, 0.3, -0.1]
        rotations[2] = [0.0, 0.7, 0.0]

        posed = create_pose(
            default_figure,
            {"left_leg": rotations},
        )

        joints = figure_joint_positions(posed)["left_leg"]
        lengths = np.linalg.norm(np.diff(joints, axis=0), axis=1)

        np.testing.assert_allclose(
            lengths,
            [0.0, 55.0, 55.0, 25.0, 3.0, 0.0],
            atol=1e-9,
        )

    def test_root_frame_position_is_fixed(self):
        rotations = np.zeros((5, 3), dtype=float)
        rotations[1] = [0.5, 0.5, 0.5]

        posed = create_pose(
            default_figure,
            {"left_arm": rotations},
        )

        transforms = chain_transforms(
            posed.left_arm,
            posed.torso.left_shoulder,
        )

        np.testing.assert_allclose(
            transforms[0][:3, 3],
            posed.torso.left_shoulder,
            atol=1e-9,
        )

    def test_rotation_matrices_are_orthonormal(self):
        rotations = np.zeros((5, 3), dtype=float)
        rotations[1] = [0.5, -0.7, 0.2]
        rotations[2] = [0.1, 0.4, -0.3]

        posed = create_pose(
            default_figure,
            {"left_arm": rotations},
        )

        transforms = chain_transforms(
            posed.left_arm,
            posed.torso.left_shoulder,
        )

        for transform in transforms:
            rotation = transform[:3, :3]

            np.testing.assert_allclose(
                rotation @ rotation.T,
                np.eye(3),
                atol=1e-9,
            )

            self.assertAlmostEqual(
                np.linalg.det(rotation),
                1.0,
                places=9,
            )

    def test_upstream_joint_moves_downstream_only(self):
        default = figure_joint_positions(default_figure)["left_arm"]

        rotations = np.zeros((5, 3), dtype=float)
        rotations[2] = [0.0, np.pi / 2, 0.0]

        posed = create_pose(
            default_figure,
            {"left_arm": rotations},
        )

        moved = figure_joint_positions(posed)["left_arm"]

        np.testing.assert_allclose(moved[:3], default[:3], atol=1e-9)
        self.assertFalse(np.allclose(moved[3:], default[3:]))

    def test_zero_pose_matches_default_pose(self):
        rotations = {
            "head": np.zeros((2, 3)),
            "left_arm": np.zeros((5, 3)),
            "right_arm": np.zeros((5, 3)),
            "left_leg": np.zeros((6, 3)),
            "right_leg": np.zeros((6, 3)),
        }

        posed = create_pose(
            default_figure,
            rotations,
            spine=np.zeros((4, 3)),
        )

        expected = figure_joint_positions(default_figure)
        actual = figure_joint_positions(posed)

        for name in expected:
            np.testing.assert_allclose(
                actual[name],
                expected[name],
                atol=1e-9,
            )


if __name__ == "__main__":
    unittest.main()