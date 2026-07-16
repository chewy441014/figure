import unittest

import numpy as np

from src.kinematics.dh import figure_joint_positions
from src.model.default_pose import figure as default_figure
from src.model.pose import create_pose, rotation_xyz


class TestPose(unittest.TestCase):

    def test_zero_rotation_is_identity(self):
        rotation = rotation_xyz([0.0, 0.0, 0.0])

        np.testing.assert_allclose(
            rotation,
            np.eye(3),
            atol=1e-9,
        )

    def test_x_rotation(self):
        rotation = rotation_xyz([np.pi / 2, 0.0, 0.0])
        vector = rotation @ np.array([0.0, 1.0, 0.0])

        np.testing.assert_allclose(
            vector,
            [0.0, 0.0, 1.0],
            atol=1e-9,
        )

    def test_pose_does_not_modify_default(self):
        original_rows = default_figure.left_arm.rows.copy()

        create_pose(
            default_figure,
            {
                "left_arm": [
                    [0.0, 0.0, 0.0],
                    [0.0, np.pi / 2, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            },
        )

        np.testing.assert_allclose(
            default_figure.left_arm.rows,
            original_rows,
        )

    def test_elbow_rotation_preserves_upper_arm(self):
        default = figure_joint_positions(default_figure)["left_arm"]

        posed_figure = create_pose(
            default_figure,
            {
                "left_arm": [
                    [0.0, 0.0, 0.0],
                    [0.0, np.pi / 2, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            },
        )

        posed = figure_joint_positions(posed_figure)["left_arm"]

        # Shoulder and elbow remain fixed.
        np.testing.assert_allclose(posed[:2], default[:2], atol=1e-9)

        # Downstream joints move.
        self.assertFalse(np.allclose(posed[-1], default[-1]))

    def test_pose_preserves_link_lengths(self):
        posed_figure = create_pose(
            default_figure,
            {
                "left_arm": [
                    [0.2, 0.3, 0.1],
                    [0.0, 0.7, 0.0],
                    [0.1, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            },
        )

        joints = figure_joint_positions(posed_figure)["left_arm"]
        lengths = np.linalg.norm(np.diff(joints, axis=0), axis=1)

        np.testing.assert_allclose(
            lengths,
            [0.0, 30.0, 30.0, 10.0, 0.0],
            atol=1e-9,
        )


if __name__ == "__main__":
    unittest.main()