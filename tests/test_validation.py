import unittest
from copy import deepcopy

import numpy as np

from src.kinematics.dh import chain_transforms
from src.model.default_pose import figure as default_figure
from src.model.pose import create_pose
from src.model.skeleton import DHChain


class TestDHChainValidation(unittest.TestCase):

    def test_rejects_non_matrix_rows(self):
        with self.assertRaisesRegex(
            ValueError,
            r"DH rows must have shape \(n, 4\)",
        ):
            DHChain(
                name="bad",
                rows=np.array([0.0, 0.0, 0.0, 0.0]),
                parent_point="neck_base",
            )

    def test_rejects_wrong_column_count(self):
        with self.assertRaisesRegex(
            ValueError,
            r"DH rows must have shape \(n, 4\)",
        ):
            DHChain(
                name="bad",
                rows=np.zeros((3, 5)),
                parent_point="neck_base",
            )

    def test_accepts_empty_valid_matrix(self):
        chain = DHChain(
            name="empty",
            rows=np.empty((0, 4)),
            parent_point="neck_base",
        )

        self.assertEqual(chain.rows.shape, (0, 4))


class TestPoseValidation(unittest.TestCase):

    def test_rejects_unknown_chain_name(self):
        with self.assertRaises(AttributeError):
            create_pose(
                default_figure,
                {"third_arm": np.zeros((5, 3))},
            )

    def test_rejects_too_few_arm_rotations(self):
        with self.assertRaisesRegex(
            ValueError,
            "left_arm requires 5 rotations",
        ):
            create_pose(
                default_figure,
                {"left_arm": np.zeros((4, 3))},
            )

    def test_rejects_too_many_arm_rotations(self):
        with self.assertRaisesRegex(
            ValueError,
            "right_arm requires 5 rotations",
        ):
            create_pose(
                default_figure,
                {"right_arm": np.zeros((6, 3))},
            )

    def test_rejects_too_few_leg_rotations(self):
        with self.assertRaisesRegex(
            ValueError,
            "left_leg requires 6 rotations",
        ):
            create_pose(
                default_figure,
                {"left_leg": np.zeros((5, 3))},
            )

    def test_rejects_incorrect_spine_count(self):
        with self.assertRaisesRegex(
            ValueError,
            "Spine requires four joint rotations",
        ):
            create_pose(
                default_figure,
                {},
                spine=np.zeros((3, 3)),
            )

    def test_pose_does_not_mutate_default_joints(self):
        posed = create_pose(
            default_figure,
            {"left_arm": np.zeros((5, 3))},
        )

        posed.left_arm.joint_rotations[0, 0, 0] = 99.0

        self.assertEqual(len(default_figure.left_arm.joint_rotations), 0)


class TestForwardKinematicsValidation(unittest.TestCase):

    def test_rejects_nan_dh_values(self):
        figure = deepcopy(default_figure)
        figure.left_arm.rows[2, 1] = np.nan

        with self.assertRaisesRegex(
            ValueError,
            "left_arm.*undefined values",
        ):
            chain_transforms(
                figure.left_arm,
                figure.torso.left_shoulder,
            )

    def test_rejects_wrong_joint_matrix_count(self):
        figure = deepcopy(default_figure)
        figure.left_arm.joint_rotations = np.repeat(
            np.eye(3)[None, :, :],
            4,
            axis=0,
        )

        with self.assertRaisesRegex(
            ValueError,
            "expected joint rotations with shape",
        ):
            chain_transforms(
                figure.left_arm,
                figure.torso.left_shoulder,
            )

    def test_rejects_wrong_joint_matrix_shape(self):
        figure = deepcopy(default_figure)
        figure.left_arm.joint_rotations = np.zeros((5, 4, 4))

        with self.assertRaisesRegex(
            ValueError,
            r"expected joint rotations with shape \(5, 3, 3\)",
        ):
            chain_transforms(
                figure.left_arm,
                figure.torso.left_shoulder,
            )

    def test_rejects_undefined_parent_position(self):
        parent = np.array([np.nan, 0.0, 0.0])

        transforms = chain_transforms(
            default_figure.left_arm,
            parent,
        )

        self.assertTrue(
            np.isnan(transforms[0][:3, 3]).any()
        )


class TestDefaultModelAssumptions(unittest.TestCase):

    def test_all_chain_rows_have_four_columns(self):
        for name in (
            "head",
            "left_arm",
            "right_arm",
            "left_leg",
            "right_leg",
        ):
            chain = getattr(default_figure, name)
            self.assertEqual(chain.rows.shape[1], 4)

    def test_parent_points_exist_on_torso(self):
        torso_points = default_figure.torso.as_dict()

        for name in (
            "head",
            "left_arm",
            "right_arm",
            "left_leg",
            "right_leg",
        ):
            chain = getattr(default_figure, name)
            self.assertIn(chain.parent_point, torso_points)

    def test_default_dh_values_are_finite(self):
        for name in (
            "head",
            "left_arm",
            "right_arm",
            "left_leg",
            "right_leg",
        ):
            chain = getattr(default_figure, name)
            self.assertTrue(np.isfinite(chain.rows).all())

    def test_spine_has_four_rotation_matrices(self):
        self.assertEqual(
            default_figure.spine_joints.shape,
            (4, 3, 3),
        )


if __name__ == "__main__":
    unittest.main()