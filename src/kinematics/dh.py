from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from ..model.skeleton import DHChain, FigureModel


def translation_matrix(position: Iterable[float]) -> np.ndarray:
    """Create a 4x4 translation transform."""

    transform = np.eye(4, dtype=float)
    transform[:3, 3] = np.asarray(position, dtype=float)
    return transform


def dh_transform(
    theta: float,
    d: float,
    a: float,
    alpha: float,
) -> np.ndarray:
    """
    Create one standard Denavit-Hartenberg transform.

    Parameter order:
        theta: rotation about local Z
        d: translation along local Z
        a: translation along the new X
        alpha: rotation about the new X
    """

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)

    return np.array(
        [
            [
                cos_theta,
                -sin_theta * cos_alpha,
                sin_theta * sin_alpha,
                a * cos_theta,
            ],
            [
                sin_theta,
                cos_theta * cos_alpha,
                -cos_theta * sin_alpha,
                a * sin_theta,
            ],
            [
                0.0,
                sin_alpha,
                cos_alpha,
                d,
            ],
            [
                0.0,
                0.0,
                0.0,
                1.0,
            ],
        ],
        dtype=float,
    )


def rotation_transform(rotation: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix into a 4x4 transform."""

    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation
    return transform


def chain_transforms(
    chain: DHChain,
    parent_position: Iterable[float],
) -> list[np.ndarray]:
    """Calculate every coordinate frame in a kinematic chain."""

    parent_transform = translation_matrix(parent_position)
    current_transform = parent_transform @ chain.base_transform

    transforms = [current_transform.copy()]

    # Default to identity rotations for an unposed chain.
    if len(chain.joint_rotations) == 0:
        joint_rotations = np.repeat(
            np.eye(3, dtype=float)[None, :, :],
            len(chain.rows),
            axis=0,
        )
    else:
        joint_rotations = chain.joint_rotations

    if joint_rotations.shape != (len(chain.rows), 3, 3):
        raise ValueError(
            f"{chain.name}: expected joint rotations with shape "
            f"({len(chain.rows)}, 3, 3), got {joint_rotations.shape}."
        )

    for row, joint_rotation in zip(chain.rows, joint_rotations):
        if np.isnan(row).any():
            raise ValueError(
                f"DH chain '{chain.name}' contains undefined values."
            )

        current_transform = (
            current_transform
            @ rotation_transform(joint_rotation)
            @ dh_transform(*row)
        )

        transforms.append(current_transform.copy())

    return transforms


def joint_positions(
    chain: DHChain,
    parent_position: Iterable[float],
) -> np.ndarray:
    """Return one XYZ position for every frame in the chain."""

    transforms = chain_transforms(chain, parent_position)

    return np.array(
        [transform[:3, 3] for transform in transforms],
        dtype=float,
    )


def figure_joint_positions(
    figure: FigureModel,
) -> dict[str, np.ndarray]:
    """
    Calculate all articulated joint positions.

    Torso points remain fixed landmarks.
    Head and limb positions come from DH forward kinematics.
    """

    torso = posed_torso_points(figure)

    return {
        "head": joint_positions(
            figure.head,
            torso[figure.head.parent_point],
        ),
        "left_arm": joint_positions(
            figure.left_arm,
            torso[figure.left_arm.parent_point],
        ),
        "right_arm": joint_positions(
            figure.right_arm,
            torso[figure.right_arm.parent_point],
        ),
        "left_leg": joint_positions(
            figure.left_leg,
            torso[figure.left_leg.parent_point],
        ),
        "right_leg": joint_positions(
            figure.right_leg,
            torso[figure.right_leg.parent_point],
        ),
    }


def figure_link_segments(
    figure: FigureModel,
) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """
    Convert joint positions into start/end line segments.

    Rendering code can later replace each line with a cylinder.
    """

    positions = figure_joint_positions(figure)
    segments: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}

    for chain_name, joints in positions.items():
        segments[chain_name] = [
            (joints[index], joints[index + 1])
            for index in range(len(joints) - 1)
            if not np.allclose(joints[index], joints[index + 1])
        ]

    return segments


def torso_segments(
    figure: FigureModel,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return the fixed torso lines used for initial rendering."""

    torso = posed_torso_points(figure)

    return [
        (torso["left_shoulder"], torso["neck_base"]),
        (torso["neck_base"], torso["right_shoulder"]),
        (torso["neck_base"], torso["spine_upper"]),
        (torso["spine_upper"], torso["spine_mid_upper"]),
        (torso["spine_mid_upper"], torso["spine_mid_lower"]),
        (torso["spine_mid_lower"], torso["spine_lower"]),
        (torso["spine_lower"], torso["left_hip"]),
        (torso["spine_lower"], torso["right_hip"]),
    ]


def posed_torso_points(figure: FigureModel) -> dict[str, np.ndarray]:
    """Apply hierarchical rotations through the spine."""

    torso = figure.torso.as_dict()

    names = [
        "spine_lower",
        "spine_mid_lower",
        "spine_mid_upper",
        "spine_upper",
        "neck_base",
    ]

    posed = {
        name: point.copy()
        for name, point in torso.items()
    }

    current_position = torso["spine_lower"].copy()
    current_rotation = np.eye(3, dtype=float)

    posed["spine_lower"] = current_position

    # Each rotation affects all following spine segments.
    for index in range(4):
        current_rotation = (
            current_rotation @ figure.spine_joints[index]
        )

        original_offset = (
            torso[names[index + 1]]
            - torso[names[index]]
        )

        current_position = (
            current_position
            + current_rotation @ original_offset
        )

        posed[names[index + 1]] = current_position.copy()

    # Shoulders remain attached relative to the neck frame.
    neck = posed["neck_base"]

    posed["left_shoulder"] = (
        neck
        + current_rotation
        @ (torso["left_shoulder"] - torso["neck_base"])
    )

    posed["right_shoulder"] = (
        neck
        + current_rotation
        @ (torso["right_shoulder"] - torso["neck_base"])
    )

    return posed