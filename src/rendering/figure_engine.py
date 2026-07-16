from __future__ import annotations

import numpy as np

from ..kinematics.dh import figure_joint_positions, torso_segments, chain_transforms, posed_torso_points
from ..model.skeleton import FigureModel
from .primitives import draw_chain, draw_cylinder, draw_sphere
from .debug import draw_chain_frames, draw_frame_axes
from OpenGL.GL import glColor3f


def draw_torso(
    figure: FigureModel,
    joint_radius: float = 3.0,
    link_radius: float = 2.5,
) -> None:
    """Draw the posed torso landmarks and links."""

    points = posed_torso_points(figure)

    for point in points.values():
        draw_sphere(point, joint_radius)

    for start, end in torso_segments(figure):
        draw_cylinder(start, end, link_radius)


def draw_figure(
    figure: FigureModel,
    joint_radius: float = 3.0,
    link_radius: float = 2.0,
) -> None:
    """Draw the torso and all articulated chains."""
    glColor3f(1.0, 1.0, 1.0)
    draw_torso(
        figure,
        joint_radius=joint_radius,
        link_radius=link_radius,
    )

    chains = figure_joint_positions(figure)

    for joints in chains.values():
        draw_chain(
            joints,
            joint_radius=joint_radius,
            link_radius=link_radius,
        )

        
def draw_figure_frames(
    figure: FigureModel,
    axis_length: float = 8.0,
) -> None:
    """Draw local frames for spine and articulated chains."""

    torso = posed_torso_points(figure)

    # Spine frames: orientation accumulates upward.
    spine_names = [
        "spine_lower",
        "spine_mid_lower",
        "spine_mid_upper",
        "spine_upper",
    ]

    current_rotation = np.eye(3, dtype=float)

    for index, name in enumerate(spine_names):
        current_rotation = (
            current_rotation @ figure.spine_joints[index]
        )

        transform = np.eye(4, dtype=float)
        transform[:3, :3] = current_rotation
        transform[:3, 3] = torso[name]

        draw_frame_axes(transform, axis_length)

    # Existing head and limb frames.
    for chain_name in (
        "head",
        "left_arm",
        "right_arm",
        "left_leg",
        "right_leg",
    ):
        chain = getattr(figure, chain_name)

        transforms = chain_transforms(
            chain,
            torso[chain.parent_point],
        )

        draw_chain_frames(transforms, axis_length)