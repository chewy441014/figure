from __future__ import annotations

import numpy as np

from ..kinematics.dh import (figure_joint_positions, 
    torso_segments, chain_transforms, 
    posed_torso_points, torso_attachment_rotations)
from ..model.skeleton import FigureModel
from .primitives import draw_chain, draw_cylinder, draw_sphere
from .debug import draw_chain_frames, draw_frame_axes
from OpenGL.GL import glColor4f


def draw_torso(
        figure: FigureModel,
        joint_radius: float = 3.0,
        link_radius: float = 2.5,
        show_joints: bool = True,
    ) -> None:
        points = posed_torso_points(figure)

        if show_joints:
            for point in points.values():
                draw_sphere(point, joint_radius)

        for start, end in torso_segments(figure):
            draw_cylinder(start, end, link_radius)


def draw_figure(
        figure: FigureModel,
        joint_radius: float = 3.0,
        link_radius: float = 2.0,
        alpha: float = 1.0,
        show_joints: bool = True,
    ) -> None:
        glColor4f(1.0, 1.0, 1.0, alpha)

        draw_torso(
            figure,
            joint_radius=joint_radius,
            link_radius=link_radius,
            show_joints=show_joints,
        )

        for joints in figure_joint_positions(figure).values():
            draw_chain(
                joints,
                joint_radius=joint_radius,
                link_radius=link_radius,
                show_joints=show_joints,
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

            orientations = torso_attachment_rotations(figure)
            transforms = chain_transforms(
                chain,
                torso[chain.parent_point],
                orientations[chain.parent_point],
            )

            draw_chain_frames(transforms, axis_length)