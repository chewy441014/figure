from __future__ import annotations

import numpy as np

from ..kinematics.dh import figure_joint_positions, torso_segments
from ..model.skeleton import FigureModel
from .primitives import draw_chain, draw_cylinder, draw_sphere


def draw_torso(
    figure: FigureModel,
    joint_radius: float = 3.0,
    link_radius: float = 2.5,
) -> None:
    """Draw fixed torso landmarks and connecting links."""

    torso = figure.torso
    points = torso.as_dict()

    for point in points.values():
        if not np.isnan(point).any():
            draw_sphere(point, joint_radius)

    for start, end in torso_segments(figure):
        if not np.isnan(start).any() and not np.isnan(end).any():
            draw_cylinder(start, end, link_radius)


def draw_figure(
    figure: FigureModel,
    joint_radius: float = 3.0,
    link_radius: float = 2.0,
) -> None:
    """Draw the torso and all articulated chains."""

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