from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy

import numpy as np

from .skeleton import FigureModel


def rotation_xyz(angles: Sequence[float]) -> np.ndarray:
    """Create a local XYZ rotation matrix from radians."""

    x, y, z = np.asarray(angles, dtype=float)

    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)
    cz, sz = np.cos(z), np.sin(z)

    rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cx, -sx],
        [0.0, sx, cx],
    ])

    ry = np.array([
        [cy, 0.0, sy],
        [0.0, 1.0, 0.0],
        [-sy, 0.0, cy],
    ])

    rz = np.array([
        [cz, -sz, 0.0],
        [sz, cz, 0.0],
        [0.0, 0.0, 1.0],
    ])

    return rz @ ry @ rx


def create_pose(
    base_figure: FigureModel,
    rotations: dict[str, Sequence[Sequence[float]]],
    spine: Sequence[Sequence[float]] | None = None,
) -> FigureModel:
    """Return an independent posed figure."""

    posed = deepcopy(base_figure)

    for chain_name, joint_angles in rotations.items():
        chain = getattr(posed, chain_name)

        if len(joint_angles) != len(chain.rows):
            raise ValueError(
                f"{chain_name} requires {len(chain.rows)} rotations."
            )

        chain.joint_rotations = np.array(
            [rotation_xyz(angles) for angles in joint_angles],
            dtype=float,
        )

    if spine is not None:
        if len(spine) != 4:
            raise ValueError("Spine requires four joint rotations.")

        posed.spine_joints = np.array(
            [rotation_xyz(angles) for angles in spine],
            dtype=float,
        )

    return posed