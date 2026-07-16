from __future__ import annotations

import numpy as np
from OpenGL.GL import (
    GL_LINES,
    glBegin,
    glColor3f,
    glEnd,
    glVertex3f,
)


def draw_frame_axes(
    transform: np.ndarray,
    length: float = 8.0,
) -> None:
    """Draw local XYZ axes from a 4×4 transform."""

    origin = transform[:3, 3]
    rotation = transform[:3, :3]

    x_end = origin + rotation[:, 0] * length
    y_end = origin + rotation[:, 1] * length
    z_end = origin + rotation[:, 2] * length

    glBegin(GL_LINES)

    # X axis: red
    glColor3f(1.0, 0.0, 0.0)
    glVertex3f(*origin)
    glVertex3f(*x_end)

    # Y axis: green
    glColor3f(0.0, 1.0, 0.0)
    glVertex3f(*origin)
    glVertex3f(*y_end)

    # Z axis: blue
    glColor3f(0.0, 0.0, 1.0)
    glVertex3f(*origin)
    glVertex3f(*z_end)

    glEnd()


def draw_chain_frames(
    transforms: list[np.ndarray],
    length: float = 8.0,
) -> None:
    """Draw axes for every frame in a chain."""

    for transform in transforms:
        draw_frame_axes(transform, length)