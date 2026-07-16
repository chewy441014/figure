from __future__ import annotations

import numpy as np
from OpenGL.GL import (
    GL_LINES,
    GL_LIGHTING,
    glBegin,
    glColor3f,
    glEnd,
    glVertex3f,
    glDisable,
    glEnable,
    glIsEnabled,
)


def draw_frame_axes(
    transform: np.ndarray,
    length: float = 8.0,
) -> None:
    """Draw unlit local XYZ axes."""

    lighting_enabled = glIsEnabled(GL_LIGHTING)

    if lighting_enabled:
        glDisable(GL_LIGHTING)

    origin = transform[:3, 3]
    rotation = transform[:3, :3]

    endpoints = [
        origin + rotation[:, 0] * length,
        origin + rotation[:, 1] * length,
        origin + rotation[:, 2] * length,
    ]

    glBegin(GL_LINES)

    glColor3f(1.0, 0.0, 0.0)
    glVertex3f(*origin)
    glVertex3f(*endpoints[0])

    glColor3f(0.0, 1.0, 0.0)
    glVertex3f(*origin)
    glVertex3f(*endpoints[1])

    glColor3f(0.0, 0.0, 1.0)
    glVertex3f(*origin)
    glVertex3f(*endpoints[2])

    glEnd()

    if lighting_enabled:
        glEnable(GL_LIGHTING)


def draw_chain_frames(
    transforms: list[np.ndarray],
    length: float = 8.0,
) -> None:
    """Draw axes for every frame in a chain."""

    for transform in transforms:
        draw_frame_axes(transform, length)