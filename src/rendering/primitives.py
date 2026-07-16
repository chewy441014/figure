from __future__ import annotations

import numpy as np
from OpenGL.GL import (
    glMultMatrixf,
    glPopMatrix,
    glPushMatrix,
    glScalef,
    glTranslatef,
)
from OpenGL.GLU import (
    gluDeleteQuadric,
    gluNewQuadric,
    gluSphere,
)

from .mesh import Mesh, cylinder_mesh, sphere_mesh


# Reuse one cylinder mesh instead of rebuilding geometry each frame.
CYLINDER_MESH: Mesh = cylinder_mesh(
    segments=24,
    capped=True,
)

SPHERE_MESH: Mesh = sphere_mesh(
    slices=24,
    stacks=16,
)


def draw_sphere(
    position: np.ndarray,
    radius: float = 3.0,
) -> None:
    """Draw the cached unit sphere at an XYZ position."""

    glPushMatrix()
    glTranslatef(*position)
    glScalef(radius, radius, radius)

    SPHERE_MESH.draw()

    glPopMatrix()


def alignment_matrix(direction: np.ndarray) -> np.ndarray:
    """Create a rotation whose local +Z follows direction."""

    z_axis = direction / np.linalg.norm(direction)
    reference = np.array([0.0, 1.0, 0.0])

    # Choose another reference when nearly parallel.
    if abs(np.dot(z_axis, reference)) > 0.999:
        reference = np.array([1.0, 0.0, 0.0])

    x_axis = np.cross(reference, z_axis)
    x_axis /= np.linalg.norm(x_axis)

    y_axis = np.cross(z_axis, x_axis)

    transform = np.eye(4, dtype=np.float32)
    transform[:3, 0] = x_axis
    transform[:3, 1] = y_axis
    transform[:3, 2] = z_axis

    return transform


def draw_cylinder(
    start: np.ndarray,
    end: np.ndarray,
    radius: float = 2.0,
) -> None:
    """Draw the cached unit cylinder between two points."""

    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)

    direction = end - start
    length = float(np.linalg.norm(direction))

    if np.isclose(length, 0.0):
        return

    transform = alignment_matrix(direction)

    glPushMatrix()

    # Place and orient the unit cylinder.
    glTranslatef(*start)
    glMultMatrixf(transform.T)

    # Unit mesh radius and length are both 1.
    glScalef(radius, radius, length)

    CYLINDER_MESH.draw()

    glPopMatrix()


def draw_chain(
    joints: np.ndarray,
    joint_radius: float = 3.0,
    link_radius: float = 2.0,
    show_joints: bool = True,
) -> None:
    """Draw joint markers and links for one chain."""

    if show_joints:
        for joint in joints:
            draw_sphere(joint, joint_radius)

    for start, end in zip(joints[:-1], joints[1:]):
        if not np.allclose(start, end):
            draw_cylinder(start, end, link_radius)