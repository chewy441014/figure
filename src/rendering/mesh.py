from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from OpenGL.GL import (
    GL_TRIANGLES,
    glBegin,
    glEnd,
    glNormal3f,
    glVertex3f,
)


@dataclass(frozen=True)
class Mesh:
    """Triangle mesh with one normal per vertex."""

    vertices: np.ndarray
    normals: np.ndarray
    indices: np.ndarray

    def draw(self) -> None:
        """Draw using the current OpenGL transform and material."""

        glBegin(GL_TRIANGLES)

        for index in self.indices.flat:
            glNormal3f(*self.normals[index])
            glVertex3f(*self.vertices[index])

        glEnd()


def cylinder_mesh(
    segments: int = 24,
    capped: bool = True,
) -> Mesh:
    """
    Create a unit cylinder along local +Z.

    Radius and length are both 1.0.
    Rendering code can scale it as needed.
    """

    vertices: list[list[float]] = []
    normals: list[list[float]] = []
    triangles: list[list[int]] = []

    # Side vertices: bottom and top rings.
    for index in range(segments):
        angle = 2.0 * np.pi * index / segments
        x, y = np.cos(angle), np.sin(angle)

        vertices.extend([
            [x, y, 0.0],
            [x, y, 1.0],
        ])

        normals.extend([
            [x, y, 0.0],
            [x, y, 0.0],
        ])

    # Side triangles.
    for index in range(segments):
        next_index = (index + 1) % segments

        bottom = 2 * index
        top = bottom + 1
        next_bottom = 2 * next_index
        next_top = next_bottom + 1

        triangles.extend([
            [bottom, next_bottom, top],
            [top, next_bottom, next_top],
        ])

    if capped:
        bottom_center = len(vertices)
        vertices.append([0.0, 0.0, 0.0])
        normals.append([0.0, 0.0, -1.0])

        top_center = len(vertices)
        vertices.append([0.0, 0.0, 1.0])
        normals.append([0.0, 0.0, 1.0])

        # Separate cap-ring vertices provide correct flat normals.
        bottom_ring = len(vertices)

        for index in range(segments):
            angle = 2.0 * np.pi * index / segments
            x, y = np.cos(angle), np.sin(angle)

            vertices.append([x, y, 0.0])
            normals.append([0.0, 0.0, -1.0])

        top_ring = len(vertices)

        for index in range(segments):
            angle = 2.0 * np.pi * index / segments
            x, y = np.cos(angle), np.sin(angle)

            vertices.append([x, y, 1.0])
            normals.append([0.0, 0.0, 1.0])

        for index in range(segments):
            next_index = (index + 1) % segments

            triangles.append([
                bottom_center,
                bottom_ring + next_index,
                bottom_ring + index,
            ])

            triangles.append([
                top_center,
                top_ring + index,
                top_ring + next_index,
            ])

    return Mesh(
        vertices=np.asarray(vertices, dtype=np.float32),
        normals=np.asarray(normals, dtype=np.float32),
        indices=np.asarray(triangles, dtype=np.uint32),
    )

def sphere_mesh(
        slices: int = 24,
        stacks: int = 16,
    ) -> Mesh:
        """Create a unit sphere centered at the origin."""

        vertices = []
        normals = []
        triangles = []

        for stack in range(stacks + 1):
            phi = np.pi * stack / stacks
            z = np.cos(phi)
            ring_radius = np.sin(phi)

            for slice_index in range(slices + 1):
                theta = 2.0 * np.pi * slice_index / slices

                x = ring_radius * np.cos(theta)
                y = ring_radius * np.sin(theta)

                normal = [x, y, z]
                vertices.append(normal)
                normals.append(normal)

        row_size = slices + 1

        for stack in range(stacks):
            for slice_index in range(slices):
                current = stack * row_size + slice_index
                next_row = current + row_size

                triangles.extend([
                    [current, next_row, current + 1],
                    [current + 1, next_row, next_row + 1],
                ])

        return Mesh(
            vertices=np.asarray(vertices, dtype=np.float32),
            normals=np.asarray(normals, dtype=np.float32),
            indices=np.asarray(triangles, dtype=np.uint32),
        )