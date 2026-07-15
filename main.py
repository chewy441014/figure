import sys

import pygame
from pygame.locals import DOUBLEBUF, OPENGL
from OpenGL.GL import (
    GL_COLOR_BUFFER_BIT,
    GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST,
    GL_MODELVIEW,
    GL_PROJECTION,
    glClear,
    glClearColor,
    glEnable,
    glLoadIdentity,
    glMatrixMode,
)
from OpenGL.GLU import gluLookAt, gluPerspective

from src.model.default_pose import figure
from src.rendering.figure_engine import draw_figure


WINDOW_SIZE = (1024, 1024)


def initialize() -> pygame.time.Clock:
    """Create the window and configure OpenGL."""

    pygame.init()
    pygame.display.set_mode(WINDOW_SIZE, DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Figure")

    # Enable proper 3D depth ordering.
    glEnable(GL_DEPTH_TEST)
    glClearColor(0.08, 0.08, 0.10, 1.0)

    # Configure the perspective camera lens.
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(
        45.0,
        WINDOW_SIZE[0] / WINDOW_SIZE[1],
        1.0,
        1000.0,
    )

    return pygame.time.Clock()


def set_camera() -> None:
    """View the figure from the front with world +Z upward."""

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    gluLookAt(
        0.0, -300.0, -50.0,  # Camera position
        0.0,    0.0, -50.0,  # Camera target
        0.0,    0.0,   1.0,  # Up direction
    )


def run() -> None:
    """Run the rendering loop."""

    clock = initialize()
    running = True

    while running:
        # Process window and keyboard events.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # Clear the previous frame.
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        set_camera()
        draw_figure(figure)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    run()