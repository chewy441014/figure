import sys

import numpy as np
import pygame
from pygame.locals import DOUBLEBUF, OPENGL
from OpenGL.GL import *
from OpenGL.GLU import gluLookAt, gluPerspective

from src.model.default_pose import figure as default_figure
from src.model.pose import create_pose
from src.rendering.figure_engine import draw_figure, draw_figure_frames
from src.model.joint_limits import JOINT_LIMITS


WINDOW_SIZE = (1024, 1024)
ROTATION_SPEED = 1.5

CHAIN_ORDER = [
    "spine",
    "head",
    "left_arm",
    "right_arm",
    "left_leg",
    "right_leg",
]


def initialize() -> pygame.time.Clock:
    pygame.init()
    pygame.display.set_mode(WINDOW_SIZE, DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Figure")

    glEnable(GL_DEPTH_TEST)
    glClearColor(0.08, 0.08, 0.10, 1.0)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, 1.0, 1.0, 1000.0)

    return pygame.time.Clock()


def set_camera() -> None:
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    gluLookAt(
        0.0, -300.0, -50.0,
        0.0, 0.0, -50.0,
        0.0, 0.0, 1.0,
    )


def create_rotation_state() -> dict[str, np.ndarray]:
    """Create editable XYZ angles for every joint."""

    rotations = {
        name: np.zeros((len(getattr(default_figure, name).rows), 3))
        for name in CHAIN_ORDER
        if name != "spine"
    }

    rotations["spine"] = np.zeros((4, 3), dtype=float)
    return rotations


def rotate_active_joint(
    rotations: dict[str, np.ndarray],
    chain_name: str,
    joint_index: int,
    delta_time: float,
) -> None:
    """Rotate the selected joint freely around local XYZ axes."""

    keys = pygame.key.get_pressed()
    angles = rotations[chain_name][joint_index]
    amount = ROTATION_SPEED * delta_time

    limits = JOINT_LIMITS.get(chain_name, [])

    # Frames without explicit constraints remain fixed.
    if joint_index >= len(limits):
        return
    
    joint = limits[joint_index]
    enabled = np.asarray(joint["enabled"], dtype=bool)

    changes = np.array([
        keys[pygame.K_q] - keys[pygame.K_a],
        keys[pygame.K_w] - keys[pygame.K_s],
        keys[pygame.K_e] - keys[pygame.K_d],
    ], dtype=float)

    angles += changes * amount * enabled

    if keys[pygame.K_r]:
        angles[:] = 0.0

    angles[:] = np.clip(
        angles,
        joint["minimum"],
        joint["maximum"],
    )


def update_spine(
    spine_angles: np.ndarray,
    delta_time: float,
) -> None:
    """Twist the spine around local Z."""

    keys = pygame.key.get_pressed()
    amount = ROTATION_SPEED * delta_time

    if keys[pygame.K_t]:
        spine_angles[2] += amount

    if keys[pygame.K_g]:
        spine_angles[2] -= amount

    if keys[pygame.K_y]:
        spine_angles[:] = 0.0

    spine_angles[2] = np.clip(
        spine_angles[2],
        np.radians(-60.0),
        np.radians(60.0),
    )


def update_caption(chain_name: str, joint_index: int) -> None:
    pygame.display.set_caption(
        f"Figure | {chain_name} | joint {joint_index}"
    )


def run() -> None:
    clock = initialize()
    rotations = create_rotation_state()
    spine_angles = np.zeros(3, dtype=float)
    chain_index = 0
    joint_index = 0
    running = True

    update_caption(CHAIN_ORDER[chain_index], joint_index)

    while running:
        delta_time = clock.tick(60) / 1000.0
        chain_name = CHAIN_ORDER[chain_index]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                # Tab selects the next body chain.
                elif event.key == pygame.K_TAB:
                    chain_index = (chain_index + 1) % len(CHAIN_ORDER)
                    chain_name = CHAIN_ORDER[chain_index]
                    joint_index = 0

                # Number keys select joints 0–5.
                elif pygame.K_0 <= event.key <= pygame.K_5:
                    requested = event.key - pygame.K_0
                    maximum = len(rotations[chain_name]) - 1
                    joint_index = min(requested, maximum)

                # Backspace resets every joint.
                elif event.key == pygame.K_BACKSPACE:
                    rotations = create_rotation_state()

                update_caption(chain_name, joint_index)

        rotate_active_joint(
            rotations,
            chain_name,
            joint_index,
            delta_time,
        )

        update_spine(spine_angles, delta_time)

        figure = create_pose(
            default_figure,
            {
                name: values
                for name, values in rotations.items()
                if name != "spine"
            },
            spine=rotations["spine"],
        )

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        set_camera()
        draw_figure(figure)
        draw_figure_frames(figure)
        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    run()