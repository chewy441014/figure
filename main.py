import sys

from dataclasses import dataclass

import numpy as np
import pygame
from pygame.locals import DOUBLEBUF, OPENGL
from OpenGL.GL import *
from OpenGL.GLU import gluLookAt, gluPerspective

from src.model.default_pose import figure as default_figure
from src.model.pose import create_pose
from src.rendering.figure_engine import draw_figure, draw_figure_frames
from src.model.joint_limits import JOINT_LIMITS
from src.model.serialization import load_scene, save_scene
from src.ui.settings_menu import AppSettings, SettingsMenu, MenuAction


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
    # Enable lighting and allow glColor calls to set material color.
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_COLOR_MATERIAL)

    glColorMaterial(
        GL_FRONT_AND_BACK,
        GL_AMBIENT_AND_DIFFUSE,
    )

    # Directional light positioned above and in front.
    glLightfv(
        GL_LIGHT0,
        GL_POSITION,
        (100.0, -150.0, 200.0, 0.0),
    )

    glLightfv(
        GL_LIGHT0,
        GL_AMBIENT,
        (0.20, 0.20, 0.20, 1.0),
    )

    glLightfv(
        GL_LIGHT0,
        GL_DIFFUSE,
        (0.85, 0.85, 0.85, 1.0),
    )

    glLightfv(
        GL_LIGHT0,
        GL_SPECULAR,
        (0.40, 0.40, 0.40, 1.0),
    )

    glMaterialf(
        GL_FRONT_AND_BACK,
        GL_SHININESS,
        32.0,
    )
    glClearColor(0.08, 0.08, 0.10, 1.0)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45.0, 1.0, 1.0, 1000.0)

    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    return pygame.time.Clock()


@dataclass
class Camera:
    yaw: float = 0.0
    pitch: float = 0.0
    distance: float = 300.0
    target: np.ndarray = None

    def __post_init__(self) -> None:
        if self.target is None:
            self.target = np.array([0.0, 0.0, -60.0], dtype=float)


def update_camera(camera: Camera, delta_time: float) -> None:
    """Orbit, pan, and zoom the camera."""

    keys = pygame.key.get_pressed()

    orbit_speed = 0.5 * delta_time
    pan_speed = 25.0 * delta_time
    zoom_speed = 60.0 * delta_time

    # Arrow keys: orbit.
    camera.yaw += orbit_speed * (
        keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]
    )

    camera.pitch += orbit_speed * (
        keys[pygame.K_UP] - keys[pygame.K_DOWN]
    )

    camera.pitch = np.clip(
        camera.pitch,
        np.radians(-85.0),
        np.radians(85.0),
    )

    # I/K: vertical pan.
    camera.target[2] += pan_speed * (
        keys[pygame.K_i] - keys[pygame.K_k]
    )

    # J/L: horizontal pan.
    camera.target[0] += pan_speed * (
        keys[pygame.K_l] - keys[pygame.K_j]
    )

    # U/O: zoom.
    camera.distance += zoom_speed * (
        keys[pygame.K_o] - keys[pygame.K_u]
    )

    camera.distance = np.clip(camera.distance, 80.0, 1000.0)


def set_camera(camera: Camera) -> None:
    """Apply the camera view transform."""

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    horizontal = camera.distance * np.cos(camera.pitch)

    position = camera.target + np.array([
        horizontal * np.sin(camera.yaw),
        -horizontal * np.cos(camera.yaw),
        camera.distance * np.sin(camera.pitch),
    ])

    gluLookAt(
        *position,
        *camera.target,
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
    skeleton_transparent = False
    camera = Camera()
    show_debug_frames = True
    scene_path = "pose.json"
    settings = AppSettings()
    settings_menu = SettingsMenu()

    update_caption(CHAIN_ORDER[chain_index], joint_index)

    while running:
        delta_time = clock.tick(60) / 1000.0
        chain_name = CHAIN_ORDER[chain_index]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                action = settings_menu.handle_event(event, settings)
                if action == MenuAction.QUIT:
                    running = False
                    continue
                if action == MenuAction.START_VERIFICATION:
                    # To do
                    continue
                if settings_menu.visible or (action == MenuAction.NONE):
                    continue
                elif event.key == pygame.K_x:
                    skeleton_transparent = not skeleton_transparent
                elif event.key == pygame.K_F5:
                    save_scene(
                        scene_path,
                        rotations,
                        camera,
                        skeleton_transparent=skeleton_transparent,
                        show_debug_frames=show_debug_frames,
                    )

                elif event.key == pygame.K_F9:
                    (
                        skeleton_transparent,
                        show_debug_frames,
                    ) = load_scene(
                        scene_path,
                        rotations,
                        camera,
                    )

                elif event.key == pygame.K_f:
                    show_debug_frames = not show_debug_frames

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

        if not settings_menu.visible:
            update_camera(camera, delta_time)

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
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        set_camera(camera)
        
        draw_figure(
            figure,
            alpha=0.25 if settings.skeleton_transparent else 1.0,
            show_joints=settings.show_joint_spheres,
        )

        if settings.show_debug_frames:
            draw_figure_frames(figure)

        settings_menu.draw(settings, WINDOW_SIZE)
        
        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    run()