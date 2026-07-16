from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import pygame
from OpenGL.GL import *


@dataclass
class AppSettings:
    show_debug_frames: bool = True
    skeleton_transparent: bool = False
    show_joint_spheres: bool = True
    verification_mode: bool = False


class SettingsMenu:
    """Simple keyboard-controlled OpenGL overlay."""

    OPTIONS = (
        ("Debug frames", "show_debug_frames"),
        ("Transparent skeleton", "skeleton_transparent"),
        ("Joint spheres", "show_joint_spheres"),
        ("Axis verification", None),
        ("Quit", None)
    )

    def __init__(self) -> None:
        pygame.font.init()
        self.font = pygame.font.SysFont("consolas", 24)
        self.visible = False
        self.selected = 0

    def handle_event(
        self,
        event: pygame.event.Event,
        settings: AppSettings,
    ) -> bool:
        """Handle menu keys. Return True when the event was consumed."""

        if event.type != pygame.KEYDOWN:
            return False

        if event.key == pygame.K_ESCAPE:
            self.visible = not self.visible
            return MenuAction.NONE

        if not self.visible:
            return False

        if event.key == pygame.K_UP:
            while True:
                self.selected = (self.selected - 1) % len(self.OPTIONS)
                if self.OPTIONS[self.selected][0] is not None:
                    break

        elif event.key == pygame.K_DOWN:
            while True:
                self.selected = (self.selected + 1) % len(self.OPTIONS)
                if self.OPTIONS[self.selected][0] is not None:
                    break

        elif event.key in (pygame.K_RETURN, pygame.K_SPACE):

            label, attribute = self.OPTIONS[self.selected]

            if label == "Quit":
                return MenuAction.QUIT

            if label == "Start Axis Verification":
                return MenuAction.START_VERIFICATION

            if attribute is not None:
                setattr(
                    settings,
                    attribute,
                    not getattr(settings, attribute),
                )

        return MenuAction.NONE

    def draw(
        self,
        settings: AppSettings,
        window_size: tuple[int, int],
    ) -> None:
        """Draw the menu over the 3D scene."""

        if not self.visible:
            return

        lines = ["SETTINGS"]

        for index, (label, attribute) in enumerate(self.OPTIONS):

            if label is None:
                lines.append("----------------------------")
                continue

            if attribute is None:
                marker = ">" if index == self.selected else " "
                lines.append(f"{marker} {label}")
            else:
                marker = ">" if index == self.selected else " "
                value = "ON" if getattr(settings, attribute) else "OFF"
                lines.append(f"{marker} {label:<24} {value}")

        lines.extend(("", "Esc: close   Up/Down: select   Enter: toggle"))

        surface = self._render_surface(lines)
        self._draw_surface(surface, window_size)

    def _render_surface(self, lines: list[str]) -> pygame.Surface:
        line_height = self.font.get_linesize()
        width = max(self.font.size(line)[0] for line in lines) + 40
        height = line_height * len(lines) + 30

        surface = pygame.Surface((width, height), pygame.SRCALPHA)
        surface.fill((20, 20, 24, 225))

        for index, line in enumerate(lines):
            text = self.font.render(line, True, (235, 235, 235))
            surface.blit(text, (20, 15 + index * line_height))

        return surface

    @staticmethod
    def _draw_surface(
        surface: pygame.Surface,
        window_size: tuple[int, int],
    ) -> None:
        width, height = surface.get_size()
        pixels = pygame.image.tostring(surface, "RGBA", True)

        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGBA,
            width, height, 0,
            GL_RGBA, GL_UNSIGNED_BYTE, pixels,
        )

        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        glEnable(GL_BLEND)
        glEnable(GL_TEXTURE_2D)

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, window_size[0], 0, window_size[1], -1, 1)

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        x = (window_size[0] - width) / 2
        y = (window_size[1] - height) / 2

        glColor4f(1.0, 1.0, 1.0, 1.0)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(x, y)
        glTexCoord2f(1, 0); glVertex2f(x + width, y)
        glTexCoord2f(1, 1); glVertex2f(x + width, y + height)
        glTexCoord2f(0, 1); glVertex2f(x, y + height)
        glEnd()

        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

        glDisable(GL_TEXTURE_2D)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)


class MenuAction(Enum):
    NONE = auto()
    START_VERIFICATION = auto()
    QUIT = auto()