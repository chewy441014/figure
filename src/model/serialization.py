from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def save_scene(
    path: str | Path,
    rotations: dict[str, np.ndarray],
    camera,
    *,
    skeleton_transparent: bool,
    show_debug_frames: bool,
) -> None:
    """Save pose, camera, and display settings."""

    data = {
        "rotations": {
            name: values.tolist()
            for name, values in rotations.items()
        },
        "camera": {
            "yaw": camera.yaw,
            "pitch": camera.pitch,
            "distance": camera.distance,
            "target": camera.target.tolist(),
        },
        "display": {
            "skeleton_transparent": skeleton_transparent,
            "show_debug_frames": show_debug_frames,
        },
    }

    Path(path).write_text(
        json.dumps(data, indent=2),
        encoding="utf-8",
    )


def load_scene(
    path: str | Path,
    rotations: dict[str, np.ndarray],
    camera,
) -> tuple[bool, bool]:
    """Load values into the existing application state."""

    data = json.loads(
        Path(path).read_text(encoding="utf-8")
    )

    saved_rotations = data.get("rotations", {})

    for name, current in rotations.items():
        if name not in saved_rotations:
            continue

        loaded = np.asarray(
            saved_rotations[name],
            dtype=float,
        )

        if loaded.shape != current.shape:
            raise ValueError(
                f"{name}: expected shape {current.shape}, "
                f"received {loaded.shape}."
            )

        current[:] = loaded

    camera_data = data.get("camera", {})

    camera.yaw = float(camera_data.get("yaw", camera.yaw))
    camera.pitch = float(camera_data.get("pitch", camera.pitch))
    camera.distance = float(
        camera_data.get("distance", camera.distance)
    )

    target = np.asarray(
        camera_data.get("target", camera.target),
        dtype=float,
    )

    if target.shape != (3,):
        raise ValueError("Camera target must contain three values.")

    camera.target[:] = target

    display = data.get("display", {})

    return (
        bool(display.get("skeleton_transparent", False)),
        bool(display.get("show_debug_frames", True)),
    )