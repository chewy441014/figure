from __future__ import annotations

import numpy as np


def limit(enabled, minimum, maximum):
    return {
        "enabled": np.array(enabled, dtype=bool),
        "minimum": np.radians(minimum),
        "maximum": np.radians(maximum),
    }


FIXED = limit(
    [False, False, False],
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0],
)


JOINT_LIMITS = {
    "head": [
        FIXED,
        limit(
            [True, True, True],
            [-45.0, -70.0, -45.0],
            [45.0, 70.0, 45.0],
        ),
    ],

    "left_arm": [
        FIXED,  # Base
        limit(  # Shoulder
            [True, True, True],
            [-180.0, -180.0, -180.0],
            [180.0, 180.0, 180.0],
        ),
        limit(  # Elbow
            [False, True, False],
            [0.0, -145.0, 0.0],
            [0.0, 0.0, 0.0],
        ),
        limit(  # Wrist
            [True, True, False],
            [-80.0, -70.0, 0.0],
            [80.0, 70.0, 0.0],
        ),
        limit(  # Index knuckle
            [False, True, False],
            [0.0, -90.0, 0.0],
            [0.0, 0.0, 0.0],
        ),
    ],

    "right_arm": [
        FIXED,
        limit(
            [True, True, True],
            [-180.0, -180.0, -180.0],
            [180.0, 180.0, 180.0],
        ),
        limit(
            [False, True, False],
            [0.0, 0.0, 0.0],
            [0.0, 145.0, 0.0],
        ),
        limit(
            [True, True, False],
            [-80.0, -70.0, 0.0],
            [80.0, 70.0, 0.0],
        ),
        limit(
            [False, True, False],
            [0.0, 0.0, 0.0],
            [0.0, 90.0, 0.0],
        ),
    ],

    "left_leg": [
        FIXED,  # Base
        limit(  # Hip
            [True, True, True],
            [-120.0, -45.0, -45.0],
            [45.0, 45.0, 45.0],
        ),
        limit(  # Knee
            [False, True, False],
            [0.0, 0.0, 0.0],
            [0.0, 145.0, 0.0],
        ),
        limit(  # Ankle
            [True, True, False],
            [-45.0, -30.0, 0.0],
            [45.0, 30.0, 0.0],
        ),
        limit(  # Ball
            [True, False, False],
            [-30.0, 0.0, 0.0],
            [45.0, 0.0, 0.0],
        ),
        limit(  # Toe
            [True, False, False],
            [-20.0, 0.0, 0.0],
            [45.0, 0.0, 0.0],
        ),
    ],

    "right_leg": [
        FIXED,
        limit(
            [True, True, True],
            [-120.0, -45.0, -45.0],
            [45.0, 45.0, 45.0],
        ),
        limit(
            [False, True, False],
            [0.0, 0.0, 0.0],
            [0.0, 145.0, 0.0],
        ),
        limit(
            [True, True, False],
            [-45.0, -30.0, 0.0],
            [45.0, 30.0, 0.0],
        ),
        limit(
            [True, False, False],
            [-30.0, 0.0, 0.0],
            [45.0, 0.0, 0.0],
        ),
        limit(
            [True, False, False],
            [-20.0, 0.0, 0.0],
            [45.0, 0.0, 0.0],
        ),
    ],
}