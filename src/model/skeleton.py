from dataclasses import dataclass, field
from enum import Enum

import numpy as np


# All distances use centimeters.
# Angles will use radians when values are added later.


class Side(Enum):
    LEFT = "left"
    RIGHT = "right"


@dataclass
class DHChain:
    """Standard DH rows: [theta, d, a, alpha]."""

    name: str
    rows: np.ndarray
    parent_point: str
    base_transform: np.ndarray = field(
        default_factory=lambda: np.eye(4, dtype=float)
    )
    joint_rotations: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3, 3), dtype=float)
    )

    def __post_init__(self) -> None:
        if self.rows.ndim != 2 or self.rows.shape[1] != 4:
            raise ValueError("DH rows must have shape (n, 4).")


@dataclass
class TorsoPoints:
    """Named torso landmarks in local torso coordinates."""

    left_shoulder: np.ndarray
    neck_base: np.ndarray
    right_shoulder: np.ndarray

    left_hip: np.ndarray
    right_hip: np.ndarray

    spine_upper: np.ndarray
    spine_mid_upper: np.ndarray
    spine_mid_lower: np.ndarray
    spine_lower: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return vars(self)


@dataclass
class FigureModel:
    """Complete kinematic description of one human figure."""

    torso: TorsoPoints

    head: DHChain

    left_arm: DHChain
    right_arm: DHChain

    left_leg: DHChain
    right_leg: DHChain

    spine_joints: np.ndarray = field(
    default_factory=lambda: np.repeat(
        np.eye(3, dtype=float)[None, :, :],
        4,
        axis=0,
    )
)

    world_origin: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=float)
    )


def empty_points() -> TorsoPoints:
    """Create placeholder torso points; dimensions are assigned later."""

    point = lambda: np.full(3, np.nan, dtype=float)

    return TorsoPoints(
        neck_base=point(),
        left_shoulder=point(),
        right_shoulder=point(),
        spine_upper=point(),
        spine_mid_upper=point(),
        spine_mid_lower=point(),
        spine_lower=point(),
        left_hip=point(),
        right_hip=point(),
    )


def empty_chain(name: str, row_count: int, parent_point: str) -> DHChain:
    """Create an unconfigured DH chain."""

    return DHChain(
        name=name,
        rows=np.full((row_count, 4), np.nan, dtype=float),
        parent_point=parent_point,
    )


def create_default_figure() -> FigureModel:
    """Create the future T-pose model without dimensions or angles."""

    return FigureModel(
        torso=empty_points(),

        head=empty_chain("head", 2, "neck_base"),

        left_arm=empty_chain("left_arm", 5, "left_shoulder"),
        right_arm=empty_chain("right_arm", 5, "right_shoulder"),

        left_leg=empty_chain("left_leg", 6, "left_hip"),
        right_leg=empty_chain("right_leg", 6, "right_hip"),
        
    )