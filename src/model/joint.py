from dataclasses import dataclass

import numpy as np


@dataclass
class Joint:
    rotation: np.ndarray      # [x, y, z] radians
    enabled: np.ndarray       # [bool, bool, bool]
    minimum: np.ndarray       # radians
    maximum: np.ndarray       # radians