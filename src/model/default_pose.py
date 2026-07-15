import numpy as np

from .skeleton import create_default_figure


def rotation_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [ c, 0.0,  s, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-s, 0.0,  c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])


figure = create_default_figure()

# Standard DH order: [theta, d, a, alpha]
# Distances: centimeters. Angles: radians.

# Head: local +Z points upward.
figure.head.rows = np.array([
    [0.0,  0.0, 0.0, 0.0],  # Neck frame
    [0.0, 10.0, 0.0, 0.0],  # Neck to head center
])


# Arms: base transforms aim local +Z along world ±X.
figure.left_arm.base_transform = rotation_y(np.pi / 2)
figure.right_arm.base_transform = rotation_y(-np.pi / 2)

arm_rows = np.array([
    [0.0,  0.0, 0.0, 0.0],  # Shoulder frame
    [0.0, 30.0, 0.0, 0.0],  # Upper arm
    [0.0, 30.0, 0.0, 0.0],  # Forearm
    [0.0, 10.0, 0.0, 0.0],  # Palm
    [0.0,  0.0, 0.0, 0.0],  # Hand endpoint
])

figure.left_arm.rows = arm_rows.copy()
figure.right_arm.rows = arm_rows.copy()


# Legs: base transform aims local +Z downward.
figure.left_leg.base_transform = rotation_y(np.pi)
figure.right_leg.base_transform = rotation_y(np.pi)

leg_rows = np.array([
    [0.0,  0.0, 0.0, 0.0],       # Hip frame
    [0.0, 55.0, 0.0, 0.0],       # Hip to knee
    [0.0, 55.0, 0.0, np.pi / 2], # Knee to ankle; next +Z becomes world -Y
    [0.0, 25.0, 0.0, 0.0],       # Ankle to ball of foot
    [0.0,  3.0, 0.0, 0.0],       # Ball to big-toe tip
    [0.0,  0.0, 0.0, 0.0],       # Toe endpoint
])

figure.left_leg.rows = leg_rows.copy()
figure.right_leg.rows = leg_rows.copy()

figure.torso.neck_base = np.array([0.0, 0.0, 0.0])

figure.torso.left_shoulder = np.array([20.0, 0.0, 0.0])
figure.torso.right_shoulder = np.array([-20.0, 0.0, 0.0])

figure.torso.spine_upper = np.array([0.0, 0.0, -17.5])
figure.torso.spine_mid_upper = np.array([0.0, 0.0, -35.0])
figure.torso.spine_mid_lower = np.array([0.0, 0.0, -52.5])
figure.torso.spine_lower = np.array([0.0, 0.0, -70.0])

figure.torso.left_hip = np.array([17.5, 0.0, -70.0])
figure.torso.right_hip = np.array([-17.5, 0.0, -70.0])