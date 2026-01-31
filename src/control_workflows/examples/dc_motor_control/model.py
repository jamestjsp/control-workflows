"""DC motor state-space model."""

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


@dataclass
class DCMotorParams:
    R: float = 2.0
    L: float = 0.5
    Km: float = 0.1
    Kb: float = 0.1
    Kf: float = 0.2
    J: float = 0.02


def create_dc_motor_ss(
    p: DCMotorParams | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Create DC motor state-space model.

    States: x = [i, omega]  (armature current, angular velocity)
    Inputs: u = [Va, Td]    (voltage, load torque disturbance)
    Output: y = omega

    Returns: A, B, C, D matrices
    """
    if p is None:
        p = DCMotorParams()

    A = np.array([
        [-p.R / p.L, -p.Kb / p.L],
        [p.Km / p.J, -p.Kf / p.J],
    ])

    B = np.array([
        [1 / p.L, 0],
        [0, -1 / p.J],
    ])

    C = np.array([[0.0, 1.0]])
    D = np.array([[0.0, 0.0]])

    return A, B, C, D
