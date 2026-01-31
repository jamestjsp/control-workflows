"""DC motor control example from MATLAB."""

from .model import DCMotorParams, create_dc_motor_ss
from .lqr import lqr_gain
from .matlab_example import run_matlab_example

__all__ = ["DCMotorParams", "create_dc_motor_ss", "lqr_gain", "run_matlab_example"]
