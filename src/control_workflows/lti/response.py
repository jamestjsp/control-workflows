"""Time response functions."""

from __future__ import annotations
from typing import Tuple
import numpy as np
from numpy.typing import NDArray

from .state_space import StateSpace


def _auto_time(sys: StateSpace, t_final: float | None, n_points: int) -> NDArray:
    """Generate time vector based on system poles."""
    if t_final is None:
        poles = sys.poles()
        real_parts = np.real(poles)
        stable_poles = real_parts[real_parts < 0]
        if len(stable_poles) > 0:
            t_final = 7.0 / np.min(np.abs(stable_poles))
        else:
            t_final = 10.0
    return np.linspace(0, t_final, n_points)


def step_response(
    sys: StateSpace,
    t: NDArray | None = None,
    t_final: float | None = None,
    n_points: int = 1000,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Step response of continuous-time system. Returns (t, y)."""
    if t is None:
        t = _auto_time(sys, t_final, n_points)
    else:
        t = np.asarray(t)

    dt = t[1] - t[0]
    sys_d = sys.discretize(dt)

    u = np.ones((len(t), sys.n_inputs))
    y = sys_d.simulate(u)

    return t, y.T


def impulse_response(
    sys: StateSpace,
    t: NDArray | None = None,
    t_final: float | None = None,
    n_points: int = 1000,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Impulse response of continuous-time system. Returns (t, y)."""
    if t is None:
        t = _auto_time(sys, t_final, n_points)
    else:
        t = np.asarray(t)

    dt = t[1] - t[0]
    sys_d = sys.discretize(dt)

    u = np.zeros((len(t), sys.n_inputs))
    u[0, :] = 1.0 / dt
    y = sys_d.simulate(u)

    return t, y.T


def initial_response(
    sys: StateSpace,
    x0: NDArray,
    t: NDArray | None = None,
    t_final: float | None = None,
    n_points: int = 1000,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Free response from initial condition. Returns (t, y)."""
    x0 = np.asarray(x0, dtype=np.float64)
    if t is None:
        t = _auto_time(sys, t_final, n_points)
    else:
        t = np.asarray(t)

    dt = t[1] - t[0]
    sys_d = sys.discretize(dt)

    u = np.zeros((len(t), sys.n_inputs))
    y = sys_d.simulate(u, x0)

    return t, y.T
