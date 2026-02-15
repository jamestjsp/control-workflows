"""Steady-state Kalman filter using ctrlsys sb02od."""

import numpy as np
from numpy.typing import NDArray
from ctrlsys import sb02od


def steady_state_kalman_gain(
    A: NDArray[np.float64],
    B: NDArray[np.float64],
    C: NDArray[np.float64],
    Q: NDArray[np.float64],
    R: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute steady-state Kalman gain via discrete-time Riccati equation.

    Solves the filter ARE for system:
        x[n+1] = A*x[n] + B*u[n] + G*w[n]
        y[n] = C*x[n] + v[n]

    where w ~ N(0, Q), v ~ N(0, R)

    Args:
        A: State transition matrix (n x n)
        B: Input matrix (n x m) - also used as process noise input
        C: Output matrix (p x n)
        Q: Process noise covariance (scalar or m x m)
        R: Measurement noise covariance (scalar or p x p)

    Returns:
        K: Steady-state Kalman gain (n x p)
        P: Steady-state error covariance (n x n)
    """
    n = A.shape[0]
    p = C.shape[0]

    Q = np.atleast_2d(Q)
    R = np.atleast_2d(R)

    Qw = B @ Q @ B.T

    A_f = np.asfortranarray(A.T)
    B_f = np.asfortranarray(C.T)
    Q_f = np.asfortranarray(Qw)
    R_f = np.asfortranarray(R)
    L_f = np.asfortranarray(np.zeros((n, p)))

    X, *_ = sb02od('D', 'B', 'N', 'U', 'Z', 'S', n, p, 0, A_f, B_f, Q_f, R_f, L_f, 0.0)

    P = X

    S_innov = C @ P @ C.T + R
    K = P @ C.T @ np.linalg.inv(S_innov)

    return K, P
