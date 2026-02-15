"""LQR controller using ctrlsys sb02od."""

import numpy as np
from numpy.typing import NDArray
from ctrlsys import sb02od


def lqr_gain(
    A: NDArray[np.float64],
    B: NDArray[np.float64],
    Q: NDArray[np.float64],
    R: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute continuous-time LQR gain via algebraic Riccati equation.

    Minimizes J = integral(x'Qx + u'Ru) dt

    Args:
        A: State matrix (n x n)
        B: Input matrix (n x m)
        Q: State cost (n x n)
        R: Input cost (m x m)

    Returns:
        K: Optimal gain (m x n), use u = -K @ x
        P: Solution to CARE (n x n)
    """
    n = A.shape[0]
    m = B.shape[1]

    A_f = np.asfortranarray(A)
    B_f = np.asfortranarray(B)
    Q_f = np.asfortranarray(Q)
    R_f = np.asfortranarray(R)
    L_f = np.asfortranarray(np.zeros((n, m)))

    X, *_ = sb02od('C', 'B', 'N', 'U', 'Z', 'S', n, m, 0, A_f, B_f, Q_f, R_f, L_f, 0.0)

    P = X
    K = np.linalg.solve(R, B.T @ P)

    return K, P
