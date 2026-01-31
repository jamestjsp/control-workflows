"""Time-varying Kalman filter using SLICOT fb01vd."""

import numpy as np
from numpy.typing import NDArray
from slicot import fb01vd


def kalman_filter(
    A: NDArray[np.float64],
    B: NDArray[np.float64],
    C: NDArray[np.float64],
    Q: NDArray[np.float64],
    R: NDArray[np.float64],
    y: NDArray[np.float64],
    u: NDArray[np.float64] | None = None,
    x0: NDArray[np.float64] | None = None,
    P0: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[NDArray[np.float64]]]:
    """
    Run Kalman filter using SLICOT fb01vd for each time step.

    Args:
        A: State transition (n x n)
        B: Process noise input matrix (n x m)
        C: Output matrix (l x n)
        Q: Process noise covariance (m x m)
        R: Measurement noise covariance (l x l)
        y: Measurements (N x l)
        u: Inputs (N x m), optional
        x0: Initial state estimate (n,)
        P0: Initial covariance (n x n)

    Returns:
        x_est: State estimates (N x n)
        y_est: Output estimates (N x l)
        P_hist: Covariance history
    """
    n = A.shape[0]
    m = B.shape[1]
    l = C.shape[0]
    N = y.shape[0]

    Q = np.atleast_2d(Q)
    R = np.atleast_2d(R)

    if u is None:
        u = np.zeros((N, m))
    x0_arr = x0 if x0 is not None else np.zeros(n)
    P0_arr = P0 if P0 is not None else np.eye(n) * 10.0

    x: NDArray[np.float64] = x0_arr.copy()
    P: NDArray[np.float64] = np.asfortranarray(P0_arr.copy())

    x_est = np.zeros((N, n))
    y_est = np.zeros((N, l))
    P_hist = []

    A_f = np.asfortranarray(A)
    B_f = np.asfortranarray(B)
    C_f = np.asfortranarray(C)
    Q_f = np.asfortranarray(Q)

    for k in range(N):
        yk = y[k : k + 1, :].T
        R_f = np.asfortranarray(R.copy())

        P_next, K, _, _, info = fb01vd(n, m, l, P, A_f, B_f, C_f, Q_f, R_f, 0.0)
        if info != 0:
            raise RuntimeError(f"fb01vd failed with info={info}")

        innov = yk - C @ x.reshape(-1, 1)
        x = x + (K @ innov).flatten()

        x_est[k] = x
        y_est[k] = (C @ x.reshape(-1, 1)).flatten()

        P = np.triu(P_next) + np.triu(P_next, 1).T
        P = np.asfortranarray(P)
        P_hist.append(P.copy())

        x = A @ x + B @ u[k]  # type: ignore[index]

    return x_est, y_est, P_hist
