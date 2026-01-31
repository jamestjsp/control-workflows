"""Reproduce MATLAB Kalman filtering example."""

import numpy as np
from numpy.typing import NDArray

from .steady_state import steady_state_kalman_gain
from .time_varying import kalman_filter


def create_system() -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    float,
    float,
    float,
]:
    """Create the MATLAB example system."""
    A = np.array([
        [1.1269, -0.4940, 0.1129],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    B = np.array([[-0.3832], [0.5919], [0.5191]])
    C = np.array([[1.0, 0.0, 0.0]])
    D = 0.0
    Q = 2.3
    R = 1.0
    return A, B, C, D, Q, R


def generate_data(
    A: NDArray[np.float64],
    B: NDArray[np.float64],
    C: NDArray[np.float64],
    Q: float,
    R: float,
    N: int,
    seed: int = 42,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Generate noisy measurements."""
    rng = np.random.default_rng(seed)

    n = A.shape[0]
    x_true = np.zeros((N, n))
    y_true = np.zeros((N, 1))
    y_noisy = np.zeros((N, 1))

    x = np.zeros(n)
    for k in range(N):
        w = rng.normal(0, np.sqrt(Q))
        v = rng.normal(0, np.sqrt(R))

        x_true[k] = x
        y_true[k] = C @ x
        y_noisy[k] = y_true[k] + v

        x = A @ x + B.flatten() * w

    return x_true, y_true, y_noisy


def run_matlab_example(N: int = 100, verbose: bool = True) -> dict:
    """
    Run the full MATLAB Kalman filtering example.

    Returns dict with results for analysis.
    """
    A, B, C, _, Q, R = create_system()

    K_ss, P_ss = steady_state_kalman_gain(
        A, B, C, np.array([[Q]]), np.array([[R]])
    )

    x_true, y_true, y_noisy = generate_data(A, B, C, Q, R, N)

    x_est, y_est, P_hist = kalman_filter(
        A, B, C, np.array([[Q]]), np.array([[R]]), y_noisy
    )

    meas_err = y_noisy[:, 0] - y_true[:, 0]
    est_err = y_est[:, 0] - y_true[:, 0]

    meas_var = np.var(meas_err)
    est_var = np.var(est_err)
    reduction = (1 - est_var / meas_var) * 100

    if verbose:
        print("MATLAB Kalman Filter Example")
        print("=" * 40)
        print(f"Steady-state Kalman gain K:\n{K_ss.flatten()}")
        print(f"\nSteady-state covariance P[0,0]: {P_ss[0,0]:.4f}")
        print(f"\nMeasurement error variance: {meas_var:.4f}")
        print(f"Estimation error variance: {est_var:.4f}")
        print(f"Error reduction: {reduction:.1f}%")

    return {
        "K_ss": K_ss,
        "P_ss": P_ss,
        "x_true": x_true,
        "y_true": y_true,
        "y_noisy": y_noisy,
        "x_est": x_est,
        "y_est": y_est,
        "P_hist": P_hist,
        "meas_var": meas_var,
        "est_var": est_var,
        "reduction": reduction,
    }


if __name__ == "__main__":
    run_matlab_example(N=200)
