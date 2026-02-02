"""Reproduce MATLAB DC motor control example using SLICOT."""

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from slicot import ab04md, tf01md, tb05ad

from .model import create_dc_motor_ss
from .lqr import lqr_gain


def dc_gain(A: NDArray, B: NDArray, C: NDArray) -> NDArray:
    """Compute DC gain using SLICOT tb05ad at freq=0."""
    A_f = np.asfortranarray(A)
    B_f = np.asfortranarray(B)
    C_f = np.asfortranarray(C)
    g, *_ = tb05ad("N", "G", A_f, B_f, C_f, 0.0 + 0.0j)
    return g.real


def discretize(
    A: NDArray, B: NDArray, C: NDArray, D: NDArray, dt: float
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Discretize continuous system using SLICOT ab04md (bilinear/Tustin)."""
    A_f = np.asfortranarray(A)
    B_f = np.asfortranarray(B)
    C_f = np.asfortranarray(C)
    D_f = np.asfortranarray(D)
    Ad, Bd, Cd, Dd, _ = ab04md("C", A_f, B_f, C_f, D_f, alpha=1.0, beta=dt / 2)
    return Ad, Bd, Cd, Dd


def simulate_discrete(
    Ad: NDArray, Bd: NDArray, Cd: NDArray, Dd: NDArray, u_seq: NDArray, x0: NDArray
) -> NDArray:
    """Simulate discrete system using SLICOT tf01md."""
    Ad_f = np.asfortranarray(Ad)
    Bd_f = np.asfortranarray(Bd)
    Cd_f = np.asfortranarray(Cd)
    Dd_f = np.asfortranarray(Dd)
    u_f = np.asfortranarray(u_seq.T)
    y, _, _ = tf01md(Ad_f, Bd_f, Cd_f, Dd_f, u_f, x0.copy())
    return y.T


def build_closed_loop(
    A: NDArray, B: NDArray, C: NDArray, K: NDArray | None, Ki: float, Kff: float
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Build closed-loop system with integral action.

    Augmented state: [x, q] where q_dot = r - y
    Inputs: [r, Td]
    Output: y = omega
    """
    n = A.shape[0]
    B_Va = B[:, 0:1]
    B_Td = B[:, 1:2]

    K_fb = np.zeros((1, n)) if K is None else K.reshape(1, -1)

    A_cl = np.block(
        [
            [A - B_Va @ K_fb, B_Va * Ki],
            [-C, np.zeros((1, 1))],
        ]
    )
    B_cl = np.block(
        [
            [B_Va * Kff, B_Td],
            [np.ones((1, 1)), np.zeros((1, 1))],
        ]
    )
    C_cl = np.block([[C, np.zeros((1, 1))]])
    D_cl = np.zeros((1, 2))

    return A_cl, B_cl, C_cl, D_cl


def run_matlab_example(show_plot: bool = True, verbose: bool = True) -> dict:
    """Run DC motor control comparison: feedforward vs integral vs LQR."""
    A, B, C, _ = create_dc_motor_ss()
    n = A.shape[0]

    B_Va = B[:, 0:1]
    Kff = 1.0 / dc_gain(A, B_Va, C).item()

    A_aug = np.block(
        [
            [A, np.zeros((n, 1))],
            [-C, np.zeros((1, 1))],
        ]
    )
    B_aug = np.vstack([B_Va, np.zeros((1, 1))])

    Q_lqr = np.diag([1.0, 1.0, 20.0])
    R_lqr = np.array([[0.01]])

    K_aug, _ = lqr_gain(A_aug, B_aug, Q_lqr, R_lqr)
    K_state = K_aug[0, :n]
    Ki_lqr = K_aug[0, n]
    Ki_rlocus = 5.0

    dt = 0.01
    t_final = 6.0
    Td_time = 1.0
    Td_mag = 0.1

    N = int(t_final / dt)
    t = np.arange(N) * dt

    u_seq = np.zeros((N, 2))
    u_seq[:, 0] = 1.0
    u_seq[int(Td_time / dt) :, 1] = Td_mag

    x0_aug = np.zeros(n + 1)

    A_ff, B_ff, C_ff, D_ff = build_closed_loop(A, B, C, None, 0.0, Kff)
    Ad, Bd, Cd, Dd = discretize(A_ff, B_ff, C_ff, D_ff, dt)
    y_ff = simulate_discrete(Ad, Bd, Cd, Dd, u_seq, x0_aug)[:, 0]

    A_int, B_int, C_int, D_int = build_closed_loop(A, B, C, None, Ki_rlocus, Kff)
    Ad, Bd, Cd, Dd = discretize(A_int, B_int, C_int, D_int, dt)
    y_int = simulate_discrete(Ad, Bd, Cd, Dd, u_seq, x0_aug)[:, 0]

    A_lqr, B_lqr, C_lqr, D_lqr = build_closed_loop(
        A, B, C, K_state.reshape(1, -1), -Ki_lqr, Kff
    )
    Ad, Bd, Cd, Dd = discretize(A_lqr, B_lqr, C_lqr, D_lqr, dt)
    y_lqr = simulate_discrete(Ad, Bd, Cd, Dd, u_seq, x0_aug)[:, 0]

    if verbose:
        print("DC Motor Control Example")
        print("=" * 40)
        print(f"Feedforward gain Kff: {Kff:.2f}")
        print(f"Integral gain (root locus): {Ki_rlocus:.2f}")
        print(f"LQR gains: K_state={K_state}, Ki={Ki_lqr:.2f}")

    if show_plot:
        _, ax = plt.subplots(figsize=(10, 6))
        ax.plot(t, np.ones_like(t), "k--", label="Reference", linewidth=1)
        ax.plot(t, y_ff, label="Feedforward only")
        ax.plot(t, y_int, label="Integral feedback (K=5)")
        ax.plot(t, y_lqr, label="LQR")
        ax.axvline(Td_time, color="gray", linestyle=":", label="Disturbance")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Angular velocity (rad/s)")
        ax.set_title("DC Motor Speed Control - Disturbance Rejection Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, t_final)
        ax.set_ylim(-0.1, 1.5)
        plt.tight_layout()
        plt.show()

    return {
        "Kff": Kff,
        "Ki_rlocus": Ki_rlocus,
        "K_lqr": K_aug,
        "t": t,
        "y_ff": y_ff,
        "y_int": y_int,
        "y_lqr": y_lqr,
    }


if __name__ == "__main__":
    run_matlab_example()
