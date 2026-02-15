"""Controller design functions."""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from ctrlsys import sb01bd


def place(
    A: NDArray[np.float64], B: NDArray[np.float64], poles: NDArray, tol: float = 0.0
) -> NDArray[np.float64]:
    """Pole placement via sb01bd. Returns F such that eig(A + B @ F) = poles."""
    A = np.atleast_2d(np.asarray(A, dtype=np.float64))
    B = np.atleast_2d(np.asarray(B, dtype=np.float64))
    poles = np.asarray(poles, dtype=np.complex128)

    n = A.shape[0]
    m = B.shape[1]
    np_poles = len(poles)

    if np_poles != n:
        raise ValueError(f"Expected {n} poles, got {np_poles}")

    wr = np.real(poles).astype(np.float64)
    wi = np.imag(poles).astype(np.float64)

    A_f = np.asfortranarray(A)
    B_f = np.asfortranarray(B)

    _, _, _, _, _, _, f, *_ = sb01bd(
        "C", n, m, np_poles, 0.0, A_f, B_f, wr.copy(), wi.copy(), tol
    )

    return f[:m, :n]


def acker(
    A: NDArray[np.float64], B: NDArray[np.float64], poles: NDArray
) -> NDArray[np.float64]:
    """Ackermann's formula for SISO pole placement. Returns F (1 x n)."""
    A = np.atleast_2d(np.asarray(A, dtype=np.float64))
    B = np.atleast_2d(np.asarray(B, dtype=np.float64))
    poles = np.asarray(poles, dtype=np.complex128)

    n = A.shape[0]
    if B.shape[1] != 1:
        raise ValueError("acker only supports SISO systems (single input)")

    if len(poles) != n:
        raise ValueError(f"Expected {n} poles, got {len(poles)}")

    ctrb = np.zeros((n, n))
    ctrb[:, 0:1] = B
    for i in range(1, n):
        ctrb[:, i : i + 1] = A @ ctrb[:, i - 1 : i]

    phi = np.eye(n, dtype=np.complex128)
    for p in poles:
        phi = phi @ (A - p * np.eye(n))
    phi = np.real(phi)

    e_n = np.zeros((1, n))
    e_n[0, -1] = 1.0

    F = -e_n @ np.linalg.solve(ctrb, phi)
    return F.astype(np.float64)
