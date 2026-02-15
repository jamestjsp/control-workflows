"""System analysis functions."""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from ctrlsys import ab01nd, sb03md

from .state_space import StateSpace


def controllability_matrix(sys: StateSpace) -> NDArray[np.float64]:
    """Compute controllability matrix [B, AB, ..., A^(n-1)B]."""
    n = sys.n_states
    m = sys.n_inputs
    ctrb = np.zeros((n, n * m))
    ctrb[:, :m] = sys.B
    for i in range(1, n):
        ctrb[:, i * m : (i + 1) * m] = sys.A @ ctrb[:, (i - 1) * m : i * m]
    return ctrb


def observability_matrix(sys: StateSpace) -> NDArray[np.float64]:
    """Compute observability matrix [C; CA; ...; CA^(n-1)]."""
    n = sys.n_states
    p = sys.n_outputs
    obsv = np.zeros((n * p, n))
    obsv[:p, :] = sys.C
    for i in range(1, n):
        obsv[i * p : (i + 1) * p, :] = obsv[(i - 1) * p : i * p, :] @ sys.A
    return obsv


def is_controllable(sys: StateSpace, tol: float = 0.0) -> bool:
    """Check controllability via ab01nd."""
    if sys.n_states == 0:
        return True
    A_f = np.asfortranarray(sys.A)
    B_f = np.asfortranarray(sys.B)
    _, _, ncont, *_ = ab01nd("N", A_f, B_f, tol)
    return ncont == sys.n_states


def is_observable(sys: StateSpace, tol: float = 0.0) -> bool:
    """Check observability via duality."""
    if sys.n_states == 0:
        return True
    dual = StateSpace(sys.A.T, sys.C.T, sys.B.T, sys.D.T)
    return is_controllable(dual, tol)


def ctrb_gramian(sys: StateSpace) -> NDArray[np.float64]:
    """Compute controllability Gramian via sb03md."""
    if sys.n_states == 0:
        return np.zeros((0, 0))
    n = sys.n_states
    A_f = np.asfortranarray(sys.A)
    C_f = np.asfortranarray(-sys.B @ sys.B.T)
    x, *_ = sb03md("C", "X", "N", "N", n, A_f, C_f)
    return x


def obsv_gramian(sys: StateSpace) -> NDArray[np.float64]:
    """Compute observability Gramian via sb03md."""
    if sys.n_states == 0:
        return np.zeros((0, 0))
    n = sys.n_states
    A_f = np.asfortranarray(sys.A)
    C_f = np.asfortranarray(-sys.C.T @ sys.C)
    x, *_ = sb03md("C", "X", "N", "T", n, A_f, C_f)
    return x


def hankel_singular_values(sys: StateSpace) -> NDArray[np.float64]:
    """Compute Hankel singular values: sqrt(eig(Wc @ Wo))."""
    if sys.n_states == 0:
        return np.array([])
    Wc = ctrb_gramian(sys)
    Wo = obsv_gramian(sys)
    eigvals = np.linalg.eigvals(Wc @ Wo)
    hsv = np.sqrt(np.maximum(np.real(eigvals), 0.0))
    return np.sort(hsv)[::-1]
