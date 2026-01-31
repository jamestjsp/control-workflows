"""State-space model representation."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np
from numpy.typing import NDArray
from slicot import tb05ad, ab04md, tf01md


@dataclass
class StateSpace:
    """
    Continuous-time state-space model.

    dx/dt = A @ x + B @ u
    y = C @ x + D @ u
    """

    A: NDArray[np.float64]
    B: NDArray[np.float64]
    C: NDArray[np.float64]
    D: NDArray[np.float64]

    def __post_init__(self) -> None:
        self.A = np.atleast_2d(np.asarray(self.A, dtype=np.float64))
        self.B = np.atleast_2d(np.asarray(self.B, dtype=np.float64))
        self.C = np.atleast_2d(np.asarray(self.C, dtype=np.float64))
        self.D = np.atleast_2d(np.asarray(self.D, dtype=np.float64))

    @property
    def n_states(self) -> int:
        return self.A.shape[0]

    @property
    def n_inputs(self) -> int:
        return self.B.shape[1]

    @property
    def n_outputs(self) -> int:
        return self.C.shape[0]

    def poles(self) -> NDArray[np.complex128]:
        return np.linalg.eigvals(self.A)

    def dc_gain(self) -> NDArray[np.float64]:
        """Compute DC gain using SLICOT tb05ad at freq=0."""
        A_f = np.asfortranarray(self.A)
        B_f = np.asfortranarray(self.B)
        C_f = np.asfortranarray(self.C)
        g, *_ = tb05ad("N", "G", A_f, B_f, C_f, 0.0 + 0.0j)
        return g.real + self.D

    def is_stable(self) -> bool:
        return bool(np.all(np.real(self.poles()) < 0))

    def discretize(self, dt: float, method: str = "tustin") -> StateSpace:
        """Convert to discrete-time using SLICOT ab04md."""
        if method != "tustin":
            raise NotImplementedError(f"Method {method} not supported")
        A_f = np.asfortranarray(self.A)
        B_f = np.asfortranarray(self.B)
        C_f = np.asfortranarray(self.C)
        D_f = np.asfortranarray(self.D)
        Ad, Bd, Cd, Dd, _ = ab04md("C", A_f, B_f, C_f, D_f, alpha=1.0, beta=dt / 2)
        return StateSpace(Ad, Bd, Cd, Dd)

    def simulate(
        self, u: NDArray[np.float64], x0: NDArray[np.float64] | None = None
    ) -> NDArray[np.float64]:
        """Simulate discrete system using SLICOT tf01md."""
        if x0 is None:
            x0 = np.zeros(self.n_states)
        A_f = np.asfortranarray(self.A)
        B_f = np.asfortranarray(self.B)
        C_f = np.asfortranarray(self.C)
        D_f = np.asfortranarray(self.D)
        u_f = np.asfortranarray(u.T)
        y, _, _ = tf01md(A_f, B_f, C_f, D_f, u_f, x0.copy())
        return y.T

    def __neg__(self) -> StateSpace:
        return StateSpace(self.A, self.B, -self.C, -self.D)

    def __add__(self, other: StateSpace) -> StateSpace:
        """Parallel connection using SLICOT ab05pd."""
        if not isinstance(other, StateSpace):
            return NotImplemented
        from .interconnect import parallel

        return parallel(self, other)

    def __mul__(self, other: StateSpace) -> StateSpace:
        """Series connection using SLICOT ab05md: y = G1 * G2 * u (G1 after G2)."""
        if not isinstance(other, StateSpace):
            return NotImplemented
        from .interconnect import series

        return series(other, self)

    def feedback(self, K: StateSpace | NDArray, sign: int = -1) -> StateSpace:
        """Closed-loop system with feedback gain/system K using SLICOT ab05nd."""
        from .interconnect import feedback as fb

        if isinstance(K, np.ndarray):
            K = StateSpace(
                np.zeros((0, 0)),
                np.zeros((0, self.n_outputs)),
                np.zeros((self.n_inputs, 0)),
                K,
            )
        return fb(self, K, sign=float(sign))

    def __repr__(self) -> str:
        return f"StateSpace(n={self.n_states}, m={self.n_inputs}, p={self.n_outputs})"


def ss(
    A: NDArray | Sequence,
    B: NDArray | Sequence,
    C: NDArray | Sequence,
    D: NDArray | Sequence,
) -> StateSpace:
    """Create state-space model from matrices."""
    return StateSpace(
        np.atleast_2d(A),
        np.atleast_2d(B),
        np.atleast_2d(C),
        np.atleast_2d(D),
    )
