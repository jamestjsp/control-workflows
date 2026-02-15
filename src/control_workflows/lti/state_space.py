"""State-space model representation."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np
from numpy.typing import NDArray
from ctrlsys import tb05ad, ab04md, tf01md


@dataclass
class StateSpace:
    """
    State-space model with optional delays.

    Continuous (dt=None): dx/dt = A @ x + B @ u(t - input_delay)
    Discrete (dt=Ts):     x[k+1] = A @ x[k] + B @ u[k - input_delay]

    y = C @ x + D @ u (with output_delay if specified)
    """

    A: NDArray[np.float64]
    B: NDArray[np.float64]
    C: NDArray[np.float64]
    D: NDArray[np.float64]
    input_delay: NDArray[np.float64] | None = None
    output_delay: NDArray[np.float64] | None = None
    dt: float | None = None

    def __post_init__(self) -> None:
        self.A = np.atleast_2d(np.asarray(self.A, dtype=np.float64))
        self.B = np.atleast_2d(np.asarray(self.B, dtype=np.float64))
        self.C = np.atleast_2d(np.asarray(self.C, dtype=np.float64))
        self.D = np.atleast_2d(np.asarray(self.D, dtype=np.float64))
        if self.input_delay is not None:
            self.input_delay = np.atleast_1d(
                np.asarray(self.input_delay, dtype=np.float64)
            )
        if self.output_delay is not None:
            self.output_delay = np.atleast_1d(
                np.asarray(self.output_delay, dtype=np.float64)
            )

    @property
    def n_states(self) -> int:
        return self.A.shape[0]

    @property
    def n_inputs(self) -> int:
        return self.B.shape[1]

    @property
    def n_outputs(self) -> int:
        return self.C.shape[0]

    @property
    def is_discrete(self) -> bool:
        return self.dt is not None

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

    def freqresp(self, omega: NDArray) -> NDArray[np.complex128]:
        """Frequency response G(jw) via tb05ad, including delay effects."""
        omega = np.atleast_1d(np.asarray(omega))
        A_f = np.asfortranarray(self.A)
        B_f = np.asfortranarray(self.B)
        C_f = np.asfortranarray(self.C)
        result = np.zeros(
            (self.n_outputs, self.n_inputs, len(omega)), dtype=np.complex128
        )
        for i, w in enumerate(omega):
            g, *_ = tb05ad("N", "G", A_f, B_f, C_f, 1j * w)
            result[:, :, i] = g + self.D

        if self.input_delay is not None:
            for j, d in enumerate(self.input_delay):
                if d > 0:
                    result[:, j, :] *= np.exp(-1j * omega * d)

        if self.output_delay is not None:
            for i, d in enumerate(self.output_delay):
                if d > 0:
                    result[i, :, :] *= np.exp(-1j * omega * d)

        return result

    def discretize(
        self,
        dt: float,
        method: str = "tustin",
        delay_handling: str = "state",
        thiran_order: int = 3,
    ) -> StateSpace:
        """
        Convert to discrete-time using SLICOT ab04md.

        Args:
            dt: Sample time
            method: Discretization method ("tustin" supported)
            delay_handling: "state" (z^(-k) augmentation) or "property" (keep as delay fields)
            thiran_order: Order of Thiran filter for fractional delay (0 = round only)
        """
        if method != "tustin":
            raise NotImplementedError(f"Method {method} not supported")
        if delay_handling not in ("state", "property"):
            raise ValueError(
                f"delay_handling must be 'state' or 'property', got {delay_handling}"
            )

        A_f = np.asfortranarray(self.A)
        B_f = np.asfortranarray(self.B)
        C_f = np.asfortranarray(self.C)
        D_f = np.asfortranarray(self.D)
        Ad, Bd, Cd, Dd, _ = ab04md("C", A_f, B_f, C_f, D_f, alpha=1.0, beta=dt / 2)

        result = StateSpace(Ad, Bd, Cd, Dd, dt=dt)

        if delay_handling == "property":
            result.input_delay = (
                self.input_delay.copy() if self.input_delay is not None else None
            )
            result.output_delay = (
                self.output_delay.copy() if self.output_delay is not None else None
            )
            return result

        from .delay import discretize_delay

        if self.input_delay is not None and np.any(self.input_delay > 0):
            for i, d in enumerate(self.input_delay):
                if d > 0:
                    delay_ss = discretize_delay(d, dt, thiran_order)
                    result = _apply_input_delay(result, delay_ss, i)

        if self.output_delay is not None and np.any(self.output_delay > 0):
            for i, d in enumerate(self.output_delay):
                if d > 0:
                    delay_ss = discretize_delay(d, dt, thiran_order)
                    result = _apply_output_delay(result, delay_ss, i)

        return result

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
        return StateSpace(
            self.A,
            self.B,
            -self.C,
            -self.D,
            self.input_delay.copy() if self.input_delay is not None else None,
            self.output_delay.copy() if self.output_delay is not None else None,
            self.dt,
        )

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
        extra = ""
        if self.input_delay is not None and np.any(self.input_delay > 0):
            extra += f", input_delay={self.input_delay.tolist()}"
        if self.output_delay is not None and np.any(self.output_delay > 0):
            extra += f", output_delay={self.output_delay.tolist()}"
        if self.dt is not None:
            extra += f", dt={self.dt}"
        return f"StateSpace(n={self.n_states}, m={self.n_inputs}, p={self.n_outputs}{extra})"


def _apply_input_delay(
    sys: StateSpace, delay: StateSpace, input_idx: int
) -> StateSpace:
    """Apply discrete delay to specific input channel."""
    nd = delay.n_states
    ns = sys.n_states
    ni = sys.n_inputs
    no = sys.n_outputs

    A_new = np.zeros((ns + nd, ns + nd))
    A_new[:ns, :ns] = sys.A
    A_new[:ns, ns:] = sys.B[:, input_idx : input_idx + 1] @ delay.C
    A_new[ns:, ns:] = delay.A

    B_new = np.zeros((ns + nd, ni))
    B_new[:ns, :] = sys.B.copy()
    B_new[:ns, input_idx] = (sys.B[:, input_idx : input_idx + 1] @ delay.D).flatten()
    B_new[ns:, input_idx] = delay.B.flatten()

    C_new = np.zeros((no, ns + nd))
    C_new[:, :ns] = sys.C
    C_new[:, ns:] = sys.D[:, input_idx : input_idx + 1] @ delay.C

    D_new = sys.D.copy()
    D_new[:, input_idx] = (sys.D[:, input_idx : input_idx + 1] @ delay.D).flatten()

    return StateSpace(A_new, B_new, C_new, D_new)


def _apply_output_delay(
    sys: StateSpace, delay: StateSpace, output_idx: int
) -> StateSpace:
    """Apply discrete delay to specific output channel."""
    nd = delay.n_states
    ns = sys.n_states
    ni = sys.n_inputs
    no = sys.n_outputs

    A_new = np.zeros((ns + nd, ns + nd))
    A_new[:ns, :ns] = sys.A
    A_new[ns:, :ns] = delay.B @ sys.C[output_idx : output_idx + 1, :]
    A_new[ns:, ns:] = delay.A

    B_new = np.zeros((ns + nd, ni))
    B_new[:ns, :] = sys.B
    B_new[ns:, :] = delay.B @ sys.D[output_idx : output_idx + 1, :]

    C_new = np.zeros((no, ns + nd))
    C_new[:, :ns] = sys.C.copy()
    C_new[output_idx, :ns] = (delay.D @ sys.C[output_idx : output_idx + 1, :]).flatten()
    C_new[output_idx, ns:] = delay.C.flatten()

    D_new = sys.D.copy()
    D_new[output_idx, :] = (delay.D @ sys.D[output_idx : output_idx + 1, :]).flatten()

    return StateSpace(A_new, B_new, C_new, D_new)


def ss(
    A: NDArray | Sequence,
    B: NDArray | Sequence,
    C: NDArray | Sequence,
    D: NDArray | Sequence,
    input_delay: NDArray | Sequence | None = None,
    output_delay: NDArray | Sequence | None = None,
    dt: float | None = None,
) -> StateSpace:
    """Create state-space model from matrices with optional delays and sample time."""
    return StateSpace(
        np.atleast_2d(A),
        np.atleast_2d(B),
        np.atleast_2d(C),
        np.atleast_2d(D),
        np.asarray(input_delay) if input_delay is not None else None,
        np.asarray(output_delay) if output_delay is not None else None,
        dt,
    )
