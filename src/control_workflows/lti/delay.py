"""Time delay support for LTI systems."""

from __future__ import annotations
from typing import TYPE_CHECKING
from math import factorial

import numpy as np

if TYPE_CHECKING:
    from .transfer_function import TransferFunction
    from .state_space import StateSpace
    from .zpk import ZPK

    LTISystem = TransferFunction | StateSpace | ZPK


def pade(tau: float, n: int = 5) -> TransferFunction:
    """
    Pade approximation of time delay e^(-tau*s).

    Returns TF with num/den such that G(s) ≈ e^(-tau*s).
    Order n gives (n,n) approximant.

    Args:
        tau: Time delay (seconds)
        n: Order of approximation (default 5)

    Returns:
        TransferFunction approximating the delay
    """
    from .transfer_function import TransferFunction

    if tau == 0:
        return TransferFunction(np.array([1.0]), np.array([1.0]))

    if tau < 0:
        raise ValueError("Delay must be non-negative")

    num = np.zeros(n + 1)
    den = np.zeros(n + 1)

    for k in range(n + 1):
        coef = (
            factorial(2 * n - k)
            * factorial(n)
            / (factorial(2 * n) * factorial(k) * factorial(n - k))
        )
        num[n - k] = coef * ((-tau) ** k)
        den[n - k] = coef * (tau**k)

    return TransferFunction(num, den)


def absorbDelay(sys: LTISystem, n: int = 5) -> LTISystem:
    """
    Absorb time delays into system dynamics using Pade approximation.

    Converts exact delay representation to rational approximation suitable
    for time-domain simulation.

    Args:
        sys: LTI system with delays
        n: Pade approximation order

    Returns:
        System with delays absorbed (delays set to zero)
    """
    from .transfer_function import TransferFunction
    from .state_space import StateSpace
    from .zpk import ZPK
    from .conversions import tf2ss

    if isinstance(sys, TransferFunction):
        if sys.input_delay == 0:
            return TransferFunction(sys.num.copy(), sys.den.copy(), input_delay=0.0)
        delay_tf = pade(sys.input_delay, n)
        result = sys * delay_tf
        return TransferFunction(result.num.copy(), result.den.copy(), input_delay=0.0)

    if isinstance(sys, ZPK):
        if sys.delay == 0:
            return ZPK(sys.z.copy(), sys.p.copy(), sys.k, delay=0.0)
        delay_tf = pade(sys.delay, n)
        delay_zpk = ZPK(delay_tf.zeros(), delay_tf.poles(), delay_tf.num[0], delay=0.0)
        result = sys * delay_zpk
        return ZPK(result.z.copy(), result.p.copy(), result.k, delay=0.0)

    if isinstance(sys, StateSpace):
        if sys.dt is not None:
            return _absorb_delay_discrete(sys)

        result = sys
        if sys.input_delay is not None and np.any(sys.input_delay > 0):
            for i, d in enumerate(sys.input_delay):
                if d > 0:
                    delay_ss = tf2ss(pade(d, n))
                    result = _apply_input_delay_ss(result, delay_ss, i)

        if sys.output_delay is not None and np.any(sys.output_delay > 0):
            for i, d in enumerate(sys.output_delay):
                if d > 0:
                    delay_ss = tf2ss(pade(d, n))
                    result = _apply_output_delay_ss(result, delay_ss, i)

        return StateSpace(
            result.A.copy(),
            result.B.copy(),
            result.C.copy(),
            result.D.copy(),
            input_delay=None,
            output_delay=None,
        )

    raise TypeError(f"Unsupported system type: {type(sys)}")


def _apply_input_delay_ss(
    sys: StateSpace, delay: StateSpace, input_idx: int
) -> StateSpace:
    """Apply delay to specific input channel of state-space system."""
    from .state_space import StateSpace

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

    D_new = sys.D.copy()

    return StateSpace(A_new, B_new, C_new, D_new)


def _apply_output_delay_ss(
    sys: StateSpace, delay: StateSpace, output_idx: int
) -> StateSpace:
    """Apply delay to specific output channel of state-space system."""
    from .state_space import StateSpace

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


def thiran(d: float, n: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """
    Thiran all-pass filter coefficients for fractional delay D = N + d.

    The Thiran filter approximates z^(-D) where D = n + d is the total delay.
    For the filter to be stable, D must be in [n-0.5, n+0.5].

    This function computes coefficients for fractional part d ∈ (0,1) with
    filter order n, resulting in total delay of n+d samples.

    H(z) = z^(-n) * A(z^(-1)) / A(z)

    Args:
        d: Fractional delay beyond integer part (0 < d < 1)
        n: Filter order (default 3)

    Returns:
        (num, den) coefficient arrays for discrete transfer function
    """
    from math import comb

    if d <= 0 or d >= 1:
        raise ValueError("Fractional delay d must be in (0, 1)")

    D = n + d

    a = np.zeros(n + 1)
    for k in range(n + 1):
        prod = 1.0
        for i in range(n + 1):
            prod *= (D - n + i) / (D - n + k + i)
        a[k] = ((-1) ** k) * comb(n, k) * prod

    return a[::-1].copy(), a.copy()


def discretize_delay(delay: float, dt: float, thiran_order: int = 3) -> StateSpace:
    """
    Create discrete delay system using shift registers and Thiran filter.

    Decomposes delay = k*dt + frac where k = floor(delay/dt).
    - Integer part (k): shift register states (z^(-k))
    - Fractional part (frac): Thiran all-pass filter of order n provides
      (n + frac) samples of delay, so we subtract n from k.

    Total states = max(k - thiran_order, 0) + thiran_order for fractional delay.

    Args:
        delay: Total delay in seconds
        dt: Sample time
        thiran_order: Order of Thiran filter for fractional delay (0 = round only)

    Returns:
        Discrete StateSpace realizing the delay
    """
    from .state_space import StateSpace

    if delay < 0:
        raise ValueError("Delay must be non-negative")

    if delay == 0:
        return StateSpace(
            np.zeros((0, 0)), np.zeros((0, 1)), np.zeros((1, 0)), np.array([[1.0]])
        )

    samples = delay / dt
    k = int(samples)
    frac = samples - k

    if frac < 1e-10:
        frac = 0.0
    if thiran_order == 0 and frac > 0.5:
        k += 1
        frac = 0.0

    if k == 0 and frac < 1e-10:
        return StateSpace(
            np.zeros((0, 0)), np.zeros((0, 1)), np.zeros((1, 0)), np.array([[1.0]])
        )

    thiran_ss = None
    shift_samples = k

    if frac > 1e-10 and thiran_order > 0:
        num, den = thiran(frac, thiran_order)
        thiran_ss = _tf_to_ss_discrete(num, den)
        shift_samples = max(0, k - thiran_order)

    if shift_samples > 0:
        A_shift = np.zeros((shift_samples, shift_samples))
        if shift_samples > 1:
            A_shift[1:, :-1] = np.eye(shift_samples - 1)
        B_shift = np.zeros((shift_samples, 1))
        B_shift[0, 0] = 1.0
        C_shift = np.zeros((1, shift_samples))
        C_shift[0, -1] = 1.0
        D_shift = np.zeros((1, 1))
        shift_ss = StateSpace(A_shift, B_shift, C_shift, D_shift)

        if thiran_ss is not None:
            return shift_ss * thiran_ss
        return shift_ss

    assert thiran_ss is not None
    return thiran_ss


def _tf_to_ss_discrete(num: np.ndarray, den: np.ndarray) -> StateSpace:
    """
    Convert discrete SISO TF coefficients to controllable canonical state-space.

    H(z) = num(z)/den(z) where polynomials are in descending powers of z.
    Uses the "z^(-1) form" controllable canonical realization.
    """
    from .state_space import StateSpace

    num = np.atleast_1d(np.trim_zeros(num, "f"))
    den = np.atleast_1d(np.trim_zeros(den, "f"))

    if len(num) == 0:
        num = np.array([0.0])

    a0 = den[0]
    den = den / a0
    num = num / a0

    n = len(den) - 1
    if n == 0:
        return StateSpace(
            np.zeros((0, 0)),
            np.zeros((0, 1)),
            np.zeros((1, 0)),
            np.array([[num[0]]]),
        )

    num_padded = np.zeros(n + 1)
    num_padded[-(len(num)) :] = num

    d = num_padded[0]

    A = np.zeros((n, n))
    A[:-1, 1:] = np.eye(n - 1)
    A[-1, :] = -den[1:][::-1]

    B = np.zeros((n, 1))
    B[-1, 0] = 1.0

    C = (num_padded[1:] - d * den[1:])[::-1].reshape(1, -1)

    D = np.array([[d]])

    return StateSpace(A, B, C, D)


def _absorb_delay_discrete(sys: StateSpace) -> StateSpace:
    """Convert discrete-time delays to z^(-k) poles (shift register states)."""
    from .state_space import StateSpace, _apply_input_delay, _apply_output_delay

    assert sys.dt is not None
    result = StateSpace(
        sys.A.copy(), sys.B.copy(), sys.C.copy(), sys.D.copy(), dt=sys.dt
    )

    if sys.input_delay is not None and np.any(sys.input_delay > 0):
        for i, d in enumerate(sys.input_delay):
            if d > 0:
                k = int(round(d))
                if k > 0:
                    delay_ss = discretize_delay(k * sys.dt, sys.dt, thiran_order=0)
                    result = _apply_input_delay(result, delay_ss, i)
                    result = StateSpace(
                        result.A, result.B, result.C, result.D, dt=sys.dt
                    )

    if sys.output_delay is not None and np.any(sys.output_delay > 0):
        for i, d in enumerate(sys.output_delay):
            if d > 0:
                k = int(round(d))
                if k > 0:
                    delay_ss = discretize_delay(k * sys.dt, sys.dt, thiran_order=0)
                    result = _apply_output_delay(result, delay_ss, i)
                    result = StateSpace(
                        result.A, result.B, result.C, result.D, dt=sys.dt
                    )

    return result
