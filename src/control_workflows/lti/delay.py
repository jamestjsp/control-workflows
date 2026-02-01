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
