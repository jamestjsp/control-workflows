"""Block diagram interconnections using SLICOT."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from slicot import ab05md, ab05nd, ab05pd

if TYPE_CHECKING:
    from .state_space import StateSpace


def series(g1: StateSpace, g2: StateSpace) -> StateSpace:
    """
    Cascade (series) connection using SLICOT ab05md.

    Y = G2(G1(U)) - output of G1 feeds input of G2.
    Delays accumulate: output delay of G1 + input delay of G2.
    """
    from .state_space import StateSpace

    a1_f = np.asfortranarray(g1.A)
    b1_f = np.asfortranarray(g1.B)
    c1_f = np.asfortranarray(g1.C)
    d1_f = np.asfortranarray(g1.D)
    a2_f = np.asfortranarray(g2.A)
    b2_f = np.asfortranarray(g2.B)
    c2_f = np.asfortranarray(g2.C)
    d2_f = np.asfortranarray(g2.D)

    a, b, c, d, n, info = ab05md(
        "U", "N", a1_f, b1_f, c1_f, d1_f, a2_f, b2_f, c2_f, d2_f
    )

    if info != 0:
        raise RuntimeError(f"ab05md failed with info={info}")

    input_delay = g1.input_delay
    output_delay = g2.output_delay
    if g1.output_delay is not None or g2.input_delay is not None:
        mid_delay = np.zeros(g1.n_outputs)
        if g1.output_delay is not None:
            mid_delay += g1.output_delay
        if g2.input_delay is not None:
            mid_delay += g2.input_delay
        if np.any(mid_delay > 0):
            if output_delay is None:
                output_delay = mid_delay
            else:
                output_delay = output_delay + mid_delay

    return StateSpace(a[:n, :n], b[:n, :], c[:, :n], d, input_delay, output_delay)


def parallel(g1: StateSpace, g2: StateSpace, alpha: float = 1.0) -> StateSpace:
    """
    Parallel connection using SLICOT ab05pd.

    Y = G1*U + alpha*G2*U - both systems share same input.
    Requires matching delays or will raise.
    """
    from .state_space import StateSpace

    if not _delays_match(g1, g2):
        raise ValueError(
            "Cannot add systems with mismatched delays. Use absorbDelay() first."
        )

    n1, m, p, n2 = g1.n_states, g1.n_inputs, g1.n_outputs, g2.n_states
    a1_f = np.asfortranarray(g1.A)
    b1_f = np.asfortranarray(g1.B)
    c1_f = np.asfortranarray(g1.C)
    d1_f = np.asfortranarray(g1.D)
    a2_f = np.asfortranarray(g2.A)
    b2_f = np.asfortranarray(g2.B)
    c2_f = np.asfortranarray(g2.C)
    d2_f = np.asfortranarray(g2.D)

    n, a, b, c, d, info = ab05pd(
        n1, m, p, n2, alpha, a1_f, b1_f, c1_f, d1_f, a2_f, b2_f, c2_f, d2_f
    )

    if info != 0:
        raise RuntimeError(f"ab05pd failed with info={info}")

    return StateSpace(a[:n, :n], b[:n, :], c[:, :n], d, g1.input_delay, g1.output_delay)


def feedback(g1: StateSpace, g2: StateSpace, sign: float = -1.0) -> StateSpace:
    """
    Feedback connection using SLICOT ab05nd.

    Closed-loop with G1 in forward path, G2 in feedback path.
    Returns T = G1 / (1 + G1*G2) for negative feedback (sign=-1).

    Loop delay = g1.output_delay + g2.input_delay + g2.output_delay + g1.input_delay
    Note: ab05nd uses opposite sign convention internally.
    """
    from .state_space import StateSpace

    alpha = -sign  # ab05nd convention is opposite
    a1_f = np.asfortranarray(g1.A)
    b1_f = np.asfortranarray(g1.B)
    c1_f = np.asfortranarray(g1.C)
    d1_f = np.asfortranarray(g1.D)
    a2_f = np.asfortranarray(g2.A)
    b2_f = np.asfortranarray(g2.B)
    c2_f = np.asfortranarray(g2.C)
    d2_f = np.asfortranarray(g2.D)

    a, b, c, d, n, info = ab05nd(
        "N", alpha, a1_f, b1_f, c1_f, d1_f, a2_f, b2_f, c2_f, d2_f
    )

    if info != 0:
        raise RuntimeError(f"ab05nd failed with info={info}")

    loop_delay = _sum_delays(
        g1.input_delay, g1.output_delay, g2.input_delay, g2.output_delay
    )
    input_delay = g1.input_delay
    output_delay = np.array([loop_delay]) if loop_delay > 0 else g1.output_delay

    return StateSpace(a[:n, :n], b[:n, :], c[:, :n], d, input_delay, output_delay)


def _delays_match(g1: StateSpace, g2: StateSpace) -> bool:
    """Check if two systems have matching delay structures for addition."""

    def _get_delay_sum(d):
        if d is None:
            return 0.0
        return np.sum(d)

    return _get_delay_sum(g1.input_delay) == _get_delay_sum(
        g2.input_delay
    ) and _get_delay_sum(g1.output_delay) == _get_delay_sum(g2.output_delay)


def _sum_delays(*delays) -> float:
    """Sum all delays, treating None as zero."""
    total = 0.0
    for d in delays:
        if d is not None:
            total += np.sum(d)
    return total
