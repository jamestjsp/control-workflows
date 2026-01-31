"""Block diagram interconnections using SLICOT."""

import numpy as np
from slicot import ab05md, ab05nd, ab05pd

from control_workflows.lti.state_space import StateSpace


def series(g1: StateSpace, g2: StateSpace) -> StateSpace:
    """
    Cascade (series) connection using SLICOT ab05md.

    Y = G2(G1(U)) - output of G1 feeds input of G2.
    """
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

    return StateSpace(a[:n, :n], b[:n, :], c[:, :n], d)


def parallel(g1: StateSpace, g2: StateSpace, alpha: float = 1.0) -> StateSpace:
    """
    Parallel connection using SLICOT ab05pd.

    Y = G1*U + alpha*G2*U - both systems share same input.
    """
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

    return StateSpace(a[:n, :n], b[:n, :], c[:, :n], d)


def feedback(g1: StateSpace, g2: StateSpace, sign: float = -1.0) -> StateSpace:
    """
    Feedback connection using SLICOT ab05nd.

    Closed-loop with G1 in forward path, G2 in feedback path.
    Returns T = G1 / (1 + G1*G2) for negative feedback (sign=-1).

    Note: ab05nd uses opposite sign convention internally.
    """
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

    return StateSpace(a[:n, :n], b[:n, :], c[:, :n], d)
