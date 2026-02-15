"""Model reduction functions."""

from __future__ import annotations
import numpy as np
from ctrlsys import ab01nd, ab09ad

from .state_space import StateSpace


def balreal(sys: StateSpace) -> StateSpace:
    """Balanced realization via ab09ad with full order."""
    if sys.n_states == 0:
        return sys

    n = sys.n_states
    m = sys.n_inputs
    p = sys.n_outputs

    A_f = np.asfortranarray(sys.A)
    B_f = np.asfortranarray(sys.B)
    C_f = np.asfortranarray(sys.C)

    ar, br, cr, hsv, nr_out, iwarn, info = ab09ad(
        "C", "B", "N", "F", n, m, p, n, A_f, B_f, C_f, 0.0
    )

    return StateSpace(ar[:nr_out, :nr_out], br[:nr_out, :], cr[:, :nr_out], sys.D)


def reduce(sys: StateSpace, order: int | None = None, tol: float = 1e-6) -> StateSpace:
    """Balanced truncation via ab09ad."""
    if sys.n_states == 0:
        return sys

    n = sys.n_states
    m = sys.n_inputs
    p = sys.n_outputs

    A_f = np.asfortranarray(sys.A)
    B_f = np.asfortranarray(sys.B)
    C_f = np.asfortranarray(sys.C)

    if order is None:
        ordsel = "A"
        nr_in = 0
    else:
        ordsel = "F"
        nr_in = order

    ar, br, cr, hsv, nr_out, iwarn, info = ab09ad(
        "C", "B", "N", ordsel, n, m, p, nr_in, A_f, B_f, C_f, tol
    )

    return StateSpace(ar[:nr_out, :nr_out], br[:nr_out, :], cr[:, :nr_out], sys.D)


def minreal(sys: StateSpace, tol: float = 0.0) -> StateSpace:
    """Minimal realization via ab01nd + duality."""
    if sys.n_states == 0:
        return sys

    A_f = np.asfortranarray(sys.A)
    B_f = np.asfortranarray(sys.B)

    a1, b1, ncont, _, _, z1, *_ = ab01nd("I", A_f, B_f, tol)

    if ncont == 0:
        return StateSpace(
            np.zeros((0, 0)),
            np.zeros((0, sys.n_inputs)),
            np.zeros((sys.n_outputs, 0)),
            sys.D,
        )

    Ac = a1[:ncont, :ncont]
    Bc = b1[:ncont, :]
    Cc = (sys.C @ z1)[:, :ncont]

    Ac_f = np.asfortranarray(Ac.T)
    Cc_f = np.asfortranarray(Cc.T)

    a2, c2, nobs, _, _, z2, *_ = ab01nd("I", Ac_f, Cc_f, tol)

    if nobs == 0:
        return StateSpace(
            np.zeros((0, 0)),
            np.zeros((0, sys.n_inputs)),
            np.zeros((sys.n_outputs, 0)),
            sys.D,
        )

    Aco = a2[:nobs, :nobs].T
    Bco = (Bc.T @ z2)[:, :nobs].T
    Cco = c2[:nobs, :].T

    return StateSpace(Aco, Bco, Cco, sys.D)
