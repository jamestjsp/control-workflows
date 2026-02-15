"""Conversions between LTI representations using SLICOT."""

from __future__ import annotations
import numpy as np
from ctrlsys import tb04ad, td04ad

from .transfer_function import TransferFunction
from .zpk import ZPK
from .state_space import StateSpace


def ss2tf(sys: StateSpace) -> TransferFunction:
    """
    Convert state-space to transfer function using SLICOT tb04ad.

    Only for SISO systems. Preserves total delay.
    """
    if sys.n_inputs != 1 or sys.n_outputs != 1:
        raise ValueError("ss2tf only supports SISO systems")

    A_f = np.asfortranarray(sys.A)
    B_f = np.asfortranarray(sys.B)
    C_f = np.asfortranarray(sys.C)
    D_f = np.asfortranarray(sys.D)

    _, _, _, _, nr, index, den, num, _ = tb04ad("R", A_f, B_f, C_f, D_f)

    deg = index[0]
    num_poly = num[0, 0, : deg + 1]
    den_poly = den[0, : deg + 1]

    total_delay = 0.0
    if sys.input_delay is not None:
        total_delay += float(np.sum(sys.input_delay))
    if sys.output_delay is not None:
        total_delay += float(np.sum(sys.output_delay))

    return TransferFunction(num_poly, den_poly, total_delay)


def tf2ss(tf: TransferFunction) -> StateSpace:
    """
    Convert transfer function to state-space using SLICOT td04ad.

    Returns controllable canonical form. Preserves delay as input_delay.
    """
    n = tf.order
    input_delay = np.array([tf.input_delay]) if tf.input_delay > 0 else None

    if n == 0:
        return StateSpace(
            np.zeros((0, 0)),
            np.zeros((0, 1)),
            np.zeros((1, 0)),
            np.array([[tf.dc_gain()]]),
            input_delay,
            None,
        )

    num_padded = np.zeros(n + 1)
    num_padded[-(len(tf.num)) :] = tf.num

    index = np.array([n], dtype=np.int32)
    dcoeff = tf.den.reshape(1, -1)
    ucoeff = num_padded.reshape(1, 1, -1)

    dcoeff_f = np.asfortranarray(dcoeff)
    ucoeff_f = np.asfortranarray(ucoeff)

    nr, A, B, C, D, _ = td04ad("R", 1, 1, index, dcoeff_f, ucoeff_f, 0.0)

    A = A[:nr, :nr]
    B = B[:nr, :]
    C = C[:, :nr]

    return StateSpace(A, B, C, D, input_delay, None)


def zpk2tf(z: ZPK) -> TransferFunction:
    """Convert zero-pole-gain to transfer function. Preserves delay."""
    num = z.k * np.poly(z.z)
    den = np.poly(z.p)
    return TransferFunction(np.real(num), np.real(den), z.delay)


def tf2zpk(tf: TransferFunction) -> ZPK:
    """Convert transfer function to zero-pole-gain. Preserves delay."""
    return ZPK(tf.zeros(), tf.poles(), tf.num[0], tf.input_delay)


def zpk2ss(z: ZPK) -> StateSpace:
    """Convert zero-pole-gain to state-space."""
    return tf2ss(zpk2tf(z))


def ss2zpk(sys: StateSpace) -> ZPK:
    """Convert state-space to zero-pole-gain."""
    return tf2zpk(ss2tf(sys))
