"""Frequency response analysis functions."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Union
import numpy as np
from numpy.typing import NDArray

from .state_space import StateSpace
from .transfer_function import TransferFunction
from .zpk import ZPK

LTISystem = Union[StateSpace, TransferFunction, ZPK]


@dataclass
class StabilityMargins:
    """Stability margins for feedback systems."""

    gm: float
    pm: float
    wcg: float
    wcp: float
    gm_dB: float
    stable: bool


def _auto_omega(sys: LTISystem, n_points: int = 1000) -> NDArray[np.float64]:
    """Auto-generate frequency range based on poles/zeros."""
    features = []
    poles = sys.poles()
    if len(poles) > 0:
        features.extend(np.abs(poles[poles != 0]))
    if hasattr(sys, "zeros"):
        zeros = sys.zeros()
        if len(zeros) > 0:
            features.extend(np.abs(zeros[zeros != 0]))

    if len(features) == 0:
        return np.logspace(-3, 3, n_points)

    min_freq = min(features) / 10
    max_freq = max(features) * 10
    return np.logspace(np.log10(min_freq), np.log10(max_freq), n_points)


def freqresp(
    sys: LTISystem, omega: NDArray | None = None, n_points: int = 1000
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    """
    Compute frequency response including delay effects.

    Returns:
        (omega, response) where response shape is:
        - SISO: (len(omega),)
        - MIMO StateSpace: (p, m, len(omega))

    For TransferFunction/ZPK, delay is included via e^(-jw*delay).
    For StateSpace, delay is handled in sys.freqresp().
    """
    if omega is None:
        omega = _auto_omega(sys, n_points)
    omega = np.atleast_1d(np.asarray(omega, dtype=np.float64))

    if isinstance(sys, StateSpace):
        return omega, sys.freqresp(omega)

    if isinstance(sys, ZPK):
        resp = np.array([sys(1j * w) for w in omega])
    else:
        resp = sys(1j * omega)
    return omega, np.atleast_1d(resp)


def bode(
    sys: LTISystem,
    omega: NDArray | None = None,
    dB: bool = True,
    deg: bool = True,
    input_idx: int = 0,
    output_idx: int = 0,
    n_points: int = 1000,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute Bode plot data.

    Returns:
        (omega, mag, phase)
    """
    omega, resp = freqresp(sys, omega, n_points)

    if isinstance(sys, StateSpace):
        resp = resp[output_idx, input_idx, :]

    mag = np.abs(resp)
    phase = np.unwrap(np.angle(resp))

    if dB:
        mag = 20 * np.log10(np.maximum(mag, 1e-15))
    if deg:
        phase = np.degrees(phase)

    return omega, mag, phase


def nyquist(
    sys: LTISystem,
    omega: NDArray | None = None,
    input_idx: int = 0,
    output_idx: int = 0,
    n_points: int = 1000,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute Nyquist plot data.

    Returns:
        (omega, real, imag)
    """
    omega, resp = freqresp(sys, omega, n_points)

    if isinstance(sys, StateSpace):
        resp = resp[output_idx, input_idx, :]

    return omega, np.real(resp), np.imag(resp)


def _find_crossings(x: NDArray, y: NDArray, level: float) -> NDArray[np.float64]:
    """Find x values where y crosses level via linear interpolation."""
    y_shifted = y - level
    sign_changes = np.where(np.diff(np.sign(y_shifted)))[0]

    crossings = []
    for i in sign_changes:
        if y_shifted[i + 1] == y_shifted[i]:
            continue
        x_cross = x[i] - y_shifted[i] * (x[i + 1] - x[i]) / (
            y_shifted[i + 1] - y_shifted[i]
        )
        crossings.append(x_cross)

    return np.array(crossings)


def margin(
    sys: LTISystem,
    input_idx: int = 0,
    output_idx: int = 0,
) -> StabilityMargins:
    """
    Compute gain and phase margins.

    Returns:
        StabilityMargins with gm, pm, wcg, wcp, gm_dB, stable
    """
    omega, resp = freqresp(sys, n_points=2000)

    if isinstance(sys, StateSpace):
        resp = resp[output_idx, input_idx, :]

    mag = np.abs(resp)
    phase = np.unwrap(np.angle(resp))
    phase_deg = np.degrees(phase)

    gain_crossings = _find_crossings(omega, mag, 1.0)
    wcg = gain_crossings[0] if len(gain_crossings) > 0 else np.inf

    phase_crossings = _find_crossings(omega, phase_deg, -180.0)
    wcp = phase_crossings[0] if len(phase_crossings) > 0 else np.inf

    if wcg < np.inf:
        phase_at_wcg = np.interp(wcg, omega, phase_deg)
        pm = 180.0 + phase_at_wcg
    else:
        pm = np.inf

    if wcp < np.inf:
        mag_at_wcp = np.interp(wcp, omega, mag)
        gm = 1.0 / mag_at_wcp if mag_at_wcp > 0 else np.inf
        gm_dB = 20 * np.log10(gm) if gm > 0 and gm < np.inf else np.inf
    else:
        gm = np.inf
        gm_dB = np.inf

    stable = (gm > 1.0 or gm == np.inf) and (pm > 0 or pm == np.inf)

    return StabilityMargins(
        gm=gm,
        pm=pm,
        wcg=wcg,
        wcp=wcp,
        gm_dB=gm_dB,
        stable=stable,
    )
