"""Zero-Pole-Gain representation."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np
from numpy.typing import NDArray


@dataclass
class ZPK:
    """
    Continuous-time zero-pole-gain model with optional delay.

    G(s) = k * prod(s - z_i) / prod(s - p_i) * e^(-s*delay)
    """

    z: NDArray[np.complex128]
    p: NDArray[np.complex128]
    k: float
    delay: float = 0.0

    def __post_init__(self) -> None:
        self.z = np.atleast_1d(np.asarray(self.z, dtype=np.complex128))
        self.p = np.atleast_1d(np.asarray(self.p, dtype=np.complex128))
        if len(self.z) == 1 and np.isnan(self.z[0]):
            self.z = np.array([], dtype=np.complex128)
        if len(self.p) == 1 and np.isnan(self.p[0]):
            self.p = np.array([], dtype=np.complex128)

    @property
    def order(self) -> int:
        return len(self.p)

    def poles(self) -> NDArray[np.complex128]:
        return self.p.copy()

    def zeros(self) -> NDArray[np.complex128]:
        return self.z.copy()

    def dc_gain(self) -> float:
        if np.any(self.p == 0):
            return np.inf if len(self.z) == 0 or not np.any(self.z == 0) else np.nan
        gain = self.k * np.prod(-self.z) / np.prod(-self.p)
        return float(np.real(gain))

    def is_stable(self) -> bool:
        return bool(np.all(np.real(self.p) < 0))

    def __call__(self, s: complex | NDArray) -> complex | NDArray:
        num = self.k * np.prod(s - self.z)
        den = np.prod(s - self.p)
        resp = num / den
        if self.delay != 0:
            resp = resp * np.exp(-s * self.delay)
        return resp

    def __neg__(self) -> ZPK:
        return ZPK(self.z, self.p, -self.k, self.delay)

    def __mul__(self, other: ZPK | float) -> ZPK:
        if isinstance(other, (int, float)):
            return ZPK(self.z, self.p, self.k * other, self.delay)
        if not isinstance(other, ZPK):
            return NotImplemented
        return ZPK(
            np.concatenate([self.z, other.z]),
            np.concatenate([self.p, other.p]),
            self.k * other.k,
            self.delay + other.delay,
        )

    def __rmul__(self, other: float) -> ZPK:
        return self.__mul__(other)

    def __truediv__(self, other: ZPK | float) -> ZPK:
        if isinstance(other, (int, float)):
            return ZPK(self.z, self.p, self.k / other, self.delay)
        if not isinstance(other, ZPK):
            return NotImplemented
        return ZPK(
            np.concatenate([self.z, other.p]),
            np.concatenate([self.p, other.z]),
            self.k / other.k,
            self.delay - other.delay,
        )

    def __repr__(self) -> str:
        delay_str = f", delay={self.delay}" if self.delay != 0 else ""
        return f"ZPK(z={self.z.tolist()}, p={self.p.tolist()}, k={self.k}{delay_str})"


def zpk(
    z: NDArray | Sequence | None,
    p: NDArray | Sequence,
    k: float,
    delay: float = 0.0,
) -> ZPK:
    """Create zero-pole-gain model with optional delay."""
    if z is None:
        z = []
    return ZPK(np.asarray(z), np.asarray(p), k, delay)
