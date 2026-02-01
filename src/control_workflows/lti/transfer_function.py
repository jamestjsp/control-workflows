"""Transfer function representation."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np
from numpy.typing import NDArray


@dataclass
class TransferFunction:
    """
    Continuous-time transfer function G(s) = num(s) / den(s) * e^(-s*delay).

    Polynomials stored highest-power-first (numpy convention).
    """

    num: NDArray[np.float64]
    den: NDArray[np.float64]
    input_delay: float = 0.0

    def __post_init__(self) -> None:
        self.num = np.atleast_1d(np.asarray(self.num, dtype=np.float64))
        self.den = np.atleast_1d(np.asarray(self.den, dtype=np.float64))
        self.num = np.trim_zeros(self.num, "f")
        self.den = np.trim_zeros(self.den, "f")
        if len(self.num) == 0:
            self.num = np.array([0.0])
        if len(self.den) == 0:
            raise ValueError("Denominator cannot be zero")
        self.num = self.num / self.den[0]
        self.den = self.den / self.den[0]

    @property
    def order(self) -> int:
        return len(self.den) - 1

    def poles(self) -> NDArray[np.complex128]:
        return np.roots(self.den)

    def zeros(self) -> NDArray[np.complex128]:
        return np.roots(self.num)

    def dc_gain(self) -> float:
        if self.den[-1] == 0:
            return np.inf if self.num[-1] != 0 else np.nan
        return float(self.num[-1] / self.den[-1])

    def is_stable(self) -> bool:
        return bool(np.all(np.real(self.poles()) < 0))

    def __call__(self, s: complex | NDArray) -> complex | NDArray:
        resp = np.polyval(self.num, s) / np.polyval(self.den, s)
        if self.input_delay != 0:
            resp = resp * np.exp(-s * self.input_delay)
        return resp

    def __neg__(self) -> TransferFunction:
        return TransferFunction(-self.num, self.den, self.input_delay)

    def __add__(self, other: TransferFunction | float) -> TransferFunction:
        if isinstance(other, (int, float)):
            other = TransferFunction([other], [1.0])
        if not isinstance(other, TransferFunction):
            return NotImplemented
        if self.input_delay != other.input_delay:
            raise ValueError(
                f"Cannot add TFs with different delays ({self.input_delay} vs {other.input_delay}). "
                "Use absorbDelay() first."
            )
        num = np.polyadd(
            np.polymul(self.num, other.den), np.polymul(other.num, self.den)
        )
        den = np.polymul(self.den, other.den)
        return TransferFunction(num, den, self.input_delay)

    def __radd__(self, other: float) -> TransferFunction:
        return self.__add__(other)

    def __sub__(self, other: TransferFunction | float) -> TransferFunction:
        return self + (-other if isinstance(other, TransferFunction) else -other)

    def __rsub__(self, other: float) -> TransferFunction:
        return (-self).__add__(other)

    def __mul__(self, other: TransferFunction | float) -> TransferFunction:
        if isinstance(other, (int, float)):
            return TransferFunction(self.num * other, self.den, self.input_delay)
        if not isinstance(other, TransferFunction):
            return NotImplemented
        return TransferFunction(
            np.polymul(self.num, other.num),
            np.polymul(self.den, other.den),
            self.input_delay + other.input_delay,
        )

    def __rmul__(self, other: float) -> TransferFunction:
        return self.__mul__(other)

    def __truediv__(self, other: TransferFunction | float) -> TransferFunction:
        if isinstance(other, (int, float)):
            return TransferFunction(self.num / other, self.den, self.input_delay)
        if not isinstance(other, TransferFunction):
            return NotImplemented
        return TransferFunction(
            np.polymul(self.num, other.den),
            np.polymul(self.den, other.num),
            self.input_delay - other.input_delay,
        )

    def __rtruediv__(self, other: float) -> TransferFunction:
        return TransferFunction([other] * len(self.den), [1.0]) / self

    def __pow__(self, n: int) -> TransferFunction:
        if n == 0:
            return TransferFunction([1.0], [1.0], 0.0)
        if n < 0:
            return TransferFunction(self.den, self.num, -self.input_delay) ** (-n)
        result = self
        for _ in range(n - 1):
            result = result * self
        return result

    def feedback(
        self, H: TransferFunction | float = 1.0, sign: int = -1
    ) -> TransferFunction:
        """Closed-loop: G / (1 - sign*G*H). Loop delay accumulates."""
        if isinstance(H, (int, float)):
            H = TransferFunction([H], [1.0])
        loop_delay = self.input_delay + H.input_delay
        if sign == -1:
            return TransferFunction(
                np.polymul(self.num, H.den),
                np.polyadd(np.polymul(self.den, H.den), np.polymul(self.num, H.num)),
                loop_delay,
            )
        else:
            return TransferFunction(
                np.polymul(self.num, H.den),
                np.polysub(np.polymul(self.den, H.den), np.polymul(self.num, H.num)),
                loop_delay,
            )

    def __repr__(self) -> str:
        delay_str = f", delay={self.input_delay}" if self.input_delay != 0 else ""
        return f"TransferFunction(num={self.num.tolist()}, den={self.den.tolist()}{delay_str})"


class LaplaceDomain:
    """Laplace variable for algebraic TF construction."""

    def __init__(self) -> None:
        pass

    def __add__(self, other: float) -> TransferFunction:
        return TransferFunction([1.0, other], [1.0])

    def __radd__(self, other: float) -> TransferFunction:
        return self.__add__(other)

    def __sub__(self, other: float) -> TransferFunction:
        return TransferFunction([1.0, -other], [1.0])

    def __rsub__(self, other: float) -> TransferFunction:
        return TransferFunction([-1.0, other], [1.0])

    def __mul__(self, other: float) -> TransferFunction:
        return TransferFunction([other, 0.0], [1.0])

    def __rmul__(self, other: float) -> TransferFunction:
        return self.__mul__(other)

    def __truediv__(self, other: float) -> TransferFunction:
        return TransferFunction([1.0 / other, 0.0], [1.0])

    def __pow__(self, n: int) -> TransferFunction:
        if n >= 0:
            return TransferFunction([1.0] + [0.0] * n, [1.0])
        return TransferFunction([1.0], [1.0] + [0.0] * (-n))

    def __repr__(self) -> str:
        return "s"


s = LaplaceDomain()


def tf(
    num: NDArray | Sequence | str,
    den: NDArray | Sequence | None = None,
    input_delay: float = 0.0,
) -> TransferFunction | LaplaceDomain:
    """
    Create transfer function.

    Usage:
        tf([1, 2], [1, 3, 2])  # (s+2)/(s^2+3s+2)
        tf([5], [1, 1], input_delay=3.4)  # 5/(s+1) * e^(-3.4s)
        tf('s')                 # returns Laplace variable for algebraic use
    """
    if isinstance(num, str) and num.lower() == "s":
        return s
    if den is None:
        den = [1.0]
    return TransferFunction(np.asarray(num), np.asarray(den), input_delay)


def pid(
    kp: float, ki: float = 0.0, kd: float = 0.0, tf_: float = 0.0
) -> TransferFunction:
    """
    Create PID controller.

    C(s) = Kp + Ki/s + Kd*s/(Tf*s + 1)

    Args:
        kp: Proportional gain
        ki: Integral gain
        kd: Derivative gain
        tf_: Derivative filter time constant (0 = ideal derivative)
    """
    if tf_ == 0:
        num = [kd, kp, ki]
        den = [1.0, 0.0]
    else:
        num = [kp * tf_ + kd, kp + ki * tf_, ki]
        den = [tf_, 1.0, 0.0]
    return TransferFunction(np.asarray(num), np.asarray(den))
