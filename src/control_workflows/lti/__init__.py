"""Linear Time-Invariant system representations."""

from .transfer_function import TransferFunction, tf, pid
from .zpk import ZPK, zpk
from .state_space import StateSpace, ss
from .conversions import tf2ss, ss2tf, zpk2ss, ss2zpk, zpk2tf, tf2zpk
from .interconnect import series, parallel, feedback
from .analysis import (
    controllability_matrix,
    observability_matrix,
    is_controllable,
    is_observable,
    ctrb_gramian,
    obsv_gramian,
    hankel_singular_values,
)
from .reduction import balreal, reduce, minreal
from .response import step_response, impulse_response, initial_response
from .design import place, acker

__all__ = [
    "TransferFunction",
    "tf",
    "pid",
    "ZPK",
    "zpk",
    "StateSpace",
    "ss",
    "tf2ss",
    "ss2tf",
    "zpk2ss",
    "ss2zpk",
    "zpk2tf",
    "tf2zpk",
    "series",
    "parallel",
    "feedback",
    "controllability_matrix",
    "observability_matrix",
    "is_controllable",
    "is_observable",
    "ctrb_gramian",
    "obsv_gramian",
    "hankel_singular_values",
    "balreal",
    "reduce",
    "minreal",
    "step_response",
    "impulse_response",
    "initial_response",
    "place",
    "acker",
]
