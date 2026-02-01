"""
Discrete delay to z^(-k) poles example.

Based on MATLAB: Convert Time Delay in Discrete-Time Model to Factors of 1/z
https://www.mathworks.com/help/control/ug/convert-time-delay-in-discrete-time-model-to-factors-of-1z.html
"""

import numpy as np

from control_workflows.lti import ss, pid, absorbDelay
from control_workflows.lti.conversions import tf2ss
from control_workflows.lti.interconnect import series, feedback


def run_matlab_example(verbose: bool = True) -> dict:
    """Run discrete delay absorption example."""
    Ts = 0.01

    G = ss([[0.9]], [[0.125]], [[0.08]], [[0]], input_delay=[7], dt=Ts)

    C_tf = pid(6, 90, Ts=Ts)
    C = tf2ss(C_tf)
    C = ss(C.A, C.B, C.C, C.D, dt=Ts)

    L = series(C, G)
    H_unity = ss(
        np.zeros((0, 0)), np.zeros((0, 1)), np.zeros((1, 0)), np.array([[1.0]]), dt=Ts
    )
    T = feedback(L, H_unity)

    if verbose:
        print("=== Discrete Delay to z^(-k) Poles ===\n")
        print("Plant G: discrete SS with 7-sample input delay")
        print(f"  {G}")
        print(f"\nController C: discrete PI (Kp=6, Ki=90, Ts={Ts})")
        print(f"  {C}")
        print("\nClosed-loop T (with delay property):")
        print(f"  {T}")
        print(f"  Order: {T.n_states}")
        if T.input_delay is not None:
            print(f"  Input delay: {T.input_delay.tolist()}")
        if T.output_delay is not None:
            print(f"  Output delay: {T.output_delay.tolist()}")

    Tnd = absorbDelay(T)

    poles_at_zero = np.sum(np.abs(Tnd.poles()) < 1e-10)

    if verbose:
        print("\nAfter absorbDelay (delay -> z^(-k) poles):")
        print(f"  {Tnd}")
        print(f"  Order: {Tnd.n_states}")
        if Tnd.input_delay is not None and np.any(Tnd.input_delay > 0):
            print(f"  Input delay: {Tnd.input_delay.tolist()}")
        else:
            print("  Input delay: (none)")
        if Tnd.output_delay is not None and np.any(Tnd.output_delay > 0):
            print(f"  Output delay: {Tnd.output_delay.tolist()}")
        else:
            print("  Output delay: (none)")
        print(f"\nPoles at z=0: {poles_at_zero}")
        print(f"All poles: {np.round(Tnd.poles(), 6).tolist()}")

    return {
        "T": T,
        "Tnd": Tnd,
        "poles_at_zero": poles_at_zero,
        "order_before": T.n_states,
        "order_after": Tnd.n_states,
    }


if __name__ == "__main__":
    run_matlab_example()
