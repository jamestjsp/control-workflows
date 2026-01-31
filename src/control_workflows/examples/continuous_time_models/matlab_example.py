"""Reproduce MATLAB 'Creating Continuous-Time Models' example using SLICOT."""

import numpy as np

from control_workflows.lti import (
    tf,
    zpk,
    ss,
    tf2ss,
    ss2tf,
    zpk2tf,
    tf2zpk,
)


def run_matlab_example(verbose: bool = True) -> dict:
    """
    Demonstrate continuous-time model creation following MATLAB workflow.

    Covers:
    - Transfer functions (polynomial and algebraic forms)
    - Zero-pole-gain models
    - State-space models
    - Conversions between representations
    - Analysis (poles, zeros, DC gain, stability)
    """
    results = {}

    if verbose:
        print("Creating Continuous-Time Models")
        print("=" * 50)

    # === Transfer Functions ===
    if verbose:
        print("\n1. TRANSFER FUNCTIONS")
        print("-" * 30)

    # Method 1: Polynomial coefficients
    # G(s) = (s + 2) / (s^2 + 3s + 2)
    G1 = tf([1, 2], [1, 3, 2])
    results["tf_poly"] = G1

    if verbose:
        print("G1 = tf([1,2], [1,3,2])")
        print(f"   Poles: {G1.poles()}")
        print(f"   Zeros: {G1.zeros()}")
        print(f"   DC gain: {G1.dc_gain():.2f}")
        print(f"   Stable: {G1.is_stable()}")

    # Method 2: Algebraic form using Laplace variable
    s = tf("s")
    G2 = (s + 2) / (s**2 + 3 * s + 2)
    results["tf_algebraic"] = G2

    if verbose:
        print("\nG2 = (s+2)/(s^2+3s+2)")
        print(f"   Poles: {G2.poles()}")
        print(f"   DC gain: {G2.dc_gain():.2f}")

    # Second-order system
    wn = 5.0
    zeta = 0.7
    G_2nd = wn**2 / (s**2 + 2 * zeta * wn * s + wn**2)
    results["tf_2nd_order"] = G_2nd

    if verbose:
        print(f"\nSecond-order: wn={wn}, zeta={zeta}")
        print(f"   Poles: {G_2nd.poles()}")
        print(f"   Stable: {G_2nd.is_stable()}")

    # === Zero-Pole-Gain ===
    if verbose:
        print("\n2. ZERO-POLE-GAIN MODELS")
        print("-" * 30)

    # H(s) = 2(s+1) / ((s+2)(s+3))
    H = zpk([-1], [-2, -3], 2.0)
    results["zpk"] = H

    if verbose:
        print("H = zpk([-1], [-2,-3], 2)")
        print(f"   Zeros: {H.zeros()}")
        print(f"   Poles: {H.poles()}")
        print(f"   Gain: {H.k}")
        print(f"   DC gain: {H.dc_gain():.2f}")

    # Complex poles
    H_complex = zpk([], [-1 + 2j, -1 - 2j], 5.0)
    results["zpk_complex"] = H_complex

    if verbose:
        print("\nComplex poles: zpk([], [-1+2j, -1-2j], 5)")
        print(f"   Poles: {H_complex.poles()}")
        print(f"   Stable: {H_complex.is_stable()}")

    # === State-Space ===
    if verbose:
        print("\n3. STATE-SPACE MODELS")
        print("-" * 30)

    # Create state-space model
    A = np.array([[0, 1], [-2, -3]])
    B = np.array([[0], [1]])
    C = np.array([[1, 0]])
    D = np.array([[0]])

    sys = ss(A, B, C, D)
    results["ss"] = sys

    if verbose:
        print("sys = ss(A, B, C, D)")
        print(f"   States: {sys.n_states}")
        print(f"   Inputs: {sys.n_inputs}")
        print(f"   Outputs: {sys.n_outputs}")
        print(f"   Poles: {sys.poles()}")
        print(f"   DC gain: {sys.dc_gain()}")
        print(f"   Stable: {sys.is_stable()}")

    # === Conversions ===
    if verbose:
        print("\n4. MODEL CONVERSIONS")
        print("-" * 30)

    # State-space to transfer function
    sys_tf = ss2tf(sys)
    results["ss_to_tf"] = sys_tf

    if verbose:
        print("ss2tf(sys)")
        print(f"   num: {sys_tf.num}")
        print(f"   den: {sys_tf.den}")

    # Transfer function to state-space
    G1_ss = tf2ss(G1)
    results["tf_to_ss"] = G1_ss

    if verbose:
        print("\ntf2ss(G1)")
        print(f"   A:\n{G1_ss.A}")
        print(f"   B:\n{G1_ss.B}")
        print(f"   C: {G1_ss.C}")

    # ZPK to transfer function
    H_tf = zpk2tf(H)
    results["zpk_to_tf"] = H_tf

    if verbose:
        print("\nzpk2tf(H)")
        print(f"   num: {H_tf.num}")
        print(f"   den: {H_tf.den}")

    # Transfer function to ZPK
    G1_zpk = tf2zpk(G1)
    results["tf_to_zpk"] = G1_zpk

    if verbose:
        print("\ntf2zpk(G1)")
        print(f"   zeros: {G1_zpk.zeros()}")
        print(f"   poles: {G1_zpk.poles()}")

    # === Algebra ===
    if verbose:
        print("\n5. SYSTEM ALGEBRA")
        print("-" * 30)

    # Parallel connection
    G_par = G1 + G2
    results["parallel"] = G_par

    if verbose:
        print("Parallel: G1 + G2")
        print(f"   Order: {G_par.order}")

    # Series connection
    G_ser = G1 * G2
    results["series"] = G_ser

    if verbose:
        print("\nSeries: G1 * G2")
        print(f"   Order: {G_ser.order}")

    # Feedback
    G_fb = G1.feedback()
    results["feedback"] = G_fb

    if verbose:
        print("\nFeedback: G1 / (1 + G1)")
        print(f"   DC gain: {G_fb.dc_gain():.2f}")
        print(f"   Stable: {G_fb.is_stable()}")

    # === Frequency evaluation ===
    if verbose:
        print("\n6. FREQUENCY RESPONSE")
        print("-" * 30)

    omega = np.logspace(-1, 2, 5)
    s_vals = 1j * omega
    resp = np.array([G1(s_val) for s_val in s_vals])
    results["freq_response"] = {"omega": omega, "response": resp}

    if verbose:
        print("G1(jw) at selected frequencies:")
        for w, r in zip(omega, resp):
            print(
                f"   w={w:6.2f}: |G|={np.abs(r):.3f}, phase={np.angle(r) * 180 / np.pi:.1f} deg"
            )

    if verbose:
        print("\n" + "=" * 50)
        print("Done!")

    return results


if __name__ == "__main__":
    run_matlab_example()
