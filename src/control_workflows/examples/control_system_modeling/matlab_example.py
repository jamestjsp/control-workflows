"""Reproduce MATLAB 'Control System Modeling with Model Objects' example using SLICOT.

Demonstrates mixing different model types (zpk, pid, tf) and connecting them
using SLICOT block diagram algebra routines (ab05md, ab05nd, ab05pd).

Source: https://www.mathworks.com/help/control/ug/control-system-modeling-with-model-objects.html
"""

import numpy as np

from control_workflows.lti import tf, zpk, pid, tf2ss, zpk2ss
from .interconnect import series, feedback


def run_matlab_example(verbose: bool = True) -> dict:
    """
    Demonstrate control system modeling with mixed model types.

    Architecture:
        r -> [F] -> (+) -> [C] -> [G] -> y
                     ^              |
                     |----[S]<------|
    """
    results = {}

    if verbose:
        print("Control System Modeling with Model Objects")
        print("=" * 50)

    # === Create Components ===
    if verbose:
        print("\n1. CREATE COMPONENTS")
        print("-" * 30)

    # Plant: double pole at -1
    G = zpk([], [-1, -1], 1.0)
    results["plant"] = G
    if verbose:
        print("G (plant) = zpk([], [-1,-1], 1)")
        print(f"   Poles: {G.poles()}")
        print(f"   DC gain: {G.dc_gain():.2f}")

    # Controller: PID with derivative filter
    C = pid(kp=2.0, ki=1.3, kd=0.3, tf_=0.5)
    results["controller"] = C
    if verbose:
        print("\nC (PID) = pid(Kp=2, Ki=1.3, Kd=0.3, Tf=0.5)")
        print(f"   Order: {C.order}")
        print(f"   DC gain: {C.dc_gain():.2f}")

    # Sensor
    s = tf("s")
    S = 5 / (s + 4)
    results["sensor"] = S
    if verbose:
        print("\nS (sensor) = 5/(s+4)")
        print(f"   Pole: {S.poles()[0]:.1f}")
        print(f"   DC gain: {S.dc_gain():.2f}")

    # Prefilter
    F = 1 / (s + 1)
    results["prefilter"] = F
    if verbose:
        print("\nF (prefilter) = 1/(s+1)")
        print(f"   Pole: {F.poles()[0]:.1f}")
        print(f"   DC gain: {F.dc_gain():.2f}")

    # === Convert to State-Space ===
    if verbose:
        print("\n2. CONVERT TO STATE-SPACE")
        print("-" * 30)

    G_ss = zpk2ss(G)
    C_ss = tf2ss(C)
    S_ss = tf2ss(S)
    F_ss = tf2ss(F)

    results["G_ss"] = G_ss
    results["C_ss"] = C_ss
    results["S_ss"] = S_ss
    results["F_ss"] = F_ss

    if verbose:
        print(f"G: {G_ss.n_states} states")
        print(f"C: {C_ss.n_states} states")
        print(f"S: {S_ss.n_states} states")
        print(f"F: {F_ss.n_states} states")

    # === Build Composite Systems ===
    if verbose:
        print("\n3. BLOCK DIAGRAM ALGEBRA (SLICOT)")
        print("-" * 30)

    # Forward path: G*C (controller before plant)
    GC = series(C_ss, G_ss)
    results["forward_path"] = GC
    if verbose:
        print("GC = series(C, G)")
        print(f"   States: {GC.n_states}")
        print(f"   Stable: {GC.is_stable()}")

    # Closed-loop: feedback(G*C, S)
    T = feedback(GC, S_ss, sign=-1.0)
    results["closed_loop"] = T
    if verbose:
        print("\nT = feedback(GC, S)")
        print(f"   States: {T.n_states}")
        print(f"   Poles: {np.round(T.poles(), 3)}")
        print(f"   Stable: {T.is_stable()}")

    # Filtered closed-loop: T*F
    Try = series(F_ss, T)
    results["filtered_closed_loop"] = Try
    if verbose:
        print("\nTry = series(F, T)")
        print(f"   States: {Try.n_states}")
        print(f"   Poles: {np.round(Try.poles(), 3)}")
        print(f"   Stable: {Try.is_stable()}")

    # Open-loop: S*G*C
    SG = series(G_ss, S_ss)
    open_loop = series(C_ss, SG)
    results["open_loop"] = open_loop
    if verbose:
        print("\nOpen-loop = series(C, series(G, S))")
        print(f"   States: {open_loop.n_states}")

    # === Analysis ===
    if verbose:
        print("\n4. ANALYSIS")
        print("-" * 30)

    dc_gain = Try.dc_gain()[0, 0]
    results["dc_gain"] = dc_gain
    if verbose:
        print(f"Closed-loop DC gain: {dc_gain:.3f}")

    poles = T.poles()
    dominant = poles[np.argmax(np.real(poles))]
    results["dominant_pole"] = dominant
    if verbose:
        print(f"Dominant pole: {dominant:.3f}")
        wn = np.abs(dominant)
        zeta = -np.real(dominant) / wn
        print(f"   Natural freq: {wn:.2f} rad/s")
        print(f"   Damping ratio: {zeta:.2f}")

    if verbose:
        print("\n" + "=" * 50)
        print("Done!")

    return results


if __name__ == "__main__":
    run_matlab_example()
