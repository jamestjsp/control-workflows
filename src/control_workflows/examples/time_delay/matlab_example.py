"""Reproduce MATLAB 'Specifying Time Delays' example.

Source: https://www.mathworks.com/help/control/ug/specifying-time-delays.html

Demonstrates:
1. Creating transfer functions with input delays
2. Creating state-space systems with delays
3. Exact frequency response with delays (e^(-jωτ))
4. Pade approximation for time-domain simulation
5. Delay propagation in series/parallel/feedback
"""

import numpy as np

from control_workflows.lti import (
    tf,
    ss,
    zpk,
    bode,
    step_response,
    pade,
    absorbDelay,
    tf2ss,
)


def run_matlab_example(verbose: bool = True) -> dict:
    """Demonstrate time delay support."""
    results = {}

    if verbose:
        print("Specifying Time Delays")
        print("=" * 50)

    # === 1. Create Delayed Transfer Function ===
    if verbose:
        print("\n1. DELAYED TRANSFER FUNCTION")
        print("-" * 30)

    G = tf([5], [1, 1], input_delay=3.4)
    results["delayed_tf"] = G
    if verbose:
        print("G = tf([5], [1,1], input_delay=3.4)")
        print("   G(s) = 5/(s+1) * e^(-3.4s)")
        print(f"   Delay: {G.input_delay} seconds")
        print(f"   DC gain: {G.dc_gain():.2f}")

    # === 2. Create Delayed State-Space ===
    if verbose:
        print("\n2. DELAYED STATE-SPACE")
        print("-" * 30)

    sys_ss = ss([[-1]], [[1]], [[12]], [[0]], input_delay=[2.5])
    results["delayed_ss"] = sys_ss
    if verbose:
        print("sys = ss([[-1]], [[1]], [[12]], [[0]], input_delay=[2.5])")
        print(f"   States: {sys_ss.n_states}")
        print(f"   Input delay: {sys_ss.input_delay}")
        print(f"   DC gain: {sys_ss.dc_gain()[0, 0]:.2f}")

    # === 3. Delayed ZPK ===
    if verbose:
        print("\n3. DELAYED ZPK")
        print("-" * 30)

    G_zpk = zpk([], [-1], 5.0, delay=3.4)
    results["delayed_zpk"] = G_zpk
    if verbose:
        print("G_zpk = zpk([], [-1], 5.0, delay=3.4)")
        print("   G(s) = 5/(s+1) * e^(-3.4s)")
        print(f"   Delay: {G_zpk.delay} seconds")

    # === 4. Frequency Response with Delay ===
    if verbose:
        print("\n4. FREQUENCY RESPONSE (EXACT)")
        print("-" * 30)

    omega = np.array([0.1, 1.0, 10.0])
    w, mag, phase = bode(G, omega=omega, dB=True, deg=True)

    G_nodelay = tf([5], [1, 1])
    w2, mag2, phase2 = bode(G_nodelay, omega=omega, dB=True, deg=True)

    results["bode_omega"] = w
    results["bode_mag"] = mag
    results["bode_phase"] = phase
    results["bode_phase_nodelay"] = phase2

    if verbose:
        print("Comparing phase with/without delay:")
        print("  ω (rad/s) | Phase (delay) | Phase (no delay) | Δ Phase")
        print("  " + "-" * 55)
        for i in range(len(omega)):
            delta = phase[i] - phase2[i]
            print(
                f"  {omega[i]:8.2f} | {phase[i]:13.1f} | {phase2[i]:16.1f} | {delta:8.1f}°"
            )
        print("\n  Note: Phase difference = -ω * τ * 180/π")
        expected_delta = -omega * 3.4 * 180 / np.pi
        print(f"  Expected Δ: {expected_delta}")

    # === 5. Pade Approximation ===
    if verbose:
        print("\n5. PADE APPROXIMATION")
        print("-" * 30)

    delay_approx = pade(3.4, n=5)
    results["pade_approx"] = delay_approx
    if verbose:
        print(f"pade(3.4, n=5) -> order {delay_approx.order} TF")
        print(f"   Poles: {np.round(delay_approx.poles(), 3)}")

        test_freqs = [0.1, 0.5, 1.0]
        print("\n   Comparing e^(-jωτ) vs Pade:")
        for w_test in test_freqs:
            exact = np.exp(-1j * w_test * 3.4)
            approx = delay_approx(1j * w_test)
            err = np.abs(exact - approx) / np.abs(exact) * 100
            print(
                f"   ω={w_test}: exact={exact:.3f}, pade={approx:.3f}, err={err:.2f}%"
            )

    # === 6. Absorb Delay ===
    if verbose:
        print("\n6. ABSORBING DELAYS")
        print("-" * 30)

    G_absorbed = absorbDelay(G, n=5)
    results["absorbed_tf"] = G_absorbed
    if verbose:
        print("absorbDelay(G, n=5)")
        print(f"   Original delay: {G.input_delay}")
        print(f"   Absorbed delay: {G_absorbed.input_delay}")
        print(f"   Original order: {G.order}")
        print(f"   Absorbed order: {G_absorbed.order}")

    # === 7. Step Response ===
    if verbose:
        print("\n7. STEP RESPONSE")
        print("-" * 30)

    G_ss = tf2ss(G)
    t, y = step_response(G_ss, t_final=15.0, n_points=500)
    results["step_t"] = t
    results["step_y"] = y

    if verbose:
        delay_idx = np.searchsorted(t, 3.4)
        print("Step response of G (with 3.4s delay)")
        print(f"   Response at t=0: {y[0, 0]:.4f} (should be ~0)")
        print(f"   Response at t=3.4: {y[0, delay_idx]:.4f} (starts rising)")
        print(f"   Steady-state: {y[0, -1]:.2f}")

    # === 8. Delay Arithmetic ===
    if verbose:
        print("\n8. DELAY ARITHMETIC")
        print("-" * 30)

    G1 = tf([1], [1, 1], input_delay=1.0)
    G2 = tf([2], [1, 2], input_delay=0.5)

    G_series = G1 * G2
    results["series_tf"] = G_series
    if verbose:
        print("G1 (delay=1.0) * G2 (delay=0.5)")
        print(f"   Series delay: {G_series.input_delay} (sum)")

    G_feedback = G1.feedback(G2)
    results["feedback_tf"] = G_feedback
    if verbose:
        print("\nfeedback(G1, G2)")
        print(f"   Loop delay: {G_feedback.input_delay}")

    # === 9. Summary ===
    if verbose:
        print("\n" + "=" * 50)
        print("SUMMARY")
        print("-" * 30)
        print("✓ Delays stored explicitly in TF/SS/ZPK")
        print("✓ Exact frequency response via e^(-jωτ)")
        print("✓ Pade approximation for time simulation")
        print("✓ Delays propagate through interconnections")
        print("=" * 50)

    return results


if __name__ == "__main__":
    run_matlab_example()
