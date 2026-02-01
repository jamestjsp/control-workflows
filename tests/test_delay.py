"""Tests for time delay functionality."""

import numpy as np
import pytest
from control_workflows.lti import (
    tf,
    ss,
    zpk,
    pade,
    absorbDelay,
    bode,
    freqresp,
    tf2ss,
    ss2tf,
    tf2zpk,
    zpk2tf,
    step_response,
)


class TestPade:
    def test_zero_delay(self):
        p = pade(0.0, n=3)
        assert p.order == 0
        assert p.dc_gain() == pytest.approx(1.0)

    def test_order_5(self):
        p = pade(1.0, n=5)
        assert p.order == 5
        assert p.dc_gain() == pytest.approx(1.0)

    def test_all_pass(self):
        p = pade(1.0, n=5)
        freqs = [0.1, 0.5, 1.0]
        for w in freqs:
            assert np.abs(p(1j * w)) == pytest.approx(1.0, rel=0.01)

    def test_phase_match(self):
        tau = 2.0
        p = pade(tau, n=5)
        w = 0.5
        exact_phase = -w * tau
        approx_phase = np.angle(p(1j * w))
        assert approx_phase == pytest.approx(exact_phase, abs=0.05)

    def test_negative_delay_raises(self):
        with pytest.raises(ValueError):
            pade(-1.0)


class TestTransferFunctionDelay:
    def test_create_with_delay(self):
        G = tf([1], [1, 1], input_delay=2.5)
        assert G.input_delay == 2.5
        assert G.order == 1

    def test_default_no_delay(self):
        G = tf([1], [1, 1])
        assert G.input_delay == 0.0

    def test_call_includes_delay(self):
        G = tf([1], [1, 1], input_delay=1.0)
        G_nodelay = tf([1], [1, 1])
        s = 1j
        ratio = G(s) / G_nodelay(s)
        expected = np.exp(-s * 1.0)
        assert ratio == pytest.approx(expected, rel=1e-10)

    def test_mul_accumulates_delay(self):
        G1 = tf([1], [1, 1], input_delay=1.0)
        G2 = tf([1], [1, 2], input_delay=0.5)
        G = G1 * G2
        assert G.input_delay == pytest.approx(1.5)

    def test_div_subtracts_delay(self):
        G1 = tf([1], [1, 1], input_delay=2.0)
        G2 = tf([1], [1, 2], input_delay=0.5)
        G = G1 / G2
        assert G.input_delay == pytest.approx(1.5)

    def test_add_same_delay_ok(self):
        G1 = tf([1], [1, 1], input_delay=1.0)
        G2 = tf([2], [1, 2], input_delay=1.0)
        G = G1 + G2
        assert G.input_delay == 1.0

    def test_add_different_delay_raises(self):
        G1 = tf([1], [1, 1], input_delay=1.0)
        G2 = tf([1], [1, 2], input_delay=0.5)
        with pytest.raises(ValueError, match="different delays"):
            G1 + G2

    def test_neg_preserves_delay(self):
        G = tf([1], [1, 1], input_delay=2.0)
        assert (-G).input_delay == 2.0

    def test_feedback_accumulates_delay(self):
        G = tf([1], [1, 1], input_delay=1.0)
        H = tf([1], [1, 2], input_delay=0.5)
        T = G.feedback(H)
        assert T.input_delay == pytest.approx(1.5)


class TestStateSpaceDelay:
    def test_create_with_input_delay(self):
        sys = ss([[-1]], [[1]], [[1]], [[0]], input_delay=[2.0])
        assert sys.input_delay is not None
        assert sys.input_delay[0] == 2.0

    def test_create_with_output_delay(self):
        sys = ss([[-1]], [[1]], [[1]], [[0]], output_delay=[1.5])
        assert sys.output_delay is not None
        assert sys.output_delay[0] == 1.5

    def test_default_no_delay(self):
        sys = ss([[-1]], [[1]], [[1]], [[0]])
        assert sys.input_delay is None
        assert sys.output_delay is None

    def test_freqresp_includes_delay(self):
        sys = ss([[-1]], [[1]], [[1]], [[0]], input_delay=[1.0])
        sys_nodelay = ss([[-1]], [[1]], [[1]], [[0]])
        omega = np.array([1.0])
        resp = sys.freqresp(omega)
        resp_nodelay = sys_nodelay.freqresp(omega)
        ratio = resp[0, 0, 0] / resp_nodelay[0, 0, 0]
        expected = np.exp(-1j * 1.0 * 1.0)
        assert ratio == pytest.approx(expected, rel=1e-10)

    def test_neg_preserves_delay(self):
        sys = ss([[-1]], [[1]], [[1]], [[0]], input_delay=[2.0])
        neg_sys = -sys
        assert neg_sys.input_delay is not None
        assert neg_sys.input_delay[0] == 2.0


class TestZPKDelay:
    def test_create_with_delay(self):
        G = zpk([], [-1], 1.0, delay=2.5)
        assert G.delay == 2.5

    def test_default_no_delay(self):
        G = zpk([], [-1], 1.0)
        assert G.delay == 0.0

    def test_call_includes_delay(self):
        G = zpk([], [-1], 1.0, delay=1.0)
        G_nodelay = zpk([], [-1], 1.0)
        s = 1j
        ratio = G(s) / G_nodelay(s)
        expected = np.exp(-s * 1.0)
        assert ratio == pytest.approx(expected, rel=1e-10)

    def test_mul_accumulates_delay(self):
        G1 = zpk([], [-1], 1.0, delay=1.0)
        G2 = zpk([], [-2], 2.0, delay=0.5)
        G = G1 * G2
        assert G.delay == pytest.approx(1.5)


class TestAbsorbDelay:
    def test_absorb_tf_delay(self):
        G = tf([1], [1, 1], input_delay=1.0)
        G_absorbed = absorbDelay(G, n=3)
        assert G_absorbed.input_delay == 0.0
        assert G_absorbed.order > G.order

    def test_absorb_zero_delay_unchanged(self):
        G = tf([1], [1, 1])
        G_absorbed = absorbDelay(G)
        assert G_absorbed.order == G.order

    def test_absorb_zpk_delay(self):
        G = zpk([], [-1], 1.0, delay=1.0)
        G_absorbed = absorbDelay(G, n=3)
        assert G_absorbed.delay == 0.0
        assert len(G_absorbed.p) > len(G.p)

    def test_absorb_ss_input_delay(self):
        sys = ss([[-1]], [[1]], [[1]], [[0]], input_delay=[1.0])
        sys_absorbed = absorbDelay(sys, n=3)
        assert sys_absorbed.input_delay is None
        assert sys_absorbed.n_states > sys.n_states


class TestConversionsPreserveDelay:
    def test_tf2ss_preserves_delay(self):
        G = tf([1], [1, 1], input_delay=2.0)
        sys = tf2ss(G)
        assert sys.input_delay is not None
        assert sys.input_delay[0] == pytest.approx(2.0)

    def test_ss2tf_preserves_delay(self):
        sys = ss([[-1]], [[1]], [[1]], [[0]], input_delay=[2.0])
        G = ss2tf(sys)
        assert G.input_delay == pytest.approx(2.0)

    def test_tf2zpk_preserves_delay(self):
        G = tf([1], [1, 1], input_delay=2.0)
        Z = tf2zpk(G)
        assert Z.delay == pytest.approx(2.0)

    def test_zpk2tf_preserves_delay(self):
        Z = zpk([], [-1], 1.0, delay=2.0)
        G = zpk2tf(Z)
        assert G.input_delay == pytest.approx(2.0)


class TestFrequencyResponseWithDelay:
    def test_bode_phase_shift(self):
        G = tf([1], [1, 1], input_delay=1.0)
        G_nodelay = tf([1], [1, 1])
        omega = np.array([1.0])

        _, _, phase = bode(G, omega=omega, deg=True)
        _, _, phase_nodelay = bode(G_nodelay, omega=omega, deg=True)

        expected_shift = -1.0 * 1.0 * 180 / np.pi
        actual_shift = phase[0] - phase_nodelay[0]
        assert actual_shift == pytest.approx(expected_shift, abs=0.5)

    def test_freqresp_magnitude_unchanged(self):
        G = tf([1], [1, 1], input_delay=1.0)
        G_nodelay = tf([1], [1, 1])
        omega = np.array([0.1, 1.0, 10.0])

        _, resp = freqresp(G, omega)
        _, resp_nodelay = freqresp(G_nodelay, omega)

        np.testing.assert_allclose(np.abs(resp), np.abs(resp_nodelay), rtol=1e-10)


class TestStepResponseWithDelay:
    def test_step_response_delayed(self):
        sys = ss([[-1]], [[1]], [[1]], [[0]], input_delay=[2.0])
        t, y = step_response(sys, t_final=10.0, n_points=200, pade_order=10)
        assert y[0, -1] == pytest.approx(1.0, rel=0.1)

    def test_step_response_delayed_increased_order(self):
        sys = ss([[-1]], [[1]], [[1]], [[0]], input_delay=[2.0])
        sys_absorbed = absorbDelay(sys, n=10)
        assert sys_absorbed.n_states > sys.n_states
