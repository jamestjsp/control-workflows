"""Tests for frequency response analysis module."""

import numpy as np
import pytest
from control_workflows.lti import (
    tf,
    zpk,
    ss,
    bode,
    nyquist,
    margin,
    freqresp,
    StabilityMargins,
)


class TestFreqresp:
    def test_tf_first_order(self):
        sys = tf([1], [1, 1])
        w, resp = freqresp(sys)
        assert len(w) == 1000
        assert len(resp) == 1000
        assert np.abs(resp[0]) == pytest.approx(1.0, rel=0.01)

    def test_tf_custom_omega(self):
        sys = tf([1], [1, 1])
        omega = np.array([0.1, 1.0, 10.0])
        w, resp = freqresp(sys, omega)
        np.testing.assert_array_equal(w, omega)
        assert len(resp) == 3

    def test_zpk(self):
        sys = zpk([-2], [-1, -3], 2)
        w, resp = freqresp(sys)
        assert len(resp) == 1000

    def test_ss(self):
        A = np.array([[-1, 0], [0, -2]])
        B = np.array([[1], [1]])
        C = np.array([[1, 1]])
        D = np.array([[0]])
        sys = ss(A, B, C, D)
        w, resp = freqresp(sys)
        assert resp.shape == (1, 1, 1000)


class TestBode:
    def test_first_order_dc_gain(self):
        sys = tf([1], [1, 1])
        w, mag, phase = bode(sys)
        assert mag[0] == pytest.approx(0.0, abs=0.1)

    def test_first_order_high_freq_phase(self):
        sys = tf([1], [1, 1])
        w, mag, phase = bode(sys)
        assert phase[-1] == pytest.approx(-90.0, abs=10.0)

    def test_dB_false(self):
        sys = tf([2], [1, 1])
        w, mag, _ = bode(sys, dB=False)
        assert mag[0] == pytest.approx(2.0, rel=0.01)

    def test_deg_false(self):
        sys = tf([1], [1, 1])
        w, _, phase = bode(sys, deg=False)
        assert phase[-1] == pytest.approx(-np.pi / 2, abs=0.2)

    def test_zpk(self):
        sys = zpk([-2], [-1, -3], 2)
        w, mag, phase = bode(sys)
        assert len(mag) == 1000
        assert len(phase) == 1000

    def test_ss(self):
        A = np.array([[-1]])
        B = np.array([[1]])
        C = np.array([[1]])
        D = np.array([[0]])
        sys = ss(A, B, C, D)
        w, mag, phase = bode(sys)
        assert mag[0] == pytest.approx(0.0, abs=0.1)


class TestNyquist:
    def test_first_order(self):
        sys = tf([1], [1, 1])
        omega = np.array([0.01, 1.0, 100.0])
        w, re, im = nyquist(sys, omega)
        assert re[0] == pytest.approx(1.0, rel=0.01)
        assert im[0] == pytest.approx(0.0, abs=0.02)

    def test_at_corner_freq(self):
        sys = tf([1], [1, 1])
        omega = np.array([1.0])
        w, re, im = nyquist(sys, omega)
        assert re[0] == pytest.approx(0.5, rel=0.01)
        assert im[0] == pytest.approx(-0.5, rel=0.01)


class TestMargin:
    def test_stable_system(self):
        sys = tf([1], [1, 1])
        m = margin(sys)
        assert isinstance(m, StabilityMargins)
        assert m.stable
        assert m.gm == np.inf
        assert m.pm == np.inf

    def test_unstable_closed_loop(self):
        sys = tf([10], [1, 3, 3, 1])
        m = margin(sys)
        assert not m.stable
        assert m.gm < 1.0
        assert m.pm < 0

    def test_marginally_stable(self):
        sys = tf([1], [1, 2, 1])
        m = margin(sys)
        assert m.stable

    def test_wcg_wcp(self):
        sys = tf([10], [1, 3, 3, 1])
        m = margin(sys)
        assert m.wcg > 0
        assert m.wcp > 0
        assert m.wcg == pytest.approx(1.91, rel=0.1)
        assert m.wcp == pytest.approx(1.73, rel=0.1)


class TestAutoOmega:
    def test_uses_poles(self):
        sys = tf([1], [1, 10])
        w, _ = freqresp(sys)
        assert w[0] < 10
        assert w[-1] > 10

    def test_no_poles(self):
        sys = tf([1], [1])
        w, _ = freqresp(sys)
        assert w[0] == pytest.approx(1e-3, rel=0.1)
        assert w[-1] == pytest.approx(1e3, rel=0.1)
