import math

import pytest

from dataset_tools.worldgen import motion as mo


def test_sigma_boundary_conditions():
    assert mo.sigma(0) == 0 and mo.sigma(1) == pytest.approx(1)
    for f in (mo.dsigma, mo.ddsigma):
        assert f(0) == 0 and f(1) == pytest.approx(0, abs=1e-12)
    assert mo.big_sigma(1) == pytest.approx(0.5)
    assert max(mo.dsigma(i / 1000) for i in range(1001)) == pytest.approx(1.875, rel=1e-5)


@pytest.mark.parametrize("D", [0.05, 0.3, 1.0, 6.0, 25.0])
def test_profile_is_consistent_and_within_limits(D):
    v_max, a_max = 0.5, 0.4
    p = mo.plan_profile(D, v_max, a_max)
    assert p.eval(p.T)[0] == pytest.approx(D)
    assert p.eval(0)[1] == 0 and p.eval(p.T)[1] == pytest.approx(0, abs=1e-12)
    n = 4000
    h = p.T / n
    prev = p.eval(0)
    for i in range(1, n + 1):
        u, v, a = p.eval(i * h)
        assert v <= v_max + 1e-12 and abs(a) <= a_max + 1e-9
        # derivatives agree with finite differences of the position/velocity
        assert (u - prev[0]) / h == pytest.approx(0.5 * (v + prev[1]), abs=1e-6 + 1e-3 * v_max)
        assert (v - prev[1]) / h == pytest.approx(0.5 * (a + prev[2]), abs=1e-6 + 2e-3 * a_max)
        prev = (u, v, a)


def test_acceleration_is_continuous_at_phase_joins():
    p = mo.plan_profile(10.0, 0.5, 0.4)
    for t in (p.Ta, p.Ta + p.Tc):
        a_lo, a_hi = p.eval(t - 1e-7)[2], p.eval(t + 1e-7)[2]
        assert abs(a_lo - a_hi) < 1e-5


def test_short_segment_is_time_optimal_at_a_max():
    p = mo.plan_profile(0.5, 0.6, 0.4)
    assert p.Tc == 0
    peak_a = max(abs(p.eval(i * p.T / 2000)[2]) for i in range(2001))
    assert peak_a == pytest.approx(0.4, rel=1e-3)
    assert p.T == pytest.approx(2 * math.sqrt(1.875 * 0.5 / 0.4))


def test_long_segment_averages_near_cruise_speed():
    p = mo.plan_profile(20.0, 0.5, 0.4)
    assert 20.0 / p.T > 0.8 * 0.5          # vs 0.5/1.875 = 0.27 for a pure min-jerk move


def test_wrap_pi():
    assert mo.wrap_pi(3 * math.pi / 2) == pytest.approx(-math.pi / 2)
    assert mo.wrap_pi(-3 * math.pi / 2) == pytest.approx(math.pi / 2)
