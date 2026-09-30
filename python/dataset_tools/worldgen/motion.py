"""Smooth S-curve motion primitives shared (by formula) with the C++ kinematic mover.

A segment moves a scalar coordinate u from 0 to D (metres along a straight line, or
radians of yaw) with a velocity profile that ramps up, cruises, and ramps down:

    ramp (tau = t/Ta in [0,1]):   v = vc * sigma(tau),  sigma = 10 tau^3 - 15 tau^4 + 6 tau^5
                                  u = vc * Ta * Sigma(tau), Sigma = 2.5 tau^4 - 3 tau^5 + tau^6
    cruise:                       v = vc
    down-ramp: mirror image.

sigma' and sigma'' vanish at tau = 0 and 1, so acceleration AND jerk are continuous
everywhere: the analytic IMU derived from this trajectory has no steps. The ramp's
peak acceleration is 1.875 * vc / Ta. plan_profile picks the fastest member of the
family within v_max and a_max (see its docstring). The mover in C++ implements exactly these
formulas from (D, Ta, vc) stored in the script; it never re-plans.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


def sigma(t: float) -> float:
    return t * t * t * (10.0 + t * (-15.0 + 6.0 * t))


def dsigma(t: float) -> float:
    return 30.0 * t * t * (1.0 + t * (-2.0 + t))


def ddsigma(t: float) -> float:
    return t * (60.0 + t * (-180.0 + 120.0 * t))


def big_sigma(t: float) -> float:
    return t * t * t * t * (2.5 + t * (-3.0 + t))


@dataclass(frozen=True)
class Profile:
    D: float        # total displacement (>= 0)
    Ta: float       # ramp duration
    vc: float       # cruise rate

    @property
    def Tc(self) -> float:
        return 0.0 if self.vc == 0.0 else max(self.D / self.vc - self.Ta, 0.0)

    @property
    def T(self) -> float:
        return 0.0 if self.D == 0.0 else 2.0 * self.Ta + self.Tc

    def eval(self, t: float) -> tuple[float, float, float]:
        """(u, du/dt, d2u/dt2) at time t (clamped to [0, T])."""
        if self.D == 0.0:
            return 0.0, 0.0, 0.0
        T, Ta, vc = self.T, self.Ta, self.vc
        t = min(max(t, 0.0), T)
        if t < Ta:
            tau = t / Ta
            return vc * Ta * big_sigma(tau), vc * sigma(tau), vc * dsigma(tau) / Ta
        if t <= Ta + self.Tc:
            return 0.5 * vc * Ta + vc * (t - Ta), vc, 0.0
        tau = (T - t) / Ta
        return self.D - vc * Ta * big_sigma(tau), vc * sigma(tau), -vc * dsigma(tau) / Ta


def plan_profile(D: float, v_max: float, a_max: float) -> Profile:
    """Fastest profile of this family within v_max and a_max.

    Long (D >= v_max * Ta_full, Ta_full = 1.875 v_max / a_max): ramps at the full
    Ta_full, cruise at v_max. Short: no cruise, and the ramp is re-timed so the peak
    acceleration is exactly a_max: vc = sqrt(a_max D / 1.875), Ta = D / vc
    (vc < v_max follows from D < v_max * Ta_full)."""
    if D < 0 or v_max <= 0 or a_max <= 0:
        raise ValueError("plan_profile: D >= 0, v_max > 0, a_max > 0")
    if D == 0:
        return Profile(0.0, 0.0, 0.0)
    Ta = 1.875 * v_max / a_max
    if D >= v_max * Ta:
        return Profile(D, Ta, v_max)
    vc = math.sqrt(a_max * D / 1.875)
    return Profile(D, D / vc, vc)


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi
