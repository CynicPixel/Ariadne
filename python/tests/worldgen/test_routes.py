import math

import numpy as np
import pytest

from dataset_tools.worldgen import routes
from dataset_tools.worldgen.motion import wrap_pi
from dataset_tools.worldgen.params import load_preset
from dataset_tools.worldgen.raster import Grid
from dataset_tools.worldgen.scene import generate

ROOT = __file__.rsplit("/python/", 1)[0]


@pytest.fixture(scope="module")
def scene():
    return generate(load_preset(f"{ROOT}/configs/worldgen/maze_short.yaml"), 2)


def test_segments_join_continuously_at_rest(scene):
    for sc in scene.scripts:
        segs = sc.segments
        for a, b in zip(segs, segs[1:]):
            assert b.t0 == pytest.approx(a.t0 + a.duration)
            pa, va, _ = a.pose(a.t0 + a.duration)
            pb, vb, _ = b.pose(b.t0)
            assert np.allclose(pa[:3], pb[:3], atol=1e-9)
            assert abs(wrap_pi(pa[3] - pb[3])) < 1e-9 and abs(pa[4] - pb[4]) < 1e-12
            assert np.allclose(va, 0, atol=1e-9) and np.allclose(vb, 0, atol=1e-9)


def rot(yaw, pitch):
    cz, sz, cy, sy = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    return Rz @ Ry


def check_imu_against_finite_differences(sc, times):
    """omega_B from R^T dR/dt (vee of the skew part), f_B = R^T (a + g e_z), by FD."""
    h = 1e-4
    for t in times:
        (x, y, z, yaw, pitch), _, _ = sc.pose(t)
        gyro, f = routes.imu(sc, t)
        p_m, _, _ = sc.pose(t - h)
        p_p, _, _ = sc.pose(t + h)
        R = rot(yaw, pitch)
        dR = (rot(p_p[3], p_p[4]) - rot(p_m[3], p_m[4])) / (2 * h)
        W = R.T @ dR
        w_fd = np.array([W[2, 1] - W[1, 2], W[0, 2] - W[2, 0], W[1, 0] - W[0, 1]]) / 2
        assert np.allclose(gyro, w_fd, atol=1e-5), t
        acc = np.array([(p_p[i] - 2 * [x, y, z][i] + p_m[i]) / (h * h) for i in range(3)])
        f_fd = R.T @ (acc + np.array([0, 0, routes.GRAVITY]))
        assert np.allclose(f, f_fd, atol=2e-3), t


def test_analytic_imu_matches_finite_differences(scene):
    sc = scene.scripts[0]
    check_imu_against_finite_differences(sc, np.linspace(0.3, sc.duration - 0.3, 700))


def test_imu_with_pitch_matches_finite_differences():
    from dataset_tools.worldgen.motion import plan_profile
    prof = plan_profile(math.hypot(1.2, 0.4), 0.44, 0.6)
    turn = routes.Segment("turn", 0.0, prof, prof.T, (1, 2, 1, 0.3, -0.1), (1, 2, 1, 1.5, 0.3))
    mprof = plan_profile(2.0, 0.5, 0.4)
    move = routes.Segment("move", prof.T, mprof, mprof.T, (1, 2, 1, 1.5, 0.3), (2.5, 3.0, 1.2, 1.5, 0.3))
    sc = routes.AgentScript(0, [turn, move])
    check_imu_against_finite_differences(sc, np.linspace(0.01, sc.duration - 0.01, 500))


def test_every_sample_is_inside_the_collision_envelope_and_speed_budget(scene):
    P = scene.preset
    M = P.motion
    g = scene.grid
    v_lk = M.lk_budget_px / (M.focal_px * M.frame_dt_s)
    w_max = min(M.omega_max_radps, M.lk_budget_px / (2 * M.focal_px * M.frame_dt_s))
    for sc in scene.scripts:
        for t in np.arange(0.0, sc.duration, 0.05):
            (x, y, z, _, _), (vx, vy, vz, wy_, wp_), _ = sc.pose(t)
            w = math.hypot(wy_, wp_)
            r, c = int(math.floor(y / g.g)) - g.oy, int(math.floor(x / g.g)) - g.ox
            e = scene.esdf_m[r, c]
            assert e >= P.clearance.rho_agent_m - 1e-9
            speed = math.sqrt(vx * vx + vy * vy + vz * vz)
            assert speed <= M.v_cap_mps + 1e-9
            assert speed <= v_lk * min(e, z, scene.wall_height_m - z) + 1e-6
            assert abs(w) <= w_max + 1e-9
            assert 0.3 <= z <= scene.wall_height_m - 0.3


def test_every_survey_is_a_full_turn(scene):
    for sc in scene.scripts:
        spins = [s for s in sc.segments if s.kind == "turn" and s.tag.startswith("survey")]
        assert len(spins) == 3                       # own entrance, target, other entrance
        assert all(s.profile.D == pytest.approx(2 * math.pi) for s in spins)


def test_line_of_sight():
    free = np.ones((20, 20), bool)
    free[5:15, 10] = False
    grid = Grid(0, 0, 20, 20, 0.1)
    assert routes.line_of_sight(free, grid, (0.25, 0.25), (1.95, 0.25))
    assert not routes.line_of_sight(free, grid, (0.55, 1.0), (1.55, 1.0))


def test_gate_reports_reasons(scene):
    assert routes.gate(scene, scene.overlap) is None
    bad = dict(scene.overlap, inter_rooms=[[0, 0]])
    assert "rooms" in routes.gate(scene, bad)


def test_script_json_reproduces_the_trajectory(scene):
    """What the C++ mover reads is sufficient: re-evaluate from JSON only."""
    from dataset_tools.worldgen.motion import Profile
    sc = scene.scripts[1]
    js = sc.to_json()
    segs = [routes.Segment(d["kind"], d["t0"], Profile(d["D"], d["Ta"], d["vc"]), d["duration"],
                           tuple(d["start"]), tuple(d["end"]), d["tag"]) for d in js["segments"]]
    again = routes.AgentScript(js["agent"], segs)
    for t in np.linspace(0, sc.duration, 400):
        assert again.pose(t) == sc.pose(t)
