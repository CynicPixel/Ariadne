"""Hilti control-point scoring (5.6, D141) on a synthetic run with known answers."""
import numpy as np
import pytest

from analysis import hilti_score as hs


def _Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _write_tum(path, t_ns, T):
    with open(path, "w") as f:
        for t, P in zip(t_ns, T):
            R = P[:3, :3]
            w = np.sqrt(max(1e-12, 1 + np.trace(R))) / 2
            q = [(R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w), w]
            f.write(f"{t * 1e-9:.9f} " + " ".join(f"{float(v):.17g}" for v in (*P[:3, 3], *q)) + "\n")


def _run(tmp_path, noise_m=(0.0,) * 5, gt_shift_s=0.0):
    t_ns = np.arange(0, 60_000_000_001, 100_000_000, dtype=np.int64) + 10**18   # 10 Hz for 60 s
    T = []
    for t in t_ns:   # a slow arc, body yawing with it
        s = (t - t_ns[0]) * 1e-9
        P = np.eye(4)
        P[:3, :3] = _Rz(0.02 * s)
        P[:3, 3] = [5 * np.sin(0.02 * s), 5 * (1 - np.cos(0.02 * s)), 0.1 * s / 60]
        T.append(P)
    _write_tum(tmp_path / "est.tum", t_ns, T)
    # GT lives in a different world frame: W_gt = A W_est. The GCP sits 0.8 m ahead, 1 m below the IMU (robot frame).
    A = np.eye(4); A[:3, :3] = _Rz(0.7); A[:3, 3] = [18.0, 24.0, -3.4]
    off = np.array([0.8, 0.0, -1.0])
    gt_t = t_ns[[55, 155, 305, 455, 555]] + 50_000_000 + int(gt_shift_s * 1e9)   # between frames: interpolation
    with open(tmp_path / "gt.txt", "w") as f, open(tmp_path / "gcp.csv", "w") as g:
        f.write("# timestamp_s id x y z\n")
        g.write("#timestamp [ns],frame_id,point_name,x,y,z\n")
        for k, t in enumerate(gt_t):
            P = hs.interpolate(t_ns, np.array(T), t, 0.5)
            if P is None:   # outside the estimate: the GT still exists, the robot pose is extrapolated by hand
                P = T[-1]
            p = (A @ P @ np.append(off, 1.0))[:3] + np.array([noise_m[k], 0, 0])
            f.write(f"{t * 1e-9:.9f} " + " ".join(f"{float(v):.17g}" for v in p) + " 0 0 0 1\n")
            g.write(f"{t},imu,P{k}," + ",".join(f"{float(v):.17g}" for v in off) + "\n")
    return tmp_path


def test_exact_run_scores_full_marks(tmp_path):
    d = _run(tmp_path)
    r = hs.score(d / "est.tum", d / "gt.txt", d / "gcp.csv", None)
    assert r["unreached"] == []
    assert r["fit"]["score_0_100"] == 100.0 and r["loo"]["score_0_100"] == 100.0
    assert r["fit"]["rms_m"] < 1e-4


def test_without_gcp_offset_the_imu_position_misses_by_the_lever_arm(tmp_path):
    d = _run(tmp_path)
    r = hs.score(d / "est.tum", d / "gt.txt")
    assert r["predicted_point"].startswith("IMU position")
    # The rigid fit absorbs most of the 1.28 m lever arm on this arc, but not all of it: the residual (~2 cm here) is
    # already at Hilti's centimetre thresholds, which is why the GCP must be placed through T_I_F p_F.
    assert r["fit"]["rms_m"] > 0.01 and r["fit"]["score_0_100"] < 100.0


def test_one_bad_point_costs_more_under_leave_one_out(tmp_path):
    d = _run(tmp_path, noise_m=(0.0, 0.0, 0.08, 0.0, 0.0))
    r = hs.score(d / "est.tum", d / "gt.txt", d / "gcp.csv", None)
    assert r["loo"]["error_m"][2] > r["fit"]["error_m"][2]
    assert r["loo"]["score_0_100"] <= r["fit"]["score_0_100"]


def test_points_outside_the_estimate_are_unreached_and_score_zero(tmp_path):
    d = _run(tmp_path, gt_shift_s=10.0)   # the last GCP now falls after the estimate ends
    r = hs.score(d / "est.tum", d / "gt.txt", d / "gcp.csv", None)
    assert r["unreached"] == [4]
    assert 4 not in r["fit"]["points"]
    assert r["fit"]["score_0_100"] == pytest.approx(80.0)   # 4 of 5 points at 20 each, over 20 * 5


@pytest.mark.parametrize("err,pts", [(0.004, 20), (0.009, 10), (0.02, 6), (0.05, 5), (0.09, 3), (0.3, 1), (0.5, 0)])
def test_hilti_point_table(err, pts):
    assert hs.points_for(err) == pts
