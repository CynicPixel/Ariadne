"""Corridor gate ordering scoring (D148 part A) on a synthetic scene and run with known answers."""
import json

import numpy as np

from analysis import corridor_gate as cg

HEADER = "a,b,score,degenerate,reasons,sigma_t_m,sigma_r_rad,err_t_m,err_r_deg,nees6,label,pcm,gnc_inlier,corridor"


def _scene(tmp_path):
    # 10 x 10 cells of 0.5 m from (-1, -1); clearance 0.2 m in row 0 (y < -0.5), 0.4 m in row 1, 1.0 m elsewhere.
    grid = np.full((10, 10), 1.0, np.float32)
    grid[0, :], grid[1, :] = 0.2, 0.4
    s = tmp_path / "scene"
    s.mkdir()
    grid.astype("<f4").tofile(s / "esdf_f32.bin")
    (s / "scene.json").write_text(json.dumps({"esdf": {"file": "esdf_f32.bin", "rows": 10, "cols": 10, "cell_m": 0.5,
                                                       "origin_m": [-1.0, -1.0]}}))
    return s


def _run(tmp_path, cands, name="agent0"):
    r = tmp_path / name
    (r / "raw").mkdir(parents=True)
    xy = [(0.0, -0.9), (0.0, -0.4), (0.0, 2.0)]          # keyframes 0 narrow, 1 corridor, 2 open
    t_ns = [0, 100_000_000, 200_000_000]
    with open(r / "vo_trace.csv", "w") as f:
        f.write("frame,t_ns,keyframe\n")
        for i, t in enumerate(t_ns):
            f.write(f"{2 * i},{t},1\n{2 * i + 1},{t + 50_000_000},0\n")
    with open(r / "raw" / "groundtruth_body.tum", "w") as f:
        for t, (x, y) in zip(t_ns, xy):
            f.write(f"{t * 1e-9:.9f} {x} {y} 0 0 0 0 1\n")
    with open(r / "loop_candidates.csv", "w") as f:
        f.write(HEADER + "\n")
        for c in cands:
            f.write(c + "\n")
    with open(r / "loop_trace.jsonl", "w") as f:
        for c in cands:
            a = int(c.split(",")[0])
            f.write(json.dumps({"t_ns": t_ns[a], "candidates": [{"a": a, "b": int(c.split(",")[1])}]}) + "\n")
    return r


def _score(tmp_path, cands):
    grid, cell, origin = cg.load_esdf(_scene(tmp_path))
    rows, d = cg.score_run(_run(tmp_path, cands), grid, cell, origin)
    return cg.summarise(rows, d)


def test_met_when_a_confined_candidate_is_refused_and_bad_ones_never_become_inliers(tmp_path):
    res = _score(tmp_path, [
        "1,0,0.9,1,collinear|parallax,0,0,0,0,0,good,0,0,0.95",   # narrow, refused
        "2,1,0.9,0,,0,0,0.5,3,0,bad,1,0,0.5",                       # corridor, GT-bad, removed by GNC
        "2,2,0.9,0,,0,0,0,0,0,good,1,1,0.4",                        # open, accepted
    ])
    assert res["keyframes"] == {"open": 1, "corridor": 1, "narrow": 1}
    assert res["ordering_violations"] == 0
    assert res["per_label"]["narrow"]["refused"] == 1
    assert res["per_label"]["narrow"]["reasons"] == {"collinear": 1, "parallax": 1}
    assert res["per_label"]["corridor"]["accepted_bad_removed_by_pcm_gnc"] == 1
    assert res["per_label"]["open"]["accepted_gt"] == {"good": 1}
    assert res["bullet"] == "met"


def test_a_degenerate_candidate_reaching_pcm_is_a_violation(tmp_path):
    res = _score(tmp_path, ["1,0,0.9,1,parallax,0,0,0,0,0,good,1,0,0.9"])
    assert res["ordering_violations"] == 1
    assert res["bullet"] == "not met"


def test_a_bad_confined_gnc_inlier_fails_the_bullet(tmp_path):
    res = _score(tmp_path, ["1,0,0.9,1,parallax,0,0,0,0,0,good,0,0,0.9", "1,2,0.9,0,,0,0,1,5,0,bad,1,1,0.5"])
    assert res["per_label"]["corridor"]["gnc_inlier_bad"] == 1
    assert res["bullet"] == "not met"


def test_no_confined_candidate_is_not_exercised(tmp_path):
    assert _score(tmp_path, ["2,2,0.9,0,,0,0,0,0,0,good,1,1,0.4"])["bullet"] == "not exercised"
