"""compose_mission_spool: schema-3 merges from pass-B estimates (D213)."""
import numpy as np
import yaml

from analysis.compose_mission_spool import compose


def _rot_z(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _pose(R, t):
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = R, t
    return m


def _spool(root, agent, kfs):
    d = root / f"in{agent}"
    (d / f"agent{agent}").mkdir(parents=True)
    bundles = []
    for kf in kfs:
        stem = f"agent{agent}/kf_{kf}"
        for suf in ("rgb", "depth", "sigma"):
            (d / f"{stem}_{suf}.png").write_bytes(b"x")
        bundles.append({"role": "keyframe", "agent": agent, "kf_id": kf, "t_ns": kf * 50_000_000,
                        "T_WiC": np.eye(4)[:3, :].reshape(-1).tolist(), "files": stem})
    (d / "spool.yaml").write_text(yaml.safe_dump({
        "schema_version": 1, "run_id": f"r{agent}", "scene_id": "s", "scale": 0.5, "keyframe_stride": 1,
        "heldout_stride": 10, "image_encoding": "e", "frames": "f",
        "merge": {"ok": False, "T_O0_O1": np.eye(4)[:3, :].reshape(-1).tolist(), "edge": ""}, "bundles": bundles}))
    return d


def _collab(root, est, kept):
    d = root / "passb"
    d.mkdir()
    (d / "results.yaml").write_text(yaml.safe_dump({"estimates": est, "kept_inter_edges": kept}))
    return d


def _est(agent, anchor, T):
    return {"agent": agent, "anchor": anchor, "T_G_W": None if T is None else T[:3, :].reshape(-1).tolist()}


def test_three_agents_chain_and_unmerged(tmp_path):
    T0 = _pose(_rot_z(10), [1, 2, 0])
    T1 = _pose(_rot_z(-30), [5, 0, 0])
    T2 = _pose(_rot_z(90), [0, 7, 1])
    sp = [_spool(tmp_path, 0, [1, 2, 3]), _spool(tmp_path, 1, [4, 5]), _spool(tmp_path, 2, [6, 7]),
          _spool(tmp_path, 3, [8])]
    est = [_est(0, 1, T0), _est(1, 1, T1), _est(2, 1, T2), _est(3, None, None)]
    kept = [{"agent_a": 1, "kf_a": 5, "agent_b": 2, "kf_b": 6, "t_s": 3.0, "weight": 1.0},   # 2 joins via 1
            {"agent_a": 0, "kf_a": 2, "agent_b": 1, "kf_b": 4, "t_s": 5.0, "weight": 0.9},
            {"agent_a": 0, "kf_a": 99, "agent_b": 2, "kf_b": 7, "t_s": 1.0, "weight": 1.0}]  # kf 99 not spooled
    doc = compose(sp, _collab(tmp_path, est, kept), tmp_path / "out")
    assert doc["schema_version"] == 3 and "merge" not in doc
    m = {x["agent"]: x for x in doc["merges"]}
    assert sorted(m) == [1, 2, 3]
    assert m[1]["ok"] and m[1]["edge"].startswith("agent0 kf 2 <- agent1 kf 4")
    assert m[2]["ok"] and m[2]["edge"].startswith("agent1 kf 5 <- agent2 kf 6")
    assert not m[3]["ok"]
    for k, Tk in ((1, T1), (2, T2)):   # T_O0_O = T_G_W0^-1 T_G_Wk
        got = np.array(m[k]["T_O0_O"]).reshape(3, 4)
        np.testing.assert_allclose(got, (np.linalg.inv(T0) @ Tk)[:3, :], atol=1e-12)
    assert len(doc["bundles"]) == 8
    assert (tmp_path / "out" / "agent2" / "kf_6_rgb.png").is_symlink()


def test_agent_in_another_frame_does_not_merge(tmp_path):
    sp = [_spool(tmp_path, 0, [1]), _spool(tmp_path, 1, [2])]
    est = [_est(0, 0, np.eye(4)), _est(1, 1, np.eye(4))]   # split brain: different anchors
    kept = [{"agent_a": 0, "kf_a": 1, "agent_b": 1, "kf_b": 2, "t_s": 1.0, "weight": 1.0}]
    doc = compose(sp, _collab(tmp_path, est, kept), tmp_path / "out")
    assert doc["merges"] == [dict(doc["merges"][0], ok=False)]
