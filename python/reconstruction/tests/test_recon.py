"""Unit tests for the 1.7 reconstruction service's local stages (design RECON_1.7).

Synthetic spools in the C++ writer's exact format (spool.yaml + RGB/fp16 PNGs):
a fronto-parallel plane seen by two agents whose odometry frames differ by a known
T_O0_O1, so every stage has an exact expected answer.
"""
import numpy as np
import cv2
import pytest
import yaml

from ariadne_recon import fuse, geometry_eval, image_metrics
from ariadne_recon.gaussians import SH_C0, init_from_points
from ariadne_recon.spool import SpoolError, load

W, H, F = 64, 48, 50.0
K = [F, F, (W - 1) / 2, (H - 1) / 2]


def row_major(T):
    return [float(x) for x in T[:3, :].ravel()]


def write_half(path, a):
    cv2.imwrite(str(path), a.astype(np.float16).view(np.uint16))


def make_spool(tmp, T_O0_O1=np.eye(4), z=2.0, drop=None):
    """Agent 0 at the origin, agent 1's camera at its own origin; both see the plane Z = z."""
    bundles = []
    for agent in (0, 1):
        (tmp / f"agent{agent}").mkdir(exist_ok=True)
        stem = f"agent{agent}/kf_{agent}"
        rgb = np.full((H, W, 3), (200, 100, 50), np.uint8)
        cv2.imwrite(str(tmp / f"{stem}_rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        Z = np.full((H, W), z, np.float32)
        Z[0, 0] = np.nan
        write_half(tmp / f"{stem}_depth.png", Z)
        write_half(tmp / f"{stem}_sigma.png", np.full((H, W), 0.01, np.float32))
        e = {"role": "keyframe", "agent": agent, "kf_id": agent, "t_ns": agent, "frame_id_of_pose": 1,
             "T_WiC": row_major(np.eye(4)), "K": K, "width": W, "height": H, "baseline_m": 0.1, "files": stem,
             "rgb_kind": "raw_radiance", "exposure_state": "not_modeled", "lamp_state": "inert", "mask": "none"}
        if drop:
            e.pop(drop)
        bundles.append(e)
    y = {"schema_version": 1, "run_id": "r", "scene_id": "s",
         "merge": {"ok": True, "T_O0_O1": row_major(T_O0_O1), "edge": "e"}, "bundles": bundles}
    (tmp / "spool.yaml").write_text(yaml.safe_dump(y))
    return tmp


def test_missing_field_is_an_error(tmp_path):
    with pytest.raises(SpoolError, match="lamp_state"):
        load(make_spool(tmp_path, drop="lamp_state"))


def test_depth_roundtrip_and_rgb_order(tmp_path):
    sp = load(make_spool(tmp_path))
    Z, S = sp.keyframes()[0].depth()
    assert np.isnan(Z[0, 0]) and Z[1, 1] == np.float32(2.0) and abs(S[1, 1] - 0.01) < 1e-5
    assert tuple(sp.keyframes()[0].rgb()[0, 0]) == (200, 100, 50)


def test_fuse_applies_merge_and_voxelizes(tmp_path):
    T = np.eye(4)
    T[:3, 3] = [5.0, 0.0, 0.0]          # agent 1's odometry origin is 5 m along x in O0
    sp = load(make_spool(tmp_path, T_O0_O1=T))
    c = fuse.fuse(sp, fuse.FuseParams(voxel_m=0.01, max_depth_m=6, max_sigma_m=0.05, max_points=10 ** 6))
    assert np.allclose(c.xyz[:, 2], 2.0, atol=1e-3)                 # the plane, in O0
    a1 = c.xyz[c.source == 1]
    assert len(a1) and a1[:, 0].min() > 5.0 - 1.3                  # agent 1's points shifted by 5 m
    assert c.stats["pixels_with_depth"] == 2 * (W * H - 1)          # NaN pixel excluded
    coarse = fuse.fuse(sp, fuse.FuseParams(voxel_m=0.5, max_depth_m=6, max_sigma_m=0.05, max_points=10 ** 6))
    assert len(coarse.xyz) < len(c.xyz)
    with pytest.raises(SpoolError, match="no depth survived"):
        fuse.fuse(sp, fuse.FuseParams(voxel_m=0.01, max_depth_m=1.0, max_sigma_m=0.05, max_points=10 ** 6))


def test_gaussian_init_matches_simple_trainer_conventions():
    s = 0.1
    g = np.stack(np.meshgrid(np.arange(10), np.arange(10), indexing="ij"), -1).reshape(-1, 2) * s
    xyz = np.c_[g, np.zeros(len(g))]
    rgb = np.full((len(xyz), 3), 255, np.uint8)
    a = init_from_points(xyz, rgb, 0.1, seed=3)
    interior = (np.abs(xyz[:, :2] - 0.45) < 0.35).all(axis=1)       # 4 neighbours at distance s
    assert np.allclose(a.scales[interior], np.log(s), atol=1e-6)
    assert np.allclose(a.opacities, np.log(0.1 / 0.9))
    assert np.allclose(a.sh0, 0.5 / SH_C0)
    b = init_from_points(xyz, rgb, 0.1, seed=3)
    assert np.array_equal(a.quats, b.quats)                          # seeded: reproducible
    with pytest.raises(ValueError):
        init_from_points(np.zeros((5, 3)), rgb[:5], 0.1, seed=0)     # coincident points


def test_geometry_metrics_on_exact_surface():
    tri = np.array([[[0, 0, 0], [1, 0, 0], [1, 1, 0]], [[0, 0, 0], [1, 1, 0], [0, 1, 0]]], float)
    gt = geometry_eval.sample_surface(tri, 0.02, seed=0)
    assert len(gt) >= 2500 and np.all(gt[:, 2] == 0) and gt[:, :2].min() >= 0 and gt[:, :2].max() <= 1
    r = geometry_eval.evaluate(gt, gt, tau=0.01)
    assert r["accuracy_m"] == 0 and r["completeness"] == 1
    half = gt[gt[:, 0] < 0.5] + [0, 0, 0.05]
    r2 = geometry_eval.evaluate(half, gt, tau=0.06)
    assert abs(r2["accuracy_m"] - 0.05) < 1e-9 and 0.4 < r2["completeness"] < 0.6


def test_image_metrics():
    rng = np.random.default_rng(0)
    a = rng.random((40, 50, 3))
    assert image_metrics.ssim(a, a) == pytest.approx(1.0)
    assert image_metrics.psnr(a, a) == float("inf")
    assert image_metrics.psnr(a, np.clip(a + 0.1, 0, 2)) == pytest.approx(20.0, abs=1e-9)
    assert image_metrics.ssim(a, rng.random((40, 50, 3))) < 0.1


def test_splat_encoding_matches_the_viewer_format():
    """32-byte rows: position f32x3, linear scale f32x3, RGBA u8, quaternion wxyz u8 (q*128+128)."""
    from ariadne_recon.gaussians import Gaussians
    from ariadne_recon.viewer import quat_mul, rot_to_quat_wxyz, to_splat
    g = init_from_points(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1.0]]), np.full((4, 3), 255, np.uint8),
                         0.5, seed=0)
    g = Gaussians(g.means, np.tile([1, 0, 0, 0], (4, 1)).astype(np.float32), np.zeros((4, 3), np.float32),
                  g.opacities, g.sh0, g.shN)
    T = np.eye(4)
    T[:3, 3] = [10, 20, 30]
    rec = np.frombuffer(to_splat({k: getattr(g, k) for k in ("means", "quats", "scales", "opacities", "sh0")}, T),
                        dtype=[("p", "<f4", 3), ("s", "<f4", 3), ("c", "u1", 4), ("r", "u1", 4)])
    assert rec.itemsize == 32 and len(rec) == 4
    assert np.allclose(rec["p"][1], [11, 20, 30]) and np.allclose(rec["s"], 1.0)      # exp(0) = 1
    assert tuple(rec["c"][0]) == (255, 255, 255, 128)                                  # sigmoid(logit .5)
    assert tuple(rec["r"][0]) == (255, 128, 128, 128)                                  # identity, clamped w
    Rz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])                                # 90 deg about z
    q = rot_to_quat_wxyz(Rz)
    assert np.allclose(q, [np.sqrt(0.5), 0, 0, np.sqrt(0.5)])
    assert np.allclose(quat_mul(q, np.array([[1.0, 0, 0, 0]]))[0], q)


def make_spool_v2(tmp, with_lamp=True):
    """Schema 2 (6.1, D195): agent 0 keyframe with an albedo proxy and an applied photometric state, plus the lamp."""
    make_spool(tmp)
    y = yaml.safe_load((tmp / "spool.yaml").read_text())
    y["schema_version"] = 2
    a = np.full((H, W), 0.25, np.float32)
    a[2, 3] = np.nan
    write_half(tmp / "agent0/kf_0_albedo.png", a)
    y["bundles"][0].update({"rgb_kind": "albedo_proxy", "exposure_state": {"exposure_gain": 2.0, "decision_id": 7},
                            "lamp_state": {"lamp_rel": 0.5}})
    if with_lamp:
        (tmp / "lamp_calibration.yaml").write_text("name: test\n")
        y["lamp"] = {"calibration": "lamp_calibration.yaml", "calibration_source": "configs/sim/x.yaml",
                     "calibration_name": "test", "rig": "rig_stereo_v2", "intensity": 3.0,
                     "R_rect_raw": [1, 0, 0, 0, 1, 0, 0, 0, 1]}
    (tmp / "spool.yaml").write_text(yaml.safe_dump(y))
    return tmp


def test_schema2_albedo_photo_and_lamp(tmp_path):
    sp = load(make_spool_v2(tmp_path))
    assert sp.schema_version == 2 and sp.lamp.intensity == 3.0 and sp.lamp.R_rect_raw.shape == (3, 3)
    kf0, kf1 = sp.keyframes()
    assert kf0.rgb_kind == "albedo_proxy" and kf0.exposure_gain == 2.0 and kf0.lamp_rel == 0.5 and kf0.decision_id == 7
    A = kf0.albedo()
    assert A[0, 0] == np.float32(0.25) and np.isnan(A[2, 3])   # undefined stays NaN, never a guessed reflectance
    assert kf1.rgb_kind == "raw_radiance" and kf1.exposure_gain is None
    with pytest.raises(SpoolError, match="no albedo proxy"):
        kf1.albedo()


def test_schema2_albedo_needs_the_lamp(tmp_path):
    with pytest.raises(SpoolError, match="lamp record"):
        load(make_spool_v2(tmp_path, with_lamp=False))


def test_schema1_still_loads_without_photo(tmp_path):
    sp = load(make_spool(tmp_path))
    assert sp.schema_version == 1 and sp.lamp is None and sp.keyframes()[0].lamp_rel is None


def test_schema3_merges_list_places_every_agent(tmp_path):
    """Schema 3 (A1/M/A2, 2026-10-23): merges: replaces merge:, one entry per non-anchor agent."""
    make_spool(tmp_path)
    y = yaml.safe_load((tmp_path / "spool.yaml").read_text())
    y["schema_version"] = 3
    y.pop("merge")
    T = np.eye(4)
    T[:3, 3] = [1.0, 2.0, 0.0]
    y["merges"] = [{"agent": 1, "ok": True, "T_O0_O": row_major(T), "edge": "agent0 kf 3 <- agent1 kf 5"}]
    (tmp_path / "spool.yaml").write_text(yaml.safe_dump(y))
    sp = load(tmp_path)
    b1 = [b for b in sp.keyframes() if b.agent == 1][0]
    np.testing.assert_allclose(sp.T_O0_C(b1), T @ b1.T_O_C)
    y["merge"] = {"ok": True, "T_O0_O1": row_major(T), "edge": "e"}
    (tmp_path / "spool.yaml").write_text(yaml.safe_dump(y))
    with pytest.raises(SpoolError, match="not merge"):
        load(tmp_path)
