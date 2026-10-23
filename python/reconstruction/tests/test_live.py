"""Phase 6 live path, CPU parts (RECON_6, D190/D193): submap frames and re-anchoring (exit-gate bullet 2's algebra),
depth seeding, chunk export. No GPU, no torch."""
import json

import numpy as np
import pytest

from ariadne_recon.live.export import SPLAT_BYTES, chunk_bytes, table
from ariadne_recon.live.seeding import SeedParams, seed
from ariadne_recon.live.submaps import SubmapManager, inv_se3


def se3(yaw, t):
    c, s = np.cos(yaw), np.sin(yaw)
    T = np.eye(4)
    T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    T[:3, 3] = t
    return T


def test_submaps_close_at_max_and_hold_anchor_relative_poses():
    m = SubmapManager(max_keyframes=3)
    for k in range(7):
        sm, opened = m.add_keyframe(0, k, se3(0.1 * k, [k, 0, 0]), pose_version=1)
        assert opened == (k % 3 == 0)
    assert [len(s.keyframes) for s in m.submaps] == [3, 3, 1]
    assert [s.closed for s in m.submaps] == [True, True, False]
    s1 = m.submaps[1]
    assert s1.anchor_kf == 3
    np.testing.assert_allclose(s1.T_anchor_C[3], np.eye(4), atol=1e-12)
    np.testing.assert_allclose(s1.T_O_anchor @ s1.T_anchor_C[5], se3(0.5, [5, 0, 0]), atol=1e-12)


def test_reanchor_moves_the_placement_and_nothing_inside():
    """Exit-gate bullet 2: a PGO update mid-training re-anchors submaps with zero retraining -- the Gaussians in the
    local frame are unchanged, only the placement moves, and it moves exactly by the anchor's correction."""
    m = SubmapManager(max_keyframes=2)
    for k in range(4):
        m.add_keyframe(0, k, se3(0.0, [k, 0, 0]), pose_version=1)
    gaussians_local = np.random.default_rng(0).normal(size=(100, 3))
    before = gaussians_local.copy()
    member_before = {k: T.copy() for k, T in m.submaps[1].T_anchor_C.items()}
    T_W_O = se3(0.3, [1, 2, 0])
    corr = se3(0.05, [0.1, -0.2, 0])
    moved = m.reanchor(0, {2: corr @ se3(0.0, [2, 0, 0]), 3: se3(1.0, [9, 9, 9])}, pose_version=2)
    assert moved == [1]                                   # kf 3 is not an anchor: ignored (no intra-submap fix)
    np.testing.assert_array_equal(gaussians_local, before)
    for k, T in m.submaps[1].T_anchor_C.items():
        np.testing.assert_array_equal(T, member_before[k])
    np.testing.assert_allclose(m.submaps[1].placement(T_W_O), T_W_O @ corr @ se3(0.0, [2, 0, 0]), atol=1e-12)
    np.testing.assert_allclose(m.submaps[0].placement(T_W_O), T_W_O, atol=1e-12)   # untouched submap
    with pytest.raises(ValueError, match="older"):
        m.reanchor(0, {2: np.eye(4)}, pose_version=1)


def test_seeding_backprojects_into_the_anchor_frame_and_respects_alpha_and_rho():
    h, w, f = 40, 60, 50.0
    K = np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1]])
    Z = np.full((h, w), 2.0, np.float32)
    Z[0, :] = np.nan                                     # no depth: never seeded
    S = np.full((h, w), 0.01, np.float32)
    S[1, :] = 0.2                                        # sigma above the gate
    A = np.full((h, w), 0.4, np.float32)
    A[2, :] = np.nan                                     # undefined albedo: never seeded
    alpha = np.zeros((h, w), np.float32)
    alpha[:, : w // 2] = 0.9                             # left half already opaque
    T = se3(0.2, [1, 0, 0])
    p = SeedParams(sample_fraction=1.0, rho_m=1e-4)
    s = seed(Z, S, A, K, T, alpha, None, p, np.random.default_rng(0))
    r, c = s.pixels[:, 0], s.pixels[:, 1]
    assert (r >= 3).all() and (c >= w // 2).all()
    assert len(s.means) == (h - 3) * (w - w // 2)
    X_C = np.stack([(c - K[0, 2]) / f * 2.0, (r - K[1, 2]) / f * 2.0, np.full(len(r), 2.0)], 1)
    np.testing.assert_allclose(s.means, X_C @ T[:3, :3].T + T[:3, 3], atol=1e-5)
    np.testing.assert_allclose(s.scales, 2.0 / f, rtol=1e-6)    # the pixel footprint
    np.testing.assert_allclose(s.colors[:, 0], 0.4)
    # rho: existing Gaussians at those exact points suppress every seed; a coarse rho merges neighbours.
    assert len(seed(Z, S, A, K, T, alpha, s.means, p, np.random.default_rng(0)).means) == 0
    coarse = seed(Z, S, A, K, T, None, None, SeedParams(sample_fraction=1.0, rho_m=0.2), np.random.default_rng(0))
    assert 0 < len(coarse.means) < (h - 3) * w / 4


def test_seeding_sample_fraction_is_deterministic():
    h, w = 30, 30
    K = np.array([[30.0, 0, 14.5], [0, 30.0, 14.5], [0, 0, 1]])
    Z = np.full((h, w), 1.5, np.float32)
    S = np.full((h, w), 0.01, np.float32)
    img = np.full((h, w, 3), 0.5, np.float32)
    p = SeedParams(sample_fraction=0.05, rho_m=1e-4)
    a = seed(Z, S, img, K, np.eye(4), None, None, p, np.random.default_rng(3))
    b = seed(Z, S, img, K, np.eye(4), None, None, p, np.random.default_rng(3))
    assert len(a.means) == round(0.05 * h * w)
    np.testing.assert_array_equal(a.pixels, b.pixels)


def test_chunk_is_32_bytes_per_gaussian_in_the_anchor_frame_and_table_places_it():
    n = 5
    params = {"means": np.arange(15, dtype=np.float32).reshape(5, 3), "scales": np.log(np.full((n, 3), 0.02, np.float32)),
              "quats": np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1)), "opacities": np.zeros(n, np.float32),
              "sh0": np.zeros((n, 1, 3), np.float32)}
    b = chunk_bytes(params)
    assert len(b) == SPLAT_BYTES * n
    pos = np.frombuffer(b, dtype=[("p", "<f4", 3), ("s", "<f4", 3), ("c", "u1", 4), ("r", "u1", 4)])["p"]
    np.testing.assert_array_equal(pos, params["means"])                       # not moved: the anchor frame
    m = SubmapManager()
    m.add_keyframe(1, 7, se3(0.0, [1, 0, 0]), pose_version=4)
    t = json.loads(table(m.submaps, {1: se3(0.0, [0, 5, 0])}, {0: n}, "albedo proxy"))
    assert t["label"] == "albedo proxy" and t["submaps"][0]["bytes"] == 32 * n
    np.testing.assert_allclose(np.reshape(t["submaps"][0]["T_W_anchor"], (4, 4)), se3(0.0, [1, 5, 0]))
    np.testing.assert_allclose(inv_se3(se3(0.4, [1, 2, 3])) @ se3(0.4, [1, 2, 3]), np.eye(4), atol=1e-12)
