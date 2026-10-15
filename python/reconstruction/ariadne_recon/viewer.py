"""Browser preview of the reconstruction (design RECON_1.7): a self-contained HTML page
with the Gaussians as an embedded standard `.splat` (32 B each: position f32x3, scale
f32x3 linear, RGBA u8, quaternion wxyz u8 as q*128+128) and both agents' ESTIMATED
paths, all in the world frame (z up) through the evaluation record's T_W_O0.

The `.splat` encoding follows the format gaussian-splats-3d's SplatParser reads
(v0.4.7: rotation bytes [w, x, y, z], each (b - 128) / 128). Colour = SH DC * C0 + 0.5,
alpha = sigmoid(opacity), scale = exp(log-scale): what gsplat renders at sh_degree 0.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import yaml

from .gaussians import SH_C0
from .spool import pose_from_row_major

TEMPLATE = Path(__file__).with_name("viewer_template.html")


def rot_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Shepperd's method; returns a unit quaternion (w, x, y, z) with w >= 0."""
    m = R
    tr = np.trace(m)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.asarray(q)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product, wxyz; a is (4,), b is (N, 4)."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b.T
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], axis=1)


def to_splat(pkg: dict, T_W_O0: np.ndarray) -> bytes:
    R, t = T_W_O0[:3, :3], T_W_O0[:3, 3]
    pos = (pkg["means"].astype(np.float64) @ R.T + t).astype("<f4")
    scale = np.exp(pkg["scales"].astype(np.float64)).astype("<f4")
    rgb = np.clip(pkg["sh0"][:, 0, :].astype(np.float64) * SH_C0 + 0.5, 0, 1)
    alpha = 1 / (1 + np.exp(-pkg["opacities"].astype(np.float64)))
    rgba = np.clip(np.round(np.c_[rgb, alpha] * 255), 0, 255).astype(np.uint8)
    q = pkg["quats"].astype(np.float64)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    qw = quat_mul(rot_to_quat_wxyz(R), q)
    rot = np.clip(np.round(qw * 128 + 128), 0, 255).astype(np.uint8)
    rec = np.empty(len(pos), dtype=[("p", "<f4", 3), ("s", "<f4", 3), ("c", "u1", 4), ("r", "u1", 4)])
    rec["p"], rec["s"], rec["c"], rec["r"] = pos, scale, rgba, rot
    return rec.tobytes()


def tum_positions(path: Path) -> np.ndarray:
    rows = [l.split() for l in path.read_text().splitlines() if l.strip() and not l.startswith("#")]
    return np.array([[float(v) for v in r[1:4]] for r in rows])


def build(prep_dir: Path, run_dir: Path, spool_dir: Path, out: Path, path_stride: int = 10) -> dict:
    pkg = dict(np.load(prep_dir / "package" / "render_package.npz"))
    prep = yaml.safe_load((prep_dir / "prepare.yaml").read_text())
    score_path = prep_dir / "score.yaml"
    score = yaml.safe_load(score_path.read_text()) if score_path.exists() else {}
    T_W_O0 = pose_from_row_major(yaml.safe_load((run_dir / "recon_eval.yaml").read_text())["T_W_O0"])
    T_O0_O1 = pose_from_row_major(yaml.safe_load((spool_dir / "spool.yaml").read_text())["merge"]["T_O0_O1"])
    splat = to_splat(pkg, T_W_O0)

    paths = {}
    for agent, T in (("agent0", T_W_O0), ("agent1", T_W_O0 @ T_O0_O1)):
        tum = next((run_dir / "raw").glob(f"*-{agent}-*/estimate_body.tum"))
        p = tum_positions(tum)[::path_stride]
        paths[agent] = np.round(p @ T[:3, :3].T + T[:3, 3], 3).tolist()

    pos = np.frombuffer(splat, dtype=[("p", "<f4", 3), ("x", "u1", 20)])["p"]
    lo, hi = np.percentile(pos, 2, axis=0), np.percentile(pos, 98, axis=0)
    c = (lo + hi) / 2
    span = float(np.max(hi[:2] - lo[:2]))
    view = {"count": int(len(pos)), "target": [float(c[0]), float(c[1]), 0.8],
            "camera": [float(c[0]), float(c[1] - 0.55 * span), float(0.9 * span)]}

    geo = prep["geometry"]
    fills = {"__SCENE__": prep["scene_id"], "__GAUSSIANS__": f"{prep['gaussians']:,}",
             "__KEYFRAMES__": str(prep["bundles"]["keyframes"]),
             "__ACCURACY__": f"{geo['accuracy_m'] * 1000:.0f}", "__COMPLETENESS__": f"{geo['completeness']:.2f}",
             "__PSNR__": f"{score['heldout_psnr_db']:.1f}" if score else "n/a", "__RUN__": prep["run_id"],
             "__VIEW_JSON__": json.dumps(view), "__PATHS_JSON__": json.dumps(paths),
             "__SPLAT_B64__": base64.b64encode(splat).decode()}
    html = TEMPLATE.read_text()
    for k, v in fills.items():
        if k not in html:
            raise ValueError(f"template lacks {k}")
        html = html.replace(k, v)
    out.write_text(html)
    return {"bytes": out.stat().st_size, "gaussians": view["count"], "view": view}

