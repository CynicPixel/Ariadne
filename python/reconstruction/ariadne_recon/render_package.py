"""The render package: the small, self-contained input of the remote (CUDA) render
step (D44, design reconstruction.md#first-reconstruction-loop). Initialized Gaussians + the held-out cameras
(world-to-camera view matrices in agent 0's odometry frame, K, size) + their
captured RGB (render targets, for LPIPS remotely; PSNR/SSIM are computed locally
from the returned renders). No spool, no depth, no GT.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .gaussians import SH_DEGREE, Gaussians
from .spool import Spool

PACKAGE_VERSION = 1


def write(spool: Spool, g: Gaussians, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    views = spool.heldout()
    if not views:
        raise ValueError("render package: the spool has no held-out frames")
    sizes = {(b.width, b.height) for b in views}
    if len(sizes) != 1:
        raise ValueError(f"render package: held-out frames differ in size {sizes}")
    (w, h), = sizes
    viewmats = np.stack([np.linalg.inv(spool.T_O0_C(b)) for b in views]).astype(np.float32)
    Ks = np.stack([b.K for b in views]).astype(np.float32)
    targets = np.stack([b.rgb() for b in views])
    names = np.array([f"agent{b.agent}_frame{b.kf_id}" for b in views])
    path = out_dir / "render_package.npz"
    np.savez_compressed(path, means=g.means, quats=g.quats, scales=g.scales, opacities=g.opacities, sh0=g.sh0,
                        viewmats=viewmats, Ks=Ks, targets=targets, names=names)
    meta = {"package_version": PACKAGE_VERSION, "run_id": spool.run_id, "scene_id": spool.scene_id,
            "gaussians": len(g), "views": len(views), "width": w, "height": h, "sh_degree": SH_DEGREE,
            "frame": "agent 0 odometry (viewmats are world-to-camera, OpenCV camera axes)",
            "rasterization": {"packed": False, "rasterize_mode": "classic", "near_plane": 0.01, "far_plane": 1e10,
                              "background": "black (gsplat default)"},
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (out_dir / "render_package.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta
