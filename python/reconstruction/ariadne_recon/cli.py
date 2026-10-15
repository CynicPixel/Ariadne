"""Reconstruction service entry point (design RECON_1.7).

  python -m ariadne_recon.cli prepare --spool SPOOL --config recon_v1.yaml --scene SCENE \
        --eval-record RUN/recon_eval.yaml --out OUT [--seed 1]
      validate spool -> fuse -> init Gaussians -> geometry metrics -> render package (local, CPU)
  python -m ariadne_recon.cli score --out OUT --remote DIR
      PSNR / SSIM from the remote renders (+ the remote LPIPS) -> results.yaml
  python -m ariadne_recon.cli viewer --out OUT --run RUN --spool SPOOL --page PAGE.html
      self-contained WebGL page: the Gaussians as an embedded .splat + both agents' paths (world frame)

Every output directory gets the inputs' identities (spool.yaml sha256, config,
seed, command) so a result can be traced back to its spool and package.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from . import fuse as fuse_mod
from . import geometry_eval, image_metrics, render_package
from .gaussians import init_from_points
from .spool import load, pose_from_row_major


def _manifest(stage: str, extra: dict) -> dict:
    """§28.2 identity of this stage (git + image from the launcher's environment; unknown is said, not guessed)."""
    import os
    return {"stage": stage, "git_commit": os.environ.get("ARIADNE_GIT_COMMIT", "unknown"),
            "git_dirty": os.environ.get("ARIADNE_GIT_DIRTY", "unknown"),
            "image_id": os.environ.get("ARIADNE_IMAGE_ID", "unknown"), **extra}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    head = ("ply\nformat binary_little_endian 1.0\nelement vertex %d\nproperty float x\nproperty float y\n"
            "property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n") % len(xyz)
    rec = np.empty(len(xyz), dtype=[("p", "<f4", 3), ("c", "u1", 3)])
    rec["p"], rec["c"] = xyz.astype("<f4"), rgb
    path.write_bytes(head.encode() + rec.tobytes())


def prepare(a) -> dict:
    cfg = yaml.safe_load(Path(a.config).read_text())
    sv, ev = cfg["service"], cfg["evaluation"]
    if sv["train_steps"] != 0:
        raise ValueError("1.7 renders the initialization: train_steps must be 0")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sp = load(a.spool)
    cloud = fuse_mod.fuse(sp, fuse_mod.FuseParams(sv["voxel_m"], sv["max_depth_m"], sv["max_sigma_m"],
                                                  sv["max_gaussians"]))
    g = init_from_points(cloud.xyz, cloud.rgb, sv["init_opacity"], a.seed)
    write_ply(out / "fused_cloud_O0.ply", cloud.xyz, cloud.rgb)
    pkg = render_package.write(sp, g, out / "package")

    rec = yaml.safe_load(Path(a.eval_record).read_text())
    T_W_O0 = pose_from_row_major(rec["T_W_O0"])
    xyz_w = cloud.xyz @ T_W_O0[:3, :3].T + T_W_O0[:3, 3]
    gt = geometry_eval.scene_gt_points(Path(a.scene), ev["gt_sample_m"])
    geo = geometry_eval.evaluate(xyz_w, gt, ev["completeness_tau_m"])

    res = {"stage": "prepare", "run_id": sp.run_id, "scene_id": sp.scene_id, "config": cfg["name"],
           "seed": a.seed, "spool_sha256": _sha(Path(a.spool) / "spool.yaml"),
           "merge": {"ok": sp.merge_ok, "edge": sp.merge_edge},
           "bundles": {"keyframes": len(sp.keyframes()), "heldout": len(sp.heldout())},
           "fusion": cloud.stats, "gaussians": len(g), "package": pkg, "geometry": geo,
           "command": " ".join(sys.argv)}
    (out / "manifest_prepare.yaml").write_text(yaml.safe_dump(_manifest("prepare", {
        "config": cfg["name"], "seed": a.seed, "spool_sha256": res["spool_sha256"], "splat_backend": "none (local)",
        "command": res["command"]}), sort_keys=False))
    (out / "prepare.yaml").write_text("# ariadne_recon prepare (design RECON_1.7). Generated.\n" +
                                      yaml.safe_dump(res, sort_keys=False))
    return res


def score(a) -> dict:
    out, remote = Path(a.out), Path(a.remote)
    prep = yaml.safe_load((out / "prepare.yaml").read_text())
    rj = json.loads((remote / "remote.json").read_text())
    if rj.get("package_sha256") != prep["package"]["sha256"]:
        raise ValueError("remote renders were made from a different render package")
    r = np.load(remote / "renders.npz")
    pk = np.load(out / "package" / "render_package.npz")
    if list(r["names"]) != list(pk["names"]):
        raise ValueError("render / target view order differs")
    tg = pk["targets"].astype(np.float64) / 255.0
    rn = r["renders"].astype(np.float64)
    ps = [image_metrics.psnr(rn[i], tg[i]) for i in range(len(tg))]
    ss = [image_metrics.ssim(rn[i], tg[i]) for i in range(len(tg))]
    lp = rj.get("lpips_alex")
    res = {"stage": "score", "views": len(tg), "heldout_psnr_db": float(np.mean(ps)),
           "heldout_ssim": float(np.mean(ss)), "heldout_lpips": float(np.mean(lp)) if lp else None,
           "per_view": [{"view": str(n), "psnr_db": p, "ssim": s, **({"lpips": l} if lp else {})}
                        for n, p, s, l in zip(pk["names"], ps, ss, lp or [None] * len(ps))],
           "remote_env": rj["env"], "remote_render_s": rj.get("render_s"),
           "renders_sha256": _sha(remote / "renders.npz"), "command": " ".join(sys.argv)}
    (out / "manifest_score.yaml").write_text(yaml.safe_dump(_manifest("score", {
        "splat_backend": "gsplat_cuda@kaggle", "remote_env": rj["env"], "package_sha256": prep["package"]["sha256"],
        "renders_sha256": res["renders_sha256"], "command": res["command"]}), sort_keys=False))
    (out / "score.yaml").write_text("# ariadne_recon score (design RECON_1.7). Generated.\n" +
                                    yaml.safe_dump(res, sort_keys=False))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ariadne_recon")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    for k in ("--spool", "--config", "--scene", "--eval-record", "--out"):
        p.add_argument(k, required=True)
    p.add_argument("--seed", type=int, default=1)
    s = sub.add_parser("score")
    s.add_argument("--out", required=True)
    s.add_argument("--remote", required=True)
    v = sub.add_parser("viewer")
    for k in ("--out", "--run", "--spool", "--page"):
        v.add_argument(k, required=True)
    v.add_argument("--mission")   # a 1.8 mission run: draw its final egress route
    v.add_argument("--scene")
    a = ap.parse_args(argv)
    if a.cmd == "viewer":
        from . import viewer
        if a.mission and not a.scene:
            raise ValueError("--mission needs --scene (the planner grid)")
        print(viewer.build(Path(a.out), Path(a.run), Path(a.spool), Path(a.page),
                           mission_run=Path(a.mission) if a.mission else None,
                           scene_dir=Path(a.scene) if a.scene else None))
        return 0
    res = prepare(a) if a.cmd == "prepare" else score(a)
    print(yaml.safe_dump({k: v for k, v in res.items() if k not in ("per_view", "command")}, sort_keys=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
