#!/usr/bin/env python3
"""Demo-video renders of the trained splat on a GPU with gsplat (6.x, M 2026-10-24; the in-region g4dn path, since the
laptop's headless Chrome only has software GL). Same deliverables and camera poses as render_orbit.mjs:

  python3 render_gsplat.py --splat DIR (table.json + chunks/) --out DIR [--w 1920 --h 1080 --seconds 9 --fps 30]

  DIR/orbit/frame_NNNN.png   closed orbit round the mission world's centre at 35 deg elevation
  DIR/stills/still_K.png      4 viewpoints at 60 deg elevation, a pose no ground agent occupied
  DIR/align/agent_K.png, align/merged.png   one overview: each agent's submaps alone, then all
  DIR/poses.json              every image's camera in mission world W (m, z-up): pos, target, fov_deg, agents

Placement is the console's: each chunk (.splat, 32 B/Gaussian, viewer.to_splat's layout) is in its submap's anchor
frame and goes to W by the table row's T_W_anchor (row-major). gsplat renders with RGB colours (SH0, the chunk's
colours), linear scales and sigmoid opacities decoded from the chunk, on a black background.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

LABEL = "trained Gaussian splat, novel viewpoints (rendered)"
REC = np.dtype([("p", "<f4", 3), ("s", "<f4", 3), ("c", "u1", 4), ("r", "u1", 4)])


def quat_wxyz_from_rot(R: np.ndarray) -> np.ndarray:
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return np.array([w, x, y, z])


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:   # a (4,) x b (N, 4), wxyz
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b.T
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], 1)


def load(splat: Path) -> tuple[dict, dict]:
    """Every submap's Gaussians in W, concatenated, with the agent of each; plus the table's anchor bounds."""
    table = json.loads((splat / "table.json").read_text())
    parts, anchors = [], []
    for row in table["submaps"]:
        if not row.get("chunk"):
            continue
        r = np.frombuffer((splat / "chunks" / row["chunk"]).read_bytes(), REC)
        T = np.array(row["T_W_anchor"], float).reshape(4, 4)
        q = (r["r"].astype(np.float64) - 128) / 128
        q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
        parts.append({"means": r["p"].astype(np.float64) @ T[:3, :3].T + T[:3, 3],
                      "quats": quat_mul(quat_wxyz_from_rot(T[:3, :3]), q), "scales": r["s"].astype(np.float64),
                      "rgb": r["c"][:, :3] / 255.0, "alpha": r["c"][:, 3] / 255.0,
                      "agent": np.full(len(r), row["agent"])})
        anchors.append(T[:3, 3])
    g = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    a = np.array(anchors)
    return g, {"min": a.min(0).tolist(), "max": a.max(0).tolist(), "label": table.get("label"), "submaps": len(parts)}


def poses(bounds: dict, seconds: float, fps: int, elev_deg: float = 35.0) -> dict:
    """render_orbit.mjs's camera plan, verbatim."""
    lo, hi = np.array(bounds["min"]), np.array(bounds["max"])
    c = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
    R = 0.75 * max(hi[0] - lo[0], hi[1] - lo[1]) + 3
    look = (c + [0, 0, 0.3]).tolist()
    at = lambda az, el, r: (c + r * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az),
                                              math.sin(el)])).tolist()
    n = round(seconds * fps)
    el = math.radians(elev_deg)
    orbit = [{"file": f"orbit/frame_{i:04d}.png", "pos": at(2 * math.pi * i / n, el, R), "target": look,
              "fov_deg": 50, "agents": None} for i in range(n)]
    stills = [{"file": f"stills/still_{k}.png", "pos": at(math.pi / 4 + k * math.pi / 2, math.pi / 3, 0.8 * R),
               "target": look, "fov_deg": 50, "agents": None} for k in range(4)]
    over = {"pos": at(-math.pi / 2, 1.25, 1.1 * R), "target": look, "fov_deg": 50}
    return {"orbit": orbit, "stills": stills, "over": over}


def viewmat(pos, target) -> np.ndarray:
    """World -> camera, OpenCV/gsplat convention (x right, y down, z forward), world z-up."""
    p, t = np.array(pos, float), np.array(target, float)
    f = (t - p) / np.linalg.norm(t - p)
    r = np.cross(f, [0, 0, 1.0])
    r = r / np.linalg.norm(r) if np.linalg.norm(r) > 1e-9 else np.array([1.0, 0, 0])
    d = np.cross(f, r)
    M = np.eye(4)
    M[:3, :3] = np.stack([r, d, f])
    M[:3, 3] = -M[:3, :3] @ p
    return M


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splat", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--w", type=int, default=1920)
    ap.add_argument("--h", type=int, default=1080)
    ap.add_argument("--seconds", type=float, default=9.0)
    ap.add_argument("--fps", type=int, default=30)
    a = ap.parse_args()
    import cv2
    import torch
    from gsplat import rasterization

    g, b = load(a.splat)
    plan = poses(b, a.seconds, a.fps)
    dev = "cuda"
    T = lambda x: torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32, device=dev)
    G = {k: T(g[k]) for k in ("means", "quats", "scales", "rgb", "alpha")}
    agent = torch.as_tensor(g["agent"], device=dev)
    fy = a.h / 2 / math.tan(math.radians(50) / 2)
    K = T([[fy, 0, a.w / 2], [0, fy, a.h / 2], [0, 0, 1]])[None]

    def shoot(view, agents, file):
        sel = torch.ones_like(agent, dtype=torch.bool) if agents is None else torch.isin(agent, torch.as_tensor(agents, device=dev))
        img, _, _ = rasterization(G["means"][sel], G["quats"][sel], G["scales"][sel], G["alpha"][sel], G["rgb"][sel],
                                  T(viewmat(view["pos"], view["target"]))[None], K, a.w, a.h,
                                  backgrounds=torch.zeros(1, 3, device=dev))
        px = (img[0].clamp(0, 1) * 255).round().byte().cpu().numpy()
        (a.out / file).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(a.out / file), cv2.cvtColor(px, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 6])

    for v in plan["orbit"] + plan["stills"]:
        shoot(v, None, v["file"])
    align = []
    for k in sorted(set(g["agent"].tolist())):
        align.append({"file": f"align/agent_{k}.png", **plan["over"], "agents": [k]})
        shoot(plan["over"], [k], align[-1]["file"])
    align.append({"file": "align/merged.png", **plan["over"], "agents": None})
    shoot(plan["over"], None, "align/merged.png")
    (a.out / "poses.json").write_text(json.dumps({
        "label": LABEL, "renderer": f"gsplat {__import__('gsplat').__version__} rasterization, {torch.cuda.get_device_name(0)}",
        "frame": "mission world W (m, z-up)", "size": [a.w, a.h], "intrinsics": {"fx": fy, "fy": fy, "cx": a.w / 2, "cy": a.h / 2},
        "table_label": b["label"], "submaps": b["submaps"], "gaussians": int(len(g["means"])), "bounds": b,
        "orbit": plan["orbit"], "stills": plan["stills"], "align": align}, indent=1))
    print(f"rendered {len(plan['orbit'])} orbit frames, 4 stills, {len(align)} align images, {len(g['means'])} Gaussians")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
