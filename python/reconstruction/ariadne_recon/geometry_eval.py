"""Geometry metrics of the fused cloud against the scene's GT meshes (§30.2 mapping:
chamfer_m, completeness). Evaluation only: GT never reaches the service.

GT surface: every scene mesh (world frame, triangles) sampled uniformly by area at
about one point per `spacing`² (seeded, so reproducible). The cloud is mapped into
the world with T_W_O0 from the evaluation record (Umeyama of the merged trajectory).
accuracy = mean cloud->GT distance; gt_to_cloud = mean GT->cloud distance;
chamfer = their mean; completeness = fraction of GT samples within tau of the cloud.
The agents see part of the maze only, so completeness and gt_to_cloud measure
coverage as much as quality: reported as measured.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def load_obj_triangles(path: Path) -> np.ndarray:
    """(T, 3, 3) triangles from an OBJ (v / f lines; polygons fan-triangulated)."""
    verts, tris = [], []
    for line in path.read_text().splitlines():
        if line.startswith("v "):
            verts.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            idx = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:]]
            for k in range(1, len(idx) - 1):
                tris.append((idx[0], idx[k], idx[k + 1]))
    if not tris:
        raise ValueError(f"{path}: no faces")
    v = np.asarray(verts, dtype=np.float64)
    return v[np.asarray(tris)]


def sample_surface(tris: np.ndarray, spacing: float, seed: int) -> np.ndarray:
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    counts = np.maximum(1, np.ceil(area / spacing ** 2).astype(np.int64))
    rng = np.random.default_rng(seed)
    tri_idx = np.repeat(np.arange(len(tris)), counts)
    r1, r2 = rng.random(len(tri_idx)), rng.random(len(tri_idx))
    s = np.sqrt(r1)
    w0, w1, w2 = 1 - s, s * (1 - r2), s * r2
    return w0[:, None] * a[tri_idx] + w1[:, None] * b[tri_idx] + w2[:, None] * c[tri_idx]


def scene_gt_points(scene_dir: Path, spacing: float, seed: int = 0) -> np.ndarray:
    meshes = sorted((scene_dir / "meshes").glob("*.obj"))
    if not meshes:
        raise ValueError(f"{scene_dir}: no meshes")
    return sample_surface(np.concatenate([load_obj_triangles(m) for m in meshes]), spacing, seed)


def evaluate(cloud_world: np.ndarray, gt: np.ndarray, tau: float) -> dict:
    d_cg, _ = cKDTree(gt).query(cloud_world, k=1)
    d_gc, _ = cKDTree(cloud_world).query(gt, k=1)
    return {"accuracy_m": float(d_cg.mean()), "accuracy_median_m": float(np.median(d_cg)),
            "gt_to_cloud_m": float(d_gc.mean()), "chamfer_m": float(0.5 * (d_cg.mean() + d_gc.mean())),
            "completeness": float((d_gc <= tau).mean()), "gt_samples": int(len(gt)),
            "cloud_points": int(len(cloud_world))}
