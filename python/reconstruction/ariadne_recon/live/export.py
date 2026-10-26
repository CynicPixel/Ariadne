"""Submap chunks for the console (reconstruction.md §1/§3, D190): one .splat per submap, in its ANCHOR frame, plus a table of
placements. The browser renderer (Spark, per-object transform) draws chunk i at table[i].T_W_anchor, so re-anchoring
re-sends a 4x4, never the chunk.

.splat record (antimatter15 layout, 32 B/Gaussian, SH0): position f32x3, scale f32x3 (linear), RGBA u8x4, rotation
u8x4 (wxyz * 128 + 128). The writer is viewer.to_splat (1.7), called with the identity so positions stay in the
anchor frame.
"""
from __future__ import annotations

import json

import numpy as np

from ..viewer import to_splat

SPLAT_BYTES = 32


def chunk_bytes(params: dict) -> bytes:
    """params: means (N,3) anchor frame, scales (N,3) log, quats (N,4) wxyz, opacities (N,) logit, sh0 (N,1,3)."""
    b = to_splat(params, np.eye(4))
    if len(b) != SPLAT_BYTES * len(params["means"]):
        raise AssertionError("splat record size")
    return b


def table(submaps, T_W_O: dict[int, np.ndarray], counts: dict[int, int], label: str) -> str:
    """JSON the /splat panel reads: per submap its placement in W, pose version, size, and what it shows."""
    rows = []
    for sm in submaps:
        rows.append({"sid": sm.sid, "agent": sm.agent, "anchor_kf": sm.anchor_kf, "closed": sm.closed,
                     "pose_version": sm.pose_version, "gaussians": int(counts.get(sm.sid, 0)),
                     "bytes": SPLAT_BYTES * int(counts.get(sm.sid, 0)),
                     "T_W_anchor": [float(x) for x in sm.placement(T_W_O[sm.agent]).ravel()]})
    return json.dumps({"label": label, "submaps": rows})
