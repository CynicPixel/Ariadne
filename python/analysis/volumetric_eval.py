"""4.1 volumetric map vs GT (VOLUMETRIC_4.1 §6, D110): the three map metrics of mission_eval.py, on a slab.png.

  python3 -I volumetric_eval.py --scene SCENE_DIR --slab NEW/slab.png --baseline OLD/slab.png --out DIR

coverage = GT-free cells the map observed free / GT-free cells; free precision = map-free cells that are GT-free;
occupied precision = map-occupied cells that are GT structure. Same definitions and the same GT layer (`free` of
scene_gt.npz) as mission_eval.py, so the two rows compare directly. Also: the distance (cells) from each map-occupied,
GT-free cell to GT structure (D45's occupied-precision cause), by brute force over the structure cells.
Reads PNGs with numpy's own zlib decoder (no cv2 dependency).
"""
from __future__ import annotations

import argparse
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dataset_tools"))
from worldgen.raster import TRAV_EPS_M, _edt_to  # noqa: E402  (the GT's clearance convention, one implementation)

UNKNOWN, FREE, OCC = 0, 1, 2


def read_png_u8(p: Path) -> np.ndarray:
    """Minimal 8-bit grayscale PNG reader (what cv::imwrite writes for CV_8U); filters 0-4."""
    b = p.read_bytes()
    assert b[:8] == b"\x89PNG\r\n\x1a\n", p
    i, idat, w, h = 8, b"", 0, 0
    while i < len(b):
        n, t = struct.unpack(">I4s", b[i:i + 8])
        d = b[i + 8:i + 8 + n]
        if t == b"IHDR":
            w, h, depth, ctype = struct.unpack(">IIBB", d[:10])
            assert depth == 8 and ctype == 0, (depth, ctype)
        elif t == b"IDAT":
            idat += d
        i += 12 + n
    raw = zlib.decompress(idat)
    out = np.zeros((h, w), np.uint8)
    prev = np.zeros(w, np.int32)
    for r in range(h):
        f, line = raw[r * (w + 1)], np.frombuffer(raw[r * (w + 1) + 1:(r + 1) * (w + 1)], np.uint8).astype(np.int32)
        cur = np.zeros(w, np.int32)
        for c in range(w):
            a = cur[c - 1] if c else 0
            up, ul = prev[c], (prev[c - 1] if c else 0)
            if f == 0: x = line[c]
            elif f == 1: x = line[c] + a
            elif f == 2: x = line[c] + up
            elif f == 3: x = line[c] + (a + up) // 2
            else:
                pa, pb, pc = abs(up - ul), abs(a - ul), abs(a + up - 2 * ul)
                x = line[c] + (a if pa <= pb and pa <= pc else (up if pb <= pc else ul))
            cur[c] = x & 0xFF
        out[r], prev = cur, cur
    return out


def metrics(slab: np.ndarray, gtfree: np.ndarray) -> dict:
    occ_false = (slab == OCC) & gtfree
    struct_cells = np.argwhere(~gtfree)
    dist = [int(np.min(np.max(np.abs(struct_cells - p), axis=1))) for p in np.argwhere(occ_false)]  # Chebyshev, cells
    hist = {k: int(sum(1 for d in dist if d == k)) for k in (1, 2, 3)}
    hist[">3"] = int(sum(1 for d in dist if d > 3))
    return {"coverage": float(((slab == FREE) & gtfree).sum() / gtfree.sum()),
            "free_precision": float(((slab == FREE) & gtfree).sum() / max((slab == FREE).sum(), 1)),
            "occupied_precision": float(((slab == OCC) & ~gtfree).sum() / max((slab == OCC).sum(), 1)),
            "false_occupied_cells": int(occ_false.sum()), "false_occupied_distance_cells": hist}


def read_png_auto(p: Path) -> np.ndarray:
    return read_png_u8(p)


def edge_clearance_ok(blocked: np.ndarray, rho: float, g: float) -> np.ndarray:
    """1.8's 2D rule (planning::EdgeClearance + Traversable), via worldgen's own GT convention (raster._edt_to:
    centre to the nearest point of a blocked square), imported, not re-derived."""
    cap = int(np.ceil((rho + 2 * g) / g))
    return _edt_to(blocked, cap) * g >= rho - TRAV_EPS_M


def connected(trav: np.ndarray, a, b) -> bool:
    """8-connected flood from a to b on trav (row, col)."""
    if not (trav[a] and trav[b]):
        return False
    seen = np.zeros_like(trav, dtype=bool)
    stack = [a]
    seen[a] = True
    while stack:
        y, x = stack.pop()
        if (y, x) == b:
            return True
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                yy, xx = y + dy, x + dx
                if 0 <= yy < trav.shape[0] and 0 <= xx < trav.shape[1] and trav[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    stack.append((yy, xx))
    return False


def trav_metrics(trav: np.ndarray, gt: np.ndarray, starts=(), target=None) -> dict:
    tp = int((trav & gt).sum())
    return {"claimed": int(trav.sum()), "precision": tp / max(int(trav.sum()), 1), "recall": tp / max(int(gt.sum()), 1),
            "false_claims": int((trav & ~gt).sum()),
            "start_to_target": [connected(trav, tuple(s), tuple(target)) for s in starts] if target is not None else []}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--slab", type=Path, required=True)
    ap.add_argument("--baseline", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rho-responder", type=float, default=0.40)
    a = ap.parse_args()
    gt = np.load(a.scene / "scene_gt.npz")
    gtfree = gt["free"].astype(bool)
    rows = {"phase4_tsdf": metrics(read_png_u8(a.slab), gtfree)}
    if a.baseline:
        rows["placeholder_1.8"] = metrics(read_png_u8(a.baseline), gtfree)
    # 4.2: responder traversability vs GT trav_responder (invariant 4: precision is what a claim must protect).
    trav_rows = {}
    gtr, st, tg = gt["trav_responder"].astype(bool), [tuple(x) for x in gt["start_cells"]], tuple(gt["target_cell"])
    for name, f in (("phase4_hybrid (the product)", "trav_responder.png"), ("phase4_strict_3d", "trav_responder_strict3d.png")):
        if (a.slab.parent / f).exists():
            trav_rows[name] = trav_metrics(read_png_u8(a.slab.parent / f).astype(bool), gtr, st, tg)
    if a.baseline:
        base = read_png_u8(a.baseline)
        b = (base == FREE) & edge_clearance_ok(base != FREE, a.rho_responder, float(gt["grid_m"]))
        trav_rows["placeholder_1.8"] = trav_metrics(b, gtr, st, tg)
    for k, m in trav_rows.items():
        print(f"{k:28s} trav_responder: claimed {m['claimed']}  precision {m['precision']:.4f}  recall {m['recall']:.3f}  "
              f"false claims {m['false_claims']}  start->target {m['start_to_target']}")
    a.out.mkdir(parents=True, exist_ok=True)
    y = ["# volumetric_eval (VOLUMETRIC_4.1 §6). Generated.", f"slab: {a.slab}", f"baseline: {a.baseline}", "rows:"]
    for k, m in rows.items():
        print(f"{k:16s} coverage {m['coverage']:.3f}  free precision {m['free_precision']:.3f}  occupied precision "
              f"{m['occupied_precision']:.3f}  false-occupied {m['false_occupied_cells']} by distance {m['false_occupied_distance_cells']}")
        y.append(f"  {k}: {m}")
    y.append("trav_responder_vs_gt:")
    for k, m in trav_rows.items():
        y.append(f"  {k}: {m}")
    (a.out / "map_vs_gt.yaml").write_text("\n".join(y) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
