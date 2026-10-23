"""The generated world's ground-truth mesh for the splat page's second viewport (6.4; user decision 2026-10-23, D199).

worldgen writes one OBJ per surface (meshes/<kind>_<id>.obj, world frame W, z up, metres). This merges them into one
OBJ for the browser: v / vt / vn indices are offset per file, and each file becomes a group named by its kind
(wall, floor, ceiling, prop, board), so the page can hide the ceiling and look in from above.

It is GROUND TRUTH from the generator, shown for setup and comparison, labelled "generated world (ground truth)",
and never blended with the reconstruction.
"""
from __future__ import annotations

import re
from pathlib import Path

KINDS = ("wall", "floor", "ceiling", "prop", "board")


def merged_obj(scene_dir: str | Path, include_ceiling: bool = False) -> str:
    meshes = sorted(Path(scene_dir, "meshes").glob("*.obj"))
    if not meshes:
        raise FileNotFoundError(f"{scene_dir}/meshes has no OBJ files")
    out = ["# Ariadne generated world (ground truth): worldgen meshes merged, world frame W, z up, metres"]
    nv = nt = nn = 0
    for p in meshes:
        kind = re.sub(r"_[0-9a-f]+$", "", p.stem)
        if kind not in KINDS:
            raise ValueError(f"{p.name}: unknown surface kind {kind!r}")
        if kind == "ceiling" and not include_ceiling:
            continue
        out.append(f"g {kind}")
        dv = dt = dn = 0
        for line in p.read_text().splitlines():
            tag = line.split(" ", 1)[0]
            if tag == "v":
                dv += 1
                out.append(line)
            elif tag == "vt":
                dt += 1
                out.append(line)
            elif tag == "vn":
                dn += 1
                out.append(line)
            elif tag == "f":
                refs = []
                for r in line.split()[1:]:
                    parts = r.split("/")
                    parts[0] = str(int(parts[0]) + nv)
                    if len(parts) > 1 and parts[1]:
                        parts[1] = str(int(parts[1]) + nt)
                    if len(parts) > 2 and parts[2]:
                        parts[2] = str(int(parts[2]) + nn)
                    refs.append("/".join(parts))
                out.append("f " + " ".join(refs))
        nv, nt, nn = nv + dv, nt + dt, nn + dn
    return "\n".join(out) + "\n"
