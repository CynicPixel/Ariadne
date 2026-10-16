"""XFeat reference exporter (D15, design PERCEPTION_2.1). OFFLINE tool, run in the pinned ariadne-xfeat-ref image
with --network none and the upstream checkout mounted read-only:

  python -I xfeat_reference.py --upstream /xf --out /out [--image NAME=PATH ...]

1. weights: every inference tensor of upstream's xfeat.pt (loaded with weights_only=True: no pickle code runs)
   -> out/xfeat_weights.atb, the Ariadne tensor blob the C++ port loads;
2. reference: upstream's OWN XFeatModel.forward and XFeat.detectAndCompute on fixed inputs
   -> out/ref_<name>.atb (the network's three outputs + keypoints/scores/descriptors) and out/input_<name>.png;
3. out/manifest.yaml: upstream commit, sha256 of xfeat.pt, of every output, torch version, the settings used.

ATB format (little-endian): b"ATB1", u32 count, then per tensor: u16 name length, name (utf-8), u8 dtype (0 = f32,
1 = i64), u8 ndim, u32 dims[ndim], raw data (C order). Small, dependency-free, read by cpp/perception/features/xfeat.
"""
from __future__ import annotations

import argparse
import hashlib
import struct
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml


def write_atb(path: Path, tensors: dict[str, np.ndarray]) -> None:
    with open(path, "wb") as f:
        f.write(b"ATB1")
        f.write(struct.pack("<I", len(tensors)))
        for name, a in tensors.items():
            a = np.ascontiguousarray(a)
            code = {np.dtype(np.float32): 0, np.dtype(np.int64): 1}[a.dtype]
            nb = name.encode()
            f.write(struct.pack("<H", len(nb)) + nb + struct.pack("<BB", code, a.ndim))
            f.write(struct.pack(f"<{a.ndim}I", *a.shape))
            f.write(a.astype(a.dtype.newbyteorder("<")).tobytes())


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git_head(repo: Path) -> str:
    """The checked-out commit, read from .git directly (the image has no git binary)."""
    head = (repo / ".git" / "HEAD").read_text().strip()
    if not head.startswith("ref: "):
        return head
    ref = head[5:]
    loose = repo / ".git" / ref
    if loose.is_file():
        return loose.read_text().strip()
    for line in (repo / ".git" / "packed-refs").read_text().splitlines():
        if line.endswith(" " + ref):
            return line.split()[0]
    raise SystemExit(f"cannot resolve {ref}")


def synthetic(h: int, w: int) -> np.ndarray:
    """A deterministic test card: blocks, a disc, a ramp and seeded noise (no dataset pixels)."""
    rng = np.random.default_rng(20261016)
    yy, xx = np.mgrid[0:h, 0:w]
    img = 40 + 0.15 * xx + 60 * (((xx // 40) + (yy // 40)) % 2)
    img += 70 * (((xx - w * 0.6) ** 2 + (yy - h * 0.4) ** 2) < (h * 0.18) ** 2)
    img += rng.normal(0, 6, size=(h, w))
    return np.clip(img, 0, 255).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--image", action="append", default=[], help="NAME=PATH (8-bit gray, cropped to /32)")
    ap.add_argument("--top-k", type=int, default=4096)
    a = ap.parse_args()
    up, out = Path(a.upstream), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(up))             # explicit: -I drops the script dir and cwd from sys.path
    from modules.xfeat import XFeat         # noqa: E402  (upstream code, read-only mount)

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    pt = up / "weights" / "xfeat.pt"
    sd = torch.load(pt, map_location="cpu", weights_only=True)
    weights = {k: v.numpy().astype(np.float32) for k, v in sd.items()
               if not k.startswith("fine_matcher") and not k.endswith("num_batches_tracked")}
    write_atb(out / "xfeat_weights.atb", weights)

    xf = XFeat(weights=sd, top_k=a.top_k)
    xf.eval()
    inputs = {"synthetic": synthetic(480, 640)}
    for spec in a.image:
        name, path = spec.split("=", 1)
        g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if g is None:
            raise SystemExit(f"cannot read {path}")
        h, w = (g.shape[0] // 32) * 32, (g.shape[1] // 32) * 32      # crop, so preprocess_tensor never resizes
        y0, x0 = (g.shape[0] - h) // 2, (g.shape[1] - w) // 2
        inputs[name] = np.ascontiguousarray(g[y0:y0 + h, x0:x0 + w])
    files = {}
    for name, g in inputs.items():
        cv2.imwrite(str(out / f"input_{name}.png"), g)
        x = torch.tensor(g, dtype=torch.float32)[None, None]          # raw 0..255, exactly as upstream feeds it
        with torch.inference_mode():
            feats, kpts, heat = xf.net(x)
            d = xf.detectAndCompute(x, top_k=a.top_k)[0]
        ref = {"feats": feats[0].numpy(), "keypoint_logits": kpts[0].numpy(), "heatmap": heat[0].numpy(),
               "keypoints": d["keypoints"].numpy().astype(np.float32), "scores": d["scores"].numpy(),
               "descriptors": d["descriptors"].numpy()}
        write_atb(out / f"ref_{name}.atb", ref)
        files[name] = {"shape": list(g.shape), "keypoints": int(len(d["keypoints"]))}

    commit = git_head(up)
    man = {"upstream": "github.com/verlab/accelerated_features", "license": "Apache-2.0",
           "commit": commit or "UNRECORDED (no .git in the mount)", "xfeat_pt_sha256": sha256(pt),
           "torch": str(torch.__version__), "threads": 1, "deterministic": True, "top_k": a.top_k,
           "detection_threshold": xf.detection_threshold, "inputs": files,
           "sha256": {p.name: sha256(p) for p in sorted(out.glob("*")) if p.suffix in (".atb", ".png")}}
    (out / "manifest.yaml").write_text("# XFeat reference export (D15). Generated.\n" + yaml.safe_dump(man, sort_keys=False))
    print(yaml.safe_dump(files))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
