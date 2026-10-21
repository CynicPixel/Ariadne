"""ALIKED + LightGlue reference exporter (2.1, design PERCEPTION_2.1 §12, SMOKE §4s). OFFLINE tool, run in the session
image ariadne-eigenplaces-ref:feat-2.1 with --network none, the upstream checkout and the weights mounted read-only:

  python -I aliked_lightglue_reference.py --upstream /lg --weights /w --out /out --size 752x480 --k 1024 --k 2048 \
      --image NAME=PATH ... --pair NAME_A,NAME_B ... --upstream-commit SHA --aliked-commit SHA

Upstream: github.com/cvg/LightGlue (Apache-2.0; its lightglue/aliked.py is ALIKED, BSD-3, Zhao et al.) with the
weights it loads: Shiaoming/ALIKED models/aliked-n16.pth and cvg/LightGlue v0.1_arxiv aliked_lightglue.pth. The
configuration is glue-factory's official one (configs/aliked+lightglue-official.yaml): ALIKED-n16, top-K keypoints
(detection_threshold 0), LightGlue with depth_confidence = width_confidence = -1 (all 9 layers, no pruning),
filter_threshold 0.1. No resize: the native image size, as upstream's extract(resize=None).

Why the graph differs from upstream's code (each rewrite is checked against upstream in torch before export, and the
reference tensors are upstream's UNMODIFIED outputs):
- DeformableConv2d: cv::dnn has no DeformConv importer. torchvision.ops.deform_conv2d (DCN v1, no mask) is rewritten
  as: sampling grid (base tap + learned offset, torchvision's layout dy then dx) -> one GridSample (bilinear, zeros,
  align_corners=True, which is torchvision's zero-outside bilinear) -> reshape -> MatMul with the conv weight.
- DKD (top-K mode): nn.Unfold and the per-batch Python loop become gathers of the 25 neighbours of each top-K index
  from a zero-padded score map; the border is zeroed by a constant mask instead of slice assignment.
- SDDH: get_patches' fancy indexing becomes a gather; the einsum "ncp,pcd->nd" becomes one MatMul.
- Keypoints are emitted in cv::ALIKED's convention, n = 2 px / W - 1, so cv::ALIKED (which maps px = (n + 1) / 2 W)
  returns upstream's pixel keypoints (upstream: px = (k + 1) / 2 (W - 1)) exactly. No batch dimension on the outputs,
  as cv::ALIKED reads them ([K,2], [K,128], [K]).
- LightGlue: unflatten(-1, (h, -1, ...)) / repeat_interleave lose the rank when traced with a dynamic keypoint
  count, so SelfBlock, CrossBlock, rotate_half and the positional encoding are respelled with explicit reshapes and
  matmuls (lightglue_export_rewrites); the match SET must stay identical to upstream's, scores within 1e-4.
- LightGlue takes keypoints ALREADY normalised by upstream's normalize_keypoints (shift W/2, H/2; scale max(W,H)/2):
  cv::LightGlueMatcher passes them through unchanged when setPairInfo gets an empty Size(). Outputs matches0 [M,2]
  int64 and mscores0 [M], the names and shapes cv::LightGlueMatcher reads.

Why the stubs: kornia is imported by upstream utils.py (resize, unused at resize=None) and aliked.py (grayscale_to_rgb,
unused: the image is replicated to RGB here); the stub raises if either is ever called. torch.hub's
load_state_dict_from_url is pointed at the mounted weights (same files, sha256 in the manifest); no network.

Writes to --out: aliked_n16_k<K>_<W>x<H>.onnx, lightglue_aliked.onnx, ref_img_<name>_k<K>.atb (input, keypoints px,
keypoints_cv, scores, descriptors), ref_pair_<a>_<b>_k<K>.atb (kpts0n, kpts1n, matches, mscores, and the adaptive
upstream's matches for information), input_<name>.png, manifest.yaml.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import struct
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import yaml


def write_atb(path: Path, tensors: dict[str, np.ndarray]) -> None:
    """The ATB blob of xfeat_reference.py (same format, read by cpp/perception/features/xfeat/atb.hpp)."""
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


def stub_kornia() -> None:
    def never(*_a, **_k):
        raise RuntimeError("kornia stub called: this path must not run at resize=None with an RGB input")

    k = types.ModuleType("kornia")
    k.color = types.SimpleNamespace(grayscale_to_rgb=never)
    k.geometry = types.SimpleNamespace(transform=types.SimpleNamespace(resize=never))
    sys.modules["kornia"] = k
    sys.modules["kornia.color"] = k.color


def import_upstream(root: Path):
    """lightglue.aliked and lightglue.lightglue without the package __init__ (which imports disk/sift/kornia.feature)."""
    pkg = types.ModuleType("lightglue")
    pkg.__path__ = [str(root / "lightglue")]
    sys.modules["lightglue"] = pkg
    return importlib.import_module("lightglue.aliked"), importlib.import_module("lightglue.lightglue")


# ---------------------------------------------------------------- export-only rewrites (checked against upstream)

def deform_conv2d_gs(x, offset, weight, bias, padding):
    """torchvision.ops.deform_conv2d, stride 1, dilation 1, one offset group, no mask, as GridSample + MatMul."""
    # int(): the ALIKED graph has a fixed input size, so every dim is a constant. Traced x.shape would become
    # Shape -> Gather -> Mul int64 arithmetic, which cv::dnn ENGINE_NEW mistypes against float (SMOKE §4s).
    b, c, h, w = (int(v) for v in x.shape)
    o, _, kh, kw = (int(v) for v in weight.shape)
    ho, wo = (int(v) for v in offset.shape[-2:])
    kk = kh * kw
    off = offset.reshape(b, kk, 2, ho, wo)
    a = torch.arange(kh, dtype=x.dtype).repeat_interleave(kw)            # tap row,  k = a * kw + b
    bb = torch.arange(kw, dtype=x.dtype).repeat(kh)                      # tap col
    gy = torch.arange(ho, dtype=x.dtype)[None, :, None] - padding[0] + a[:, None, None]   # [kk, ho, 1]
    gx = torch.arange(wo, dtype=x.dtype)[None, None, :] - padding[1] + bb[:, None, None]  # [kk, 1, wo]
    y = gy[None] + off[:, :, 0]                                          # [b, kk, ho, wo]
    xx = gx[None] + off[:, :, 1]
    grid = torch.stack([2.0 * xx / (w - 1) - 1.0, 2.0 * y / (h - 1) - 1.0], -1).reshape(b, kk * ho, wo, 2)
    s = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=True)  # [b, c, kk*ho, wo]
    s = s.reshape(b, c * kk, ho * wo)                                    # c major, tap minor = weight's layout
    out = torch.matmul(weight.reshape(o, c * kk), s)
    if bias is not None:
        out = out + bias[None, :, None]
    return out.reshape(b, o, ho, wo)


def dcn_forward_gs(self, x):
    """DeformableConv2d.forward with the decomposition (mask=False in ALIKED's configs)."""
    h, w = (int(v) for v in x.shape[2:])
    max_offset = max(h, w) / 4.0
    assert not self.mask
    offset = self.offset_conv(x).clamp(-max_offset, max_offset)
    return deform_conv2d_gs(x, offset, self.regular_conv.weight, self.regular_conv.bias, self.padding)


class AlikedExport(torch.nn.Module):
    """ALIKED-n16 top-K with static shapes: image [1,3,H,W] RGB in [0,1] -> keypoints (cv) [K,2], descriptors
    [K,128], scores [K]."""

    def __init__(self, m, k: int, h: int, w: int):
        super().__init__()
        self.m, self.k, self.h, self.w = m, k, h, w
        r = m.dkd.radius
        mask = torch.ones(1, 1, h, w)
        mask[..., :r, :] = 0
        mask[..., -r:, :] = 0
        mask[..., :, :r] = 0
        mask[..., :, -r:] = 0
        self.register_buffer("border", mask)
        d = torch.arange(-r, r + 1)
        dy, dx = torch.meshgrid(d, d, indexing="ij")                    # Unfold's order: row-major over the patch
        self.register_buffer("nb", (dy.reshape(-1) * (w + 2 * r) + dx.reshape(-1)).long())
        ks = m.desc_head.kernel_size
        py, px = torch.meshgrid(torch.arange(ks), torch.arange(ks), indexing="ij")
        self.register_buffer("patch_dy", py.reshape(-1).long())
        self.register_buffer("patch_dx", px.reshape(-1).long())

    def dkd(self, scores_map):
        m, k, h, w = self.m.dkd, self.k, self.h, self.w
        r = m.radius
        nms = self.m_nms(scores_map) * self.border
        top = torch.topk(nms.reshape(1, -1), k)
        idx = top.indices[0]                                             # [K]
        iy, ix = torch.div(idx, w, rounding_mode="trunc"), idx % w
        padded = F.pad(scores_map, (r, r, r, r)).reshape(-1)             # Unfold pads with zeros
        centre = (iy + r) * (w + 2 * r) + (ix + r)
        patch = padded[centre[:, None] + self.nb[None, :]]               # [K, (2r+1)^2]
        hw_grid = m.hw_grid.to(scores_map)
        max_v = patch.max(dim=1).values[:, None]
        x_exp = ((patch - max_v) / m.temperature).exp()
        xy_res = x_exp @ hw_grid / x_exp.sum(dim=1)[:, None]
        xy = torch.stack([ix, iy], 1).to(scores_map) + xy_res
        wh = torch.tensor([w - 1, h - 1], dtype=scores_map.dtype)
        kn = xy / wh * 2 - 1
        score = F.grid_sample(scores_map, kn.view(1, 1, -1, 2), mode="bilinear", align_corners=True)[0, 0, 0, :]
        return kn, score

    def m_nms(self, s):
        return self.aliked_mod.simple_nms(s, self.m.dkd.radius)

    def sddh(self, x, kn):
        d = self.m.desc_head
        _, c, h, w = (int(v) for v in x.shape)
        wh = torch.tensor([[w - 1, h - 1]], dtype=x.dtype)
        max_offset = max(h, w) / 4.0
        kw_ = (kn / 2 + 0.5) * wh                                        # [K, 2] pixels
        ks = d.kernel_size
        # upstream: corner = long(long(px) - ks/2 + 1), then clamp. For ks = 3 and px >= 0 that is floor(px) - 1
        # clamped at 0 (long(-0.5) = 0). Computed with Floor in float so every Cast below sees an integral value.
        # Observed (SMOKE §4s): with the truncating .long() spelling, cv::dnn ENGINE_NEW gave a different patch on
        # ~75% of keypoints, i.e. as if a fractional part >= 0.5 rounded up. Cast2's CPU path itself truncates
        # (cast2_layer.cpp), so where the rounding happens in that graph is not pinned down.
        assert ks == 3
        corner = torch.floor(kw_) - 1.0
        cx = corner[:, 0].clamp(min=0, max=w - 1 - ks).long()
        cy = corner[:, 1].clamp(min=0, max=h - 1 - ks).long()
        yy = cy[:, None] + self.patch_dy[None]                           # [K, ks*ks]
        xx = cx[:, None] + self.patch_dx[None]
        flat = x[0].reshape(c, h * w)
        patch = flat[:, (yy * w + xx).reshape(-1)].reshape(c, -1, ks, ks).permute(1, 0, 2, 3)  # [K, C, ks, ks]
        n = self.k
        offset = d.offset_conv(patch).clamp(-max_offset, max_offset)
        offset = offset[:, :, 0, 0].view(n, 2, d.n_pos).permute(0, 2, 1)  # [K, n_pos, 2]
        pos = kw_.unsqueeze(1) + offset
        pos = (2.0 * pos / wh[None] - 1).reshape(1, n * d.n_pos, 1, 2)
        feats = F.grid_sample(x, pos, mode="bilinear", align_corners=True)
        feats = feats.reshape(c, n, d.n_pos, 1).permute(1, 0, 2, 3)
        feats = torch.selu(d.sf_conv(feats)).squeeze(-1)                 # [K, C, n_pos]
        aggw = d.agg_weights.reshape(d.n_pos * c, -1)                    # [p, c, d] -> [(p c), d]
        descs = torch.matmul(feats.permute(0, 2, 1).reshape(n, d.n_pos * c), aggw)
        return F.normalize(descs, p=2.0, dim=1)

    def forward(self, image):
        fmap, smap = self.m.extract_dense_map(image)
        kn, score = self.dkd(smap)
        desc = self.sddh(fmap, kn)
        w = float(self.w)
        px = kn[:, 0:1].new_tensor([self.w - 1, self.h - 1]) * (kn + 1) / 2.0
        kcv = 2.0 * px / px.new_tensor([w, float(self.h)]) - 1.0         # cv::ALIKED's convention
        return kcv, desc, score


class lightglue_export_rewrites:
    """Export-only LightGlue rewrites: unflatten(-1, (h, -1, ...)) and repeat_interleave lose the rank under tracing
    with a dynamic keypoint count (torch.onnx's transpose symbolic then fails), so the same reshapes are spelled with
    explicit (b, n, heads, -1, ...) dims. CPU, mask-free paths only (the ones LightGlueExport runs)."""

    def __init__(self, lg_mod):
        self.m = lg_mod

    def __enter__(self):
        m = self.m
        self.saved = (m.rotate_half, m.SelfBlock.forward, m.CrossBlock.forward,
                      m.LearnableFourierPositionalEncoding.forward, m.MatchAssignment.forward)

        def assign_forward(self, desc0, desc1):
            # einsum "bmd,bnd->bmn" as a MatMul: cv::dnn's Einsum layer keeps the first call's shapes (SMOKE §4s)
            mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
            d = mdesc0.shape[-1]
            mdesc0, mdesc1 = mdesc0 / d**0.25, mdesc1 / d**0.25
            sim = torch.matmul(mdesc0, mdesc1.transpose(-1, -2))
            return m.sigmoid_log_double_softmax(sim, self.matchability(desc0), self.matchability(desc1)), sim

        def rotate_half(x):
            x = x.reshape(*x.shape[:-1], -1, 2)
            x1, x2 = x[..., 0], x[..., 1]
            return torch.stack((-x2, x1), dim=-1).reshape(*x.shape[:-2], -1)

        def posenc_forward(self, x):
            projected = self.Wr(x)                                       # [b, n, F/2]
            e = torch.stack([torch.cos(projected), torch.sin(projected)], 0).unsqueeze(-3)   # [2, b, 1, n, F/2]
            return torch.stack([e, e], -1).reshape(*e.shape[:-1], -1)    # repeat_interleave(2, dim=-1)

        def self_forward(self, x, encoding, mask=None):
            assert mask is None
            b, n, _ = x.shape
            qkv = self.Wqkv(x).reshape(b, n, self.num_heads, -1, 3).transpose(1, 2)
            q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
            q = m.apply_cached_rotary_emb(encoding, q)
            k = m.apply_cached_rotary_emb(encoding, k)
            context = self.inner_attn(q, k, v, mask=None)
            message = self.out_proj(context.transpose(1, 2).reshape(b, n, -1))
            return x + self.ffn(torch.cat([x, message], -1))

        def cross_forward(self, x0, x1, mask=None):
            assert mask is None
            b, n0, _ = x0.shape
            n1 = x1.shape[1]
            qk0 = self.to_qk(x0).reshape(b, n0, self.heads, -1).transpose(1, 2)
            qk1 = self.to_qk(x1).reshape(b, n1, self.heads, -1).transpose(1, 2)
            v0 = self.to_v(x0).reshape(b, n0, self.heads, -1).transpose(1, 2)
            v1 = self.to_v(x1).reshape(b, n1, self.heads, -1).transpose(1, 2)
            qk0, qk1 = qk0 * self.scale**0.5, qk1 * self.scale**0.5
            sim = torch.matmul(qk0, qk1.transpose(-1, -2))               # einsum "bhid, bhjd -> bhij"
            attn01 = F.softmax(sim, dim=-1)
            attn10 = F.softmax(sim.transpose(-2, -1).contiguous(), dim=-1)
            m0 = torch.matmul(attn01, v1)                                # "bhij, bhjd -> bhid"
            m1 = torch.matmul(attn10, v0)                                # "bhji, bhjd -> bhid" with attn10^T^T
            m0 = self.to_out(m0.transpose(1, 2).reshape(b, n0, -1))
            m1 = self.to_out(m1.transpose(1, 2).reshape(b, n1, -1))
            return x0 + self.ffn(torch.cat([x0, m0], -1)), x1 + self.ffn(torch.cat([x1, m1], -1))

        m.rotate_half = rotate_half
        m.SelfBlock.forward = self_forward
        m.CrossBlock.forward = cross_forward
        m.LearnableFourierPositionalEncoding.forward = posenc_forward
        m.MatchAssignment.forward = assign_forward
        return self

    def __exit__(self, *exc):
        m = self.m
        (m.rotate_half, m.SelfBlock.forward, m.CrossBlock.forward,
         m.LearnableFourierPositionalEncoding.forward, m.MatchAssignment.forward) = self.saved
        return False


class LightGlueExport(torch.nn.Module):
    """LightGlue (aliked weights), all layers, upstream's filter: normalised kpts -> matches0 [M,2], mscores0 [M]."""

    def __init__(self, lg, lg_mod):
        super().__init__()
        self.lg, self.filter_matches = lg, lg_mod.filter_matches

    def forward(self, kpts0, kpts1, desc0, desc1):
        lg = self.lg
        d0, d1 = lg.input_proj(desc0), lg.input_proj(desc1)
        e0, e1 = lg.posenc(kpts0), lg.posenc(kpts1)
        for i in range(lg.conf.n_layers):
            d0, d1 = lg.transformers[i](d0, d1, e0, e1)
        scores, _ = lg.log_assignment[lg.conf.n_layers - 1](d0, d1)
        m0, _, ms0, _ = self.filter_matches(scores, lg.conf.filter_threshold)
        i0 = torch.nonzero(m0[0] > -1)[:, 0]
        return torch.stack([i0, m0[0][i0]], -1), ms0[0][i0]


def set_compare(px_r, d_r, s_r, px_e, d_e, s_e, tol_px: float = 1e-3) -> dict:
    """Keypoint SETS, not index order: a 1e-6 change of the score map (any other engine's float error) flips
    near-tied NMS maxima (scores == maxpool(scores)) and top-K order. Each upstream keypoint is paired with the
    nearest exported one; matched = within tol_px. Descriptor and score errors are over the matched pairs."""
    d2 = ((px_r[:, None, :] - px_e[None, :, :]) ** 2).sum(-1)
    j = d2.argmin(1)
    ok = np.sqrt(d2[np.arange(len(px_r)), j]) <= tol_px
    return {"matched_fraction": float(ok.mean()), "n": int(len(px_r)),
            "max_px": float(np.sqrt(d2[np.arange(len(px_r)), j][ok]).max()) if ok.any() else float("nan"),
            "max_desc": float(np.abs(d_r[ok] - d_e[j[ok]]).max()) if ok.any() else float("nan"),
            "max_score": float(np.abs(s_r[ok] - s_e[j[ok]]).max()) if ok.any() else float("nan")}


def normalize_np(px: np.ndarray, w: int, h: int) -> np.ndarray:
    """upstream normalize_keypoints with size = (w, h)."""
    shift = np.array([w / 2, h / 2], np.float32)
    return ((px - shift) / np.float32(max(w, h) / 2)).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", required=True, type=Path)
    ap.add_argument("--weights", required=True, type=Path, help="dir with aliked-n16.pth and aliked_lightglue.pth")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--size", default="752x480")
    ap.add_argument("--k", action="append", type=int, required=True)
    ap.add_argument("--image", action="append", default=[], help="NAME=PATH (gray)")
    ap.add_argument("--pair", action="append", default=[], help="NAME_A,NAME_B")
    ap.add_argument("--upstream-commit", required=True)
    ap.add_argument("--aliked-commit", required=True)
    a = ap.parse_args()

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    stub_kornia()
    wfiles = {"aliked-n16.pth": a.weights / "aliked-n16.pth", "aliked_lightglue.pth": a.weights / "aliked_lightglue.pth"}

    def local_weights(url, *args, file_name=None, **kw):
        name = url.rsplit("/", 1)[-1]
        return torch.load(wfiles[name], map_location="cpu", weights_only=True)

    torch.hub.load_state_dict_from_url = local_weights
    aliked_mod, lg_mod = import_upstream(a.upstream)
    w, h = (int(v) for v in a.size.lower().split("x"))
    a.out.mkdir(parents=True, exist_ok=True)

    lg = lg_mod.LightGlue(features="aliked", depth_confidence=-1, width_confidence=-1, filter_threshold=0.1,
                          flash=False).eval()
    lg_adaptive = lg_mod.LightGlue(features="aliked", flash=False).eval()   # upstream defaults, for information

    imgs = {}
    for spec in a.image:
        name, path = spec.split("=", 1)
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None or gray.shape != (h, w):
            raise SystemExit(f"cannot read {path} at {w}x{h}")
        cv2.imwrite(str(a.out / f"input_{name}.png"), gray)
        f = gray.astype(np.float32) * np.float32(1.0 / 255.0)              # cv::dnn::blobFromImage's scaling
        imgs[name] = np.stack([f, f, f], 0)[None].astype(np.float32)       # gray -> RGB, NCHW

    checks = {}
    orig = aliked_mod.DeformableConv2d.forward
    for k in a.k:
        model = aliked_mod.ALIKED(model_name="aliked-n16", max_num_keypoints=k, detection_threshold=0.0,
                                  nms_radius=2).eval()
        exp = AlikedExport(model, k, h, w).eval()
        exp.aliked_mod = aliked_mod
        feats = {}
        for name, x in imgs.items():
            t = torch.from_numpy(x)
            with torch.no_grad():
                ref = model({"image": t})                                  # UNMODIFIED upstream (torchvision DCN)
                aliked_mod.DeformableConv2d.forward = dcn_forward_gs
                try:
                    kcv, desc, score = exp(t)
                finally:
                    aliked_mod.DeformableConv2d.forward = orig
            px = ref["keypoints"][0].numpy().astype(np.float32)
            px_exp = ((kcv.numpy() + 1) / 2 * np.array([w, h], np.float32)).astype(np.float32)
            c = set_compare(px, ref["descriptors"][0].numpy(), ref["keypoint_scores"][0].numpy(),
                            px_exp, desc.numpy(), score.numpy())
            checks[f"{name}_k{k}"] = c
            print(f"{name} k={k}: rewrite vs upstream: {c}")
            if c["matched_fraction"] < 0.99 or c["max_desc"] > 1e-4 or c["max_score"] > 1e-4:
                raise SystemExit("export rewrite does not reproduce upstream")
            feats[name] = (px, ref["descriptors"][0].numpy().astype(np.float32))
            write_atb(a.out / f"ref_img_{name}_k{k}.atb", {
                "input": x, "keypoints": px, "keypoints_cv": kcv.numpy().astype(np.float32),
                "descriptors": feats[name][1], "scores": ref["keypoint_scores"][0].numpy().astype(np.float32)})

        aliked_mod.DeformableConv2d.forward = dcn_forward_gs
        try:
            torch.onnx.export(exp, torch.from_numpy(next(iter(imgs.values()))), str(a.out / f"aliked_n16_k{k}_{w}x{h}.onnx"),
                              opset_version=17, input_names=["image"],
                              output_names=["keypoints", "descriptors", "scores"], do_constant_folding=True,
                              dynamo=False)
        finally:
            aliked_mod.DeformableConv2d.forward = orig

        for spec in a.pair:
            na, nb = spec.split(",")
            (p0, d0), (p1, d1) = feats[na], feats[nb]
            k0n, k1n = normalize_np(p0, w, h), normalize_np(p1, w, h)
            size = torch.tensor([[w, h]], dtype=torch.float32)
            with torch.no_grad():
                data = {"image0": {"keypoints": torch.from_numpy(p0)[None], "descriptors": torch.from_numpy(d0)[None],
                                   "image_size": size},
                        "image1": {"keypoints": torch.from_numpy(p1)[None], "descriptors": torch.from_numpy(d1)[None],
                                   "image_size": size}}
                r = lg(data)                                               # UNMODIFIED upstream
                ra = lg_adaptive(data)
                with lightglue_export_rewrites(lg_mod):
                    em, es = LightGlueExport(lg, lg_mod)(torch.from_numpy(k0n)[None], torch.from_numpy(k1n)[None],
                                                         torch.from_numpy(d0)[None], torch.from_numpy(d1)[None])
            mt, sc = r["matches"][0].numpy().astype(np.int64), r["scores"][0].numpy().astype(np.float32)
            same = em.numpy().shape == mt.shape and bool((em.numpy() == mt).all())
            dsc = float(np.abs(es.numpy() - sc).max()) if same and len(sc) else float("nan")
            ma = {tuple(x) for x in ra["matches"][0].numpy().tolist()}
            inter = len(ma & {tuple(x) for x in mt.tolist()})
            print(f"pair {na},{nb} k={k}: {len(mt)} matches; rewrite identical {same}, |score| {dsc:.2e}; "
                  f"adaptive upstream {len(ma)} matches, {inter} shared, stopped at layer {int(ra['stop'])}")
            if not same or dsc > 1e-4:
                raise SystemExit("LightGlue export wrapper does not reproduce upstream")
            checks[f"pair_{na}_{nb}_k{k}"] = {"matches": int(len(mt)), "adaptive_matches": len(ma),
                                              "adaptive_shared": inter, "adaptive_stop": int(ra["stop"])}
            write_atb(a.out / f"ref_pair_{na}_{nb}_k{k}.atb", {
                "kpts0n": k0n, "kpts1n": k1n, "desc0": d0, "desc1": d1, "matches": mt, "mscores": sc})

    n0 = max(a.k)
    dummy = (torch.zeros(1, n0, 2), torch.zeros(1, n0, 2), torch.zeros(1, n0, 128), torch.zeros(1, n0, 128))
    with lightglue_export_rewrites(lg_mod):
        torch.onnx.export(LightGlueExport(lg, lg_mod), dummy, str(a.out / "lightglue_aliked.onnx"), opset_version=17,
                          input_names=["kpts0", "kpts1", "desc0", "desc1"], output_names=["matches0", "mscores0"],
                          dynamic_axes={"kpts0": {1: "num_keypoints0"}, "kpts1": {1: "num_keypoints1"},
                                        "desc0": {1: "num_keypoints0"}, "desc1": {1: "num_keypoints1"},
                                        "matches0": {0: "num_matches0"}, "mscores0": {0: "num_matches0"}},
                          do_constant_folding=True, dynamo=False)

    manifest = {
        "upstream": "github.com/cvg/LightGlue (Apache-2.0); lightglue/aliked.py = ALIKED (BSD-3, Zhao Xiaoming)",
        "commit": a.upstream_commit,
        "aliked_weights": {"repo": "github.com/Shiaoming/ALIKED", "commit": a.aliked_commit, "file": "models/aliked-n16.pth",
                           "sha256": sha256(wfiles["aliked-n16.pth"]), "license": "BSD-3-Clause"},
        "lightglue_weights": {"release": "cvg/LightGlue v0.1_arxiv", "file": "aliked_lightglue.pth",
                              "sha256": sha256(wfiles["aliked_lightglue.pth"]), "license": "Apache-2.0"},
        "config": "glue-factory configs/aliked+lightglue-official.yaml: aliked-n16, top-K (detection_threshold 0), "
                  "nms_radius 2; LightGlue depth_confidence -1, width_confidence -1, filter_threshold 0.1",
        "size": [w, h], "k": a.k, "resize": None,
        "preprocessing": "gray -> x * (1/255) float32 -> replicate to RGB, NCHW (= cv::dnn::blobFromImage(1/255, swapRB))",
        "keypoints_output": "cv::ALIKED convention n = 2 px / W - 1 (px = upstream (k + 1) / 2 (W - 1))",
        "lightglue_input": "keypoints normalised as upstream normalize_keypoints(size=(W,H))",
        "torch": str(torch.__version__), "torchvision": str(torchvision.__version__), "opencv_python": cv2.__version__,
        "threads": 1, "deterministic": True, "opset": 17,
        "rewrite_checks": checks,
        "sha256": {p.name: sha256(p) for p in sorted(a.out.iterdir()) if p.name != "manifest.yaml"},
    }
    (a.out / "manifest.yaml").write_text("# ALIKED + LightGlue reference export (PERCEPTION_2.1 §12). Generated.\n"
                                         + yaml.safe_dump(manifest, sort_keys=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
