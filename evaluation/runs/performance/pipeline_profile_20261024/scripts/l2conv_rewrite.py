#!/usr/bin/env python3
"""Graph-level check of ALIKED's dense-map normalisation (M, 2026-10-24: "worth a graph-level check").

In our ALIKED export the dense 128-channel feature map is normalised as upstream's F.normalize(x, dim=1):
ReduceL2(axes=1) -> Clip(eps) -> Expand(to x's shape) -> Div. cv::dnn ENGINE_NEW runs it as ONE serial Normalize
(the bench's per-op profile on c7i: ~580 ms at 1 thread and ~535 ms at 8-16, 64 % of the forward there), and the Expand
materialises a 128 x H x W float copy. This rewrites ONLY that site, exactly in arithmetic, as
    n = Sqrt(Conv(Mul(x, x), W = ones[1, C, 1, 1]))  -> Clip(eps) (unchanged) -> Div(x, n) by broadcasting
(the sum of squares over channels as a 1x1 convolution: the parallel conv path), dropping the Expand. The descriptor
normalisation after sampling (/ReduceL2_1, 1024 x 128) is left as it is. Not a model change in maths; the summation
order differs, so outputs may differ by float rounding - measured by --check, never assumed.

  python -I l2conv_rewrite.py --in aliked_n16_k1024_1280x720.onnx --out aliked_n16_k1024_1280x720_l2conv.onnx
      [--check ref_img_sim_a0_f200_k1024.atb]     (cv2.dnn ENGINE_NEW: rewritten vs original vs upstream torch)
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def read_atb(p: Path) -> dict[str, np.ndarray]:
    b = p.read_bytes()
    (n,), o, out = struct.unpack_from("<I", b, 4), 8, {}
    for _ in range(n):
        (ln,) = struct.unpack_from("<H", b, o)
        name = b[o + 2:o + 2 + ln].decode(); o += 2 + ln
        code, nd = struct.unpack_from("<BB", b, o); o += 2
        shape = struct.unpack_from(f"<{nd}I", b, o); o += 4 * nd
        dt = {0: np.float32, 1: np.int64}[code]
        cnt = int(np.prod(shape)) if nd else 1
        out[name] = np.frombuffer(b, dt, cnt, o).reshape(shape); o += cnt * np.dtype(dt).itemsize
    return out


def rewrite(m: onnx.ModelProto) -> onnx.ModelProto:
    g = m.graph
    by_out = {o: n for n in g.node for o in n.output}
    cons = {}
    for n in g.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)
    red = next(n for n in g.node if n.op_type == "ReduceL2" and n.name == "/ReduceL2")
    x = red.input[0]
    clip = next(c for c in cons[red.output[0]] if c.op_type == "Clip")
    expand = next(c for c in cons[clip.output[0]] if c.op_type == "Expand")
    div = next(c for c in cons[x] if c.op_type == "Div" and expand.output[0] in c.input)
    C = next(int(v.type.tensor_type.shape.dim[1].dim_value) for v in onnx.shape_inference.infer_shapes(m).graph.value_info
             if v.name == x)
    w = numpy_helper.from_array(np.ones((1, C, 1, 1), np.float32), "a2_l2conv_ones")
    g.initializer.append(w)
    new = [helper.make_node("Mul", [x, x], ["a2_l2conv_sq"], name="/a2_l2conv/Mul"),
           helper.make_node("Conv", ["a2_l2conv_sq", "a2_l2conv_ones"], ["a2_l2conv_sumsq"], name="/a2_l2conv/Conv",
                            kernel_shape=[1, 1]),
           helper.make_node("Sqrt", ["a2_l2conv_sumsq"], [red.output[0]], name="/a2_l2conv/Sqrt")]
    i = list(g.node).index(red)
    g.node.remove(red)
    for k, n in enumerate(new):
        g.node.insert(i + k, n)
    div.input[list(div.input).index(expand.output[0])] = clip.output[0]   # broadcast [1,1,H,W] over [1,C,H,W]
    if not cons.get(expand.output[0], [None])[1:]:
        g.node.remove(expand)
    onnx.checker.check_model(m)
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--check", type=Path)
    a = ap.parse_args()
    m = rewrite(onnx.load(str(a.inp)))
    onnx.save(m, str(a.out))
    print(f"rewrote /ReduceL2 -> Mul+Conv(ones)+Sqrt, Expand dropped: {a.out}")
    if a.check:
        import cv2
        ref = read_atb(a.check)
        x = np.ascontiguousarray(ref["input"], np.float32)
        outs = {}
        for tag, p in (("orig", a.inp), ("l2conv", a.out)):
            net = cv2.dnn.readNetFromONNX(str(p), cv2.dnn.ENGINE_NEW)
            net.setInput(x)
            names = net.getUnconnectedOutLayersNames()
            outs[tag] = dict(zip(names, net.forward(names)))
        for k in outs["orig"]:
            o, r = outs["orig"][k], outs["l2conv"][k]
            print(f"{k}: shape {o.shape}  max |l2conv - orig| = {float(np.abs(o - r).max()):.3e}")
        kp_r = ref["keypoints_cv"]
        for tag in ("orig", "l2conv"):
            kp = next(v for k, v in outs[tag].items() if "keypoint" in k)
            d = np.sqrt(((kp_r[:, None] - kp[None]) ** 2).sum(-1)).min(1)
            print(f"{tag} vs upstream torch: keypoints within 1e-3 px (cv units 2/W): "
                  f"{float((d < 2e-3 / x.shape[3]).mean()):.4f}, max {float(d.max()) * x.shape[3] / 2:.2e} px")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
