"""EigenPlaces reference exporter (2.1, design PERCEPTION_2.1 §11.4). OFFLINE tool, run in the session image
ariadne-eigenplaces-ref:feat-2.1 with --network none, the upstream checkout and the weights mounted read-only:

  python -I eigenplaces_reference.py --upstream /ep --weights /w/ResNet50_256_eigenplaces.pth --out /out \
      --size 752x480 --size 376x240 --image NAME=PATH ... --upstream-commit SHA

Upstream: github.com/gmberton/EigenPlaces (MIT), GeoLocalizationNet_("ResNet50", 256) = torchvision ResNet-50 minus
avgpool/fc, then L2Norm(dim=1) -> GeM -> Flatten -> Linear(2048, 256) -> L2Norm. Upstream's test transform is
ToTensor + ImageNet Normalize at the image's own resolution (datasets/test_dataset.py); EuRoC is gray, so the channel
is replicated to RGB. Per --size the image is resized with cv2.INTER_AREA (identity at the native size), which the C++
side reproduces with the same OpenCV 5.0.0 call.

Writes to --out:
1. eigenplaces_r50_256_<W>x<H>.onnx: the whole model, opset 17, FIXED input size (GeM's kernel is the map size, so a
   dynamic shape would become Shape/Gather ops in the graph);
2. eigenplaces_r50_layer4_<W>x<H>.onnx: the backbone only, and head.atb (GeM p, fc weight/bias): the fallback if the
   importer cannot run the head, which is then computed in C++;
3. ref_<name>_<W>x<H>.atb: the exact preprocessed input (1x3xHxW), upstream's descriptor (256) and layer4 output;
4. manifest.yaml: upstream commit, weights sha256, package versions, sizes, preprocessing, sha256 of every output.

Why the torch.hub stub: GeoLocalizationNet_'s constructor initialises its backbone from CosPlace over torch.hub (a
network fetch). Those values are then entirely replaced by load_state_dict(strict=True) with the EigenPlaces
weights, so the stub returns an untrained torchvision ResNet-50 instead and nothing else changes.
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
import torchvision
import yaml

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


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


def preprocess(gray: np.ndarray, w: int, h: int) -> np.ndarray:
    """gray uint8 HxW -> 1x3xhxw float32, exactly as the C++ PlaceRecognizer does it."""
    f = gray.astype(np.float32) / 255.0                      # ToTensor
    if (f.shape[1], f.shape[0]) != (w, h):
        f = cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA)
    rgb = np.stack([f, f, f], axis=0)                         # gray -> RGB, CHW
    return ((rgb - MEAN[:, None, None]) / STD[:, None, None])[None].astype(np.float32)


def preprocess_rgb(bgr: np.ndarray, w: int, h: int) -> np.ndarray:
    """colour uint8 HxWx3 as cv2.imread gives it (BGR) -> 1x3xhxw float32 in RGB order, as PlaceRecognizer::DescribeRgb
    does it from an RGB image: /255 -> INTER_AREA on all three channels together -> ImageNet mean/std."""
    f = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if (f.shape[1], f.shape[0]) != (w, h):
        f = cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA)
    return ((f.transpose(2, 0, 1) - MEAN[:, None, None]) / STD[:, None, None])[None].astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", required=True, type=Path)
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--size", action="append", required=True, help="WxH")
    ap.add_argument("--image", action="append", default=[], help="NAME=PATH (gray)")
    ap.add_argument("--rgb-image", action="append", default=[], help="NAME=PATH (colour PNG; the sim rig, RGB path)")
    ap.add_argument("--upstream-commit", required=True, help="git rev-parse HEAD of the checkout (no git in the image)")
    a = ap.parse_args()

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.hub.load = lambda *args, **kw: torchvision.models.resnet50()   # see the module docstring
    sys.path.insert(0, str(a.upstream))
    from eigenplaces_model import eigenplaces_network  # noqa: E402  (upstream code, read-only mount)

    model = eigenplaces_network.GeoLocalizationNet_("ResNet50", 256)
    state = torch.load(a.weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    gem = model.aggregation[1]
    fc = model.aggregation[3]
    a.out.mkdir(parents=True, exist_ok=True)
    write_atb(a.out / "head.atb", {"gem_p": gem.p.detach().numpy().astype(np.float32).reshape(1),
                                   "fc_weight": fc.weight.detach().numpy().astype(np.float32),
                                   "fc_bias": fc.bias.detach().numpy().astype(np.float32)})

    outputs: dict[str, str] = {}
    sizes = []
    for s in a.size:
        w, h = (int(v) for v in s.lower().split("x"))
        sizes.append([w, h])
        dummy = torch.zeros(1, 3, h, w)
        full = a.out / f"eigenplaces_r50_256_{w}x{h}.onnx"
        torch.onnx.export(model, dummy, str(full), opset_version=17, input_names=["image"],
                          output_names=["descriptor"], do_constant_folding=True, dynamo=False)
        trunk = a.out / f"eigenplaces_r50_layer4_{w}x{h}.onnx"
        torch.onnx.export(model.backbone, dummy, str(trunk), opset_version=17, input_names=["image"],
                          output_names=["layer4"], do_constant_folding=True, dynamo=False)
        for spec, rgb in [(s, False) for s in a.image] + [(s, True) for s in a.rgb_image]:
            name, path = spec.split("=", 1)
            gray = cv2.imread(path, cv2.IMREAD_COLOR if rgb else cv2.IMREAD_GRAYSCALE)   # rgb: BGR in memory
            if gray is None:
                raise SystemExit(f"cannot read {path}")
            x = preprocess_rgb(gray, w, h) if rgb else preprocess(gray, w, h)
            with torch.no_grad():
                t = torch.from_numpy(x)
                l4 = model.backbone(t)
                d = model.aggregation(l4)
            if (a.out / f"input_{name}.png").exists() is False:
                cv2.imwrite(str(a.out / f"input_{name}.png"), gray)
            ref = a.out / f"ref_{name}_{w}x{h}.atb"
            write_atb(ref, {"input": x, "layer4": l4.numpy().astype(np.float32),
                            "descriptor": d.numpy().astype(np.float32).reshape(-1)})
            print(f"{name} {w}x{h}: |d| = {float(d.norm()):.6f}, layer4 {tuple(l4.shape)}")

    for p in sorted(a.out.iterdir()):
        if p.name != "manifest.yaml":
            outputs[p.name] = sha256(p)
    manifest = {
        "upstream": "github.com/gmberton/EigenPlaces",
        "license": "MIT",
        "commit": a.upstream_commit,
        "model": "GeoLocalizationNet_(ResNet50, fc_output_dim=256)",
        "weights": a.weights.name,
        "weights_sha256": sha256(a.weights),
        "torch": str(torch.__version__), "torchvision": str(torchvision.__version__),
        "opencv_python": cv2.__version__,
        "threads": 1, "deterministic": True, "opset": 17,
        "sizes": sizes,
        "preprocessing": "gray/255 -> cv2.resize INTER_AREA to WxH (if different) -> replicate to RGB -> "
                         "(x - [0.485,0.456,0.406]) / [0.229,0.224,0.225], NCHW",
        "sha256": outputs,
    }
    if a.rgb_image:
        manifest["preprocessing_rgb"] = ("colour PNG -> RGB order -> /255 -> cv2.resize INTER_AREA (3 channels) to WxH "
                                         "(if different) -> ImageNet mean/std, NCHW (PlaceRecognizer::DescribeRgb)")
        manifest["rgb_images"] = [s.split("=", 1)[0] for s in a.rgb_image]
    (a.out / "manifest.yaml").write_text("# EigenPlaces reference export (PERCEPTION_2.1 §11.4). Generated.\n"
                                         + yaml.safe_dump(manifest, sort_keys=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
