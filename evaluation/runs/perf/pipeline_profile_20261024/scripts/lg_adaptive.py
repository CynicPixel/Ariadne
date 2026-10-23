#!/usr/bin/env python3
"""LightGlue adaptive depth/width on the knob study's real pairs (M's request, 2026-10-24: size the payoff before any
design). Runs UPSTREAM LightGlue (cvg/LightGlue eb42fee, aliked_lightglue.pth, the exporter's pins and loader) in
torch on the exact features the fleet's LightGlue sees (ariadne_perf_bench knobs --dump-features: 1280x720 K=1024,
descriptors after the int8 wire round trip), in two modes:
  all9      depth_confidence = width_confidence = -1, filter 0.1: what our ONNX export computes (the control: its
            matches must equal the cv::dnn matches of knobs.csv's 1280x720:1024 rows)
  adaptive  upstream's defaults, depth 0.95 / width 0.99, filter 0.1 (flash off; on CPU pruning_keypoint_thresholds
            is -1, so point pruning is active at any K)
Per pair and mode: the exit layer (`stop`, 1..9), the keypoints still active after each layer in A and B (from
prune0/prune1: a point's value = 1 + the number of layers it survived), the match count, torch wall ms, and the match
list for ariadne_perf_bench gate (the D71 verdict).

  python -I lg_adaptive.py --upstream LG_SRC --weights W --exporter python/model_tools --feat-dir DIR --pairs CSV
      --threads 16 --out matches.csv
"""
from __future__ import annotations

import argparse
import csv
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch


def read_feat(p: Path):
    b = p.read_bytes()
    (n,) = struct.unpack_from("<i", b, 0)
    o = 4
    kp = np.frombuffer(b, np.float32, 2 * n, o).reshape(n, 2); o += 8 * n
    d = np.frombuffer(b, np.float32, 128 * n, o).reshape(n, 128); o += 512 * n
    return kp, d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--exporter", type=Path, required=True, help="dir of aliked_lightglue_reference.py (its loader)")
    ap.add_argument("--feat-dir", type=Path, required=True)
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.use_deterministic_algorithms(True)
    sys.path.insert(0, str(a.exporter))
    import aliked_lightglue_reference as ref                       # stub_kornia, import_upstream (the pinned path)
    ref.stub_kornia()

    def local_weights(url, *args, file_name=None, **kw):
        name = url.rsplit("/", 1)[-1]                              # as the exporter: the URL names the file
        return torch.load(a.weights / name, map_location="cpu", weights_only=True)
    torch.hub.load_state_dict_from_url = local_weights
    _, lg_mod = ref.import_upstream(a.upstream)
    nets = {"all9": lg_mod.LightGlue(features="aliked", depth_confidence=-1, width_confidence=-1,
                                     filter_threshold=0.1, flash=False).eval(),
            "adaptive": lg_mod.LightGlue(features="aliked", depth_confidence=0.95, width_confidence=0.99,
                                         filter_threshold=0.1, flash=False).eval()}
    size = torch.tensor([[1280.0, 720.0]])
    rows = list(csv.DictReader(open(a.pairs)))
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["agent_a", "kf_a", "agent_b", "kf_b", "mode", "stop", "kept_a_by_layer", "kept_b_by_layer",
                    "n_matches", "torch_ms", "matches"])
        for r in rows:
            ka, da = read_feat(a.feat_dir / f"a{r['agent_a']}_kf{r['kf_a']}.feat")
            kb, db = read_feat(a.feat_dir / f"a{r['agent_b']}_kf{r['kf_b']}.feat")
            data = {"image0": {"keypoints": torch.from_numpy(ka.copy())[None],
                               "descriptors": torch.from_numpy(da.copy())[None], "image_size": size},
                    "image1": {"keypoints": torch.from_numpy(kb.copy())[None],
                               "descriptors": torch.from_numpy(db.copy())[None], "image_size": size}}
            for mode, net in nets.items():
                with torch.inference_mode():
                    t0 = time.perf_counter()
                    out = net(data)
                    ms = 1000 * (time.perf_counter() - t0)
                stop = int(out["stop"])
                p0, p1 = out["prune0"][0].numpy(), out["prune1"][0].numpy()
                kept = lambda p: ";".join(str(int((p > i).sum())) for i in range(stop))
                m = out["matches"][0].numpy()
                w.writerow([r["agent_a"], r["kf_a"], r["agent_b"], r["kf_b"], mode, stop, kept(p0), kept(p1), len(m),
                            f"{ms:.1f}", " ".join(f"{i}-{j}" for i, j in m.tolist())])
            fh.flush()
    print(f"lg_adaptive: {len(rows)} pairs x {len(nets)} modes -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
