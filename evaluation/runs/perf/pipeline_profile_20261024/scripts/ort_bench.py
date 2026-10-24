#!/usr/bin/env python3
"""The ORT comparison row of the dnn matrix (pipeline profile, A2 2026-10-24). This OpenCV build has no ONNX Runtime
(ENGINE_ORT needs WITH_ONNXRUNTIME; the bench's build.txt records the build information either way), so ORT is timed
here, from Python, on the SAME ONNX files and the exporters' own preprocessed input tensors (the `input` of each
reference ATB), single session, intra-op threads = --threads, inter-op 1, sequential execution.

  python3 -I ort_bench.py --models-dir DIR [--threads 4] [--reps 20] [--warmup 3] --label L --out ort_timing.csv
  (appends rows; DIR holds aliked_lg_sim_ref/, aliked_lg_sim_960x540/, aliked_lg_sim_640x360/, eigenplaces_sim_ref/)

Rows: label, model, variant, threads, p50_ms, p95_ms, mean_ms, n, check. The check compares the ORT output with
the upstream torch reference in the same ATB: for ALIKED the fraction of reference keypoints (cv convention) within
1e-3 px of an ORT keypoint and the max descriptor |diff| over those; for LightGlue the match count vs the reference
pair's; for EigenPlaces the cosine with the reference descriptor.
Timing is wall time of session.run only (the inputs are prepared once): what cv::dnn's forward is compared with.
"""
from __future__ import annotations

import argparse
import csv
import struct
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def read_atb(p: Path) -> dict[str, np.ndarray]:
    b = p.read_bytes()
    assert b[:4] == b"ATB1", p
    (n,), o, out = struct.unpack_from("<I", b, 4), 8, {}
    for _ in range(n):
        (ln,) = struct.unpack_from("<H", b, o)
        name = b[o + 2:o + 2 + ln].decode()
        o += 2 + ln
        code, nd = struct.unpack_from("<BB", b, o)
        o += 2
        shape = struct.unpack_from(f"<{nd}I", b, o)
        o += 4 * nd
        dt = {0: np.float32, 1: np.int64}[code]
        cnt = int(np.prod(shape)) if nd else 1
        out[name] = np.frombuffer(b, dt, cnt, o).reshape(shape)
        o += cnt * np.dtype(dt).itemsize
    return out


def session(path: Path, threads: int) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads, so.inter_op_num_threads = threads, 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def timed(f, warmup: int, reps: int) -> tuple[float, float, float]:
    for _ in range(warmup):
        f()
    ms = []
    for _ in range(reps):
        t0 = time.perf_counter()
        f()
        ms.append(1000 * (time.perf_counter() - t0))
    ms.sort()
    q = lambda p: ms[min(len(ms) - 1, int(p * (len(ms) - 1) + 0.5))]
    return q(0.5), q(0.95), sum(ms) / len(ms)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rows = []
    T = a.threads
    ref_lg_pairs = {}
    for d, size in (("aliked_lg_sim_ref", "1280x720"), ("aliked_lg_sim_960x540", "960x540"),
                    ("aliked_lg_sim_640x360", "640x360")):
        D = a.models_dir / d
        s = session(D / f"aliked_n16_k1024_{size}.onnx", T)
        ref = read_atb(D / "ref_img_sim_a0_f200_k1024.atb")
        x = np.ascontiguousarray(ref["input"], np.float32)
        feed = {s.get_inputs()[0].name: x}
        outs = {o.name: v for o, v in zip(s.get_outputs(), s.run(None, feed))}
        kp, desc = outs["keypoints"], outs["descriptors"]
        rk = ref["keypoints_cv"]
        dist = np.sqrt(((rk[:, None, :] - kp[None, :, :]) ** 2).sum(-1))
        j = dist.argmin(1)
        ok = dist[np.arange(len(rk)), j] < 1e-3 * 2 / int(size.split("x")[0])   # cv convention: n = 2 px / W - 1
        dmax = float(np.abs(desc[j[ok]] - ref["descriptors"][ok]).max()) if ok.any() else float("nan")
        p50, p95, mean = timed(lambda: s.run(None, feed), a.warmup, a.reps)
        rows.append([a.label, "aliked_extract", size, T, p50, p95, mean, a.reps,
                     f"kp_match {ok.mean():.4f} desc_max {dmax:.2e}"])
        if size == "1280x720":
            ref_lg_pairs = {"pair": read_atb(D / "ref_pair_sim_a0_f200_sim_a0_f205_k1024.atb"), "dir": D}
    if ref_lg_pairs:
        D, pr = ref_lg_pairs["dir"], ref_lg_pairs["pair"]
        s = session(D / "lightglue_aliked.onnx", T)
        da = read_atb(D / "ref_img_sim_a0_f200_k1024.atb")["descriptors"]
        db = read_atb(D / "ref_img_sim_a0_f205_k1024.atb")["descriptors"]
        sa = read_atb(D / "ref_img_sim_a0_f200_k1024.atb")["scores"]
        sb = read_atb(D / "ref_img_sim_a0_f205_k1024.atb")["scores"]
        for k in (1024, 512, 256):
            ia, ib = np.argsort(-sa, kind="stable")[:k], np.argsort(-sb, kind="stable")[:k]
            feed = {"kpts0": np.ascontiguousarray(pr["kpts0n"].reshape(1, -1, 2)[:, ia], np.float32),
                    "kpts1": np.ascontiguousarray(pr["kpts1n"].reshape(1, -1, 2)[:, ib], np.float32),
                    "desc0": np.ascontiguousarray(da[ia][None], np.float32),
                    "desc1": np.ascontiguousarray(db[ib][None], np.float32)}
            m = s.run(None, feed)[0]
            p50, p95, mean = timed(lambda: s.run(None, feed), a.warmup, a.reps)
            chk = f"matches {len(m)} ref {len(pr['matches'])}" if k == 1024 else f"matches {len(m)}"
            rows.append([a.label, "lightglue_match", f"K{k}", T, p50, p95, mean, a.reps, chk])
    D = a.models_dir / "eigenplaces_sim_ref"
    for size in ("640x360",):
        s = session(D / f"eigenplaces_r50_256_{size}.onnx", T)
        ref = read_atb(D / f"ref_sim_a0_f200_{size}.atb")
        name = next(k for k in ref if k.startswith("input"))
        dname = next(k for k in ref if "desc" in k or k == "output")
        feed = {s.get_inputs()[0].name: np.ascontiguousarray(ref[name], np.float32)}
        d = s.run(None, feed)[0].ravel()
        r = ref[dname].ravel()
        cos = float(d @ r / (np.linalg.norm(d) * np.linalg.norm(r)))
        p50, p95, mean = timed(lambda: s.run(None, feed), a.warmup, a.reps)
        rows.append([a.label, "eigenplaces_describe", size, T, p50, p95, mean, a.reps, f"cosine {cos:.7f}"])
    new = not a.out.exists()
    with open(a.out, "a", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        if new:
            w.writerow(["label", "stage", "variant", "threads", "p50_ms", "p95_ms", "mean_ms", "n", "check"])
        w.writerows(rows)
    for r in rows:
        print(f"ort {r[1]:22s} {r[2]:9s} threads {T:2d} p50 {r[4]:8.1f} ms  {r[8]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
