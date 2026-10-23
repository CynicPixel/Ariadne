"""The reconstruction service loop (RECON_6 §1/§3; D190, D194, D195): keyframes in -> submap chunks + table out.

  python -m ariadne_recon.live.service --spool DIR --out SPLAT_DIR [--backend gpu|cpu_init]
         [--training-image albedo_proxy|rgb] [--iters N | --budget-ms MS] [--export-period-s 2]

Input: a spool (schema 2 for the albedo proxy; replay from files alone). The live `#kf` socket feeds the same
`ingest()` (B's tee, RECON_6 §3). Output, in SPLAT_DIR (what operator_console.splat serves):
  table.json, chunks/<sid>_<n>.splat, service_log.jsonl (one line per keyframe), manifest.json

Backends:
  gpu       LiveTrainer (gsplat on CUDA)
  cpu_init  no GPU (D194): the same submaps and depth seeding, no optimisation; labelled "initialisation only, no GPU"

Training image (D195): the albedo proxy by default; a keyframe without one is refused unless --training-image rgb
(the lit-world ablation) is given. Nothing falls back silently.

Placement: T_W_O per agent comes with the live tee (B); for a spool replay it is the identity [replay has no
briefing], so W = agent 0's odometry frame.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from ..spool import Spool, load
from .export import SPLAT_BYTES, chunk_bytes
from .seeding import SeedParams, seed
from .submaps import SubmapManager

LABELS = {("gpu", "albedo_proxy"): "albedo proxy (gsplat, live)",
          ("gpu", "rgb"): "RGB ablation (lit world; not the albedo default)",
          ("cpu_init", "albedo_proxy"): "albedo proxy — initialisation only, no GPU",
          ("cpu_init", "rgb"): "RGB ablation — initialisation only, no GPU"}


def _atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


class _CpuInit:
    """D194: submaps + seeding, no optimisation. Gaussians stay at their seeds."""

    def __init__(self, channels: int, max_keyframes: int, sp: SeedParams, seed_value: int):
        self.manager = SubmapManager(max_keyframes)
        self.sp, self.channels = sp, channels
        self.g: dict[int, dict] = {}
        self.rng = np.random.default_rng(seed_value)

    def add_keyframe(self, agent, kf, T_O_C, pose_version, K, image, Z, sigma) -> dict:
        t0 = time.perf_counter()
        sm, _ = self.manager.add_keyframe(agent, kf, T_O_C, pose_version)
        g = self.g.setdefault(sm.sid, {"means": np.zeros((0, 3), np.float32), "scales": np.zeros(0, np.float32),
                                       "colors": np.zeros((0, self.channels), np.float32)})
        s = seed(Z, sigma, image, K, sm.T_anchor_C[kf], None, g["means"], self.sp, self.rng)
        for k in ("means", "scales", "colors"):
            g[k] = np.concatenate([g[k], getattr(s, k)])
        return {"agent": agent, "kf": kf, "sid": sm.sid, "seeded": len(s.means), "gaussians": len(g["means"]),
                "iters": 0, "ms_total": 1e3 * (time.perf_counter() - t0)}

    def params_numpy(self, sid: int) -> dict:
        g = self.g[sid]
        n = len(g["means"])
        c = np.clip(g["colors"], 0, 1)
        rgb = np.repeat(c, 3, axis=1) if c.shape[1] == 1 else c[:, :3]
        q = np.zeros((n, 4), np.float32)
        q[:, 0] = 1
        return {"means": g["means"], "scales": np.log(np.repeat(g["scales"][:, None], 3, axis=1)), "quats": q,
                "opacities": np.full(n, np.log(0.99 / 0.01), np.float32),
                "sh0": ((rgb - 0.5) / 0.28209479177387814)[:, None, :].astype(np.float32)}


class Service:
    def __init__(self, out: Path, backend: str, training_image: str, trainer, export_period_s: float = 2.0):
        self.out, self.backend, self.training_image, self.trainer = out, backend, training_image, trainer
        (out / "chunks").mkdir(parents=True, exist_ok=True)
        self.label = LABELS[(backend, training_image)]
        self.T_W_O: dict[int, np.ndarray] = {}
        self.chunk: dict[int, str] = {}
        self.counts: dict[int, int] = {}
        self.n_export: dict[int, int] = {}
        self.last_export: dict[int, float] = {}
        self.version = 0
        self.period = export_period_s
        self.log = open(out / "service_log.jsonl", "w")

    def ingest(self, agent: int, kf: int, T_O_C, pose_version: int, K, image, Z, sigma) -> dict:
        if agent not in self.T_W_O:
            self.T_W_O[agent] = np.eye(4)
        before = {s.sid for s in self.trainer.manager.submaps if not s.closed}
        rec = self.trainer.add_keyframe(agent, kf, T_O_C, pose_version, K, image, Z, sigma)
        now = time.monotonic()
        closed_now = {s.sid for s in self.trainer.manager.submaps if s.closed} & before
        for sid in closed_now:
            self._export(sid)                                      # a closing submap is sent once, final
        if now - self.last_export.get(rec["sid"], -1e9) >= self.period:
            self._export(rec["sid"])
        self.log.write(json.dumps(rec) + "\n")
        return rec

    def _export(self, sid: int) -> None:
        params = self.trainer.params_numpy(sid)
        n = self.n_export.get(sid, 0) + 1
        name = f"{sid}_{n}.splat"
        _atomic(self.out / "chunks" / name, chunk_bytes(params))
        old = self.chunk.get(sid)
        self.chunk[sid], self.n_export[sid], self.counts[sid] = name, n, len(params["means"])
        self.last_export[sid] = time.monotonic()
        self._table()
        if old:
            (self.out / "chunks" / old).unlink(missing_ok=True)

    def _table(self) -> None:
        self.version += 1
        rows = []
        for sm in self.trainer.manager.submaps:
            if sm.sid not in self.chunk:
                continue
            rows.append({"sid": sm.sid, "agent": sm.agent, "anchor_kf": sm.anchor_kf, "closed": sm.closed,
                         "pose_version": sm.pose_version, "chunk": self.chunk[sm.sid],
                         "gaussians": self.counts[sm.sid], "bytes": SPLAT_BYTES * self.counts[sm.sid],
                         "T_W_anchor": [float(x) for x in sm.placement(self.T_W_O[sm.agent]).ravel()]})
        _atomic(self.out / "table.json", json.dumps({"label": self.label, "version": self.version,
                                                     "backend": self.backend, "training_image": self.training_image,
                                                     "submaps": rows}).encode())

    def finish(self) -> None:
        for sm in self.trainer.manager.submaps:
            self._export(sm.sid)
        self.log.close()


def replay(sp: Spool, svc: Service, training_image: str, max_keyframes: int = 0) -> int:
    n = 0
    for b in sorted(sp.keyframes(), key=lambda b: (b.t_ns, b.agent)):
        if training_image == "albedo_proxy":
            if b.rgb_kind != "albedo_proxy":
                raise SystemExit(f"keyframe {b.kf_id}: no albedo proxy (rgb_kind {b.rgb_kind}); the RGB ablation "
                                 "must be asked for (--training-image rgb), never a fallback (D195)")
            image = b.albedo()
        else:
            image = b.rgb().astype(np.float32) / 255.0
        Z, S = b.depth()
        svc.ingest(b.agent, b.kf_id, sp.T_O0_C(b) if b.agent else b.T_O_C, b.frame_id_of_pose, b.K, image, Z, S)
        n += 1
        if max_keyframes and n >= max_keyframes:
            break
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", choices=("gpu", "cpu_init"), default="gpu")
    ap.add_argument("--training-image", choices=("albedo_proxy", "rgb"), default="albedo_proxy")
    ap.add_argument("--iters", type=int, default=0)
    ap.add_argument("--budget-ms", type=float, default=0.0)
    ap.add_argument("--export-period-s", type=float, default=2.0)
    ap.add_argument("--max-keyframes", type=int, default=0)
    a = ap.parse_args(argv)
    sp = load(a.spool)
    channels = 1 if a.training_image == "albedo_proxy" else 3
    if a.backend == "gpu":
        from .trainer import LiveTrainer, TrainParams
        trainer = LiveTrainer(channels, TrainParams(iters=a.iters, budget_ms=a.budget_ms))
    else:
        trainer = _CpuInit(channels, 30, SeedParams(), 0)
    out = Path(a.out)
    svc = Service(out, a.backend, a.training_image, trainer, a.export_period_s)
    t0 = time.time()
    n = replay(sp, svc, a.training_image, a.max_keyframes)
    svc.finish()
    (out / "manifest.json").write_text(json.dumps({
        "tool": "ariadne_recon.live.service", "spool": str(a.spool), "spool_run_id": sp.run_id,
        "spool_schema": sp.schema_version, "backend": a.backend,
        "splat_backend": "gsplat_cuda" if a.backend == "gpu" else "cpu_init", "training_image": a.training_image,
        "iters": a.iters, "budget_ms": a.budget_ms, "keyframes": n, "wall_s": round(time.time() - t0, 2)}, indent=2))
    print(f"service: {n} keyframes -> {out} ({len(svc.chunk)} submaps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
