"""The reconstruction service loop (RECON_6 §1/§3; D190, D194, D195): keyframes in -> submap chunks + table out.

  python -m ariadne_recon.live.service --spool DIR [DIR ...] --out SPLAT_DIR [--scene DIR] [--placement briefing|merges] [--backend gpu|cpu_init]
         [--training-image albedo_proxy|rgb] [--iters N | --budget-ms MS] [--export-period-s 2]

Input: a spool (schema 2 for the albedo proxy; replay from files alone). The live `#kf` socket feeds the same
`ingest()` (B's tee, RECON_6 §3). Output, in SPLAT_DIR (what operator_console.splat serves):
  table.json, chunks/<sid>_<n>.splat, service_log.jsonl (one line per keyframe), manifest.json

Backends:
  gpu       LiveTrainer (gsplat on CUDA)
  cpu_init  no GPU (D194): the same submaps and depth seeding, no optimisation; labelled "initialisation only, no GPU"

Training image (D195, D199): `rgb` on lit worlds (the live demo's default, D199); `albedo_proxy` where the agents'
lamp is the light (D195), and then a keyframe without the proxy is refused. Nothing falls back silently.

Placement (placement.py, D230): every keyframe enters in its own agent's odometry frame O_k; the table places a
submap at T_W_O[k] @ T_O_anchor. Live, T_W_O[k] comes from B's tee (the briefing constant until collab moves it,
then a `t_w_o` record: set_placement, one matrix, no retraining). A replay takes it from the briefing (--scene,
--placement briefing: every agent's own entrance, the demo's known-starts rule) or from the spool's merges
(--placement merges: agent 0's entrance, agent k via T_O0_Ok). An agent without a placement is not drawn.
Without --scene, W = agent 0's odometry frame.
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
from .placement import placements
from .seeding import SeedParams, seed
from .submaps import SubmapManager

LABELS = {("gpu", "albedo_proxy"): "albedo proxy (gsplat, live)",
          ("gpu", "rgb"): "RGB, lit world (gsplat, live)",
          ("cpu_init", "albedo_proxy"): "albedo proxy — initialisation only, no GPU",
          ("cpu_init", "rgb"): "RGB, lit world — initialisation only, no GPU"}


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
        self.T_W_O: dict[int, np.ndarray] = {}   # agent -> placement; set_placement() before its first keyframe
        self.chunk: dict[int, str] = {}
        self.counts: dict[int, int] = {}
        self.n_export: dict[int, int] = {}
        self.last_export: dict[int, float] = {}
        self.version = 0
        self.period = export_period_s
        self.log = open(out / "service_log.jsonl", "w")

    def set_placement(self, agent: int, T_W_O) -> None:
        """Agent k's odometry frame in W (briefing, merge, or a live `t_w_o` update): every submap of the agent moves
        with it in the next table; chunks are not re-exported."""
        self.T_W_O[agent] = np.asarray(T_W_O, float).copy()
        if self.chunk:
            self._table()

    def ingest(self, agent: int, kf: int, T_O_C, pose_version: int, K, image, Z, sigma) -> dict:
        if agent not in self.T_W_O:
            raise ValueError(f"agent {agent}: no placement (set_placement first; an unplaced agent is never drawn)")
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


def replay(spools: list[Spool], svc: Service, training_image: str, max_keyframes: int = 0, pace: float = 0.0) -> int:
    """Every keyframe of a placed agent, all spools merged in time order; agents without a placement are skipped.
    pace > 0 plays the keyframes at their mission times (1 = mission speed): what the console sees during a live run,
    so a recorded mission shows the splat growing. A keyframe is never dropped to keep up; a late one is just late."""
    n = 0
    t_first, wall0 = None, time.monotonic()
    keyframes = [b for sp in spools for b in sp.keyframes()]
    if len({(b.agent, b.kf_id) for b in keyframes}) != len(keyframes):
        raise SystemExit("the spools repeat an (agent, keyframe): one spool per agent")
    for b in sorted(keyframes, key=lambda b: (b.t_ns, b.agent)):
        if b.agent not in svc.T_W_O:
            continue
        if pace > 0:
            t_first = b.t_ns if t_first is None else t_first
            wait = (b.t_ns - t_first) * 1e-9 / pace - (time.monotonic() - wall0)
            if wait > 0:
                time.sleep(wait)
        if training_image == "albedo_proxy":
            if b.rgb_kind != "albedo_proxy":
                raise SystemExit(f"keyframe {b.kf_id}: no albedo proxy (rgb_kind {b.rgb_kind}); RGB training must be "
                                 "asked for (--training-image rgb, lit worlds), never a fallback (D195)")
            image = b.albedo()
        else:
            image = b.rgb().astype(np.float32) / 255.0
        Z, S = b.depth()
        svc.ingest(b.agent, b.kf_id, b.T_O_C, b.frame_id_of_pose, b.K, image, Z, S)
        n += 1
        if max_keyframes and n >= max_keyframes:
            break
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spool", required=True, nargs="+", help="one composed spool, or one spool per agent")
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", choices=("gpu", "cpu_init"), default="gpu")
    ap.add_argument("--training-image", choices=("albedo_proxy", "rgb"), default="rgb")   # D199: lit demo
    ap.add_argument("--iters", type=int, default=0)
    ap.add_argument("--budget-ms", type=float, default=0.0)
    ap.add_argument("--export-period-s", type=float, default=2.0)
    ap.add_argument("--max-keyframes", type=int, default=0)
    ap.add_argument("--pace", type=float, default=0.0, help="0 = as fast as possible; 1 = mission speed")
    ap.add_argument("--scene", default=None, help="worldgen scene dir: W = the mission world (the GT viewport's)")
    ap.add_argument("--placement", choices=("briefing", "merges"), default=None,
                    help="default: briefing with --scene, merges without")
    a = ap.parse_args(argv)
    spools = [load(s) for s in a.spool]
    channels = 1 if a.training_image == "albedo_proxy" else 3
    if a.backend == "gpu":
        from .trainer import LiveTrainer, TrainParams
        trainer = LiveTrainer(channels, TrainParams(iters=a.iters, budget_ms=a.budget_ms))
    else:
        trainer = _CpuInit(channels, 30, SeedParams(), 0)
    out = Path(a.out)
    svc = Service(out, a.backend, a.training_image, trainer, a.export_period_s)
    mode = a.placement or ("briefing" if a.scene else "merges")
    placed, unplaced = placements(spools, mode, a.scene)
    for k, T in placed.items():
        svc.set_placement(k, T)
    for k, why in unplaced.items():
        print(f"service: agent {k} not placed, not drawn: {why}")
    t0 = time.time()
    n = replay(spools, svc, a.training_image, a.max_keyframes, a.pace)
    svc.finish()
    (out / "manifest.json").write_text(json.dumps({
        "tool": "ariadne_recon.live.service", "spool": a.spool, "spool_run_id": [sp.run_id for sp in spools],
        "spool_schema": [sp.schema_version for sp in spools], "backend": a.backend,
        "splat_backend": "gsplat_cuda" if a.backend == "gpu" else "cpu_init", "training_image": a.training_image,
        "iters": a.iters, "budget_ms": a.budget_ms, "pace": a.pace, "keyframes": n, "wall_s": round(time.time() - t0, 2),
        "placement": {"mode": mode, "scene": a.scene, "world": "mission W" if a.scene else "agent 0 odometry frame",
                      "T_W_O": {str(k): [float(x) for x in T.ravel()] for k, T in placed.items()},
                      "unplaced": {str(k): why for k, why in unplaced.items()}}}, indent=2))
    print(f"service: {n} keyframes -> {out} ({len(svc.chunk)} submaps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
