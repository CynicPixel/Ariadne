"""Compose an N-agent mission spool (schema 3) from per-agent spools and a pass-B collab run (Phase 8, D213).

  python3 -I compose_mission_spool.py --spool AGENT_SPOOL [--spool ...] --collab PASSB_DIR --out DIR

Inputs:
  AGENT_SPOOL  one spool directory per agent (`ariadne_sim_vo --spool` on the mission recording, schema 1 or 2), each
               holding that agent's bundles (agent{k}/...) in its own odometry frame O_k.
  PASSB_DIR    an `ariadne_collab_sim` output directory: results.yaml's `estimates` (per agent: the adopted frame G =
               W of `anchor`, and T_G_W) and `kept_inter_edges` (every inter-agent loop the robust stage still kept
               in its last solve, never an injected one, with when it was first kept).

Output DIR/spool.yaml, schema_version 3 (A1's N-agent reader; A2's schema-2 bundle fields unchanged): the per-agent
bundles concatenated (files are symlinked, never copied), and instead of schema 1/2's single `merge:` block a
`merges:` list with one entry per non-anchor agent k:

  {agent: k, ok: true, T_O0_O: [3x4 row-major], edge: "agent{j} kf {a} <- agent{k} kf {b} (...)"}

T_O0_O = T_G_W0^-1 T_G_Wk maps O_k into agent 0's frame (agent 0 anchors W, D45), so T_O0_C = T_O0_O T_Ok_C. It is
defined only when agents 0 and k adopted the SAME frame (same anchor). The edge is the earliest kept loop joining k to
an agent already in agent 0's frame, with both keyframes present in the bundles (the reader looks up t_merge from
them). An agent with no such frame or edge gets ok: false. Pass A and sim_vo run the same front end on the same
recording, so their keyframe ids coincide; the composer checks that every edge keyframe exists in the bundles.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import yaml

SCHEMA = 3


def _pose34(v) -> np.ndarray:
    m = np.eye(4)
    m[:3, :] = np.asarray(v, float).reshape(3, 4)
    return m


def compose(spools: list[Path], collab: Path, out: Path) -> dict:
    heads, bundles = [], []
    for sp in spools:
        y = yaml.safe_load((sp / "spool.yaml").read_text())
        if y.get("schema_version") not in (1, 2):
            raise ValueError(f"{sp}: per-agent spools must be schema 1 or 2")
        agents = {b["agent"] for b in y["bundles"]}
        if len(agents) != 1:
            raise ValueError(f"{sp}: expected one agent per input spool, got {sorted(agents)}")
        heads.append((sp, y, agents.pop()))
    ids = [a for _, _, a in heads]
    if len(set(ids)) != len(ids) or 0 not in ids:
        raise ValueError(f"agents must be distinct and include agent 0, got {ids}")
    heads.sort(key=lambda h: h[2])
    for sp, y, _ in heads:
        bundles += [dict(b, _src=sp) for b in y["bundles"]]
    have = {(b["agent"], b["kf_id"]) for b in bundles}

    r = yaml.safe_load((collab / "results.yaml").read_text())
    est = {e["agent"]: e for e in r.get("estimates") or []}
    e0 = est.get(0)
    frame0 = e0["anchor"] if e0 and e0.get("T_G_W") else None
    in_frame = {0} if frame0 is not None else set()
    T = {a: _pose34(e["T_G_W"]) for a, e in est.items() if e.get("T_G_W") and e.get("anchor") == frame0}
    kept = sorted(r.get("kept_inter_edges") or [], key=lambda k: (k["t_s"], k["agent_a"], k["agent_b"]))

    merges, pending = [], [a for a in ids if a != 0]
    progress = True
    while pending and progress:   # agents join through any agent already in agent 0's frame
        progress = False
        for k in list(pending):
            edge = None
            if k in T:
                for e in kept:
                    pair = {(e["agent_a"], e["kf_a"]), (e["agent_b"], e["kf_b"])}
                    other = [p for p in pair if p[0] != k]
                    mine = [p for p in pair if p[0] == k]
                    if len(other) == 1 and len(mine) == 1 and other[0][0] in in_frame and pair <= have:
                        edge = (other[0], mine[0], e)
                        break
            if edge:
                (j, kj), (_, kk), e = edge
                T_O0_O = np.linalg.inv(T[0]) @ T[k]
                merges.append({"agent": k, "ok": True, "T_O0_O": [float(x) for x in T_O0_O[:3, :].reshape(-1)],
                               "edge": f"agent{j} kf {kj} <- agent{k} kf {kk} (collab_sim kept loop, first kept "
                                       f"t_s={e['t_s']:.3f}, weight={e['weight']:.3f})"})
                in_frame.add(k)
                pending.remove(k)
                progress = True
    for k in pending:
        merges.append({"agent": k, "ok": False, "T_O0_O": [float(x) for x in np.eye(4)[:3, :].reshape(-1)],
                       "edge": "none: no shared frame with agent 0 or no kept loop with both keyframes spooled"})
    merges.sort(key=lambda m: m["agent"])

    out.mkdir(parents=True, exist_ok=True)
    for b in bundles:   # symlink each bundle's files (stem + every suffix present) into out/agent{k}/
        src_dir = b["_src"] / Path(b["files"]).parent
        (out / Path(b["files"]).parent).mkdir(parents=True, exist_ok=True)
        stem = Path(b["files"]).name
        for f in src_dir.glob(stem + "_*"):
            dst = out / Path(b["files"]).parent / f.name
            if not dst.exists():
                os.symlink(f.resolve(), dst)
    _, y0, _ = heads[0]
    lamp_files = {h[1].get("lamp", {}).get("calibration") for h in heads if h[1].get("lamp")}
    for sp, y, _ in heads:
        if y.get("lamp") and not (out / y["lamp"]["calibration"]).exists():
            os.symlink((sp / y["lamp"]["calibration"]).resolve(), out / y["lamp"]["calibration"])
    doc = {"schema_version": SCHEMA,
           "run_id": "+".join(str(h[1]["run_id"]) for h in heads),
           "scene_id": y0["scene_id"]}
    for k in ("scale", "keyframe_stride", "heldout_stride", "image_encoding", "frames"):
        vals = {yaml.safe_dump(h[1].get(k)) for h in heads}
        if len(vals) != 1:
            raise ValueError(f"per-agent spools disagree on {k}")
        doc[k] = y0.get(k)
    if len(lamp_files) > 1:
        raise ValueError("per-agent spools carry different lamp calibrations")
    if y0.get("lamp"):
        doc["lamp"] = y0["lamp"]
    doc["merges"] = merges
    doc["merge_source"] = {"collab_run": str(collab.resolve()), "frame_anchor": frame0}
    doc["bundles"] = [{k: v for k, v in b.items() if k != "_src"} for b in bundles]
    (out / "spool.yaml").write_text("# Ariadne N-agent mission spool (compose_mission_spool.py, D213). Generated.\n"
                                    + yaml.safe_dump(doc, sort_keys=False))
    return doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spool", type=Path, action="append", required=True)
    ap.add_argument("--collab", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    doc = compose(a.spool, a.collab, a.out)
    for m in doc["merges"]:
        print(f"agent {m['agent']}: {'ok' if m['ok'] else 'NOT merged'}  {m['edge']}")


if __name__ == "__main__":
    main()
