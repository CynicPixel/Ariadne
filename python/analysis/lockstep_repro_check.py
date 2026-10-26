"""Are two LIVE lockstep runs of one scene and seed the same run? (design live-runtime.md D165; A1's 4.3 gate)

  python3 python/analysis/lockstep_repro_check.py RUN_A RUN_B [OUT.json]   (live.sh's live-<id> directories)

Defined before the runs:
  1. both ran in lockstep (results.yaml `lockstep: true`), and every barrier's commands were applied AT the barrier
     (lockstep.jsonl: applied_ns == T for every agent commanded, answered before the resume); the pause point is
     reported from the acks (paused_clock_ns − T per barrier: the sim clock when the next run_to was requested);
  2. the consumed stream is identical (results.yaml consumed digest and message count): same messages, same order;
  3. per agent: estimate_body.tum sha256-identical, identical frame and keyframe counts;
  4. trace.jsonl identical (both live: no labels to strip) and mover_state.jsonl identical;
  5. the link's per-topic receive counts (link.jsonl, the last #stats) are reported side by side: a drop on one side
     is the first suspect when 2-4 differ.
A difference is reported per check, never smoothed. Renderer dependence: state which one ran (llvmpipe / NVIDIA EGL);
this checks the runs given, it does not claim bit-stability elsewhere. std-lib only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from live_replay_check import run_dir, sha, summary  # noqa: E402


def lockstep(d: Path) -> dict:
    res = (d / "results.yaml").read_text()
    on = any(l.strip() == "lockstep: true" for l in res.splitlines())
    bars, acks = [], []
    p = d / "lockstep.jsonl"
    for line in (p.read_text().splitlines() if p.exists() else []):
        r = json.loads(line)
        (acks if "ack" in r else bars).append(r.get("ack", r))
    late = [b for b in bars if not b["answered"] or any(a not in (-1, b["T_ns"]) for a in b["applied_ns"])]
    by_target = {a["run_to_ns"]: a for a in acks}
    offsets = [by_target[b["resume_to_ns"]]["paused_clock_ns"] - b["T_ns"] for b in bars if b["resume_to_ns"] in by_target]
    return {"lockstep": on, "barriers": len(bars), "commands_not_applied_at_T": len(late),
            "pause_offset_ns": {"min": min(offsets), "max": max(offsets)} if offsets else None,
            "acks_not_accepted": sum(not a["accepted"] for a in acks),
            "wall_paused_s_max": max((b["wall_paused_s"] for b in bars), default=None)}


def link_stats(base: Path, d: Path) -> dict:
    last = None
    for p in (d / "link.jsonl", base / "link.jsonl"):
        if p.exists():
            for line in p.read_text().splitlines():
                r = json.loads(line)
                if r.get("control") == "#stats":
                    last = r["payload"]
            break
    return last or {}


def main() -> None:
    a_base, b_base = Path(sys.argv[1]), Path(sys.argv[2])
    a, b = run_dir(a_base), run_dir(b_base)
    sa, sb = summary(a), summary(b)
    la, lb = lockstep(a), lockstep(b)
    checks = {
        "lockstep.both": la["lockstep"] and lb["lockstep"],
        "lockstep.applied_at_T": la["commands_not_applied_at_T"] == 0 and lb["commands_not_applied_at_T"] == 0,
        "consumed": sa["consumed"] == sb["consumed"],
        "agent0": sa["agent0"] == sb["agent0"],
        "agent1": sa["agent1"] == sb["agent1"],
        "trace_sha256": sa["trace_sha256"] == sb["trace_sha256"],
        "mover_state_sha256": sha(a / "mover_state.jsonl") == sha(b / "mover_state.jsonl"),
    }
    out = {"verdict": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
           "a": {"run": str(a), "summary": sa, "lockstep": la, "link_received": link_stats(a_base, a).get("received")},
           "b": {"run": str(b), "summary": sb, "lockstep": lb, "link_received": link_stats(b_base, b).get("received")}}
    txt = json.dumps(out, indent=1, sort_keys=True)
    if len(sys.argv) > 3:
        Path(sys.argv[3]).write_text(txt + "\n")
    print(json.dumps({"verdict": out["verdict"], "failed": [k for k, v in checks.items() if not v]}))
    sys.exit(0 if out["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
