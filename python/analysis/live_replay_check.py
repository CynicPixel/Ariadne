"""Does the recording of a live run reproduce it? (design LIVE_RUNTIME_L1 D152; register L1 acceptance 2)

  python3 python/analysis/live_replay_check.py RUN       (RUN = live.sh's live-<id> directory, with replay1/ replay2/)

Defined before the first run:
  1. consumed sequence: the replays consume the SAME messages in the same order (results.yaml consumed digest
     and count equal the live run's);
  2. per agent: estimate_body.tum sha256-identical, identical frame and keyframe counts;
  3. mission: trace.jsonl identical once the two mode labels are removed (live records say mode live, executed
     true; replays say mode replay, executed false: invariant 3 labels which decisions moved anything);
  4. mover_state.jsonl identical (it is part of the recorded input);
  5. replay1 == replay2 on all of the above (two VO threads share OpenCV's pool: the replay itself is deterministic).
Writes RUN/replay_check.json and exits 1 on any mismatch. std-lib only (runs on the AWS host's python3).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def run_dir(base: Path) -> Path:
    ds = [d for d in base.iterdir() if d.is_dir() and (d / "results.yaml").exists()]
    if len(ds) != 1:
        raise SystemExit(f"{base}: expected one ariadne_live run directory, found {[d.name for d in ds]}")
    return ds[0]


def results(d: Path) -> dict:
    out = {}
    for line in (d / "results.yaml").read_text().splitlines():
        if line.startswith("consumed:"):
            out["consumed"] = line.split(":", 1)[1].strip()
        if line.strip().startswith("- {agent:"):
            body = line.strip()[3:-1]
            kv = dict(p.split(": ") for p in body.split(", "))
            out[f"agent{kv['agent']}"] = {"frames": int(kv["frames"]), "keyframes": int(kv["keyframes"])}
    return out


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def summary(d: Path) -> dict:
    s = results(d)
    for k in (0, 1):
        tums = sorted((d / "raw").glob(f"*-agent{k}-*/estimate_body.tum"))
        if len(tums) != 1:
            raise SystemExit(f"{d}: expected one estimate_body.tum for agent {k}, found {len(tums)}")
        s[f"agent{k}"]["estimate_body_sha256"] = sha(tums[0])
    trace = []
    for line in (d / "trace.jsonl").read_text().splitlines():
        r = json.loads(line)
        r.pop("mode", None)
        r.pop("executed", None)
        trace.append(json.dumps(r, sort_keys=True))
    s["trace_sha256"] = hashlib.sha256("\n".join(trace).encode()).hexdigest()
    s["trace_records"] = len(trace)
    s["mover_state_sha256"] = sha(d / "mover_state.jsonl")
    if (d / "directives.jsonl").exists():   # operator directives are recorded inputs (AGENTIC_7 §0b)
        s["directives_sha256"] = sha(d / "directives.jsonl")
    return s


def main() -> int:
    base = Path(sys.argv[1])
    live = summary(run_dir(base))
    reps = [summary(run_dir(base / f"replay{i}")) for i in (1, 2)]
    checks = {}
    for i, r in enumerate(reps, 1):
        for key in live:
            checks[f"replay{i}.{key}"] = live[key] == r[key]
    checks["replay1==replay2"] = reps[0] == reps[1]
    ok = all(checks.values())
    out = {"verdict": "PASS" if ok else "FAIL", "live": live, "replay1": reps[0], "replay2": reps[1], "checks": checks}
    (base / "replay_check.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"verdict": out["verdict"], "failed": [k for k, v in checks.items() if not v]}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
