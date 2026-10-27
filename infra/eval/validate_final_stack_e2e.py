#!/usr/bin/env python3
"""Acceptance gate for the integrated Ariadne mission bundle.

This distinguishes a launched container from a mission that actually exercised
the estimator, planner, collaboration and live reconstruction paths.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def require(ok: bool, message: str, failures: list[str]) -> None:
    if not ok:
        failures.append(message)


def main() -> int:
    archive_root = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    expected_recon = sys.argv[3] if len(sys.argv) > 3 else None
    failures: list[str] = []
    # A reconstruction submap may also contain results.yaml. Select the runtime
    # summary by its schema, rather than assuming it is the only non-raw report.
    live_results = []
    for p in archive_root.rglob("results.yaml"):
        summary = p.read_text(errors="replace")
        if re.search(r"^mode:\s*(?:live|live-demo)\s*$", summary, re.M) and re.search(
            r"^consumed:\s*", summary, re.M
        ):
            live_results.append(p)
    require(len(live_results) == 1, f"expected one live results.yaml, found {len(live_results)}", failures)
    if len(live_results) != 1:
        print(json.dumps({"verdict": "FAIL", "failures": failures}, indent=2))
        return 1
    run_dir = live_results[0].parent
    text = live_results[0].read_text()
    replans_match = re.search(r"^replans:\s*(\d+)", text, re.M)
    commands_match = re.search(r"^commands_sent:\s*(\d+)", text, re.M)
    agents = [
        {k: int(v) for k, v in re.findall(r"(agent|frames|keyframes|keyframes_integrated):\s*(\d+)", line)}
        for line in text.splitlines() if line.strip().startswith("- {agent:")
    ]
    require(len(agents) >= 2, f"expected >=2 agent result rows, found {len(agents)}", failures)
    for a in agents:
        require(a.get("frames", 0) > 0, f"agent {a.get('agent')} processed no frames", failures)
        require(a.get("keyframes", 0) > 0, f"agent {a.get('agent')} produced no keyframes", failures)
    require(bool(replans_match) and int(replans_match.group(1)) > 0, "mission made no planner replans", failures)
    require(bool(commands_match) and int(commands_match.group(1)) > 0, "mission issued no movement commands", failures)

    raw_results = list((run_dir / "raw").rglob("results.yaml"))
    require(len(raw_results) >= 2, f"expected per-agent SLAM reports, found {len(raw_results)}", failures)
    for p in raw_results:
        t = p.read_text()
        require("mode: full" in t and "imu_full:" in t,
                f"{p.parent.name} did not report the full inertial estimator", failures)

    splat_manifest = run_dir.parent / "splat" / "manifest.json"
    require(splat_manifest.exists(), "live reconstruction service produced no manifest", failures)
    splat = {}
    if splat_manifest.exists():
        splat = json.loads(splat_manifest.read_text())
        require(splat.get("keyframes", 0) > 0, "reconstruction service ingested no keyframes", failures)
        require(splat.get("training_image") == "rgb", "reconstruction image mode was not RGB", failures)
        if expected_recon:
            require(splat.get("backend") == expected_recon,
                    f"reconstruction backend was not {expected_recon}", failures)
        if expected_recon == "gpu":
            train_log = run_dir.parent / "splat" / "service_log.jsonl"
            records = [json.loads(line) for line in train_log.read_text().splitlines()] if train_log.exists() else []
            require(any(r.get("iters", 0) > 0 for r in records),
                    "GPU reconstruction performed no optimization iterations", failures)

    replay_check = run_dir.parent / "replay_check.json"
    require(replay_check.exists(), "saved-input replay check did not produce replay_check.json", failures)
    replay = json.loads(replay_check.read_text()) if replay_check.exists() else {}
    require(replay.get("verdict") == "PASS", "saved-input replay was not deterministic PASS", failures)
    status = output_dir / "mission_exit_status.txt"
    require(status.exists(), "AWS wrapper did not preserve live/replay exit statuses", failures)
    if status.exists():
        require("live_exit=0" in status.read_text() and "replay_exit=0" in status.read_text(),
                "AWS wrapper reported a live or replay failure", failures)

    report = {
        "verdict": "PASS" if not failures else "FAIL",
        "live_results": str(live_results[0]),
        "agents": agents,
        "replans": int(replans_match.group(1)) if replans_match else None,
        "commands_sent": int(commands_match.group(1)) if commands_match else None,
        "slam_reports": [str(p) for p in raw_results],
        "reconstruction": {k: splat.get(k) for k in ("backend", "training_image", "keyframes", "wall_s")},
        "replay": replay.get("verdict"),
        "failures": failures,
    }
    out = output_dir / "final_stack_acceptance.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
