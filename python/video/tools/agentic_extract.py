"""Beat 8 data: one decision from the final integrated live run, copied into data/agentic_final/.

Source: the final integrated live run (Nova 2 Lite via Amazon Bedrock, two robots, 200 s) at
  /data/code/Ariadne-e2e-final/evaluation/runs/live-agentic/final_stack_e2e/20261026T000432Z/.../89e61fecbb277f69-live-20261026T001641Z
Copies the t = 80 s decision images the model was given (observe_image: the map and both robots' keyframes), the
trimmed trace records of that decision (tool-call names and inputs, assignments, candidates' risk and approval flag,
validation, the command), the mover's response to the command, and whole-run counts:
validation ok, decisions with an approval-gated option on offer, gated options picked, mover refusals.

  python3 python/video/tools/agentic_extract.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

RUN = Path("/data/code/Ariadne-e2e-final/evaluation/runs/live-agentic/final_stack_e2e/20261026T000432Z/aws_run/build/live/runs/"
           "live-20261026T001605Z-ff97/89e61fecbb277f69-live-20261026T001641Z")
T_S = 80
OUT = Path(__file__).resolve().parents[1] / "data" / "agentic_final"


def main():
    rows = [json.loads(l) for l in (RUN / "trace.jsonl").read_text().splitlines() if l.strip()]
    mover = [json.loads(l) for l in (RUN / "mover_state.jsonl").read_text().splitlines() if l.strip()]
    OUT.mkdir(parents=True, exist_ok=True)
    for name in (f"t{T_S * 1000}ms_map.png", f"t{T_S * 1000}ms_a0_kf.png", f"t{T_S * 1000}ms_a1_kf.png"):
        shutil.copy(RUN / "decisions" / name, OUT / name)

    recs = sorted((r for r in rows if r["t_s"] == T_S), key=lambda r: r["agent"])
    orch = recs[0]["choice"]["answer"]["orchestrator"]
    decision = dict(
        t_s=T_S, model=orch["model"], latency_s=orch["latency_s"], fallback_used=orch["fallback_used"],
        tool_calls=[dict(tool=c["tool"], input=c["input"], status=c["status"]) for c in orch["tool_calls"]],
        assignments=recs[0]["choice"]["answer"]["assignments"],
        agents=[dict(agent=r["agent"], chosen=r["chosen"], classical_choice=r["classical_choice"],
                     validation=r["choice"]["validation"], command=r["command"], result=r["result"],
                     candidates_total=r["candidates_total"], autonomous_threshold=r["autonomous_threshold"],
                     candidates=[{k: c[k] for k in ("index", "utility", "risk", "needs_approval", "path_m", "cells")}
                                 for c in r["candidates"][:8]],
                     mover=[m["state"] for m in mover if m["agent"] == r["agent"]
                            and m["state"]["cmd_stamp_ns"] == r["command"]["stamp_ns"]]) for r in recs])

    def gated_pick(r):
        c = next((c for c in r["candidates"] if c["index"] == r["chosen"]), None)
        return bool(c and c["needs_approval"])

    totals = dict(
        decisions=len(rows), validation_ok=sum(r["choice"].get("validation") == "ok" for r in rows),
        gated_offered=sum(any(c["needs_approval"] for c in r["candidates"]) for r in rows),
        gated_picked=sum(gated_pick(r) for r in rows),
        mover_commands=len(mover), mover_refused=sum(not m["state"]["accepted"] for m in mover),
        refusal_reasons=[m["state"]["reason"] for m in mover if not m["state"]["accepted"]])
    (OUT / "decision.json").write_text(json.dumps(dict(source=str(RUN), decision=decision, totals=totals), indent=1))
    json.dump(totals, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
