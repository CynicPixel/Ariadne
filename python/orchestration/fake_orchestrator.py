"""A scripted stand-in for the orchestrator (no Bedrock, no spend): exercises every validation branch of ariadne_live's
LLM chooser (design agentic.md D156, D158) deterministically, by replaying a live recording against it.

  python -m orchestration.fake_orchestrator --socket SOCK --dir DIR --approvals DIR

Replan n (0-based) gets scenario n % 7, in SCEN's order. Each answer names the scenario in orchestrator.scenario, and
python/tests/orchestration/check_fake_run.py checks the runtime's verdict for it:
  0 ok           both agents: their last candidate (not the classical one) -> source orchestrator
  1 out_of_set   agent 0: candidate 99                                      -> agent 0 fallback, agent 1 ok
  2 malformed    a line that is not JSON for this decision                  -> timeout fallback (both)
  3 same_cluster both agents: the same cluster, another one eligible        -> agent 1 fallback (exactly-once)
  4 hold         both agents: -1                                            -> source orchestrator, hold
  5 no_approval  agent 0: an above-threshold candidate, no approval file    -> agent 0 fallback (only if any is above)
  6 approved     agent 0: an above-threshold candidate, the console approval file written first -> ok
Scenarios 5/6 need a candidate above the threshold; with none, the record says so and the check skips them.
"""
from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path

SCEN = ["approved", "no_approval", "ok", "out_of_set", "malformed", "same_cluster", "hold"]   # the first replans carry above-threshold candidates on the L1 recording


def answer(req: dict, n: int, approvals: Path) -> str | None:
    s = SCEN[n % len(SCEN)]
    did = req["decision_id"]
    ag = {a["agent"]: a for a in req["agents"]}
    asg = []
    for k, a in ag.items():
        c = len(a["candidates"]) - 1 if a["candidates"] else -1
        asg.append({"agent": k, "candidate": c, "reason": "scripted", "approval_id": ""})
    if s == "out_of_set":
        asg[0]["candidate"] = 99
    elif s == "malformed":
        return "{this is not json\n"
    elif s == "same_cluster" and len(ag) == 2:
        # pick, for agent 1, a candidate in agent 0's chosen cluster, if one exists
        c0 = ag[0]["candidates"][asg[0]["candidate"]]["cluster"] if asg[0]["candidate"] >= 0 else None
        same = [i for i, c in enumerate(ag[1]["candidates"]) if c["cluster"] == c0]
        if same:
            asg[1]["candidate"] = same[0]
    elif s == "hold":
        for e in asg:
            e["candidate"] = -1
    elif s in ("no_approval", "approved"):
        hi = [i for i, c in enumerate(ag[asg[0]["agent"]]["candidates"]) if c["needs_approval"]]
        if hi:
            asg[0]["candidate"] = hi[0]
            if s == "approved":
                aid = f"{did}-a{asg[0]['agent']}-c{hi[0]}"
                asg[0]["approval_id"] = aid
                adir = Path(req["approvals_dir"])   # where the runtime looks (as the MCP server does)
                adir.mkdir(parents=True, exist_ok=True)
                (adir / f"{aid}.json").write_text(json.dumps({"verdict": "approved", "operator": "test"}))
        else:
            s += ":none_above_threshold"
    return json.dumps({"decision_id": did, "assignments": asg,
                       "orchestrator": {"model": "fake", "scenario": s, "calls": 0, "tool_calls": []}}) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--socket", required=True)
    ap.add_argument("--approvals", required=True)
    a = ap.parse_args()
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    Path(a.socket).unlink(missing_ok=True)
    srv.bind(a.socket)
    srv.listen(1)
    conn, _ = srv.accept()
    f = conn.makefile("rwb")
    n = 0
    for line in f:
        req = json.loads(line)
        out = answer(req, n, Path(a.approvals))
        n += 1
        if out:
            f.write(out.encode())
            f.flush()
    time.sleep(0.1)


if __name__ == "__main__":
    main()
