"""Chain checks for the console replay (design live-runtime.md#operator-console). Exit 1 on any failure.

  python -m operator_console.check COLLAB_RUN MISSION_RUN SCENE

- ledger.csv totals (wire bytes AND message counts per class) == sim_collab results.yaml;
- the console's merge event is at the mission's t_merge_s, its first-valid-route event at the mission's
  time_to_first_valid_route_s (one time base);
- causal order: no POSE_EDGE before its MATCH_REQUEST, no edge event after the merge it enables;
- frames are a pure function: two builds at the same speed are byte-identical.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import yaml

from .replay import frames_json, load_chain


def main(argv=None) -> int:
    a = (argv or sys.argv[1:])
    collab, mission, scene = Path(a[0]), Path(a[1]), Path(a[2])
    ch = load_chain(collab, mission, scene)
    res = yaml.safe_load((collab / "results.yaml").read_text())
    mres = yaml.safe_load((mission / "results.yaml").read_text())
    fails = []
    b, n = Counter(), Counter()
    for t, cls, wire, *_ in ch.ledger:
        b[cls] += wire
        n[cls] += 1
    for cls, v in res["bytes"].items():
        if not isinstance(v, dict):
            continue
        if b[cls] != v["wire_bytes"] or n[cls] != v["messages"]:
            fails.append(f"ledger {cls}: {b[cls]} B / {n[cls]} msgs != results {v['wire_bytes']} / {v['messages']}")
    if sum(b.values()) != res["bytes"]["total_wire_bytes"]:
        fails.append("ledger total != results total")
    merge = [e for e in ch.events if e["kind"] == "merge"]
    if len(merge) != 1 or abs(merge[0]["t_s"] - float(mres["t_merge_s"])) > 1e-6:
        fails.append(f"merge event {merge} != mission t_merge_s {mres['t_merge_s']}")
    first = next((e for e in ch.events if e["kind"] == "route"), None)
    if first is None or abs(first["t_s"] - float(mres["time_to_first_valid_route_s"])) > 1e-6:
        fails.append(f"first route event {first} != mission {mres['time_to_first_valid_route_s']}")
    req = min((t for t, c, *_ in ch.ledger if c == "MATCH_REQUEST"), default=None)
    edge = min((t for t, c, *_ in ch.ledger if c == "POSE_EDGE"), default=None)
    if req is None or edge is None or edge < req:
        fails.append(f"causality: first POSE_EDGE {edge} before first MATCH_REQUEST {req}")
    if edge is not None and edge > float(mres["t_merge_s"]) + 1e-6:
        fails.append(f"the first verified edge ({edge} s) comes after the merge it enables")
    f1, f2 = frames_json(ch, 10.0), frames_json(ch, 10.0)
    if f1 != f2:
        fails.append("frames are not deterministic")
    print(f"ledger: {sum(n.values())} messages, {sum(b.values())} B over {len(n)} classes")
    print(f"events: {len(ch.events)}; merge {merge[0]['t_s'] if merge else None} s; first route {first['t_s'] if first else None} s")
    print(f"frames at 10x: {len(f1)} ({sum(len(x) for x in f1)} B of JSON)")
    for f in fails:
        print("FAIL", f)
    print("VERDICT", "PASS" if not fails else "FAIL")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
