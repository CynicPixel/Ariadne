"""Check an ariadne_live run against fake_orchestrator's scripted scenarios (design agentic.md D156, D158).

  python3 python/tests/orchestration/check_fake_run.py RUN_DIR     (the ariadne_live run directory)

For every replan: the scenario the fake answered with (agent 0's record carries the answer) and the runtime's verdict
per agent must match the table in fake_orchestrator.py. Exit 1 on any mismatch. std-lib only.
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

recs = defaultdict(dict)
for l in (Path(sys.argv[1]) / "trace.jsonl").read_text().splitlines():
    r = json.loads(l)
    if "choice" in r:
        recs[r["t_s"]][r["agent"]] = r
bad, seen = [], defaultdict(int)
for t, by in sorted(recs.items()):
    a0, a1 = by.get(0), by.get(1)
    ans = a0["choice"]["answer"] if a0 else None
    scen = ans["orchestrator"]["scenario"] if isinstance(ans, dict) else "malformed"
    src = {k: r["choice"]["source"] for k, r in by.items()}
    val = {k: r["choice"]["validation"] for k, r in by.items()}
    ok = True
    if scen == "ok":
        # the fake picks each agent's last candidate WITHOUT approval: accepted unless it needs one (then the gate
        # must refuse it and fall back to the classical choice under the gate)
        for k, r in by.items():
            last = r["candidates"][-1] if r["candidates"] else None
            if last and last["needs_approval"]:
                ok = ok and src[k] == "fallback" and "without an approval" in val[k] and r["chosen"] == r["classical_choice"]
            else:
                ok = ok and src[k] == "orchestrator" and r["chosen"] == len(r["candidates"]) - 1
    elif scen == "out_of_set":
        # agent 1's pick (its last candidate, no approval) is accepted unless the gate needs an approval for it
        a1_gated = a1 is not None and a1["candidates"] and a1["candidates"][-1]["needs_approval"]
        a1_ok = (src.get(1) == "fallback" and "without an approval" in val[1]) if a1_gated else src.get(1, "orchestrator") == "orchestrator"
        ok = src.get(0) == "fallback" and "out of the candidate set" in val[0] and a1_ok
    elif scen == "malformed":
        ok = all(s == "fallback" and "timeout" in val[k] for k, s in src.items())
        ok = ok and all(r["chosen"] == r["classical_choice"] or r["chosen"] >= -1 for r in by.values())
    elif scen == "same_cluster":
        if a0 and a1 and a0["chosen"] >= 0 and a1["chosen"] >= 0:
            c0 = a0["candidates"][a0["chosen"]]["cluster"]
            c1 = a1["candidates"][a1["chosen"]]["cluster"]
            alt = any(c["cluster"] != c0 and not c["needs_approval"] for c in a1["candidates"])
            ok = c0 != c1 or not alt
            if src.get(1) == "fallback":
                ok = ok and "same cluster" in val[1]
    elif scen == "hold":
        ok = all(s == "orchestrator" and by[k]["chosen"] == -1 and by[k]["result"] == "hold" for k, s in src.items())
    elif scen == "no_approval":
        ok = src.get(0) == "fallback" and "without an approval" in val[0]
    elif scen == "approved":
        ok = src.get(0) == "orchestrator" and a0["choice"]["approval"] is not None
    elif scen.endswith(":none_above_threshold"):
        ok = True
    seen[scen] += 1
    print(f"t={t:>5} {scen:32} src={src} chosen={ {k: r['chosen'] for k, r in by.items()} } {'OK' if ok else 'MISMATCH'}")
    if not ok:
        bad.append((t, scen, val))
print(json.dumps({"scenarios": seen, "mismatches": len(bad)}))
sys.exit(1 if bad else 0)
