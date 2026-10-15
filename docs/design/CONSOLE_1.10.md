# Design — Phase 1.10: operator console stub (WebSocket telemetry, 2D top-down map, byte gauges)

Plan 1.10: "Console stub: WebSocket telemetry, 2D top-down map, byte gauges (§29)." Arch §29.3: FastAPI backend,
WebSocket telemetry at 2–10 Hz, REST for approvals; Three.js + a splat renderer frontend; incident timeline (§29.2);
hosted live for judging week (Phase 8). The stub fixes the **interfaces** (telemetry schema, endpoints) and shows
the three things the plan names; the 3D splat view, approvals and live hosting come later.

## Data source in Phase 1: a replayed run chain

The console replays a committed run chain (sim_collab → recon → mission) at a chosen speed. Live runs later publish
the SAME telemetry frames, so the frontend never knows the difference. Two producers gain time-resolved outputs:

- `ariadne_sim_collab` writes `ledger.csv`: every message as the ByteLedger saw it (sequence, class, sender,
  receiver, payload/compressed/wire bytes, the wire stamp, and a **causal send time**). Totals per class (bytes
  and counts) must equal `results.yaml` (asserted). The wire stamp is a keyframe time and is NOT causal (a request
  is stamped with agent 0's keyframe at ~138 s, the response and edge with agent 1's at ~20 s), so the console uses
  the causal time: descriptor = its keyframe time; request = the later of the two keyframes; response = the
  request's time; edge = the response's time — the rule 1.8 uses for t_merge. The wire stamps are unchanged (the
  encoded bytes, and so the 1.6 evidence, stay identical). `[ASSUMPTION]` zero processing and link latency.
- `ariadne_mission` writes `snapshots/slab_<t_ms>.png` at every replan (the 2D tri-state map then).

## Telemetry (WebSocket `/ws/telemetry`, JSON, one frame per tick)

```text
{ "t_s": mission time, "seq": n,
  "agents": [ {"id": 0, "x": m, "y": m, "yaw": rad, "merged": bool} ],   # world frame (briefing anchor + 1.6 merge)
  "bytes":  { "<CLASS>": cumulative wire bytes, ... , "total": B },          # from ledger.csv, by CAUSAL send time
  "rate_Bps": bytes in the last second,
  "map_version": index of the latest snapshot at or before t_s (fetch /api/map/<i>.png),
  "route": {"egress": [[x, y], ...] | null, "agent": [...] | null},          # latest replan at or before t_s
  "events": [ {"t_s", "kind", "text"} ]                                      # new since the last frame
}
```

Agents are drawn from their ESTIMATED poses only, through the same world anchor the mission uses (agent 0's entry
pose; agent 1 through T_O0_O1 and only once merged — before that it is shown as "not merged" with no position, which
is what the broker actually knows). Events come from the chain's own outputs: keyframe counts, each verified edge
(1.6), the merge (t_merge), each replan's frontier decision, the first valid egress route, routes changing exits.

REST: `/api/meta` (grid, exits, target, scene outline, chain ids, speed), `/api/map/<i>.png`, `/healthz`.
Time base: mission time = t_ns − t0 with the mission's t0, for the ledger, trajectories and snapshots alike;
tests assert the console's merge and first-valid-route events land at the mission's 138 s and 140 s. Frontier
decisions appear on the timeline only when the chosen frontier changes, labelled "logged decision · replay · not
executed" (invariant 3). Ticks at 5 Hz wall time; `speed` (default 10× mission time) is a server option. Deterministic: frame n at a given
speed is a pure function of the chain (tested byte for byte).

## Frontend (static, served by the backend)

One page: a canvas top-down map (snapshot image, GT-free outline not shown: GT is not operator knowledge; agents
with trails; egress route at ρ_responder bold, agent route thin), byte gauges per class (cumulative bar + rate), the
incident timeline (§29.2), a status strip (mission time, merged?, route valid?). No CDN libraries needed (Canvas 2D).

## Packaging

`python/operator_console/` (server.py: FastAPI app; replay.py: chain → frames; static/index.html), pinned
`infra/docker/console.Dockerfile` (FastAPI, uvicorn, numpy, pyyaml, opencv-headless), `infra/scripts/console.sh
CHAIN... [--speed]` → http://localhost:8080. Tests: frame builder determinism, ledger totals = results.yaml, the
WebSocket endpoint through FastAPI's TestClient.

## Not in 1.10

3D splat view (the 1.7 viewer exists separately), approvals (§23), raw vs restored feeds (Phase 5), live hosting
(Phase 8), budget gauges against a policy budget (Phase 3 defines budgets; the stub shows cumulative + rate).
