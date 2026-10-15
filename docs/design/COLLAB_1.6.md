# Design — Phase 1.6: placeholder collaboration (one verified inter-agent loop)

Plan 1.6: "broadcast every global descriptor, verify one inter-agent loop with `findEssentialMat`, exchange pose
edges through the real serializer so bytes are counted from day one; trivial broker = agent 0 (§15)". Everything
runs in **replay** from one sim recording (both agents; the evidence recording `ci_tiny_0-20261001T133914Z`),
deterministically. The scenes are built for this: worldgen's closure gate guarantees >= 3 inter-agent pose pairs
within 1.5 m / 30° with line of sight (design SIM §4.4), so retrieval can be scored against GT opportunities.

## Pipeline (arch §15.1's two-stage exchange, placeholder internals, real interfaces)

1. **Per agent:** the 1.5 VO (`stereo_vo_runner.hpp`) yields keyframes. Each keyframe keeps: id, stamp, pose in the
   agent's odometry frame, left-image ORB keypoints + descriptors, and stereo 3D points (camera frame) for the
   keypoints that have SGBM/LK depth. The runner gains a keyframe hook; EuRoC output must stay byte-identical.
2. **Global descriptor (placeholder):** the per-bit mean of the keyframe's 256-bit ORB descriptors — a 256-D
   float vector, L2-normalized ("mean-pooled", plan 1.6). Agent 1 sends one per keyframe to the broker (agent 0,
   the lowest AgentId, §15.4) as `GLOBAL_DESCRIPTOR`.
3. **Retrieval at the broker:** cosine top-1 among agent 0's keyframes; a candidate if above a threshold
   (`collab_v1` config). Placeholder: no λ₂ prioritization (Phase 3).
4. **Candidate request → feature response:** broker sends `MATCH_REQUEST` (query kf, candidate kf); agent 1
   answers `MATCH_RESPONSE` with keypoints, ORB descriptors, its 3D points and camera metadata (§15.2 "feature
   response"). Compressed class (serializer already compresses 4-5).
5. **Verification:** ORB Hamming matching with a ratio test, then `findEssentialMat` RANSAC as the geometric gate
   (min inliers, min inlier ratio). Metric SE(3) from `solvePnPRansac` on agent 1's 3D points against agent 0's
   2D keypoints (the essential matrix alone has no scale). Edges are SE(3) (locked decision).
6. **Pose edge:** the broker emits `POSE_EDGE` (the existing `LoopEdge` table: agents, keyframes, measurement,
   21-float information upper triangle, frame tags, inlier count, retrieval score). Information: placeholder
   diagonal scaled by inlier count, labelled `[PROXY]`.
7. **Merge (trivial broker):** the first accepted edge gives T_W0_W1; agent 1's trajectory is mapped into agent
   0's frame. No PGO yet (Phase 3).

## Wire additions (`wire.fbs`; classes already reserved)

`GlobalDescriptor {keyframe_id, stamp_ns, descriptor:[float], quality}`,
`MatchRequest {request_id, query_keyframe, candidate_keyframe, feature_type}`,
`FeatureResponse {request_id, keyframe_id, keypoints:[float], descriptors:[ubyte], points3d:[float] (NaN = no depth),
camera:[float] (fx fy cx cy baseline)}` — added to `Envelope`; encode/decode in `WireCodec`; bytes into `ByteLedger`
per class. Idempotent receive (§15.3): per-sender monotonic sequence high-water mark + small out-of-order buffer;
duplicates acknowledged and dropped — the Phase 1 exit's protocol test injects duplicates and reordering.

## Measured outputs (`ariadne_sim_collab`, results.yaml + manifest)

- Retrieval vs GT opportunities: candidates proposed, true/false (GT relative pose within 1.5 m / 30°).
- Each accepted edge vs GT: translation/rotation error of T_kfA_kfB against the canonical GT
  ((T_W_BA T_B_C)^-1 (T_W_BB T_B_C)).
- Merged two-agent ATE in one frame (one alignment for both agents).
- Bytes and message counts per class from the `ByteLedger` (the §15.7 table's first real numbers).
- Edge correctness separately from the cone: an edge is correct within `edge_correct_t_m` / `_r_deg` of GT
  (the cone labels retrieval opportunities, it does not judge an edge).
- Retrieval over all queries (before threshold and cap): queries with any GT opportunity, top-1 true count, top-1
  score min/median/max.

## Fault injection (`--inject-faults 1`)

Duplicate every 5th send; hold every 3rd delivery back so it arrives after the next flush's newer messages (a
same-class sequence gap the inbox must buffer); drain at the end. Outputs are built from a rank-keyed outcome map
after the exchange (merge = best-ranked verified edge, CSV by rank), so they cannot depend on arrival order. The
run fails unless both inboxes saw duplicates and buffered reorders; acceptance is `candidates.csv` byte-identical
and `results.yaml` identical except `inject_faults` / `protocol`. Results: D43.

## Code layout

`cpp/collaboration/retrieval/` (descriptor + top-1), `cpp/collaboration/protocol/` (payload helpers + idempotent
inbox), `cpp/collaboration/broker/` (verification + trivial merge), tool `cpp/runtime/tools/ariadne_sim_collab.cpp`,
`configs/collab/collab_v1.yaml`, `core.sh sim-collab`.
