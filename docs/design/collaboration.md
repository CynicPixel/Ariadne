# Multi-agent collaboration protocol: two-stage exchange, verification, broker, merging

Plan 3.2: "two-stage exchange, bookkeeping/high-water marks, store-and-forward with priority classes and TTLs
(§15.3), heartbeats, anchor propagation and frame-tagged poses (§15.5), broker election with epochs (§15.4)".
Exit (I5): under loss, latency, partition, reorder and disconnect, **every run ends `consistent` or `explicitly
incomplete`, never silently corrupted**. The split-brain test is a required part of that exit: two partitions each
elect a broker, reconnect, and merge under a new epoch with no pose-graph corruption.

Decisions: D80 (Phase 3 opened, ID block) and D81 (this design). Code: `cpp/collaboration/protocol/`. Everything
in it is OpenCV-free, ROS-free and wall-clock-free: it runs on the sim clock, so every test is a seeded
discrete-event run, executed as a unit test.

## 0. Preflight (design-preflight rule)

**Literature.** Each source below either confirms or changes this design.

- **Swarm-SLAM**, Lajoie & Beltrame, RA-L 2024, arXiv:2301.06230. The arch's reference.
  - Verified in its code (`lajoiepy/cslam`, `src/back_end/decentralized_pgo.cpp`, `is_optimizer()`):
    - The optimizer is the agent with the lexicographically lowest **(origin_robot_id, robot_id)** among its
      **current one-hop neighbours**. The origin is the anchor agent, and it propagates with every returned
      estimate (`optimized_estimates_callback`).
    - So "lowest AgentId" in arch §15.4 is a simplification.
  - Ariadne adopts the exact rule (§4 below).
  - Swarm-SLAM has no epochs. Its partition story is "re-elect and re-optimise". The epochs are ours (arch §15.4).
- **Tian et al., "Near-optimal budgeted data exchange for distributed loop closure detection"** (RSS 2018,
  arXiv:1806.00188).
  - The minimum data needed to verify a candidate set is a minimum vertex cover of the exchange graph.
  - It is NP-complete for three or more robots.
  - This is the basis of the `sent_local_feature_set` bookkeeping (§3), which 3.4 prices against.
- **CERBERUS / M3RM** (DARPA SubT): only completed submap transmissions are integrated. This is the same principle
  as our "explicitly incomplete" status (§6).
- **Lajoie et al. 2026, planetary-analogue C-SLAM** (arXiv:2601.21063).
  - Real peer-to-peer links drop at roughly 40 m.
  - A budget counted in loop closures can exceed the throughput actually available.
  - This supports budgeting in bytes (§16.3) and an explicit link seam (§2).
- **COSMO-Bench** (McGann, Potokar, Kaess 2025, arXiv:2508.16731): a back-end benchmark with a communication model
  built from real Wi-Fi data. It is a candidate realism reference for 3.3's link model, not for this item.
- **Delay-tolerant networking (DTN):** no 2024–26 paper found that combines DTN or bundle-protocol routing with
  collaborative SLAM. The multi-hop store-carry-forward literature is non-SLAM (e.g. data ferrying with contact-graph
  routing). Hence the narrow reading in §3: messages are held until the destination is reachable, and nothing is
  relayed.

**OpenCV-native check.** Nothing in this layer is image analysis, and OpenCV offers no networking or
state-machine primitives that fit.
- FlatBuffers stays the single serializer, as locked.
- 3.4 note: OpenCV has only dense `cv::eigen` (Jacobi), and no sparse Lanczos or LOBPCG. The Fiedler solver will be
  ours, on Eigen.

## 1. Gaps in the Phase-1 protocol this design closes

1. **`Inbox` delivers strictly in order per (sender, class).**
   - A descriptor that expires on its TTL, or a class 4–5 message the policy cancels, never arrives. That leaves a
     permanent sequence hole: every later message buffers until the 4096 cap throws.
   - A lost heartbeat freezes all later heartbeats, which looks like a false partition.
   - Phase 1 injected duplicates and reorders, never drops, so this never surfaced.
   - Fix: delivery semantics per class (§3.1).
2. **No acknowledgement or retransmission.** Without acks there is no way to compute "explicitly incomplete".
3. **No cross-class prerequisites.** An edge can name a keyframe the broker has not yet seen.
4. **No membership, election or epochs.** The broker is hard-wired to agent 0.

## 2. The link seam (published now, so 3.3 implements it rather than redesigns it)

```cpp
struct LinkState { bool up; double latency_s; double loss_p; double bandwidth_Bps; };
class LinkModel {             // 3.3 implements this from RFComms + GT-mesh ray-cast attenuation
  public: virtual LinkState Sample(AgentId from, AgentId to, core::Timestamp t) const = 0;
};
```

- **`Network`** is a discrete-event simulator over a `LinkModel`.
  - **Send.** `Send(from, to, framed_bytes, t)` charges a serialization delay `bytes/bandwidth` on a per-directed-link
    transmit queue.
  - **Fragments and loss.** It splits the message into `DatagramCount` fragments, which keeps the serializer's MTU
    accounting. It drops each fragment with probability `loss_p`. The draw is Philox, keyed by
    `(seed, from, to, msg_seq, fragment)`, so the outcome is independent of event order.
  - **Delivery.** Each fragment arrives at `t + queueing + latency`, and the **receiver reassembles per fragment**
    (§3.2). A fragment sent while the link is down is lost; it is not deferred.
  - **Why fragment-level.** Whole-message delivery would not converge under loss: a 15-datagram graph at 30% loss
    survives with p = 0.7^15 ≈ 0.5%, and a 39-datagram feature response at 10% loss with 0.9^39 ≈ 1.6%.
- **`FaultPlan`** layers seeded operators over any `LinkModel`:
  - scripted partitions as component lists over time windows;
  - disconnect of one agent over a window;
  - extra latency and jitter;
  - duplication, and reorder by holding a message back.
- **Ownership.** 3.3's comms broker node owns the real link model and the ray-cast. This item's tests use a
  `ScriptedLinkModel`.
- **Who writes `ByteRecord`s.** Each `Endpoint` records every first transmission, retransmission, heartbeat and ack
  at the serialization boundary (§15.7). In the runtime, all Endpoints record into **one `ByteLedger` owned by 3.3's
  comms node**, which makes the comms node the single owner of the byte record (§15.6). The comms node enforces link
  state on datagrams it forwards and must **not** re-record them, or bytes are counted twice.

## 3. Agent-side protocol state

### 3.1 Delivery semantics per class (replaces strict per-stream ordering)

| class | priority (§15.3) | delivery | reliability | TTL |
|---|---|---|---|---|
| HEARTBEAT (0) | 1 | **latest-wins** (older sequence dropped) | none, periodic (1 s) | superseded |
| ACK (12) | 1 | latest-wins per (peer, class) | none; the ack is cumulative and repeated | superseded |
| POSE_EDGE (5), POSE_GRAPH (6) | 2 | **reliable, in order** per (sender, class): the high-water mark and reorder buffer are kept | acks + retransmit | never; a graph batch is superseded (§3.3) |
| GLOBAL_DESCRIPTOR (1) | 3 | **idempotent set**, keyed by (sender, keyframe), unordered | **acked, resent until TTL** (D82: a broker cannot re-request a descriptor it never heard of) | 120 s |
| MATCH_REQUEST (3) | 3 | idempotent set keyed by request_id | best effort; the broker re-requests on timeout | 120 s |
| MATCH_RESPONSE (4) | 4 | idempotent set keyed by request_id; a response to an unknown or cancelled request is dropped | best effort | cancellable by policy |
| SUBMAP_* (7, 8) | 5 | idempotent set keyed by (sender, submap id) | best effort | cancellable by policy |

Two placements need comment:
- MATCH_REQUEST is not one of §15.3's five classes. It is placed with descriptors: it is small and belongs to the
  same retrieval round.
- Making best-effort classes re-requestable instead of reliable is what makes TTL and cancellation safe. A hole in
  an unordered set stalls nothing.

### 3.2 Acks and retransmission

- **Fragments.**
  - Every message larger than one datagram carries `(msg_id, frag_index, frag_count)` in a 6 B fragment header. The
    header is charged, and it sits inside the 1472 B datagram payload.
  - The receiver reassembles, and delivers a message once all its fragments are in.
  - Partial reassemblies of best-effort classes are discarded at their TTL or cancellation.
- **Ack.**
  - The receiver acks each reliable stream with its `next_expected`, which is cumulative.
  - For the one message being reassembled at the head of that stream, the ack also carries a **fragment bitmap**.
    Feature responses are best effort but multi-datagram, so they get a bitmap ack too. Only missing fragments are
    resent.
  - The ack rides on the heartbeat (new field `acks:[AckEntry]`). It also goes out as a standalone `ACK` within
    0.2 s of a reliable delivery if no heartbeat is due, so retransmit timing does not hinge on the heartbeat period.
- **Retransmit (selective).**
  - Fragments unacked past the timeout are resent with the same sequence and fragment ids. The high-water mark and
    the reassembly table remove duplicates.
  - The timeout is `max(1 s, 4·latency estimate)`, doubling per retry up to 16 s.
  - Retries are unbounded **to the current broker**. These classes are the product; under a partition they wait
    (§3.3).
  - On an election change, unacked reliable messages to the old broker are **abandoned and counted**. They are
    re-sent to the new broker anyway, since broker state is soft (§4).
- **Sequence numbers are assigned at first transmission, not at enqueue.** A queued message that is superseded
  (§3.3), expires or is cancelled therefore never consumed a number, and can never open a hole in a reliable stream.
- **Accounting.**
  - Every fragment, retransmit and ack passes through the serializer.
  - A `ByteRecord` gains `retransmit:bool`; `ledger.csv` is not a pinned reference.
  - "First transmission" in §7 means each fragment's first send.

### 3.3 Outbox: store-and-forward, narrow reading

- **Structure.** One outbox per destination. Strict priority classes 1–5; first-in first-out within a class.
- **Drain.** It drains only while `LinkModel` reports the link up, and the bytes it sends each tick fit the link's
  bandwidth.
- **Unreachable destinations.** While the destination is unreachable, messages are **held** at the origin. No
  relaying: sequence numbers stay per origin and destination, and multi-hop relay is out of scope (§0 DTN note).
- **Expiry and cancellation.**
  - Descriptors and requests expire 120 s after creation; the check happens at dequeue.
  - Classes 4–5 can be cancelled by `request_id`. This is the only hook §16's policy has.
- **Superseding pose-graph batches (a recorded deviation, D81).**
  - §15.3 says graphs "never expire". Read literally, the queue is unbounded under a long partition.
  - Instead, a newer POSE_GRAPH batch from the same agent **replaces** a queued, unsent older one. "Never expire"
    still holds in substance: a graph is a full snapshot of the agent's subgraph, so the newer batch contains
    everything the older one did.
  - Batches already in flight or unacked are never pulled back.
  - Edges are never superseded.
- **Graph-batch cadence.** This drives the byte table, so it is stated here. An agent sends a full snapshot of its own
  subgraph (nodes, odometry edges, covariances; §15.2) to its broker:
  - on every election change;
  - when its keyframe watermark is 10 or more keyframes past the last acked batch;
  - otherwise, at most every 10 s, if anything changed.

  These are snapshots, not deltas, because a delta stream cannot be superseded. The quadratic cost is reported, not
  hidden. Deltas are a later byte optimisation and need their own decision.
- **Bookkeeping (§15.3).** For each peer: `sent_global_descriptor_set`, `sent_local_feature_set`,
  `sent_pose_edge_set` and `sent_submap_set`.
  - They are recorded when a message is first transmitted. For reliable classes, "delivered" is a separate state set
    on ack.
  - When the broker changes, the new broker's sets start empty, because broker state is soft (§15.4). Edges and the
    latest graph are re-sent to it.
  - 3.4's cost model reads `sent_local_feature_set` (a feature response already sent costs 0, and keyframes shared
    by several candidates are priced once, the vertex cover).

### 3.4 Prerequisite hold (separate from the reorder buffer)

- **Holding.** At the broker, a POSE_EDGE whose endpoint keyframe is not yet in that agent's latest graph batch is
  held, keyed by the missing `(agent, keyframe)`.
- **Release.** It is released when a graph batch covers that keyframe; the heartbeat's `keyframe_watermark` says one
  is coming.
- **Accounting.** Held edges count as incomplete (§6).
- **What is not a prerequisite.** A MATCH_RESPONSE answers a request the broker issued, so a response with an unknown
  request id is stale and is dropped.

## 4. Membership, election, epochs

The heartbeat gains these fields, appended so older encodings still decode:
- `neighbors:[uint]`: agents heard, with the hysteresis below;
- `anchor:uint`: the agent's current anchor, i.e. origin;
- `broker:uint`: the broker it elected;
- `epoch:Epoch`: the latest epoch it accepted;
- `acks:[AckEntry]`.

The 1.6 tool sends no heartbeats, so the pinned `figures/bytes.json` is unaffected.

**Neighbours are symmetric, with hysteresis.**
- `j` becomes a neighbour of `i` when `i` has heard `j` in `m = 2` of the last 3 heartbeat periods **and** `j`'s latest
  heartbeat lists `i`.
- It stops being one after `k = 12` consecutive missed periods at a 0.5 s heartbeat (6 s), or after `j`'s heartbeats
  stop listing `i` for `k` periods.
- **k is sized from the loss rate.** A false drop needs k consecutive lost heartbeats, `p^k` per period.
  - The first choice, `k = 4` at 1 Hz, gave 0.3^4 ≈ 0.8%/period: measured 40 broker changes in 290 s with 4 agents.
  - `k = 12` gives 0.3^12 ≈ 5·10⁻⁷, i.e. ≈ 4·10⁻³ expected false drops per 300 s run over 12 directed links.
  - The price is 6 s to detect a real partition.
- Hysteresis stops loss-driven flapping. Test: steady 30% loss with no partition gives zero broker changes.
- An asymmetric link, where only one side hears the other, is therefore not a neighbour link.
- This is what stops two agents with opposite views from both electing themselves.
- Test: asymmetric link, one broker or none, never two brokers claiming one agent.

**Election (Swarm-SLAM's exact rule).**
- Each agent elects `argmin over (anchor_id, agent_id)` across itself and its symmetric neighbours.
- The comparison is lexicographic, so the lowest anchor wins first and the lowest agent id breaks ties.
- This is the one-hop rule Swarm-SLAM runs. With no relaying (§3.3), a broker can only serve agents it reaches
  directly, so the "component" of arch §15.4 is the broker's symmetric one-hop neighbourhood. Arch §15.4 is updated
  to say so (§49 note in D81).
- With a chain topology `0–1–2`, agent 2 elects 1 and agent 1 elects 0.
  - Agent 0 serves only agents that both reach it and elected it.
  - Agent 2 carries its data until it can reach a broker. This is the "store-and-forward" mode of §5.7.

**Epochs.**
- A broker solve is stamped `Epoch{counter, members_hash, broker}`.
- `members` is the sorted set of agents that, at solve time, are symmetric neighbours of the broker **and** elected
  it, plus the broker itself.
- `counter = 1 + max(the counters in the members' heartbeats)`. This is the merge rule. After two partitions each
  reach counter 5, the merged component's first solve is 6, so it is strictly newer than everything either side
  accepted.
- **Acceptance.** An agent accepts a returned estimate only if all three hold:
  1. it is in `members`;
  2. the sender is the broker it currently elects;
  3. `counter > last accepted counter`.
- **Staleness.** Anything else is stale and is dropped, and the drop is counted. "Stale" means precisely: not a
  member, not my broker, or not newer.

**Broker state is soft.**
- On election change, an agent re-sends its latest graph batch and all its edges to the new broker. The edges come
  from `sent_pose_edge_set`, which is kept per agent and is not discarded.
- Broker loss therefore costs bytes and time, never consistency.

## 5. Frames (§15.5)

- **Frames.**
  - `FrameTag{agent, epoch}`: `epoch = 0` means `W_agent`, the agent's odometry frame.
  - `epoch > 0` means `G` as defined by that epoch's anchor agent, `FrameTag.agent = anchor`.
- **Each agent's registry.**
  - It keeps the frames it knows: its own `W_i`, plus every `G` frame from an epoch it accepted.
  - It keeps `T_G_Wi` for the current `G`.
- **Rejection (strengthened from Phase 1).** The decoder already rejects untagged poses. It now also rejects a pose
  whose `FrameTag` is not in the receiver's registry: a deserialization error, counted, never composed.
- **Anchor propagation.**
  - In a merge, the lower anchor id wins.
  - Members from the losing side update `T_G_Wi <- T_Gnew_Gold · T_Gold_Wi`. Here `T_Gnew_Gold` is part of the
    estimate the broker returns.
  - Stored geometry stays in `W_i` and is never rewritten (§18.6).
- **Broker solve gauge.**
  - The anchor agent's first pose is fixed in full (rotation and translation).
  - The yaw-only gauge (§15.4, with gravity priors) sits behind `gauge: yaw_only`. 2.5 turns it on once its solver
    consumes `GRAVITY_PRIOR`.
  - **When the anchor is not in the component.** Example: partition {2,3} inherited anchor 0 from an earlier merge.
    There are no anchor nodes to fix, so the broker fixes the lowest member's `T_G_Wi` at its last accepted value.
    `G` stays the same frame, so the side's estimate still composes with the other side's at reconnection.
  - A component that never had an anchor (first contact) anchors at its own lowest member, as in §15.5.
- **Robust stage.** The solve uses 2.5's robust pipeline (`cpp/slam/robust/`: degeneracy gate → PCM → GNC-TLS) once
  2.5 merges. Until then it uses chordal initialisation plus PGO, which are on this branch already.
  - PCM and GNC are not duplicated under `cpp/collaboration/`: inter-agent edges use `cpp/slam/robust/{pcm,gnc}.hpp`.

## 6. Run outcome: `consistent` | `explicitly incomplete` | `corrupt`

At end of run (after a drain period), the harness computes three things.

**Per agent:**
- unacked reliable messages, by class and destination;
- held prerequisites;
- its last accepted epoch;
- the digest of the estimate it adopted.

**Corrupt**, which is a test failure:
- an agent adopted an estimate whose digest differs from the broker's for that epoch. The digest is taken over the
  **wire-encoded** estimate (float32 poses). The double-precision solve would differ; this is the Phase-1 gate lesson;
- **or** a pose was composed in a frame it was not expressed in. The frame registry asserts this; it does not merely
  count it;
- **or** an accepted epoch's member set does not contain the agent;
- **or** two agents accepted different estimates under the **same full epoch `(counter, members_hash, broker)`**. The
  counter alone legitimately repeats across a split brain.

**Consistent:**
- no unacked reliable messages;
- no held prerequisites;
- every agent in each final component accepted the same latest epoch;
- that epoch's graph contains every edge any member produced.

**Explicitly incomplete:** everything else. The reason list says exactly what is missing (which messages, held
edges, absent members), never just "incomplete".

## 7. Byte-gate definition (I6), fixed before 3.4 produces numbers

- **Predicted.** The policy's cost model of §16, evaluated before sending: the §15.7 table applied to the message as
  planned. For example, a feature response is priced from its keypoint count and descriptor encoding.
- **Measured.** The serializer's `wire_bytes` for the same messages, **first transmission only**: payload, plus
  compression, plus 28 B per datagram.
- **The gate.** It is computed per class over classes 1–6, and fails if `|measured − predicted| / measured > 15%` in
  any class.
- **Compressed classes (4, 5, 6).** §15.7 states uncompressed sizes, and zstd can expand small edges. The predictor
  therefore includes a per-class compression-ratio model, **calibrated on runs disjoint from the gate runs** (another
  scene or other seeds, recorded in the manifest). So classes 5–6 can neither fail on an unmodelled ratio nor pass by
  fitting the gate data.
- **Overhead reported separately, never folded in or omitted.**
  - Retransmits, acks and heartbeats are link and protocol overhead, not prediction error.
  - They appear as their own rows of the byte table.
  - They also appear as a total-including-overhead curve on the ATE-vs-bytes plot. So the headline curve is never the
    flattering one alone.

## 8. Tests (unit tests: seeded discrete-event runs, sim clock; no premerge run needed)

1. **Inbox per-class semantics:**
   - drop descriptor sequence N, and later sequences are still delivered;
   - lose a heartbeat, and the next one is delivered;
   - duplicates are dropped;
   - reorder holds back reliable classes only.
2. **Outbox:** strict priority, first-in first-out, the 120 s TTL at dequeue, cancellation by request id,
   superseding graph batches (and never edges), and hold-while-down.
3. **Acks and retransmit:** 30% loss on a reliable stream of 15-datagram graphs ends fully acked, with only the
   missing fragments resent. Retransmits are counted in the ledger.
4. **Network:**
   - the same seed gives byte-identical event logs;
   - per-fragment loss matches `1 − (1 − p)^k` within a binomial bound;
   - nothing is delivered across a partition.
5. **Election:**
   - asymmetric link;
   - chain `0–1–2`;
   - anchor-first ordering, where a lower agent id with a higher anchor loses.
6. **Split-brain:**
   - Partition {0,1} | {2,3}: each side elects its own broker and solves under its own epochs.
   - Reconnect: one broker (the lower anchor), a counter above both sides, and anchor propagation.
   - The final estimate is `consistent`.
   - The final **graph contents** (node and edge sets) equal the no-partition run's exactly.
   - The **estimate** agrees within a tolerance. A different init or warm-start path legitimately changes the floats,
     and can change GNC's inlier set.
7. **Sequence numbers and superseding:**
   - a queued graph batch superseded before transmission leaves no sequence hole;
   - an election change abandons (and counts) unacked messages to the old broker, and the run still ends
     `consistent`.
8. **Flapping:** steady 30% loss with no partition gives zero broker changes.
9. **Frames:** an unknown frame is rejected at decode, and the anchor-merge composition is checked against ground
   truth.
10. **Fault campaign:** {loss 0/10/30%} × {latency/jitter, reorder, duplicate, partition, disconnect of one agent,
   broker disconnect} × 3 seeds, on a synthetic 4-agent scenario (drifting odometry, ground-truth loop edges). Every
   run ends `consistent` or `explicitly incomplete`; zero `corrupt`. These are synthetic graphs, so the runs are
   fast. Sim and EuRoC integration comes after 3.3 and 3.4.

## 9. What stays

- **The Phase-1 exchange** (`ariadne_sim_collab`, the in-order `Inbox`) stays as the 1.6 baseline. Phase-1
  placeholders are demoted to baselines, never deleted.
- **The new protocol** gets its own runtime node. The sim tool switches to it under `protocol: v2` once 3.4's policy
  exists, so the Phase-1 gate's pinned `bytes.json` keeps describing the Phase-1 path until a recorded re-pin.

## 10. Changes forced by the tests (D82)

The fault campaign (§8.10) and the split-brain test found four design gaps. Each is fixed and recorded here.

1. **Descriptors are acked and resent until their TTL.**
   - Fire-and-forget descriptors sent into a partition were lost but recorded as sent, so no later broker ever got them.
     The split-brain run lost every cross pair near the partition start.
   - Now: a descriptor is resent until acked or 120 s old. An expired one leaves the sent set, so the next broker is
     sent it again.
2. **Acks are repeated, and the retransmit timeout is capped at 8 s.**
   - A done id rode one ack. At 30% loss a round (data + ack) succeeds 0.7·0.7 ≈ 49%. With a 16 s timeout cap, a 90 s
     drain left ≈1% of ~600 edge messages unacked (measured: 1–4 per run).
   - Now: each id rides 3 acks (ack success 1 − 0.3³ ≈ 97%) and the timeout cap is 8 s.
3. **Reliable streams have incarnations: `seq = incarnation << 40 | n`.**
   - Abandoning in-flight reliable messages on a broker change spent sequence numbers that never arrived. The new
     broker's in-order stream from that agent then waited forever: 435 edges held for snapshots in every partition and
     broker-disconnect run.
   - Now: `AbandonReliable` starts a new incarnation. A receiver that sees one resets the stream and drops what it had
     buffered from the old one; the application re-sends that content anyway.
   - This is the same hole class as superseding (§3.2), entered through abandonment.
4. **The broker re-solves when an edge arrives or its member set changes**, not only when a new snapshot arrives.
   Otherwise edges re-sent after a merge never reached a solve.

**Result:** 54 runs (6 faults × loss 0/10/30% × 3 seeds) all end `consistent`, zero `corrupt`. Split-brain merges under
counter > both sides, the lower anchor wins, every T_G_W matches ground truth within 2 mm / 2 mrad, and the edge set
equals the no-partition run's.

## 11. Runtime `verify` hook: what may accept an inter-agent edge (L7, from D63)

D63 measured **4 confident wrong verifications** on EuRoC Vicon rooms: 210–264 essential-matrix inliers, rotation error
37–59°. An inlier count alone is therefore not a loop gate. The runtime `NodeHooks::verify` must apply, in order:
1. geometric verification (ALIKED+LightGlue or ORB; essential-matrix RANSAC);
2. metric SE(3) from PnP on the candidate's stereo 3D points;
3. the D71 degeneracy gate on the edge's own covariance.

Then the broker's robust stage (PCM → GNC-TLS) runs over the batch before the solve.

An edge that passes step 1 alone is never emitted. The protocol tests' oracle `verify` is exempt: it is ground truth
by construction.

---

<a id="first-collaboration-loop"></a>

# First collaboration loop

Plan 1.6: "broadcast every global descriptor, verify one inter-agent loop with `findEssentialMat`, exchange pose
edges through the real serializer so bytes are counted from day one; trivial broker = agent 0 (§15)". Everything
runs in **replay** from one sim recording (both agents; the evidence recording `ci_tiny_0-20261001T133914Z`),
deterministically. The scenes are built for this: worldgen's closure gate guarantees >= 3 inter-agent pose pairs
within 1.5 m / 30° with line of sight (design SIM §4.4), so retrieval can be scored against GT opportunities.

### Pipeline (arch §15.1's two-stage exchange, placeholder internals, real interfaces)

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

### Wire additions (`wire.fbs`; classes already reserved)

`GlobalDescriptor {keyframe_id, stamp_ns, descriptor:[float], quality}`,
`MatchRequest {request_id, query_keyframe, candidate_keyframe, feature_type}`,
`FeatureResponse {request_id, keyframe_id, keypoints:[float], descriptors:[ubyte], points3d:[float] (NaN = no depth),
camera:[float] (fx fy cx cy baseline)}` — added to `Envelope`; encode/decode in `WireCodec`; bytes into `ByteLedger`
per class. Idempotent receive (§15.3): per-sender monotonic sequence high-water mark + small out-of-order buffer;
duplicates acknowledged and dropped — the Phase 1 exit's protocol test injects duplicates and reordering.

### Measured outputs (`ariadne_sim_collab`, results.yaml + manifest)

- Retrieval vs GT opportunities: candidates proposed, true/false (GT relative pose within 1.5 m / 30°).
- Each accepted edge vs GT: translation/rotation error of T_kfA_kfB against the canonical GT
  ((T_W_BA T_B_C)^-1 (T_W_BB T_B_C)).
- Merged two-agent ATE in one frame (one alignment for both agents).
- Bytes and message counts per class from the `ByteLedger` (the §15.7 table's first real numbers).
- Edge correctness separately from the cone: an edge is correct within `edge_correct_t_m` / `_r_deg` of GT
  (the cone labels retrieval opportunities, it does not judge an edge).
- Retrieval over all queries (before threshold and cap): queries with any GT opportunity, top-1 true count, top-1
  score min/median/max.

### Fault injection (`--inject-faults 1`)

Duplicate every 5th send; hold every 3rd delivery back so it arrives after the next flush's newer messages (a
same-class sequence gap the inbox must buffer); drain at the end. Outputs are built from a rank-keyed outcome map
after the exchange (merge = best-ranked verified edge, CSV by rank), so they cannot depend on arrival order. The
run fails unless both inboxes saw duplicates and buffered reorders; acceptance is `candidates.csv` byte-identical
and `results.yaml` identical except `inject_faults` / `protocol`. Results: D43.

### Code layout

`cpp/collaboration/retrieval/` (descriptor + top-1), `cpp/collaboration/protocol/` (payload helpers + idempotent
inbox), `cpp/collaboration/broker/` (verification + trivial merge), tool `cpp/runtime/tools/ariadne_sim_collab.cpp`,
`configs/collab/collab_v1.yaml`, `core.sh sim-collab`.

---

<a id="simulated-collaboration"></a>

# Collaboration on simulator recordings

Plan 3.2/3.3/3.4 meet here. collaboration.md §9 says the sim tool switches to the new protocol under `protocol: v2` once
3.4's policy exists; D102 says no protocol traffic runs over the real link until this exists. All three Phase 3 gate
rows wait on it:
- fault injection and split-brain on the sim;
- the §15.7 byte table, with a hard 15% gate;
- ATE against cumulative bytes, spectral vs greedy vs similarity, on 2 or more scenes × 3 seeds.

Status: revised after design review 1 (2026-10-23). Nothing is built yet.

### 0. Preflight

**Literature.**
- **COSMO-Bench** (McGann, Potokar & Kaess, 2025, arXiv 2508.16731, §IV-B/C) is the closest methodological precedent.
  - It runs each robot's front-end over real recorded data.
  - Inter-robot exchange (scans, then loop-closure detection) is gated over time by a communication model.
    Connectivity is a function of inter-robot distance; bandwidth is shared under spatially local interference.
  - The model is fitted to the real ad-hoc Wi-Fi traces of Lajoie et al. 2026 (arXiv 2601.21063).
  - They also criticise "single-robot benchmarks partitioned to simulate multi-robot". Our recordings are genuinely
    multi-agent: two rigs in one Gazebo world, on one clock.
- **Swarm-SLAM** (Lajoie & Beltrame, RA-L 2024).
  - The front-end is local; inter-robot results feed only the back-end (verified in `lajoiepy/cslam`). So the
    front-end can run open-loop on its recording.
  - The evaluation compares spectral vs greedy at equal candidate budgets, reporting λ₂ and ATE. Our budget is bytes
    (arch §16.3), so the x-axis is cumulative wire bytes.
- **Tian et al., RSS 2018:** budgeted exchange, where the vertex cover decides what must be sent. This is the
  `sent_local_feature_set` that 3.4 prices.
- **RFComms** (D102): PER is a function of the datagram size, `1 − (1 − BER)^(8L)`. That is why §3 puts the size on
  the seam.

**OpenCV-native check.**
- Retrieval: EigenPlaces through `cv::dnn` (3.1, D90 sim export).
- Hard-pair local features: `cv::ALIKED` + `cv::LightGlueMatcher` through cv::dnn (D63, with D90's sim export).
- Geometry: `cv::findEssentialMat` with USAC, `cv::solvePnPRansac`, and SGBM depth (2.1).
- Nothing in the event simulation or the link is image analysis, so there is nothing OpenCV-native to reuse there.

**Arch.** §5.7, §15.2–15.7, §16.3, §31.1, and collaboration.md §6 (outcome) and §7 (byte gate, fixed earlier), D102.

### 1. Time model: two passes, causal release

**Pass A, per agent, per seed; cached by input sha256.** The agent's shipped front-end runs over its recording:
sim_v1 perception plus sim_v2 SLAM with D101 IMU noise, the D69 configuration. Per keyframe it writes a store record:
- `id`;
- `t_final`: the frame time at which the keyframe left the BA window, so its odometry edge became final. This is
  what IntraLoopClosure already waits for (D73);
- `T_O_C` at `t_final`;
- the odometry edge to the previous keyframe, with its covariance;
- the EigenPlaces descriptor;
- the hard-pair local features with stereo 3D points (§2).

**Pass B: a discrete-event run on the recording clock.**
- `CollabNode::AddKeyframe` takes the pose, the odometry edge and the descriptor in one call (collab_node.hpp:187).
  So a keyframe is released whole at `t_final`.
  - That is causal: nothing is known before the agent would know it.
  - The cost is one BA window of latency on the descriptor. This is stated, not modelled away.
- Each 20 ms step of sim time runs `Network::DeliverUntil(t)`, then every `CollabNode::Tick(t, net)`, then the
  releases due by `t`.
- The 3.4 selection runs inside `Tick` on its 5 s period.
- After the last release comes a drain period (`drain_s` 30 s), then the collaboration.md §6 outcome.

**The startup prefix (D87's carry).** VO keyframes before the window anchor never get a final exit edge, so they
are **never released**.
- An agent's graph starts at its first released keyframe id.
- 3.4's Swarm-SLAM chain construction changes from "ids 0..max" to "ids min..max seen". Swarm-SLAM sends every
  keyframe from 0, so the two are the same there. Here this is a one-line change, with a test.
- The count of unreleased prefix keyframes per agent goes into the manifest.

**Seeds.** "3 seeds" means 3 values of `seed`, and each one drives both passes:
- in pass A: the D101 IMU noise operator and the VO's RANSAC seeds;
- in pass B: `DeriveSeed(seed, ·)` for Philox link loss and the verifier's RANSAC.
- D is `none`: the gate scenes are lit. Dark rows belong to Phase 5.
- Pass A therefore runs 2 scenes × 3 seeds = 6 times. All arms, budgets and comms scenarios of one (scene, seed)
  share that pass A.

**Why two passes.**
- **The front-end does not consume collaboration output:** arch §15.5 says stored geometry stays in W_i and is never
  rewritten. That is Swarm-SLAM's split, and COSMO-Bench's method.
- **Cost:** pass A is shared across arms, budgets and scenarios.

**Stated [ASSUMPTION]s.**
- Broker compute (verify, solve) takes zero sim time. Measured compute time is reported beside each run (it shifts
  when edges become available, not bytes). The live runtime (L1) is the closed-loop number.
- No lookahead: a unit test perturbs every store value after its release time and checks that the outputs before
  that time are unchanged.

### 2. Hooks (full spec; the Phase-1 ORB/mean-bit path becomes the `features: orb_meanbit` ablation)

- **`retrieve`:**
  - EigenPlaces R50-256 at the D90 sim size;
  - on the wire as int8 with a per-vector scale (D93);
  - exact cosine search (one gemm, D62), threshold from the 3.1 sim evaluation.
- **`features` and `verify`: ALIKED-n16 + LightGlue, the hard-pair path (D63; locked: "ALIKED+LightGlue hard pairs").**
  - Inter-agent candidates are hard pairs by construction: a different agent, a different viewpoint, a different
    time.
  - Evidence (D63, EuRoC, 4815 EP-376 top-5 pairs with the 1.6 verifier's RANSAC E):
    - pose AUC@10°: 0.607 against the ORB placeholder's 0.284;
    - C ≥ A on 11/11 flights;
    - AP_geom 0.807 against 0.303.
  - D63 lists "3.x calls it" as its first runtime consumer. D90 supplies the 1280×720 RGB sim export.
  - **Why not XFeat:** XFeat ships in no runtime path. D60 measured it as the VO engine and did not ship it, and no
    loop-verification row for XFeat exists. It stays a §31.3 row.
  - **Extraction:** at keyframe cadence in pass A, K = 1024 (D63), 3D points from the masked SGBM depth.
  - **Wire encoding:** fp16 descriptors (128 × 2 B). int8 is a 3.6/§17.2 row, gated on a LightGlue-match-set check
    before it may become the default. The predictor prices `n_features × feature_bytes` (3.4).
  - **Verify chain (collaboration.md §11):** LightGlue matches, then the RANSAC essential-matrix gate, then PnP for the
    metric SE(3), then the D71 degeneracy gate. The broker then runs degeneracy → PCM → GNC-TLS (2.5) over the batch.
  - **Thresholds** (min matches/inliers) are re-calibrated for ALIKED+LG on the other scene (§5), never on gate data.
    D63 states that τ does not transfer MH→V, so this cannot be skipped.
  - **Cost:** about 1.2 s per warm pair (D63). Broker CPU time per run is reported, and it becomes a §16.3 secondary
    budget (time) row.
- **`solve`:** the broker's chordal init plus PGO with the 2.5 robust stage, gauge per collaboration.md §5.
- **Carry, stated:** intra-agent loop closure (D87's EuRoC numbers) stays ORB/mean-bit. Moving it to the same stack
  is a separate item that would re-run D87. It is registered, not done here.

### 3. Link

- `CommsNode(MakeLinkModel(scenario, tracks, walls), seed)`.
- Positions come from GT body tracks. The physical channel depends on where the radios really are; this is not an
  estimation leak.
- **Seam change (D102's open item): datagram size goes on the seam.**
  - `LinkModel::Sample` returns the link state. Loss is evaluated per datagram at its actual size with
    `rf::ExpectedPer(rx_dbm, σ, noise, L)`, the RFComms expression D102 already transcribes.
  - `kPathMtuBytes` stays 1500 (D102 L8, under RFComms as modelled).
  - `ScriptedLinkModel` and `FaultPlan` keep size-independent loss, so the protocol tests are unchanged.
- **Scenarios:** `ideal`, `collapsed_v1`, `collapsed_partition_v1` (configs/comms), plus `DatagramFaults`
  (duplicate, reorder, jitter) for the campaign.

### 4. Arms, budgets, runs

**Arms**, all sharing the same pass A and link seed. The plan's gate plot names three arms: spectral, greedy and
similarity.

| plot label | D170 name | §31.1 row | meaning |
|---|---|---|---|
| — | — | solo | each agent alone, per-agent ATE, no exchange |
| — | — | independent multi-agent | both agents, no exchange: per-agent ATE in its own frame, no merged map |
| similarity | `similarity` (select unset) | (reference) | every candidate above the retrieval threshold fetched; no budget |
| greedy | `greedy` (`SelectBySimilarity`) | greedy similarity | top similarity under the byte budget |
| — | — | greedy coverage | **after the gate arms**: definition pending (open item 1) |
| spectral | `spectral` (`SelectSpectral`) | Ariadne spectral + vertex-cover exchange | D170 |
| spectral, no VC | `spectral` + `vertex_cover: off` | Ariadne spectral | features re-sent per candidate |

**Budget levels: derived, not chosen.**
- Measure the similarity arm's mean fetch rate R (bytes/s of classes 3–4) on the calibration scene (the other
  scene, as in §5).
- The levels are {10, 25, 50, 100} % of R, with burst = one 5 s selection period of the level.
- The runtime default (D170's unsourced 20 kB/s and 200 kB) is replaced by 25 % of R from scene 0, recorded with
  the run that measured it.

**Scenes:** ci_tiny seed 0 and seed 1 (two worldgen seeds, 2 agents each). Scene 0 has 17 true inter-agent
opportunities (v1, the 1.6 retrieval row).
- **Precondition, measured first on AWS:** scene 1's true opportunities, from pass A and GT co-visibility.
- If scene 1 has fewer than 10, record a third scene (maze_short, a GPU render on g4dn spot) before the gate runs.
  The threshold comes from scene 0's 17: it keeps the cross-scene calibration from resting on a handful of pairs.

**Gate matrix:**
- 2 scenes × 3 seeds × {similarity, greedy, spectral} × 4 budgets, on `collapsed_v1`;
  - the similarity arm has no budget, so it runs once per (scene, seed);
- the §6 campaign, on the spectral arm.

### 5. Outputs and the byte gate

**Per run:**
- `ledger.csv`: every datagram, tagged first-transmission / retransmit / ack / heartbeat;
- `byte_table.yaml`: predicted vs measured per class 1–6, overhead rows separate (collaboration.md §7);
- `ate_vs_bytes.csv`: one row per broker epoch. Columns:
  - `t`;
  - cumulative first-transmission bytes;
  - cumulative all bytes;
  - merged ATE over released keyframes, in G;
  - per-component counts.
  The merged ATE uses one SE(3) Umeyama alignment to GT per row, and agents not yet merged are reported separately,
  never dropped silently.
- `outcome.yaml`: `consistent` / `explicitly incomplete` / `corrupt`, with reasons;
- `manifest.yaml` (§28.2).

**Byte gate (collaboration.md §7, completed here):**
- **Pairing:** every message the protocol plans carries its predicted size, keyed by message id. The ledger pairs
  each first transmission with its prediction. Predictions are made where the message is planned, by its sender:
  the broker for requests, a member for descriptors, responses, edges and graph batches. They are summed over
  every broker the run had: 3.4's open item.
- **Predictors per class:**
  - heartbeat, descriptor and request: the codec's size function on the planned fields. Exact by construction; a
    test pins it;
  - feature response: 3.4's `FetchCostPredictor` (n_features × feature_bytes plus framing, response ratio);
  - pose edge and graph batch: the §15.7 rows times a calibrated zstd ratio.
- **Overhead rows, never prediction error:**
  - retransmits;
  - acks;
  - **resync after an election.** Broker state is soft, so a new broker receives the latest graph batch and the
    full edge set again (collaboration.md §4). These sends are tagged `resync` at plan time.
- **Calibration** of the compression ratios and verify thresholds is cross-scene: scene 1's runs calibrate scene 0's
  gate and vice versa.
- The gate fails if any class is off by more than 15%.
- `python/analysis/collab_gate.py` produces the plot, the table and the verdict from the run directories, with no
  hand-edited numbers (§36).

### 6. Fault campaign and split-brain on the sim

- **Campaign:**
  - loss levels {0, 0.1, 0.3};
  - duplicate, reorder and jitter;
  - `collapsed_partition_v1` (partition 20–40 s, agent 1 disconnected 60–65 s);
  - a forced split-brain: each agent alone elects itself, then reconnects under a new epoch.
- **Gate:** every run ends `consistent` or `explicitly incomplete`; none ends `corrupt`. A replay with the same seeds
  is byte-identical: the ledger and `outcome.yaml` hashes are compared.

### 7. What stays

- `protocol: v1` (`ariadne_sim_collab` today) stays the 1.6 baseline and the Phase-1 gate path, unchanged; `bytes.json`
  is unchanged.
- v2 lives in its own translation unit, behind `protocol: v2` in the collab config.

### 8. Order and cost

1. 3.4 merges first. It needs a rebase onto main and its own AWS pre-merge.
2. Then this:
   - the seam change and tests;
   - pass A with its store;
   - pass B over CommsNode with its tests: no lookahead, determinism, outcome;
   - the gate harness.
3. The sweep runs on AWS (cpux86): pass A once per scene, then pass B in parallel there.

### Open items

1. **"greedy coverage" (§31.1): sourced, built after the gate arms.**
   - Tian, Khosoussi, Giamou, How & Kelly, RSS 2018 (arXiv 1806.00188) pose budgeted exchange as monotone submodular
     maximisation under a transmission budget. The utility is the expected number of candidate loop closures whose
     verification the exchanged vertex set enables. It generalises maximum coverage and is NP-hard, and a greedy
     over vertices is a (1 − 1/e) approximation (their RSS/WAFR slides).
   - The row is that greedy over keyframes priced in bytes, with candidate probability set to the retrieval
     similarity.
   - [UNVERIFIED] the exact theorem statement. Transcribe it from the paper when the row is built.
2. **Seam change.** Confirm that `ExpectedPer(L)` is D102 L8's "RFComms as modelled" case, so the 1500 MTU stands.
   Under the physical case (PER in bits, slow fading) the recommendation flips to capping fragment size, and that
   becomes its own decision.
   - After the change, these must be byte-identical: the protocol tests, the 54-run campaign, and Phase 1's
     `bytes.json`.
