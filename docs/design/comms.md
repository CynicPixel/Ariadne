# 3.3 Comms broker node — RFComms link model, wall attenuation, partitions (D102)

Session A2, 2026-10-22, branch `feat/3.3-comms`. Arch §5.7; the seam is collaboration.md §2.

## 1. What it is

The accounting/policy layer of arch §5.7. One `CommsNode` per mission owns three things:
- the **link stack**: a base `LinkModel` under a `FaultPlan`;
- the **`Network`** that every datagram crosses;
- the **one `ByteLedger`** that every `Endpoint` records into.

Code:
- `cpp/collaboration/comms/rf_link_model.hpp`: the physics.
- `cpp/collaboration/comms/comms_node.hpp`: the node and the scenario config.
- `cpp/runtime/datasets/scene_walls.hpp`: GT walls and agent tracks from the sim scene.
- `cpp/runtime/tools/ariadne_comms_probe.cpp`: link state along a mission.

## 2. The physical layer: RFComms, transcribed

Transcribed from gz-sim8 `src/systems/rf_comms/RFComms.cc` (read 2026-10-22) and reproduced as is, quirks
included:
- Beyond `max_range` there is no link.
- PL(d) = l0 + 10·n·log10(d).
  - **Quirk:** `l0` is documented as "received power at 1 m" but is used as the path loss at 1 m.
- P_rx ~ N(tx − PL − walls·L_w, σ²), redrawn **per packet**.
- BER = erfc(√(P_rx/N₀)), with powers in W.
  - **Quirk:** there is no ½ factor, and it uses a power ratio rather than Eb/N₀.
- PER = 1 − (1 − BER)^bytes.
  - **Quirk:** the exponent is in bytes, not bits.

**Our additions:**
- **The wall term:** the number of solid wall blocks the straight path crosses, times `wall_loss_db` (the COST-231
  multi-wall form).
  - Worldgen walls are the faces between free and solid cells, so a doorway has no face and a block has two.
  - Counting face **entries** (n·d < 0), once per surface, counts blocks.
- **`latency_s`**, which RFComms does not model.
- **Capacity as a link rate:** `Network` queues at that rate, whereas RFComms drops over a 1 s window. This is a
  stated deviation.

**`loss_p`** is E[PER] over the shadowing. That is RFComms' exact per-packet marginal, and it leaves the Bernoulli
draw in `Network`'s keyed Philox stream.
- PER is nearly a step function of P_rx, so the expectation is integrated on a 4001-point grid over ±8σ.
- It is tabulated at 0.1 dB against the mean P_rx.
- It matches a Monte Carlo of RFComms' per-packet procedure (draw P_rx, then Bernoulli) within 4 standard errors
  at margins −10…30 dB.

**Seam limit:** `LinkModel::Sample` carries no datagram size, so `loss_p` is computed for one size,
`datagram_bytes` (the 1500-byte MTU). §4 quantifies the error. Faithful per-datagram PER would need a size on the
seam; that is the 3.2 owner's call.

**Time base:** `Sample(t)` reads `t.ns` as the sim clock, the GT `t_ns`. Positions are clamped outside a
trajectory. The radio sits at the body origin.

## 3. Scenarios (`configs/comms/`)

The arch §5.7 modes:

| scenario | mode | parameters |
|---|---|---|
| `ideal` | ideal | — |
| `rf_gz_default` | attenuated | gz defaults + 6.9 dB per wall |
| `collapsed_v1` | attenuated | tx 0 dBm, n 3.0, 250 kb/s, 6.9 dB per wall, all [ASSUMPTION] |
| `collapsed_partition_v1` | attenuated | collapsed_v1 + a 20 s partition + a 5 s disconnect |

- `range_limited` is RF without walls.
- The 6.9 dB "heavy wall" is COST-231's value as it is commonly cited. I could not confirm it against the primary
  report: [UNVERIFIED-primary].
- Faults are scripted on `FaultPlan` in seconds from the mission's first stamp.

## 4. Measured: the link state along ci_tiny (`ariadne_comms_probe`, 1 Hz, 199 s, 2 agents)

| scenario | 0 walls (4.5–6.7 m) | 1 wall (5.0–10.7 m) | 2 walls (5.6–12.0 m) |
|---|---|---|---|
| rf_gz_default, `loss_p` at 1500 B | 0.0000 | 0.0000 | 0.0009 |
| collapsed_v1, `loss_p` at 1500 B | 0.016 | 0.17 | 0.49 |
| collapsed_v1, at 40 B (a heartbeat) | 0.007 | 0.10 | 0.36 |

- Time per wall count: 63 / 71 / 66 s.
- **gz-default radio is a null.** It sits about 50 dB above the BER knee at these ranges, so walls barely register
  (stated, as expected).
- **The partition and disconnect work.** In `collapsed_partition_v1` the link is down for exactly 25 of 200 s.

### L8: the MTU against the real link model

At the median P_rx of each scenario, goodput is (payload / wire) × (1 − E[PER]):

| payload B | 64 | 256 | 512 | 1024 | 1472 |
|---|---|---|---|---|---|
| rf_gz_default (−41.5 dBm) | 0.70 | 0.90 | 0.95 | 0.97 | 0.98 |
| collapsed_v1 (−72.9 dBm) | 0.61 | 0.77 | 0.79 | 0.80 | 0.80 |

- **At the median,** goodput rises with size. RFComms redraws a 10 dB shadowing per packet, so a packet is lost
  mostly when its draw fades below the knee, whatever its size. Bigger datagrams then win by amortising the 28 B
  overhead.
- **At the edge, the verdict depends on RFComms' quirks.** Computed offline from the same formulas (no run), at
  collapsed_v1's 2-wall mean power (−81.7 dBm) and its 10th-percentile power (−81.3 dBm):

  | model at the edge | 64 | 256 | 512 | 1024 | 1472 | best |
  |---|---|---|---|---|---|---|
  | RFComms as modelled (bytes, per-packet σ) | 0.42 | 0.51 | 0.52 | 0.51 | 0.51 | 512 (1472 within 3 %) |
  | bit exponent | 0.37 | 0.46 | 0.46 | 0.46 | 0.46 | 512 (flat) |
  | per-link σ (slow fading) | 0.68 | 0.85 | 0.84 | 0.77 | 0.70 | 256 |
  | bits + per-link σ (the physical case) | 0.59 | 0.54 | 0.35 | 0.14 | 0.06 | **64** |

  Rows are at the 2-wall mean; p10 differs by ≤ 0.1.

- **Verdict, conditional:**
  - **Under RFComms as modelled,** keep `kPathMtuBytes = 1500`. It costs ≤ 3 % goodput at the edge.
  - **Under the physical case** (PER in bits, slow fading), full-MTU fragments lose 3–10× goodput at the edge,
    where 64–256 B wins.

  L8 is therefore a **recommendation to the 3.2 owner**, not closed. If the link model is ever made physical, size
  edge-link fragments adaptively, or cap them at ~256 B. The serializer is untouched.
- **The cost of the seam:** a single-size `loss_p` overstates heartbeat and ack loss by 1.3–2.3× near the edge.
  That is conservative for the election: heartbeats are lost more often than RFComms would lose them.

## 5. Ledger rule (tested)

- `CommsNode::ledger()` is handed to every `Endpoint`. The `Network` has no ledger.
- Test, over a lossy RF link with a partition:
  - every non-retransmit row is an `Endpoint` first transmission, heartbeat or ack;
  - retransmit rows exist;
  - draining the network, which forwards or drops everything in flight, adds **no** row.

## 6. Not done: 3.3 is partial

- **No protocol traffic over the real link yet.** That needs 3.2's sim integration (`protocol: v2`), which does not
  exist (register 3.2).
- **No live gz RFComms.** Replay is the substrate; the plugin's equations are reproduced instead.
