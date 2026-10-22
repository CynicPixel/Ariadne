# Design — Phase 3.4: spectral prioritisation of feature requests (I6, arch §16)

**Owner:** M. **Decision:** D170. **Status:** design; selector library and unit tests first, broker wiring second, the
sim integration of 3.2 over 3.3's CommsNode right after (the Phase 3 gate rows wait on it).

## 0. Preflight (design-preflight rule)

**Literature, read 2026-10-23:**
- **Swarm-SLAM** (Lajoie & Beltrame, RA-L 2024; ALGORITHMS §1.4 M2):
  - picks the size-B candidate subset maximising λ₂ of a weighted Laplacian;
  - weights: fixed edges κ = 1, candidates κ = retrieval similarity;
  - procedure: relax, solve, round; bootstrap greedily until the robots' graphs are connected.
  - [UNVERIFIED this session: its exact graph construction in the `cslam` submodule. The crawl failed on a
    saturated uplink; retry and record any deviation in §8.]
- **MAC** (Doherty, Papalia, Huang, Rosen, Englot, Leonard, arXiv 2403.19879; IROS 2022 arXiv 2203.13897):
  - Frank-Wolfe on the Boolean relaxation; the supergradient is g_e = κ_e (v[a] − v[b])²;
  - rounding: nearest or Madow;
  - a post-hoc dual bound f(x) + gᵀ(s − x) ≥ f(x*), valid by concavity of λ₂(L(x));
  - the Fiedler pair comes from TRACEMIN-Fiedler with a sparse Cholesky, warm-started.
- **Ghosh & Boyd 2006:** the first-order gain v₂ᵀ L_e v₂ is the derivative of λ₂, and is exact only while λ₂ is
  simple.
- **Giamou, Khosoussi & How (ICRA 2018); Tian, Khosoussi, Giamou, How & Kelly (RSS 2018, WAFR 2018):**
  - the minimum data needed to verify a candidate set is a minimum vertex cover of the exchange graph;
  - budgeted selection is monotone-submodular in their metrics, so greedy has guarantees there.
  - λ₂ is not submodular, so no such guarantee here, hence the dual bound.

**OpenCV-native check:**
- `cv::eigen` is dense and symmetric only. OpenCV has no sparse eigensolver and no matching or vertex-cover routine.
- `cv::solveLP` (simplex: maximise cᵀx subject to Ax ≤ b, x ≥ 0) **is** used, for the bound's LP (§4).
- The eigen solve uses Eigen (already in core): `SimplicialLDLT` on the grounded Laplacian.

**Advisor** (2026-10-23), five corrections, all taken:
1. MAC's bound needs a relaxation of the **shared-cost** feasible set (§4).
2. No Kron reduction of the objective; superseded by the full-graph solve of §3.
3. Today's behaviour stays the default; `n_features` is v2-only; v1 bytes stay identical (§6).
4. The byte prediction keeps §7's compression model (§5).
5. The vertex-cover gap is named (§8).

## 1. What it decides

At the broker, retrieval proposes candidate pairs (a, b) of keyframes from different agents. Verifying one needs the
local features (a FEATURE_RESPONSE, about 57 KB at 800 kp, §15.7) of every endpoint the broker does not hold. Under a
byte budget B per selection round, the selector decides which candidates' feature requests to send. The rest stay
pending: they are re-scored next round as the graph changes, and dropped after `candidate_ttl_s`.

## 2. The candidate graph (arch §16.1)

- **Nodes:** every keyframe the broker knows of, for all members.
- **Fixed edges:**
  - odometry, from each agent's pose graph;
  - intra-agent loops;
  - inter-agent edges already accepted.

  All have κ = 1 (Swarm-SLAM's convention: no noise model crosses the wire for this).
- **Candidate edges:** κ_e = the retrieval similarity s_e ∈ (0, 1].
- **Objective:** λ₂(L_fixed + Σ_{e∈S} κ_e L_e), the full graph (no reduction).

## 3. λ₂ and the Fiedler space

- **Method:** block inverse iteration on L⁺ restricted to 1⊥.
  - Ground one node, so L_g is SPD, and factor it once with `SimplicialLDLT`. Odometry chains plus a few loops
    give O(n) fill.
  - Each apply solves L x = b for b ⊥ 1, then removes the mean.
  - Block size p = 4, Rayleigh–Ritz each iteration. Warm-started from the previous round's block, as arch §16 asks.
    The graph changes by one edge per greedy step.
- **Stopping:** Ritz residual ‖L v − λ v‖ ≤ 1e-9 ‖L‖ or 200 iterations, reported.
- **Near-repeated λ₂ (determinism):**
  - When λ₃ − λ₂ ≤ 1e-6 λ₂, the gain is taken over the whole cluster V, as κ_e ‖Vᵀ(e_a − e_b)‖².
  - Ties break by candidate id.

  A Fiedler vector picked arbitrarily inside a degenerate space would make selections differ across hosts.
- **Test reference:** a dense `SelfAdjointEigenSolver` on small graphs.

## 4. Selection (arch §16.2) and the bound

1. **Connectivity bootstrap.** A candidate joining two components of the current graph has gain +∞ class. These are
   taken in order of similarity per marginal cost until one component remains or the budget is spent.
2. **Greedy.** Pick argmax_e gain_e / Δcost_e, where Δcost_e is the bytes of the endpoints not yet held or selected.
   Add it, then recompute the Fiedler space (warm). Stop when nothing affordable has positive gain.
3. **1-swap refinement** (arch §16.2's "local refinement", `refine_passes`, default 2):
   - Replace one selected candidate by one unselected; keep the best swap that stays in budget, stays connected and
     raises λ₂; refill greedily.
   - Cost: |S|·|C| Fiedler solves per pass, so it is measured on the sim before the default is kept.
   - Oracle, 60 instances: selected/optimum mean 0.936 → 0.970, worst 0.533 → 0.650. The misses came mostly from the
     bootstrap's bridge choice (similarity per byte, blind to λ₂).
4. **Dual bound** (MAC §IV-D, adapted to shared costs):
   - At the greedy set x, with supergradient g, solve
     `max gᵀs` subject to `s_e ≤ y_k` for each needed endpoint k of e, `Σ_k bytes_k y_k ≤ B`, `0 ≤ s, y ≤ 1`.
     This is `cv::solveLP` with s and y stacked. It is the LP relaxation of the true shared-cost feasible set, so it
     contains the integer optimum.
   - bound = λ₂(x) + gᵀ(s* − x) ≥ λ₂(x*) by concavity.
   - Reported: `lambda2`, `bound`, `gap = bound − lambda2`. If the LP is not SINGLE/MULTI, the bound is reported
     **absent**, never guessed.
   - **Measured:** valid on 60/60 oracle instances, but loose: 1.3–6.4× the optimum on these small graphs, the
     usual LP-relaxation gap. It is reported as a bound, never as a tightness claim.

## 5. Cost model (§15.7, COLLAB_3.2 §7)

- **Raw size:** `predicted_raw(k)` is exact from the keyframe's feature count. The global-descriptor message carries a
  new `n_features` field (v2 only, §6) and the descriptor encoding.
- **Wire size:** wire(k) = ratio_class · raw(k) + fragments(raw) · (6 B fragment header + 28 B datagram), using
  D102's MTU.
- **Compression:** `ratio_class` is the per-class compression model of COLLAB_3.2 §7, calibrated on runs disjoint
  from the gate runs (recorded in the manifest). The feature response is a compressed class.
- Held or already-requested keyframes cost 0. One fetch serves every candidate sharing that keyframe.

## 6. Defaults, compatibility

- `selection: similarity | greedy | spectral`. The default `similarity` is today's request-everything behaviour,
  kept as the ablation baseline (placeholder rule). `greedy` is top similarity under the same byte budget.
  `spectral` is this design. The gate plot compares all three.
- `n_features` is written on the protocol-v2 path only, and verified absent in v1 bytes (FlatBuffers omits defaults
  unless forced). Phase 1's `bytes.json` and the 54-run campaign must reproduce unchanged.

## 7. Tests (unit, synthetic, no premerge)

1. The Fiedler solver against dense eigen: random connected graphs and long chains with loops; λ₂ to 1e-9, and the
   space by principal angles.
2. Warm start: the iteration count drops after a one-edge update, and the result is identical to a cold start.
3. Degenerate λ₂ (a symmetric cycle): the selection is identical under node relabelling and permuted inputs.
4. **Exact oracle, |C| ≤ 15:** enumerate every subset under the **shared** cost.
   - Assert bound ≥ the oracle optimum on every instance.
   - Report greedy / optimum, measured and not asserted beyond ≥ 0.
5. Bootstrap: two disconnected agents get one bridging edge first, even at lower gain.
6. Sharing: k candidates on one remote keyframe cost one response.
7. Budget: the selected cost is ≤ B always; nothing affordable is left with positive gain.

## 8. Stated gaps (as-built vs locked wording)

- **Vertex cover:** verification is at the broker, so every non-broker endpoint is fetched. The direction choice of
  Giamou et al. (send the one keyframe to the agent holding five) is not built. On the sim runs, the count of
  one-to-many candidate stars is reported, so the bytes this leaves are known.
- **First-order gain:** greedy has no approximation guarantee for λ₂. The dual bound is what is claimed, per instance.
- **Swarm-SLAM fidelity:** its graph construction is to be confirmed from source (§0).
