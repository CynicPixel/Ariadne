# Design — Phase 2: the sparse SE(3) pose-graph solver (arch §11.6, §12, §13 groundwork)

The first Phase 2 component (`local-slam.md#estimator-plan` §1, step 1). It optimises a pose graph:

- **in:** keyframe poses `X_i = T_W_C` (nodes) and relative-pose measurements (edges, §12.1 types);
- **out:** the maximum-likelihood poses, the marginal covariances of chosen nodes, and a solve report.

It is the engine for 2.3 (local pose graph, window exit), for 2.5 (chordal init → GNC-TLS → certificate) and for
Phase 3's broker PGO.

## 1. What a pose graph is, and the residual

Each edge i→j says "seen from i, j sits at `Z_ij`". If the estimates agree, `Xᵢ⁻¹ Xⱼ = Z_ij`. The disagreement is
measured in the tangent space (§12.2):

```text
E_ij = Z_ij⁻¹ · Xᵢ⁻¹ · Xⱼ          r_ij = Log(E_ij) ∈ R⁶   ([ρ; φ], translation first, D1)
cost = ½ Σ_e  w_e · r_eᵀ Ω_e r_e  (w = GNC weight; Ω = information in r's tangent, see pose_graph.hpp)
```

A **unary** edge (ANCHOR; GRAVITY_PRIOR in 2.4) is the same expression with `Xᵢ = I`, i.e. `r = Log(Z⁻¹ Xⱼ)`.

## 2. Jacobians in OUR convention (derived here, finite-difference tested)

Left perturbation of `T_W_C`: `X ← exp(δ) X`, δ = [ρ; φ].

- Perturb Xⱼ: `E' = Z⁻¹ Xᵢ⁻¹ exp(δⱼ) Xⱼ = exp(Ad(Z⁻¹Xᵢ⁻¹) δⱼ) E`.
- Perturb Xᵢ: `Xᵢ⁻¹ → Xᵢ⁻¹ exp(−δᵢ)`, so `E' = exp(−Ad(Z⁻¹Xᵢ⁻¹) δᵢ) E`.
- With `Log(exp(a) E) ≈ Log(E) + J_l⁻¹(r) a`:

```text
∂r/∂δⱼ = + J_l⁻¹(r) · Ad(Z⁻¹ Xᵢ⁻¹)        ∂r/∂δᵢ = − J_l⁻¹(r) · Ad(Z⁻¹ Xᵢ⁻¹)
```

`J_l⁻¹` is the **exact** inverse SE(3) left Jacobian (Barfoot, *State Estimation for Robotics*, §7.1.5:
`𝒥 = [[J, Q], [0, J]]` for [ρ; φ], so `𝒥⁻¹ = [[J⁻¹, −J⁻¹QJ⁻¹], [0, J⁻¹]]`). The approximation `I + ½ ad(r)` is not
used: it moves the stationary point off GTSAM's whenever residuals are non-zero, and the oracle demands 1e-6.

**Gauge (the catalogue's "best 20 lines", re-derived for our convention).** Setting δᵢ = δⱼ = ξ makes the two
Jacobians cancel exactly. So the six null vectors of the unanchored H are "the same ξ on every node", and the test
asserts `‖H·N‖ ≈ 0` with N = stacked identity blocks. (The catalogue's basis is for the RIGHT convention and is not
copied.) The anchored system must then be positive definite; every LDLT pivot `> 0` is asserted, which also catches
GNC weight collapse (catalogue §7.8).

## 3. Who does what: OpenCV 5's LM driver, our backend (D51)

`cv::detail::LevMarqBase` (OpenCV 5 `geometry/detail/optimizer.hpp`, `levmarq.cpp`) is a Ceres-style LM loop: Marquardt
damping `JᵀJ + λ·diag(JᵀJ)` with clamping, step-quality-adaptive λ, optional Jacobi column scaling, optional
geodesic acceleration (Transtrum & Sethna), and four termination tests. It calls a `LevMarqBackend` that we provide.
The contract below is read from `levmarq.cpp` and checked against OpenCV's own `PoseGraphLevMarqBackend`:

| backend call | our implementation |
|---|---|
| `calcFunc(E, calcEnergy, calcJacobian)` | at the PROBE poses: `E = ½ Σ w‖√Ω r‖²`; with the Jacobian, the block-sparse JᵀJ and `jtb = +Jᵀr` |
| `prepareVars()` | probe ← current; allocate |
| `getJtb / getDiag / setDiag` | the gradient and the JᵀJ diagonal (the driver rewrites the diagonal on each attempt) |
| `decompose()` / `solveDecomposed(right, x)` | Eigen `SimplicialLDLT` on the sparse system; solves `(JᵀJ+D) x = −right` (x is the descent step); asserts every pivot > 0 |
| `currentOplusX(x)` | probe ← `exp(x_i) X_i` for every free node (left update) |
| `acceptProbe()` | current ← probe |
| `enableGeo / calcJtbv` | geodesic acceleration: supported; OFF by default, an ablation |

**The gauge lives in the backend.** The base class has no fixed variables, so the anchor node (the lowest id, or a
named one) is simply not in the parameter vector.

**Layering (arch §3, CI).** Residuals, the exact `J_l⁻¹`, block assembly and the sparse factorisation live in
OpenCV-free `cpp/slam/pose_graph/`. Only the thin `LevMarqBackend` adapter lives in `slam_opencv`. GitHub `ci.yml`
(no OpenCV) therefore still runs the finite-difference tests, the null-space test and an assembly digest across
architectures. The full-solve digest runs in the core image only.

**Settings are config + manifest, never defaults.** OpenCV's defaults (`relEnergyDeltaTolerance` 1e-6, `maxIterations`
500, step quality on, clamp on, Jacobi scaling off) are written out explicitly in `configs/slam/pgo_v1.yaml`. The oracle
test uses tighter tolerances.

**Placeholder BA unchanged.** The two-frame BA stub keeps its own §11.6 LM, because Phase 1 outputs must stay
byte-identical (arch §0.2).

## 4. OpenCV-native, as far as it honestly goes

- **Used:** `LevMarqBase` (the optimizer); `cv::SVDecomp` for the chordal SO(3) projection with the determinant fix
  (2.5); `geometry/mst.hpp` `buildMST` as the spanning-tree initialisation baseline in the init ablation (odometry
  chain vs MST vs chordal, 2.5); `cv::Rodrigues` with its Jacobian as a finite-difference oracle for the SO(3)
  blocks; dense `cv::eigen` as the oracle for the certificate's λ_min on small graphs (2.5).
- **Not used, with reasons:**
  - **`cv::detail::PoseGraph`.** In our pinned build `HAVE_EIGEN` is undefined, so `optimize()` throws
    `StsNotImplemented`. Rebuilding OpenCV with Eigen would change the OpenCV build hash, breaking the Phase 0 gate
    and the COOL Arm-vs-Arm baseline (invariant 2). Its residual is also a quaternion-space error over float
    `Affine3f` edges, not §12.2's SE(3) Log, so its optimum differs.
  - **A sparse solver.** OpenCV has none; Eigen's `SimplicialLDLT` is what §11.6 names.

## 5. Verification

- **Finite differences:** both edge Jacobians, unary edges, `J_l⁻¹` against numerical inversion; random poses, large
  rotations (near π) and near-identity.
- **Null space:** `‖H·N‖ / ‖H‖ < 1e-12` unanchored; anchored LDLT pivots all `> 0`.
- **Backend contract:** a linear least-squares problem converges in one accepted step to the closed form; a noise-free
  SE(3) graph from a perturbed start returns the truth to 1e-9.
- **Covariance:** marginal blocks from `H X = E_block` (never an explicit inverse) vs a dense inverse on small graphs.
- **GTSAM oracle (exit-gate item):** GTSAM 4.3 (pip, exact version pinned from the index). The oracle script first
  asserts that Pose3 `localCoordinates` equals `Logmap` (so both minimise the same §12.2 cost), and swaps [rot; trans]
  ↔ [ρ; φ] at that boundary only. Cases: 100 seeded random graphs plus the SE-Sync benchmark graphs (sphere, torus,
  garage, cubicle, rim; licences checked first). Agreement: final cost ≤ 1e-6 relative, poses ≤ 1e-6. The SE-Sync
  graphs carry published certified results, which also check the sign convention of the 2.5 certificate.
- **Determinism:** `pgo_digest` (`pgo_digest.hpp`) is FNV-1a over the bits of a fixed graph's assembled energy, H
  and g. It is measured into every run manifest and required (D21). It is reproducible for the same binary on the
  same architecture only: the SE(3) maths uses libm, so the cross-architecture guarantee stays with `det_math`
  (Phase 0). The full OpenCV-driven solve is asserted bit-identical across two in-process runs
  (`PgoLevMarq.SolveIsBitIdenticalTwice`).

**As built (2026-10-16), with the deviations recorded:**

- **Shared initial estimate.** The plan was GTSAM's chordal `InitializePose3`. It threw `IndeterminateSystemException`
  on random case 2, cause **[UNVERIFIED], open**: to diagnose when 2.5 opens, where GTSAM's chordal init is the
  natural oracle. The oracle instead starts both solvers from dead reckoning along the odometry chain (random graphs)
  or the file's vertices (SE-Sync). The comparison stays valid because both start from one point, and this start is
  harder than chordal.
- **Information.** Half of the 100 random graphs carry full, correlated, anisotropic information (axis scales 0.03–1×
  the base sigma, about 1000× information spread), so no permutation or ordering bug can hide in a diagonal. The
  first version scaled noise up to 6× sigma: dead reckoning over 200 such edges started far outside the basin, and
  both solvers reached different local minima, correctly. That tests initialisation (2.5), not solvers, and was
  fixed in the generator.
- **Pose criterion.** Energy within 1e-6 relative, and poses within 1e-6 absolute **or** within 1e-6 Mahalanobis²
  under the node's marginal. With about 1000× information spread, two solvers that agree on the energy to 1e-14 can
  differ by about 1e-6 m along a weak direction: a flat minimum, which only a scale-invariant measure can tell from
  a disagreement.
- **Result:** 100 / 100 agree; 99 also within 1e-6 absolute. Worst relative energy 1.5e-12, worst Δt 3.2e-6 m at a
  Mahalanobis² of 3e-12. Initial energies match GTSAM's to 1e-9, so it is the same §12.2 cost.
- **SE-Sync graphs:** a separate test (`ARIADNE_PGO_SESYNC`, not in the pre-merge gate). Criterion fixed before the
  results: the same minimum must agree to 1e-6; different minima require both to be stationary, and the lower cost
  is reported. GTSAM stopped rim at its 1000-iteration cap (error 81,076.5), not converged. The objective is the
  g2o-information cost, not SE-Sync's isotropic κ/τ one, so SE-Sync's published optima are not compared here.
- **SE-Sync results (2026-10-16), and the criterion revised with that evidence.** Run from c18c8b5, 1 h 48 min,
  i5-1135G7:

  | graph | nodes | ours (its) | GTSAM (its) | verdict |
  |---|---|---|---|---|
  | tinyGrid3D | 9 | 9.313908074 (10) | 9.313908372 (10) | same minimum; poses Δt 2.1e-7 m |
  | smallGrid3D | 125 | 517.925324 (12) | 517.9253315 (12) | same minimum; Δt 2.8e-6 m |
  | sphere2500 | 2500 | 675.7009666 (21) | 675.7007397 (7) | same minimum (3e-7 relative); Δt 1.7e-3 m |
  | torus3D | 5000 | 29950.00592 (342) | 29950.00577 (359) | same minimum; Δt 4.0e-5 m |
  | grid3D | 8000 | 253372.1012 (137) | 253372.1003 (138) | same minimum; Δt 2.1e-4 m |
  | parking-garage | 1661 | 0.6341922679 (1000, cap) | 0.634188936 (8) | 5e-6 apart; ours stationary (max\|g\| 1.2e-9) but slow |
  | cubicle | 5750 | — | 1372.78 (41) | **excluded**: 5021 / 16869 information matrices indefinite |
  | rim | 10195 | — | 81076.5 (1000, cap) | **excluded**: 8815 / 29743 information matrices indefinite |

  - **cubicle and rim are not valid inputs for this cost.** In the files as published (SE-Sync commit `9b631b6b`),
    about 30% of the edges carry a 6×6 information matrix whose diagonal blocks are PD but whose
    translation–rotation cross terms make it **indefinite**. For example, cubicle's edge 1→2 has
    `I(2,3) = 84022.3` against `I(2,2) = I(3,3) = 10`, with minimum eigenvalue −1.57e5. ½ rᵀIr is then unbounded
    below. Our LDLT guard refused every factorisation: 15/15 failed, the energy stayed at E0, and the driver stopped.
    That is the correct behaviour. GTSAM's `noiseModel::Gaussian::Information` takes a Cholesky factor without
    checking `[UNVERIFIED in GTSAM's source; inferred from it returning finite results]`, so its numbers are for a
    different problem. SE-Sync itself uses only the isotropic κ and τ from the diagonal blocks. Since 2026-10-16,
    `PoseGraph::AddEdge` refuses any information that is not finite, symmetric and PSD (test
    `PoseGraph.AddEdgeRefusesInformationThatIsNotPsd`, using cubicle's edge 1→2). The SE-Sync test excludes such files
    by name and fails if the excluded set is not exactly {cubicle, rim}.
  - **The 1e-6 pose criterion was ill-posed for large graphs.** sphere2500 agrees with GTSAM to 3e-7 relative in
    energy, yet differs by 1.7e-3 m along flat directions. Same-minimum is now energy agreement plus stationarity,
    with poses reported, not gated. The per-pose test stays in the 100-case oracle, where Mahalanobis² against the
    marginal tells a flat minimum from a disagreement. Computing marginals for 8000 nodes is not worth it here.
  - **parking-garage exposes slow convergence on extreme conditioning.** The edge information spans 3.7e-10 in
    min/max eigenvalue ratio. Our Marquardt damping (λ·diag H, `clamp_diagonal`) creeps along a flat valley for 1000
    iterations, while GTSAM's LM takes 8. Ours ends stationary, 5e-6 above GTSAM. This is a real efficiency gap,
    recorded open for 2.5 (damping choice: λI vs λ·diag, and Jacobi scaling). It is not hidden by the criterion: the
    test reports it as "different minima, lower: gtsam".


## 6. Literature: what the 2026 search added (searched 2026-10-16, after the maintainer asked)

- **SE-Sync benchmark graphs:** a real-structure oracle set with known certified optima (Rosen et al., IJRR 2019).
- **COSMO-Bench** (arXiv 2508.16731, 2025): 24 real collaborative-SLAM pose-graph datasets from LiDAR. It is the right
  real-data source for Phase 3 and for the 2.5 outlier ramp (licence to check).
- **TACO** (arXiv 2606.29851, 2026): incremental probabilistic consensus plus switchable-constraint "sanitization"
  for robust PGO, online. **EGNC** (Int. J. Control Autom. Syst., 2025): a GNC μ schedule from convexity boundary
  points. Both are ablation candidates for 2.5 against Yang's schedule. **Neither changes the §14 pipeline or its
  ordering** (degeneracy gate → PCM → GNC).
- **Certifiable FGO** (arXiv 2603.01267) and **Certi-GNC** (arXiv 2603.20932), already in the catalogue, stay the
  references for the certificate and the robust certified composition.
