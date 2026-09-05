# ARIADNE — Implementation Gap Fill and Refactor (v2 addendum)

**Status:** Execution-layer addendum to `ARIADNE_IMPLEMENTATION_ARCHITECTURE.md` (v1). v1 remains the master
architecture; this document overrides it where a `[CORRECTION]` block says so, and fills every module where v1
specified *what* but not *how*. Written for consumption by subsequent build agents.

**Evidence tags used throughout:**
- `[VERIFIED web 2026-08-30]` — checked against a live primary source during this revision.
- `[PAPER]` — sourced from the cited paper; equations should be re-checked against the paper before implementation
  (sign and convention errors are the classic transcription failure).
- `[DESIGN]` — an Ariadne design decision, internally consistent but not externally sourced.
- `[SMOKE-TEST]` — must be confirmed against the installed binary on day 1; do not trust this document or v1 for it.

---

# 0. Corrections register — internal inconsistencies in v1 that must be resolved before code

## C1 — The IMU is claimed but never implemented anywhere in v1 (blocking)

v1 §0 declares Ariadne "stereo-inertial", Plane A produces IMU streams, §11.1 defines a `GRAVITY_PRIOR` edge, and
Correction B lists "gravity priors" as a benefit of the stereo configuration. But no section of v1 specifies IMU
preintegration, bias states, gravity alignment, or how an inertial residual enters the BA or the pose graph. As
written, the IMU data would be produced by the simulator and consumed by nothing, and the `GRAVITY_PRIOR` edge has
no producer.

**Resolution:** implement the inertial layer as specified in §A below (on-manifold IMU preintegration, Forster et
al.), staged as new Iteration 3.5. If Iteration 3.5 slips, the documented fallback (§A.7) is a reduced inertial
mode that still produces the gravity prior and motion prediction, so the pose graph edge type keeps a producer
under every outcome. Under no circumstances should "stereo-inertial" appear in the report while the estimator is
vision-only; the rules explicitly allow rejection for misrepresenting capabilities.

## C2 — `Sim3 or SE3` in the LoopEdge struct is not a decision (blocking for the protocol)

With stereo as the primary rig (v1 Correction B), scale is observable per agent and inter-agent edges are SE(3).
`Sim3` appears in exactly two places, neither of which is the production wire format:

1. The monocular ablation mode, where inter-agent registration is Sim(3) and the estimator carries per-agent scale.
2. Trajectory evaluation, where ATE alignment uses Umeyama Sim(3)/SE(3) alignment (Umeyama, TPAMI 1991) exactly as
   the `evo` toolchain does.

**Resolution:** `LoopEdge.measurement` is `SE3` with `Matrix6d information`. The monocular ablation uses a separate
`Sim3LoopEdge` type that never crosses the production serializer. This also fixes the wire-format ambiguity in §G:
a message schema cannot contain "or".

## C3 — OpenCV 5 ground truth has moved since the research passes; re-baseline `[VERIFIED web 2026-08-30]`

OpenCV 5.0 is now released (announcement dated early June 2026). Confirmed release facts that affect Ariadne:

- C++17 is the minimum standard; Python 2 and the legacy C API are removed.
  (https://docs.opencv.org/5.0/ , https://github.com/opencv/opencv/wiki/OpenCV-5)
- The former `calib3d` is split into geometry/calibration/stereo/point-cloud functionality, and **USAC is the
  default robust-estimation backend** for homography, essential/fundamental, and all PnP solvers, with
  multi-threaded execution. (wiki, verified)
- A **multi-view (N-camera) calibration pipeline** was added to `calib` (three stages: per-camera intrinsics,
  pairwise extrinsics, global refinement). Relevant to the stereo rig plus any stretch multi-camera agent.
- `ptcloud`-area functionality includes mesh/point-cloud I/O (.ply/.obj), ICP, and **dense RGB-D fusion with TSDF,
  HashTSDF, and ColorTSDF** (opencv.org OpenCV 5 article; Stereolabs engineer's write-up). §I builds the occupancy
  layer on HashTSDF rather than writing a fuser from scratch.
- New DNN engine covers **>80% of the ONNX spec**, with ONNX Runtime integration available and models hosted on
  Hugging Face. (docs.opencv.org/5.0)
- **Universal Intrinsics 2.0** targets SSE/AVX2/AVX-512/NEON/SVE/RVV from one vector codebase, with reported
  **3–4x speedups on common Arm operations such as resize and warp** (opencv.org). This is directly load-bearing
  for the COOL benchmark narrative in §N: the baseline (stock OpenCV 5 on Graviton) is already Arm-optimized, so a
  COOL delta measured on top of it is a strong, honest result, and if the delta is small on some ops, report it.
- SURF, BRIEF, FREAK, DAISY and several other classic extractors moved to `opencv_contrib` (`xfeatures2d`)
  (migration guide). Consequence: **Ariadne must build `opencv_contrib`** anyway, because `ximgproc`
  (DisparityWLSFilter, §D), `quality` (PSNR/SSIM harness), and `optflow` live in contrib.

**Two `[SMOKE-TEST]` items that this revision could not fully resolve from secondary sources:**

1. **Module naming discrepancy.** docs.opencv.org/5.0 and the wiki describe the split as `geometry`/`calib`/
   `stereo`/`ptcloud`; several third-party release write-ups describe a `3d` module covering "geometry, I/O, ICP,
   and parts of SLAM". These may be the pre-release vs release layouts, or the same code described loosely. Day 1:
   `find $PREFIX/include/opencv2 -maxdepth 1` and record the actual header set in the run manifest. All v1/v2
   references to "the geometry module" mean "wherever `findEssentialMat`, `recoverPose`, `solvePnPRansac`,
   `triangulatePoints` actually live", which the migration guide confirms are unchanged in name.
2. **PyPI wheel availability.** As of the June 2026 release coverage, the OpenCV 5 pip wheel was announced but not
   yet visible on PyPI. The containers in §N therefore **build OpenCV 5 + contrib from a pinned git tag** as the
   authoritative path; a pip wheel, if now present, is a convenience for the Python analysis env only. This also
   removes any wheel-content risk for contrib modules on arm64.

## C4 — The COOL workload must survive the restoration ablation (strategic)

v1 §36.1 and Iteration 14 anchor the COOL benchmark on the restoration front-end. v1 §6.2 Stage 4 simultaneously
commits to ablating restoration and accepting a negative result. If the ablation concludes that restoration hurts
geometry (a live possibility per the MID/low-light findings in the algorithm catalogue), the benchmarked COOL
workload would be a stage the project just removed.

**Resolution:** define the COOL benchmark workload as the **always-on per-frame pipeline**: undistort/rectify
(`remap`), pyramid construction, `calcOpticalFlowPyrLK`, image format conversion, flat-field multiply, VST
transform, resize for DNN input, and descriptor quantization/serialization. Restoration stages are benchmarked as
an *additional* configuration, clearly labeled. The COOL claim then reads "COOL executes the per-frame image
pipeline that runs on every frame from every agent" regardless of the restoration verdict.

## C5 — Novelty framing must match the sensor decision (report-level)

v1 Correction B (stereo primary) is right, but it forfeits the "monocular multi-agent gap" claim from the
algorithm catalogue §3.1 unless framed carefully. The claim Ariadne can honestly make: the system is
**vision-only** (no LiDAR, no depth sensor; stereo RGB + IMU + carried light), which is exactly the low-cost
degraded-conditions gap named by the SubT survey, and the monocular ablation quantifies what scale observability
buys. Do not write "monocular-first" anywhere.

## C6 — Determinism: Gazebo camera rendering is not a reproducible function (architectural)

v1 §33 requires identical image streams across baselines and v1 §34 requires figure-level reproducibility, but a
live Gazebo run is not bit-deterministic across executions (GPU rendering, scheduling, transport timing).

**Resolution (this is the single most important architectural addition in v2):** split simulation from consumption
via **record/replay**, §M.8. Gazebo runs once per (scene, trajectory, seed) and produces a clean recording; all
perception/SLAM/collaboration runs consume recordings deterministically; all visual degradations are applied at
consumption time by a seeded operator (§M.5). Consequences: identical pixels across every baseline and ablation by
construction; degradation parameter sweeps do not re-run Gazebo; the GPU node renders once while the Graviton/COOL
node runs the many-replica sweeps, which is also the correct economics; and "clean ground truth beneath every
degradation" (proposal §8) becomes a property of the pipeline rather than a promise. Closed-loop active-perception
experiments still run live (§M.8.3) and are exempt from bit-determinism, reported over seeds instead.

## C7 — Contrib dependence must be explicit (build system)

v1 references `xphoto`, `ximgproc`, `optflow`, `quality`, `fastNlMeansDenoising` context. Several are contrib
modules. Every container in §N builds `opencv` + `opencv_contrib` at the same pinned tag with an explicit
`-DOPENCV_EXTRA_MODULES_PATH`. The run manifest records `cv::getBuildInformation()` hash.

## C8 — v1's per-frame ALIKED ambiguity

v1 §0 diagram says "XFeat/ALIKED keypoints + sparse tracking" per frame; v1 §7.2 correctly reserves ALIKED for
hard pairs. The per-frame path is: LK tracking on existing keypoints, XFeat only at keyframes and when track count
drops below threshold. ALIKED never runs at frame rate. (At CPU frame rates ALIKED is ~5 FPS class per the
catalogue's benchmark table; it cannot be in the 20–30 Hz path.)

---

# A. Inertial layer (new module: `cpp/slam/imu/`) — fills gap C1

## A.1 Scope decision

Full stereo-VIO is not required for Ariadne's research question, but three inertial products are: (1) a gravity
direction per agent, which yields the `GRAVITY_PRIOR` pose-graph edge and removes 2 of the 4 gauge rotational
DOF ambiguities across agents; (2) motion prediction between frames, which seeds LK search windows and makes the
tracker robust to fast rotation; (3) scale/robustness reinforcement of stereo odometry in texture-poor intervals.
The design below delivers all three with the standard machinery.

## A.2 Preintegration `[PAPER]`

Implement on-manifold IMU preintegration exactly per Forster, Carlone, Dellaert, Scaramuzza, *On-Manifold
Preintegration for Real-Time Visual-Inertial Odometry*, T-RO 2017 (arXiv:1512.02363; earlier RSS 2015). Between
keyframes i and j, integrate gyro/accel samples into preintegrated ΔR_ij, Δv_ij, Δp_ij with first-order bias
Jacobians (∂ΔR/∂b_g, ∂Δv/∂b_g, ∂Δv/∂b_a, ∂Δp/∂b_g, ∂Δp/∂b_a) and the 9x9 preintegration covariance propagated
per sample. State per keyframe becomes:

```text
T_WB ∈ SE(3)   (body = IMU frame; camera via fixed T_BC extrinsic)
v_W  ∈ R³
b_g, b_a ∈ R³  (random-walk between keyframes; σ from the simulated IMU config, §M.3)
```

Residuals: the standard three-block inertial residual (rotation via Log(ΔR̃_ijᵀ R_iᵀ R_j), velocity, position with
gravity g_W), plus a bias random-walk residual between consecutive keyframes. Add these as factors in the windowed
BA of v1 §10, alongside the reprojection factors; the Schur elimination structure is unchanged because inertial
factors touch only camera-block variables. Transcribe residual and Jacobian expressions from the paper's
equations, not from this document, and validate every Jacobian by the finite-difference harness of v1 §10.4.

## A.3 Initialization `[DESIGN, standard practice]`

Agents start stationary for a configurable 1–2 s. During this window: b_g = mean gyro; gravity direction = mean
accel normalized; R_WB initialized so that g_W = (0,0,-9.81) maps to the measured direction (yaw unobservable,
fixed to 0 per agent and reconciled later by inter-agent registration); b_a initialized to the residual after
gravity removal (weakly observable at rest; give it a loose prior). Stereo provides metric scale from the first
pair, so none of the monocular VI-initialization machinery (visual-inertial alignment solves) is needed.

## A.4 Gravity prior edge (producer for v1 §11.1)

For each keyframe, after windowed BA, emit a unary `GRAVITY_PRIOR` factor on the roll/pitch components of T_WB
with information from the current velocity/bias covariance. In the broker's global PGO these priors constrain all
agents' maps to a shared gravity direction, shrinking the inter-agent alignment problem from 6 DOF to 4
(yaw + translation) whenever both agents carry valid priors. This measurably improves chordal initialization
conditioning in the corridor scenes (test in Iteration 7).

## A.5 Motion prediction

Between frames, propagate T, v with raw gyro/accel (no preintegration needed at frame rate) to predict keypoint
locations for LK; on prediction/track disagreement above threshold, flag `DEGRADED` per v1 §8.2.

## A.6 Time synchronization contract

IMU and camera timestamps come from the same simulated clock (§M.7), so no online time-offset estimation is
implemented. `t_offset_cam_imu` exists in the calibration struct, is 0 in sim, and is estimated offline if a real
rig ever appears. Document this as a stated simplification.

## A.7 Fallback mode (if Iteration 3.5 slips)

`imu_mode: attitude_only` — complementary filter (e.g., Mahony) producing gravity direction and rotation
prediction only; no velocity/bias states in BA. Products (1) and (2) of §A.1 survive; product (3) is lost. The run
manifest records which mode produced every result.

---

# B. Perception front-end: exact OpenCV 5 usage (extends v1 §6–§8)

## B.1 DNN execution contract `[SMOKE-TEST for each model]`

- One `cv::dnn::Net` instance per thread; `Net` is not treated as thread-safe. The feature worker pool owns
  N nets loaded from the same ONNX file.
- XFeat: grayscale N×1×H×W float in [0,1]; 64-D float descriptors (catalogue header audit). Fix inference size at
  640×640 (the class default) initially; measure keypoint localization error vs full-res before changing.
- DISK: input dims must be positive multiples of 16; pad-and-crop wrapper owns this.
- ALIKED: RGB [1,3,H,W]; sub-pixel keypoints; 128-D.
- LightGlue path: `cv::LightGlueMatcher` requires `setPairInfo(qKpts, tKpts, qSize, tSize)` before `match()`
  (catalogue header audit). Wrap so it drops into the same `DescriptorMatcher` seam as BF/ANN.
- EigenPlaces: export ResNet-50 backbone + head to ONNX at opset 17 with fixed input 224×224 (or the repo's
  inference resolution; record it), ImageNet normalization baked into the export or applied in the wrapper, output
  L2-normalized 256-D. If the new OpenCV DNN engine rejects any op, the sanctioned fallback is the ONNX Runtime
  integration path that OpenCV 5 itself ships `[VERIFIED web 2026-08-30, docs.opencv.org/5.0]`, which keeps the
  "through OpenCV" claim intact; record which engine ran in the manifest.

## B.2 Retrieval-path vs geometry-path preprocessing split `[DESIGN]`

The catalogue's empirical finding is that photometric enhancement can help appearance matching while hurting
geometry. Exploit the asymmetry structurally: the **geometry path** (LK, stereo, triangulation, XFeat for
correspondence) receives only calibrated corrections (flat-field, VST-domain denoise, falloff normalization); the
**retrieval path** (EigenPlaces input) may additionally receive CLAHE/contrast normalization, because retrieval is
an appearance decision and its output never enters a residual. Ablate the retrieval-path enhancement separately
(add one row to v1 §29.5).

## B.3 Radiometric stages, concretely (fills v1 §6.2's function signatures)

1. **Flat-field:** per-camera gain image `K(x)` calibrated offline in sim from a uniform-albedo wall sweep;
   correction is one `cv::multiply` by the precomputed reciprocal. Store fp16.
2. **Poisson–Gaussian handling:** estimate per-(exposure, gain) noise parameters (α photon gain, σ read) offline
   from static captures (Foi et al., *Practical Poissonian-Gaussian noise modeling and fitting for single-image
   raw-data*, TIP 2008 `[PAPER]`). Online: generalized Anscombe transform
   `f(I) = (2/α)·sqrt(αI + (3/8)α² + σ²)`, denoise in the stabilized domain (start: `cv::fastNlMeansDenoising`
   with small h; also try `medianBlur` 3x3 for salt noise from dust), then the **exact unbiased inverse** of
   Mäkitalo & Foi (TIP 2013) `[PAPER]` — the naive algebraic inverse is biased at low counts and the bias lands
   exactly in the dark regions Ariadne cares about. Gate the whole stage on measured noise level per v1 §6.3.
3. **Falloff normalization:** with calibrated lamp intensity P, lamp pose T_C_L, and current depth estimate D(x),
   compute r(x) and divide by P/r² (optionally ·cosθ after §M.4 calibration decides). Output is the
   *albedo-proxy image* used by (a) feature extraction when the ablation says it helps and (b) the reconstruction
   input path (§L.3), where it is load-bearing.
4. **Exposure/gain normalization:** maintain photometric state (exposure t, analog gain g) from the controller
   (§B.4); normalize working images to a canonical (t₀, g₀) before any cross-frame photometric comparison.

## B.4 Active exposure + lamp co-control (new module: `cpp/perception/exposure_control/`)

This is the CERBERUS-named unfixed failure (catalogue §20, Counter 2) and a genuine novelty axis; it must be a
real controller, not a rendering setting.

- **Metric:** gradient-information score of Shim et al. (*Auto-adjusting camera exposure for outdoor robotics
  using gradient information*, IROS 2014) `[PAPER]`; alternative comparator: Zhang, Forster, Scaramuzza,
  *Active exposure control for robust visual odometry in HDR environments*, ICRA 2017 `[PAPER]`.
- **Controller:** at keyframe rate, evaluate the metric on the current frame and its saturation/shadow fractions;
  gradient-descent step on log-exposure with slew-rate limit (≤ 0.5 stop/frame) to avoid destabilizing LK; hard
  constraints from motion blur (t_max from predicted angular rate, §A.5) and from the noise model (t_min).
- **Lamp channel:** lamp power is a second actuator with the same objective plus an energy cost term; in sim it is
  actuated via the Gazebo light service (§M.4). Log both actuators in the decision trace; "overexposure event →
  controller response" is Failure E of v1 §16 and a demo beat.

## B.5 Keypoint budget arithmetic `[DESIGN]`

Targets, to be tuned by profiling: 1000–1500 tracked points per frame budgeted across an 8×6 grid (v1 §7.3),
XFeat max 4096 raw detections at keyframes filtered to ≤ 1200 stored, of which ≤ 800 are eligible for
transmission (§G.4 byte table depends on these caps; changing them changes the headline metric, so they are
manifest fields, not constants).

---

# C. Stereo depth, concretely (extends v1 §9)

## C.1 SGBM configuration `[DESIGN, standard]`

`StereoSGBM` with: blockSize 5–7; P1 = 8·ch·blockSize², P2 = 32·ch·blockSize²; uniquenessRatio 10–15;
speckleWindowSize 100–200, speckleRange 2; disp12MaxDiff 1; MODE_SGBM_3WAY for speed. Left-right consistency via a
right-matcher pass and `cv::ximgproc::DisparityWLSFilter` (contrib; also yields a confidence map — use it, do not
invent one). Subpixel: SGBM outputs fixed-point disparity in 1/16 px; carry σ_d = 0.25–0.5 px into
σ_Z = Z²/(f·B)·σ_d (equivalent to v1 §9.4's form).

## C.2 Confined-space disparity arithmetic (fills a v1 hole)

numDisparities bounds minimum range: Z_min = f·B/d_max. At 1280×720, HFOV 90° (f ≈ 640 px), B = 10 cm,
numDisparities = 256 gives Z_min ≈ 0.25 m; a narrow void demands Z_min ≤ 0.3 m, so 256 is the floor at this
baseline, and the B = 20 cm ablation config must raise numDisparities or accept Z_min ≈ 0.5 m. Encode this
constraint as a config validator that refuses inconsistent (f, B, numDisparities, Z_min) combinations rather than
producing silent blind zones.

## C.3 Depth output contract to downstream

`StereoDepth = {disparity fp16, Z fp16, σ_Z fp16, confidence u8, valid mask}` at keyframe resolution. TSDF
integration (§I) consumes Z/σ_Z/mask; landmark triangulation cross-checks LK-tracked points against stereo Z
(v1 §10.2's "inconsistent stereo depth" test is |Z_tri − Z_stereo| > 3·σ).

---

# D. Windowed BA and marginalization policy (fills v1 §10's silent gap)

v1 specifies Schur + LM but not what happens when a keyframe leaves the window. The two correct options are
(a) proper marginalization with first-estimate Jacobians, or (b) drop-with-prior-free window plus pose-graph
odometry edges. The catalogue (§7.11, "the two subtle killers") documents why (a) is a consistency minefield.

**Decision `[DESIGN]`:** option (b). When a keyframe exits the window: its final windowed pose and the relative
pose + joint covariance to its predecessor are frozen into an `ODOMETRY` pose-graph edge (composed covariance via
the adjoint transport of v1 core/uncertainty); its landmarks with <2 in-window observations are retired to the
static local map (still usable for relocalization, no longer optimized). No marginal prior is retained inside the
window. This trades some estimator optimality for eliminating an entire class of subtle inconsistency bugs, and
the pose graph recovers the correlations at the level Ariadne actually uses them. State the trade in the report.

Additional specifications v1 lacks:
- Robust kernel on reprojection residuals inside BA: Huber, δ = 1.5–2.5 px (loop-closure robustness stays in the
  pose graph per v1 §11.1; BA-level Huber only handles residual mismatches, not wrong data association).
- Gauge: fix the oldest in-window pose; no free-gauge solving inside the window.
- Covariance recovery: pose covariance of the newest keyframe from the corresponding block of S⁻¹ (S from the
  Schur system), computed by solving S·X = E_block rather than inverting S; this feeds the keyframe
  `UncertaintySummary`, the keyframe-insertion score, and the IG terms in §K.
- LM damping: λ₀ = 1e-4·max(diag H), ×2 on reject, ×1/3 on accept; terminate on relative cost < 1e-6 or 15 iters.

---

# E. Pose graph numerics, filled to implementation precision

## E.1 Chordal rotation initialization `[PAPER]`

Martinec & Pajdla (CVPR 2007) linear rotation synchronization as surveyed for SLAM by Carlone, Tron, Daniilidis,
Dellaert, *Initialization techniques for 3D pose graph optimization* (ICRA 2015): stack relative-rotation
constraints R_j ≈ R_ij R_i into a sparse linear system over the 9 rotation entries per node (3 independent RHS
solves with anchor rotation fixed), then project each 3×3 block to SO(3) via SVD with `R = U·diag(1,1,det(UVᵀ))·Vᵀ`
— the determinant correction is mandatory and is the catalogue's §7.7 warning. Then translations by linear least
squares with rotations fixed, then joint SE(3) refinement.

## E.2 PCM `[PAPER]`

Mangelson, Dominic, Eustice, Vasudevan, *Pairwise Consistent Measurement Set Maximization for Robust Multi-robot
Map Merging*, ICRA 2018. Consistency statistic for candidate inter-agent closures z_ik, z_jl:
`c = || Log( z_ik⁻¹ · x̂_i,j · z_jl · x̂_l,k ) ||²_Σ` with Σ composed along the path (adjoint transport);
consistent if c < χ²₆(0.95) = 12.59 (expose the quantile as config). v1 §13.5's greedy maximal consistent set
stands; implement exact max-clique only behind a flag for candidate sets ≤ 25 (the catalogue's §17.2 blind-spot
warning — self-consistent wrong cliques — is precisely why the §13.4 degeneracy gate runs *before* PCM; keep that
ordering inviolable).

## E.3 GNC-TLS update equations `[PAPER — re-derive signs from the paper before shipping]`

Yang, Antonante, Tzoumas, Carlone, *Graduated Non-Convexity for Robust Spatial Perception*, RA-L 2020
(arXiv:1909.08605). With residual r_i, inlier threshold ε (set ε² = χ²₆(0.99)·σ² for 6-DOF edge residuals), and
control parameter μ, GNC-TLS closed-form weights:

```text
w_i = 0                                    if r_i² ≥ ((μ+1)/μ)·ε²
w_i = 1                                    if r_i² ≤ (μ/(μ+1))·ε²
w_i = (ε/ r_i)·sqrt(μ(μ+1)) − μ            otherwise
```

μ schedule: μ₀ = ε² / (2·r_max² − ε²) (r_max = largest initial residual), multiply μ by 1.4 each outer iteration
until the surrogate matches TLS; each outer iteration is one weighted PGO solve with the existing LM solver.
Expose per-edge weight trajectories in the log (v1 §13.6's judge-visible trace). Odometry edges always w = 1.

## E.4 Rotation optimality certificate `[PAPER — convention-sensitive]`

Follow the duality line of Carlone, Rosen et al. (*Lagrangian duality in 3D SLAM*, IROS 2015), SE-Sync (Rosen,
Carlone, Bandeira, Leonard, IJRR 2019, arXiv:1612.07386) and Eriksson et al. (*Rotation Averaging and Strong
Duality*, CVPR 2018). Given the connection Laplacian Q̃ of the rotation subproblem and the solved rotations R̂:
build the block-diagonal dual variable Λ with `Λ_i = sym( Σ_j Q̃_ij R̂_j R̂_iᵀ )`, form the certificate matrix
S = Λ − Q̃ (sign convention differs across papers — lock it against whichever paper's formulation is transcribed,
and unit-test on a 5-node graph with a known-global-optimal solution), and verify S ⪰ 0 by computing λ_min with
shifted Lanczos on the sparse matrix. Report `certified / not certified (λ_min = …)`; never claim global
optimality on `not certified`. This is v1 §1.1's "keep" made executable; ~300 lines plus tests.

## E.5 Gauge handling in the broker solve

Global PGO gauge: fix the anchor agent's first pose (translation + yaw; roll/pitch already constrained by gravity
priors §A.4 when present, else fix full rotation). Never solve the free-gauge system and "handle it later".

---

# F. Broker, reference frames, and protocol state machines (fills v1 §14's distributed-systems gaps)

## F.1 Broker election with epochs `[DESIGN, follows Swarm-SLAM's choice of centralized-at-rendezvous]`

v1 adopts Swarm-SLAM's elected broker but never defines election under partitions. Specification:

- Connected components are discovered from heartbeats (v1 §14.2). Within a component, the broker is the lowest
  AgentId — deterministic, no voting protocol needed (matches Swarm-SLAM's "lowest ID" rule; RA-L 2024,
  arXiv:2301.06230).
- Every broker solve is stamped with an **epoch** `(component_member_set_hash, monotonic_counter)`. Agents accept
  a returned global estimate only if its epoch's member set contains them; stale epochs are discarded. Two
  simultaneous brokers in two partitions are *correct* under this rule — each optimizes its component — and the
  merge on reconnection is just a new epoch over the union.
- Broker state is soft: everything it needs (local subgraphs, verified edges) is re-sendable from agents'
  bookkeeping, so broker loss costs recomputation, never consistency. This is the property v1 §3.3 promised
  ("consistent or explicitly incomplete, never silent corruption") made concrete.

## F.2 Reference-frame algebra

Each agent i owns frame W_i (its startup pose, gravity-aligned). The global frame G is defined as W_a of the
current **anchor agent** a. Anchor propagation per Swarm-SLAM M5: on first inter-agent verification within a
component, anchor = lowest AgentId present; on later merges, the lower *anchor* id wins and the losing side's
agents update `T_G_Wi ← T_Ganchor_Gold · T_Gold_Wi`. Every pose that crosses the wire is tagged with the frame id
it is expressed in; the deserializer rejects untagged or unknown-frame poses. All Gaussians and routes are stored
in keyframe-local or W_i frames and composed with the current `T_G_Wi` at read time, so anchor changes and PGO
updates never rewrite stored geometry (§L.4).

## F.3 Store-and-forward queue policy (fills v1 §14.3)

Per-peer outbox with strict priority classes: 1 heartbeat, 2 pose edges + pose-graph batches, 3 global
descriptors, 4 feature responses, 5 submap summaries. Within class: FIFO. TTLs: descriptors 120 s (stale
retrieval is cheap to redo), edges and graphs infinite (they are the product). Under a link budget, class 4–5
messages are the ones the spectral policy (§H) is entitled to cancel. Idempotency: every message carries
`(sender, class, monotonic_seq)`; receivers keep per-sender high-water marks plus a small out-of-order buffer;
duplicates ack-only.

## F.4 Wire format and byte accounting `[DESIGN]`

Serializer: **FlatBuffers** (schema-versioned, zero-copy read, deterministic size accounting at the single
serialization boundary v1 §4.4 requires), plus zstd level 3 on class 4–5 payloads only (descriptors are
high-entropy; measure and report the actual ratio rather than assuming). `wire_bytes` = serialized + compression
+ 28 B emulated UDP/IP header per datagram so the metric is honest about framing. Quantization at the boundary:

| Payload | Encoding | Size |
|---|---|---|
| Global descriptor (256-D) | fp16 | 512 B + ~48 B envelope |
| Keypoint | u,v as uint16 (in 1/4 px), score u8, level u8 | 6 B |
| XFeat descriptor | int8 with per-keyframe scale (ablate vs fp16) | 64 B (128 B fp16) |
| Keyframe feature response, 800 kp | keypoints + int8 desc + K, T hint | ≈ 57 KB |
| SE(3) pose edge | 7×f32 pose + 21×f32 upper-tri info + ids/stats | ≈ 150 B |
| Pose-graph batch, n nodes / m edges | nodes 28 B + edges 150 B | e.g. 200 kf + 40 edges ≈ 12 KB |

These numbers reproduce the catalogue's key quote (front-end descriptor traffic dominates backend traffic by
orders of magnitude) from first principles: one feature response ≈ 380 pose edges. The int8-descriptor ablation
belongs in v1 §16.2's compression experiment with matching precision/recall curves.

## F.5 Comms simulation split (physical vs accounting layer)

Physical layer: gz-sim's **RFComms system plugin** — log-distance path-loss model ported from the SubT
competition infrastructure, with `max_range`, `fading_exponent`, `l0`, `sigma`, and radio `capacity` SDF
parameters `[VERIFIED web 2026-08-30, gazebosim.org RFComms API]`. Its SubT lineage is itself citable realism.
Accounting/policy layer: RFComms carries only a per-pair link-state estimate; **actual Ariadne payloads flow
through the Ariadne comms broker node** (§M.6), which enforces bandwidth/latency/loss sampled from the link
state, injects partitions from scenario scripts, and is the single point where ByteRecords are written. Rationale:
byte accounting and fault injection must be deterministic and replayable (§C6), which gz-transport delivery is
not. Wall/rubble penetration count (v1 §5.5) is computed in the broker by ray-casting the GT mesh between agents
and adding a per-wall attenuation term to the RFComms-derived loss.

---

# G. Spectral prioritization numerics (fills v1 §15's solver gap)

λ₂ and the Fiedler vector on the candidate Laplacian: LOBPCG or shifted-inverse Lanczos on the sparse L, warm-
started from the previous Fiedler vector (the graph changes by one edge per iteration of the greedy loop, so
warm starts converge in a few iterations). Marginal gain of candidate e with endpoints (a,b):
`gain_e = κ_e · (v₂[a] − v₂[b])²` (this equals v₂ᵀL_e v₂ for a weighted edge Laplacian — cheaper and clearer than
forming L_e). Selection maximizes gain_e / cost_e where cost_e is the *predicted wire bytes* of verifying e from
the §F.4 table (feature response size, minus keyframes already sent per bookkeeping, with vertex-cover sharing
per Swarm-SLAM M3). Connectivity bootstrap: if the candidate's endpoints lie in different connected components of
the fixed graph, gain is +∞ class (connectivity first, per Swarm-SLAM's note that the graph must be connected
before λ₂ is meaningful). Exact enumeration oracle for |candidates| ≤ 15 in tests (v1 Iteration 6).

---

# H. Volumetric mapping, ESDF, frontiers (new module: `cpp/planning/volumetric/` — v1 has no producer for
frontiers, clearance, or traversability)

v1 §19–20 consume occupancy, clearance, frontiers, and traversability that nothing in v1 produces. Specification:

1. **TSDF:** integrate keyframe depth (§C.3) into OpenCV 5 **HashTSDF** (dense RGB-D fusion is in the 5.0 release
   `[VERIFIED web 2026-08-30]`) at 8–10 cm voxels, per agent, in W_i; weights from 1/σ_Z². `[SMOKE-TEST]` the
   exact class names/API in the installed build; if the API is awkward for incremental multi-block use, the
   fallback is a from-scratch voxel-hash TSDF (~600 lines) with the OpenCV version kept as the correctness oracle.
2. **Occupancy tri-state** (free / occupied / unknown) derived from TSDF weight and sign; unknown = weight 0.
3. **ESDF** for clearance: 3D Euclidean distance transform over the occupied set via three-pass separable
   Felzenszwalb–Huttenlocher 1D lower-envelope transforms (*Distance Transforms of Sampled Functions*, Theory of
   Computing 2012) `[PAPER]`; recompute incrementally per updated block, not globally.
4. **Frontiers:** free voxels 6-adjacent to unknown, clustered (connected components), each cluster summarized by
   centroid + normal + visible-unknown volume estimate; these are §K's candidate generators A.
5. **Traversability + risk:** traversable(x) = ESDF(x) ≥ ρ where ρ is the **consumer cross-section radius** — a
   first-class parameter with two values, ρ_agent (0.15–0.25 m) for agent motion planning and ρ_responder
   (default 0.40 m, config) for the egress product. Every egress claim in the report and demo is computed at
   ρ_responder; an agent-passable void is not a human-passable route, and conflating them is the single easiest
   demo embarrassment to preempt. Structural risk proxy: overhang score from TSDF normals + unsupported-mass
   heuristic over the GT-labeled sim zones for calibration; label it a proxy everywhere (v1 §42's honesty list).
6. **Global coarse layer:** 0.5 m downsampled occupancy/uncertainty over G for the TARE-style global stage (§K)
   and for cross-agent coverage deconfliction; exchanged as run-length-encoded submap summaries (class 5).

---

# I. Egress subsystem, executable form (extends v1 §20)

Graph: 26-connected lattice over free voxels at ESDF ≥ ρ_responder, edge cost
`c(x→y) = ||x−y|| · (1 + w_u·U(x,y) + w_r·Risk(x,y) + w_c·max(0, ρ_safe − ESDF)/ρ_safe)` with U from observation
count/TSDF weight. Planner: **D\* Lite** (Koenig & Likhachev, AAAI 2002) `[PAPER]` — incremental replanning
matches the "voxels change under you" regime; the FAR-style visibility graph (v1 §19.6) remains the exploration-
time router and the comparator row in v1 §29.6. Route confidence: 5th-percentile of per-voxel support along the
route (v1 §20.2's conservative aggregation made specific). Metrics emitted per mission: time-to-first-valid-route
(first route whose every voxel is *observed* free at ρ_responder), route validity vs GT traversability, and
route-confidence calibration (predicted confidence vs realized validity across runs — a genuinely novel little
plot; add to §28.2 Mission block).

---

# J. Active perception: information gain, weights, and allocation (fills v1 §19.3–19.7)

1. **IG terms, computable forms:** Δcoverage = expected newly-observed unknown voxels by ray-casting the candidate
   frustum through the occupancy map (sparse ray bundle, 200–500 rays); Δentropy over the same rays with
   p(occupied) from TSDF weight; Δpose_connectivity = §G's spectral gain of the inter-agent edge the view is
   predicted to enable (this term is what makes "fly toward a peer's trajectory" competitive with "explore");
   Δegress_confidence = predicted removal of low-confidence voxels on the current best route (raycast
   intersection with the route corridor); Δloop_disambiguation = for alias-flagged place pairs (v1 §16.3), value
   of a viewpoint predicted to be discriminative (different geometry visible from the two hypotheses).
   Normalize each term by its running 90th percentile across current candidates (self-normalizing, no magic
   constants).
2. **λ scheduling (the proposal's "weights are not fixed" made real):** λ_e(t) = λ_e⁰ · (1 + (E_used/E_total)³) —
   energy cost inflates cubically toward endurance limit; λ_t similarly against the mission clock; when
   E_remaining < 1.3 × E(return path), the agent's candidate set is forcibly reduced to
   {return_to_safe_zone, hold} — a hard rule outside the utility function, logged as a supervisor event, so no
   tuning error can strand an agent.
3. **Allocation:** CBBA (Choi, Brunet, How, *Consensus-Based Decentralized Auctions for Robust Task Allocation*,
   T-RO 2009) `[PAPER]` restricted to bundle size 1–2 (tasks are short-horizon views; long bundles buy little and
   complicate consensus), event-triggered: bid rounds run only on (new task cluster, completed task, topology
   change), consensus messages ride the class-2 priority. Broker-connected components may shortcut to a
   sequential greedy auction at the broker; both paths are logged identically, and the ablation row "auction vs
   broker-greedy" is nearly free.

---

# K. Dense reconstruction: the co-moving-light coherence fix, and the gsplat boundary (extends v1 §17–18)

## K.1 The problem statement v1 must adopt explicitly

3DGS reconstructs a static radiance field; under carried illumination the scene's radiance is a function of the
observer, so naively-trained splats are photometrically incoherent and held-out PSNR measures the wrong thing.
v1 §18 gestures at this; v2 makes the resolution the *default* pipeline:

**Reconstruct an albedo-proxy field, not a radiance field.** Training images are the falloff-normalized outputs
of §B.3 step 3 (divide out P·cosθ/r² using calibrated lamp and current depth). The Gaussian color then
approximates albedo up to a global scale. For display, the console renders either the albedo-proxy directly
(honest, slightly flat) or relights it with a virtual lamp using the same forward model (photorealistic and
physically consistent). Held-out evaluation renders **with the light model applied at the held-out camera's lamp
pose**, so PSNR/SSIM compare like with like. The NFL-BA-inspired L_light term (v1 §18.2) then optimizes residual
near-field effects the normalization missed, rather than carrying the whole illumination burden. GLOW-style
specular downweighting (v1 §18.3) applies to the normalized images unchanged.

## K.2 Process boundary `[DESIGN]`

Reconstruction is a **Python service** (`python/reconstruction/`): PyTorch + gsplat (nerfstudio-project/gsplat;
rasterization backend) fed by `KeyframeBundle` records over gRPC or spooled files:

```text
KeyframeBundle = { kf_id, agent, frame_id_of_pose (graph version), T_WiC, K,
                  rgb (normalized, §K.1), depth+σ (fp16), exposure_state, lamp_state, mask }
```

Poses carry the pose-graph **version** they were exported at; the service subscribes to PGO updates and
re-anchors (below) rather than retraining.

## K.3 Submap anchoring so PGO never invalidates splats

Each submap is anchored to a reference keyframe; all its Gaussians live in that keyframe's frame. A PGO update
changes only the anchor's global pose; Gaussians move rigidly for free. Intra-submap drift correction is *not*
attempted (submaps are kept short: 20–40 keyframes, new submap on distance/rotation threshold or after any loop
closure that bends the local trajectory beyond a residual threshold). This is the standard consequence of
decoupling mapping from localization (v1 Correction D) carried to its implementation conclusion.

## K.4 Merge and appearance, pinned to the catalogue's verified mechanisms

- Overlap: nearest-chunk pruning from *Hierarchical 3D Gaussians* (Kerbl et al., SIGGRAPH 2024) as the baseline —
  a Gaussian outside its own submap's cell that is nearer another submap's cell is deleted; measure before
  adopting anything fancier (v1 §17.5's fusion becomes the *measured alternative*, not the default).
- Exposure: per-keyframe 3×4 affine color transform, initialized to identity, **enabled only after ~5000
  optimization steps** so geometry stabilizes first (Hierarchical 3DGS detail; the catalogue flags skipping the
  delay as the classic failure). With §K.1 normalization in place this transform absorbs residual auto-exposure
  differences between agents, which is its correct, small job.
- Never voxel-average Gaussians during merge (Fed3DGS negative result); if union growth becomes a problem, apply
  Fed3DGS-style distillation with a replay camera bank as the documented remedy.

## K.5 Compute placement

gsplat training runs on the g5/g6 node (§N); target per-scene budgets from the audit (~30 min/room-scale run on
g5.xlarge spot). The service checkpoints submaps to S3 so the console can stream `.ply`/`.splat` exports.

---

# L. Simulation stack: exact Gazebo Harmonic / ROS 2 Jazzy build-out (fills v1 §5 at the SDF/topic level)

## L.1 Versions `[VERIFIED web 2026-08-30]`

ROS 2 **Jazzy** (Ubuntu 24.04) officially pairs with Gazebo **Harmonic**; the Jazzy vendor packages vendor
Harmonic libraries (gazebosim.org ROS/Gazebo vendor-package and installation docs). Harmonic component versions:
gz-sim8, sdformat14, gz-transport13, gz-msgs10, gz-sensors8, gz-rendering8. Install `ros-jazzy-ros-gz` (bridge +
sim launch integration). Pin all of: ROS distro, gz-harmonic apt versions, ros_gz version, in the manifest.

## L.2 Agent model SDF, stereo + IMU + lamp

One `<model>` per agent, namespaced `ariadne/agent_<k>`:

- **Stereo:** two `<sensor type="camera">` on one rigid link with `<pose>` offsets = baseline; identical
  `<update_rate>`, resolution 1280×720, `<clip near>` 0.05. Hardware-synchronous capture is emulated with
  gz-sensors **triggered cameras** (`<triggered>true</triggered>` + shared `<trigger_topic>`, driven by a fixed-
  rate publisher) `[SMOKE-TEST: confirm triggered-camera support in the installed gz-sensors8; fallback is
  same-update-rate free-running cameras plus timestamp pairing, acceptable because both render at the same sim
  step]`. Set `<camera><distortion>` to a mild plumb-bob so the undistort path is exercised (Harmonic supports
  camera distortion `[VERIFIED — Harmonic release features list]`).
- **Noise:** configure gz's built-in Gaussian camera noise to *zero* and apply all sensor degradation in the
  seeded degradation operator (§L.5) — this is what makes C6's record/replay clean-signal design work. IMU noise
  *is* configured in SDF (`<imu><noise>` gyro/accel densities + bias random walk) because inertial noise must be
  integrated by the estimator, not overlaid on images; choose consumer-MEMS-class values and record them (they
  are §A.2's σ inputs).
- **Lamp:** `<light type="spot">` attached to the camera link (inner/outer cone 25°/40°, quadratic attenuation,
  `<cast_shadows>true</cast_shadows>`); runtime dimming/aiming via the world's
  `/world/<name>/light_config` service with `gz::msgs::Light` `[VERIFIED web 2026-08-30, gz-sim light_config
  tutorial]`, wrapped by a small ariadne_sim node exposing `set_lamp(power)` to §B.4. Spot rather than point:
  matches a real headlamp lobe and gives ogre2 well-supported shadow mapping; shadow texture size is tunable
  (gz-sim shadow-texture-size tutorial exists) — raise it if shadow aliasing pollutes features. World ambient
  light set to near-zero for dark scenarios.
- **Motion:** no physics locomotion (v1 §5.6): a kinematic mover system sets the model pose along commanded
  trajectories (velocity-limited SE(3) interpolation with the collision envelope checked against the GT mesh by
  the mover itself). Implement as a gz system plugin or by driving `/world/<w>/set_pose` services from the agent
  runtime; the plugin is preferred for rate stability. GT pose from gz's pose publisher; GT depth from a
  `<sensor type="depth_camera">` co-located with the left camera (recorded, never visible to perception).

## L.3 Bridge and topics

`ros_gz_bridge parameter_bridge` from a YAML config: per agent, left/right `gz.msgs.Image → sensor_msgs/Image`,
`camera_info`, `gz.msgs.IMU → sensor_msgs/Imu`, GT pose, depth; world `/clock` (`gz.msgs.Clock →
rosgraph_msgs/Clock`); every ROS node runs `use_sim_time:=true`. All ROS topics under `/ariadne/agent_<k>/…`.

## L.4 World generation pipeline (fills v1 §5.3's "generate parametrically")

`python/dataset_tools/worldgen`: (1) sample a procedural floorplan (rooms/corridors grammar), (2) apply collapse
operators (slab drop with rotation, wall shear, beam scatter, rubble piles as convex-decomposed heaps, void
carving) with tunable severity and *repetition operators* that deliberately clone structural elements for the
aliasing lab (Iteration 7), (3) assemble meshes with `trimesh`, assign low-texture concrete PBR materials,
(4) emit SDF via Jinja templates against sdformat14 (Harmonic also ships a Python SDFormat API — use whichever is
less friction, but pin it), (5) bake the hidden GT layer: watertight mesh → voxel occupancy → signed distance
(mesh-to-sdf) → GT traversability/clearance at both ρ values → GT egress routes by running §I's planner on GT →
store as `scene_gt.npz` beside the world. Scene = (grammar seed, severity vector) — two numbers reproduce a world.

## L.5 Degradation operator (consumption-time, seeded — the C6 mechanism)

A deterministic operator `D(image, params, seed, frame_id)` applied when frames are read from the recording (and
identically in live mode): sensor response chain in order — scene radiance scale (darkness level) → Poisson shot
noise (per-pixel, seeded by (seed, frame_id, pixel-block)) → read noise → PRNU/vignette field → exposure/gain →
quantization/clipping → optional dust compositing `I = J·t + A·(1−t)`, `t = exp(−β·depth_gt)` plus a Perlin local
particulate field (uses the recorded GT depth, which is the correct physics and is only possible because GT depth
rides in the recording). Params vectors are named configs (`dark2_dust1`, …) referenced by manifests. The lamp's
*optical* effect (falloff, shadows) is real Gazebo rendering, not part of D — D degrades the *sensor*, the
renderer degrades the *scene*. Exposure control in replay mode acts on D's exposure parameters, closing the §B.4
loop without re-rendering; document that replayed exposure control changes sensing but not the light transport
(a stated, measured approximation — live runs quantify the difference in Iteration 9).

## L.6 Recording format

rosbag2 **MCAP** (default storage since ROS 2 Iron), one bag per (scene, trajectory, seed): clean images, IMU, GT
streams, tf, lamp state, camera infos. Bags are the S3 `datasets/` payload; every downstream figure's manifest
names its bag checksums. Live closed-loop runs record the same topics *plus* all Ariadne decision/telemetry
topics, giving v1 §40.17's replayable trace.

## L.7 Headless operation on AWS `[VERIFIED web 2026-08-30]`

Gazebo supports EGL headless rendering with ogre2 via `gz sim -s -r --headless-rendering world.sdf` on a GPU
instance without X (gazebosim.org headless-rendering tutorial, which itself walks through the AWS GPU setup).
Camera sensors require the render engine, hence rendering-bearing sim runs live on the g5/g6 node; Graviton runs
no Gazebo. NVIDIA container runtime with EGL libs in the sim image (§N.2).

## L.8 Run topology recap

Batch evaluation: Gazebo (GPU, once per scene/trajectory) → MCAP → S3 → N deterministic consumer replicas
(Graviton/COOL for CPU pipeline sweeps; GPU node for reconstruction) — this is the sweep economics *and* the COOL
multi-agent throughput benchmark ("agents per instance" = concurrent replay pipelines per m8g.4xlarge). Live
mode: full ROS 2 graph, agents + broker + console, for active perception, failure campaign, and the demo video.

---

# M. AWS, COOL, containers, CI (fills v1 §24–26 to runnable precision)

## M.1 Image matrix (supersedes v1 §25.1's list with contents)

| Image | Base | Contents | Runs on |
|---|---|---|---|
| `ariadne-core` (amd64+arm64, one Dockerfile, buildx) | ubuntu:24.04 | gcc-13, CMake, Eigen 3.4, OpenCV 5 + contrib built from pinned tag, FlatBuffers, zstd, ONNX models, ariadne cpp | dev, CI, Graviton (stock baseline) |
| `ariadne-cool` (arm64) | COOL AMI toolchain layered per vendor instructions | same ariadne cpp linked/`LD_LIBRARY_PATH`-switched against `/opt/cool` OpenCV | m8g.4xlarge |
| `ariadne-sim` (amd64) | ros:jazzy | gz-harmonic, ros-jazzy-ros-gz, worlds, EGL libs, degradation op | g5/g6 |
| `ariadne-gpu` (amd64) | nvidia/cuda:12.x-runtime-ubuntu24.04 | PyTorch (pinned cu12x), gsplat (pinned), reconstruction service, OpenCV 5 py | g5/g6 |
| `ariadne-console` (amd64) | python:3.12-slim | FastAPI + WebSocket API, static Three.js frontend | t4g/t3.small or ECS |

COOL benchmark discipline: same binary, same instance, same inputs; the *only* varied element is which OpenCV the
loader resolves (stock build vs `/opt/cool`). Pin `cv::setNumThreads` and thread affinity identically; report
p50/p95 per-op and per-pipeline latency, frames/s, utilization (`perf stat`), and cost/frame = (instance
$/hr)/(3600·fps); sweep concurrent agent-pipeline count to the knee. This satisfies every line of the COOL rubric
evidence list and honors v1 §24.3's Arm-vs-Arm rule. Given Universal Intrinsics 2.0 (§C3), also report the stock
OpenCV 5 Arm numbers vs OpenCV 4.x once, as context for how much the baseline itself moved.

## M.2 Build/CI

GitHub Actions: amd64 jobs on hosted runners; arm64 jobs on GitHub's Arm hosted runners if available to the
account `[SMOKE-TEST]`, else a self-hosted runner on the Graviton box (register once; it doubles as the benchmark
machine, with benchmark jobs serialized behind a lock). OpenCV 5 compile is cached via ccache + a prebuilt
"opencv-base" layer rebuilt only when the pinned tag changes. Push to ECR; images referenced by digest, never by
tag, in run manifests.

## M.3 Infra as code

Terraform modules: `network` (one VPC, public subnet, SGs), `storage` (S3 buckets from v1 §26.1 + lifecycle:
runs/ → IA at 30 d), `registry` (ECR), `compute` (launch templates for m8g.4xlarge, g5.xlarge/g6.xlarge with
spot + on-demand fallback), `sweeps` (SQS queue + tiny ASG of worker instances whose user-data pulls a manifest
message, runs the container, uploads artifacts, terminates on idle), `budget` (AWS Budgets alarm at 50/80% of
credits). No EKS, no Kubernetes; the sweep pattern is SQS + containers, which one person can operate.

## M.4 Agentic layer placement

Run the orchestrator LLM through **Amazon Bedrock** (any strong tool-use model; invoke via the converse/tool-use
API), which makes the agentic layer itself part of the "meaningful AWS component" story rather than an external
dependency. MCP server (`python/orchestration/`) exposes exactly the v1 §23.2 tool list; every tool call and the
vision-derived inputs that motivated it are appended to the decision trace (JSONL in the run artifact — the
Agentic Vision "trace" deliverable). Hard rule from v1 §23.1 enforced in code: the deterministic planner computes
the candidate action set and the safety envelope; the LLM may only *choose among* presented candidates and
compose escalation packets; an out-of-set response falls back to the deterministic choice and logs the event
(that log line is itself good failure-handling evidence). Cadence 0.2–1 Hz. The classical-planner ablation
(v1 §23.3) is then literally "skip the choose step".

## M.5 Operator console

FastAPI backend (WebSocket telemetry at 2–10 Hz, REST for approvals) + Three.js frontend with a Gaussian-splat
web renderer (e.g. the GaussianSplats3D Three.js library) streaming submap exports from S3; panels per v1 §27:
raw vs restored feeds side by side, budget gauges, negotiation timeline, uncertainty overlay, the thread, and the
approval modal wired to `HUMAN_APPROVAL_REQUEST`. This is the "working web endpoint" submission artifact; host it
live for judging week on one small instance.

---

# N. Iteration-plan deltas (v1 §31 otherwise stands)

- **Iteration 0/1:** MCAP record/replay and the seeded degradation operator move here (they are the determinism
  substrate everything else tests against). Exit condition addendum: same bag + same seed ⇒ bit-identical metrics.
- **New Iteration 3.5 — inertial layer (§A):** after oracle-validated BA, before robustness work; exit = EuRoC
  sequences (stereo+IMU, the natural first dataset for this iteration) within published stereo-VIO error class,
  and gravity priors active in the pose graph.
- **Iteration 6:** add the §F.4 byte table as a measured artifact (predicted vs actual wire bytes per class);
  prediction error > 15% fails the iteration (the spectral policy's cost model depends on it).
- **Iteration 9:** add live-vs-replay exposure-control comparison (§L.5's stated approximation, quantified).
- **Iteration 12:** egress metrics computed at ρ_responder; add route-confidence calibration plot (§I).
- **Iteration 14:** COOL workload defined per C4; add the OpenCV4-vs-OpenCV5 Arm context measurement (§M.1).

---

# O. Consolidated new-source list for subsequent agents (beyond v1 §46)

Verified live this revision: OpenCV 5.0 release facts — https://docs.opencv.org/5.0/ ,
https://github.com/opencv/opencv/wiki/OpenCV-5 , https://github.com/opencv/opencv/wiki/OpenCV-4-to-5-migration ,
https://opencv.org/opencv-5/ · ROS 2 Jazzy ↔ Gazebo Harmonic pairing — gazebosim.org ROS/Gazebo vendor and
installation docs · Gazebo headless EGL — https://gazebosim.org/api/sim/9/headless_rendering.html · RFComms
(SubT-ported log-distance path loss) — https://gazebosim.org/api/sim/9/classgz_1_1sim_1_1systems_1_1RFComms.html ·
gz light_config service — gz-sim light_config tutorial.

Papers newly load-bearing in v2: Forster et al., IMU preintegration, T-RO 2017, arXiv:1512.02363 · Foi et al.,
Poisson–Gaussian noise fitting, TIP 2008 · Mäkitalo & Foi, exact unbiased inverse GAT, TIP 2013 · Shim et al.,
gradient-based auto-exposure, IROS 2014 · Zhang et al., active exposure control, ICRA 2017 · Mangelson et al.,
PCM, ICRA 2018 · Yang et al., GNC, RA-L 2020, arXiv:1909.08605 · Carlone et al., PGO initialization, ICRA 2015 ·
Rosen et al., SE-Sync, IJRR 2019, arXiv:1612.07386 · Eriksson et al., Rotation Averaging and Strong Duality,
CVPR 2018 · Felzenszwalb & Huttenlocher, distance transforms, ToC 2012 · Koenig & Likhachev, D* Lite, AAAI 2002 ·
Choi, Brunet, How, CBBA, T-RO 2009 · Umeyama, TPAMI 1991 · Kerbl et al., Hierarchical 3D Gaussians, SIGGRAPH
2024, arXiv:2406.12080 · Lajoie & Beltrame, Swarm-SLAM, RA-L 2024, arXiv:2301.06230.

**Standing instruction to build agents:** every equation tagged `[PAPER]` in this document is a faithful sketch,
not a transcription; open the paper, transcribe the exact expression, and land the unit test from v1 §35 in the
same commit. Every `[SMOKE-TEST]` item is a day-1 task whose result is written into `docs/SMOKE_TEST_RESULTS.md`
and the run-manifest schema before dependent code is written.
