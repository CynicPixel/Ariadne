# ARIADNE — End-to-End Technical Architecture & Scoped Implementation Plan

**Project:** Ariadne — collaborative visual reconstruction and risk-weighted egress mapping for autonomous agents in GPS-denied, communication-constrained, visually degraded post-disaster structures.

**Purpose of this document:** Convert `ALGORITHMS(1).md` into an implementation-ready architecture, module boundary specification, algorithm selection, simulator/deployment stack, staged development program, test strategy, evaluation harness, and AWS/COOL deployment design.

**Design stance:** ambitious by design. The implementation should use the strongest technically defensible methods identified in the algorithm research, reimplemented from the papers where appropriate. No algorithm is excluded merely because OpenCV 5 does not expose a one-call API: OpenCV 5 is the primitive substrate; Ariadne owns the system-level implementations.

**Important:** This document is an implementation specification, not a claim that every component is already implemented. Each algorithm is tied to a validation test and a staged integration checkpoint.

---

## 0. Executive architectural decision

Ariadne should be built as a **vision-primary, stereo-inertial, multi-agent system with classical geometric state estimation and learned visual primitives**, not as an end-to-end learned SLAM stack.

The canonical runtime path is:

```text
Gazebo Harmonic / ROS 2 Jazzy simulation
        |
        v
per-agent stereo RGB + IMU + co-mounted-light streams
        |
        v
Radiometric calibration + low-light/noise normalization
        |
        v
XFeat/ALIKED keypoints + sparse tracking
        |
        v
stereo geometry + PnP/essential geometry + local landmark map
        |
        v
windowed SE(3) bundle adjustment
        |
        v
local pose graph + uncertainty
        |
        +-----------------------------+
        |                             |
        v                             v
EigenPlaces global retrieval      local map / frontier state
        |                             |
        v                             v
candidate inter-agent matches     active perception candidate generation
        |                             |
        v                             v
USAC -> geometric verification -> spectral communication prioritization
        |                             |
        +--------------+--------------+
                       |
                       v
              PCM pre-filter
                       |
                       v
                GNC robust PGO
                       |
                       v
         elected-broker global map state
                       |
             +---------+---------+
             |                   |
             v                   v
      reconstruction       traversability /
      / Gaussian maps      uncertainty volume
             |                   |
             +---------+---------+
                       |
                       v
       risk-aware multi-agent task allocation
                       |
                       v
               next observations
                       |
                       v
             continuous closed loop

Parallel operational plane:
    human approval / failure supervisor / observability / byte accounting
```

### 0.1 Why this is the chosen shape

The algorithm catalogue converges independently on the same architectural pattern from several directions:

- SVO/ORB-SLAM3-style systems motivate inexpensive frame-to-frame tracking plus expensive keyframe processing.
- Photo-SLAM separates classical localization from photorealistic mapping.
- EC3R-SLAM separates a lightweight sparse tracker from expensive feed-forward dense reconstruction.
- MAGS-SLAM demonstrates the emerging feasibility of **RGB-only multi-agent Gaussian SLAM**, using local maps and compact summaries.
- Swarm-SLAM demonstrates that communication-aware sparse collaboration can be more efficient than repeatedly running distributed back-end optimization.
- SubT experience shows that autonomy and disconnection tolerance matter more than assuming a perfect network.

These are not interchangeable systems; Ariadne synthesizes their strongest mechanisms while preserving a deliberately classical, inspectable geometric core.

References:
- XFeat: https://arxiv.org/abs/2404.19174
- LightGlue: https://arxiv.org/abs/2306.13643
- Photo-SLAM: https://arxiv.org/abs/2311.16728
- RTG-SLAM: https://arxiv.org/abs/2404.19706
- EC3R-SLAM: https://arxiv.org/abs/2510.02080
- MAGS-SLAM: https://arxiv.org/abs/2605.10760
- Swarm-SLAM: https://arxiv.org/abs/2301.06230
- OpenCV 5: https://github.com/opencv/opencv/wiki/OpenCV-5

---

# 1. Audit of `ALGORITHMS(1).md`

## 1.1 What the algorithm catalogue gets exactly right

### Keep: OpenCV 5 as a real computational substrate

The catalogue's central architectural claim is correct: OpenCV 5 is not merely a preprocessing dependency. Current OpenCV 5 documentation exposes learned local features and modern matching interfaces, and OpenCV 5 has reorganized the former `calib3d` stack into geometry/calibration/stereo/point-cloud functionality. The official OpenCV material confirms ALIKED, DISK, LightGlue, ANN search, and the reworked geometric stack.

Sources:
- https://github.com/opencv/opencv/wiki/OpenCV-5
- https://docs.opencv.org/5.0/main_modules/features_main.html
- https://docs.opencv.org/doc/doxygen/html/d8/dc3/classcv_1_1LightGlueMatcher.html
- https://docs.opencv.org/5.0/main_modules/features_annoy.html
- https://docs.opencv.org/5.0/main_modules/ptcloud.html

Implementation consequence: make `cv::*` primitives visible at the module boundary and log them in the run manifest.

---

### Keep: XFeat as the primary local feature engine

The algorithm catalogue's choice of XFeat is particularly well motivated for Ariadne:

- lightweight,
- compact 64-D descriptors,
- explicitly designed for resource-limited devices,
- strong localization/pose behavior relative to lightweight alternatives,
- useful semi-dense matching mode,
- good fit for a byte-constrained fleet.

XFeat's paper describes a large-resolution/low-channel design aimed at efficient correspondence and reports real-time operation on inexpensive CPUs.

Source:
- https://arxiv.org/abs/2404.19174

Implementation consequence:

```text
tracking:
    XFeat descriptors only when needed

keyframe:
    XFeat sparse features
    optional XFeat semi-dense refinement

inter-agent candidate:
    XFeat candidate descriptors
    ALIKED + LightGlue verification for selected hard pairs
```

Do not replace XFeat with a heavier generic descriptor simply because a benchmark has a better raw matching score. Ariadne's objective includes communication cost.

---

### Keep: ALIKED + LightGlue as the difficult-pair verifier

OpenCV 5 exposes ALIKED and LightGlue-compatible interfaces. The official LightGlue documentation describes contextual matching rather than plain descriptor distance. LightGlue's own paper emphasizes adaptive computation as image-pair difficulty changes.

Sources:
- https://arxiv.org/abs/2306.13643
- https://docs.opencv.org/doc/doxygen/html/d8/dc3/classcv_1_1LightGlueMatcher.html
- https://docs.opencv.org/5.0/main_modules/features_main.html

Implementation consequence:

- XFeat is optimized for compactness and fast repeated use.
- ALIKED + LightGlue is the expensive verification path for the relatively small number of candidate pairs selected by the communication policy.
- This is not an API limitation workaround; it is deliberate algorithmic specialization.

---

### Keep: EigenPlaces for global place retrieval, with a domain-adaptation experiment

EigenPlaces was specifically designed to improve viewpoint robustness in visual place recognition by training on clusters of different views of the same place. That is highly relevant to agents seeing the same corridor from different trajectories/heights.

Source:
- https://arxiv.org/abs/2308.10832

However, the algorithm catalogue correctly flags a real risk: standard VPR training domains are not equivalent to dark, texture-poor collapsed interiors.

Implementation consequence:

1. Start with pretrained EigenPlaces R50 256-D.
2. Export to ONNX.
3. Run through OpenCV DNN.
4. Store/transmit fp16 or int8 representation as an optional communication compression experiment.
5. Fine-tune on synthetic Ariadne views only after baseline retrieval is characterized.
6. Measure retrieval quality separately from descriptor size.

Use `cv::ANNIndex` for retrieval/index persistence.

---

### Keep: Swarm-SLAM communication architecture

This is the strongest direct analogue for Ariadne's communication layer. Swarm-SLAM is explicitly designed around sparse collaborative SLAM and introduces an inter-robot loop-closure prioritization mechanism.

Source:
- https://arxiv.org/abs/2301.06230

Adopt:

- global descriptor broadcast,
- candidate selection before expensive local-descriptor exchange,
- communication bookkeeping,
- sparse exchange,
- elected-broker global optimization,
- anchor/reference-frame propagation,
- evaluation in bytes versus map quality.

The algorithm catalogue reports that Swarm-SLAM's real-world experiment exchanged 94.95 MB over a heterogeneous three-robot setup and reports large reductions in communication relative to distributed alternatives. That becomes an excellent benchmark target, but do not copy the exact metric envelope across different sensors/tasks; use it as architectural evidence.

---

### Keep: GNC as the primary robust backend mechanism

GNC is unusually attractive because it is a wrapper around a weighted least-squares solver rather than a replacement optimizer.

Source:
- https://arxiv.org/abs/1909.08605

Implementation:

```text
initialize robust parameter μ
while not converged:
    solve weighted PGO
    compute whitened residuals
    update loop-edge weights analytically
    decrease μ
```

Use robustification **only on candidate loop-closure/inter-agent edges**. Do not indiscriminately robustify trusted odometry edges.

The GNC paper reports robustness to high outlier ratios and compares favorably with RANSAC and specialized robust methods. These are published results, not an Ariadne guarantee; Ariadne must measure its own operating regime.

---

### Keep: SE-Sync-style certificate as a verification artifact, not runtime solver

The certificate is the right "depth per line of code" component. The runtime does not need to become a full certifiably optimal solver.

Implementation:

- solve Ariadne's practical PGO with GNC,
- construct the rotation connection matrix / certificate,
- estimate the smallest eigenvalue with a Lanczos or shifted power-iteration procedure,
- report whether the sufficient certificate condition is satisfied.

Never label the map "provably globally optimal" unless the certificate condition actually holds. Otherwise report the numerical certificate as inconclusive.

---

### Keep: layered loop-closure rejection

Use:

```text
retrieval
 -> cheap candidate filters
 -> USAC / 2-view geometry
 -> degeneracy test
 -> PCM consistency
 -> robust PGO/GNC
 -> post-fit residual + certificate
```

Crucially, **not every candidate should reach PCM or PGO**.

The algorithm catalogue identifies a blind spot: pairwise-consistency methods become less meaningful if the underlying relative-geometry hypotheses are themselves geometrically degenerate. Therefore:

```text
detect degeneracy BEFORE trusting a loop hypothesis
```

This makes the loop closure stack more honest than "run robust optimization on everything."

---

## 1.2 Corrections / decisions that should override simplistic readings of the catalogue

### Correction A: do not build a fully distributed optimizer just because the problem is multi-agent

Swarm-SLAM's own architecture strongly motivates the opposite at Ariadne's scale: use sparse inter-agent communication and an elected broker to run the expensive global solve at rendezvous/contact events.

Reason:

- few agents,
- intermittent connectivity,
- expensive dense optimizer state,
- explicit byte budget,
- need for deterministic convergence behavior,
- easy debugging.

A decentralized optimizer can still be implemented as an experimental baseline, but should not be the production path.

---

### Correction B: "monocular-first" is not the best-performing physical configuration

The literature gap around multi-agent monocular 3DGS is valuable for novelty, but Ariadne's operational output is a **metric traversability/egress route**, which strongly favors metric sensing.

Therefore the primary simulated agent should be:

```text
stereo RGB
+
co-mounted controllable illumination
+
IMU
```

Use monocular mode as a benchmark/ablation. This gives:

- metric depth,
- better degeneracy handling,
- gravity priors,
- stronger egress geometry,
- more defensible risk distance estimates,
- a direct OpenCV `stereo` / `ptcloud` workload.

MAGS-SLAM should remain the literature comparator showing why an RGB-only version is interesting, rather than forcing the entire project into monocular scale ambiguity.

---

### Correction C: NFL-BA belongs primarily in the reconstruction/photometric layer

The NFL-BA paper is compelling evidence that co-located lights create systematic near-field effects and that modeling the light geometry improves mapping/tracking.

Source:
- https://arxiv.org/abs/2412.13176

Do not make every frame's geometric tracking depend on a full photometric near-field optimization. Ariadne's primary tracking remains geometric.

Instead:

1. calibrate the light,
2. compensate obvious falloff/radiometric effects before feature extraction,
3. use near-field illumination terms in dense reconstruction and keyframe photometric refinement,
4. use the illumination model as an additional depth/quality cue where justified,
5. measure whether it improves geometry in the actual simulator.

This preserves runtime reliability while exploiting the strongest physics contribution.

---

### Correction D: 3DGS must be decoupled from localization

Photo-SLAM's key architectural point is highly relevant: photorealistic mapping can be maintained alongside a geometric localization system rather than forcing pose estimation to depend upon differentiable rendering.

Source:
- https://arxiv.org/abs/2311.16728

RTG-SLAM is likewise useful as a design source for Gaussian lifecycle/stability ideas.

Source:
- https://arxiv.org/abs/2404.19706

Ariadne's geometry is therefore authoritative. The Gaussian representation is a consumer of validated cameras and depths.

---

### Correction E: do not make VGGT the tracker

The algorithm catalogue flags the memory wall correctly. VGGT-family models are better treated as expensive submap processors/verifiers, not as per-frame tracking.

VGGT-SLAM 2.0 specifically studies efficient incremental submap alignment and reports onboard robot operation.

Source:
- https://arxiv.org/abs/2601.19887

EC3R-SLAM independently reinforces the architectural pattern of a lightweight tracker plus expensive feed-forward mapping.

Source:
- https://arxiv.org/abs/2510.02080

Ariadne therefore uses classical geometric tracking and can optionally use a feed-forward model for **submap initialization, verification or offline accuracy comparison**.

---

# 2. System-of-systems architecture

## 2.1 Major planes

Ariadne has six logical planes.

### Plane A — Sensor and simulation

Produces timestamped:

- left RGB,
- right RGB,
- IMU,
- light-control state,
- ground-truth pose,
- ground-truth depth,
- ground-truth normals,
- ground-truth traversability,
- communication channel state.

### Plane B — Agent perception/state estimation

Produces:

- radiometrically normalized frames,
- keypoints/descriptors,
- tracked landmarks,
- stereo depths,
- local camera states,
- uncertainty,
- keyframes,
- local pose graph.

### Plane C — Collaboration

Produces:

- global descriptors,
- candidate loop edges,
- verified relative poses,
- communication decisions,
- synchronized subgraphs,
- global reference-frame transforms.

### Plane D — Reconstruction

Consumes validated keyframes and camera geometry and produces:

- local point clouds,
- local Gaussian submaps,
- merged Gaussian scene,
- confidence/coverage volumes,
- held-out-view renders.

### Plane E — Active mission intelligence

Consumes:

- uncertainty,
- coverage,
- traversability,
- communication connectivity,
- agent energy state,
- time remaining,
- existing task assignments.

Produces:

- candidate observation poses,
- task assignments,
- rendezvous tasks,
- revisit requests,
- communication actions,
- human escalation requests.

### Plane F — Operator / evaluation / reliability

Provides:

- live map,
- agent health,
- communication statistics,
- uncertainty,
- route confidence,
- audit traces,
- failure replay,
- evaluation metrics,
- restart/checkpoint behavior.

---

# 3. Repository structure

Use a monorepo with strict dependency direction.

```text
ariadne/
├── CMakeLists.txt
├── pyproject.toml
├── README.md
├── LICENSES/
├── docs/
│   ├── ARCHITECTURE.md
│   ├── ALGORITHMS.md
│   ├── IMPLEMENTATION.md
│   ├── EVALUATION.md
│   ├── PROTOCOL.md
│   ├── SIMULATION.md
│   └── FAILURE_MODES.md
│
├── configs/
│   ├── agent.yaml
│   ├── fleet.yaml
│   ├── camera.yaml
│   ├── illumination.yaml
│   ├── comms.yaml
│   ├── planner.yaml
│   ├── reconstruction.yaml
│   └── benchmark/*.yaml
│
├── third_party/
│   └── reference_only/
│
├── opencv/
│   └── patches/
│
├── cpp/
│   ├── core/
│   │   ├── math/
│   │   ├── geometry/
│   │   ├── uncertainty/
│   │   ├── time/
│   │   └── serialization/
│   │
│   ├── perception/
│   │   ├── radiometric/
│   │   ├── denoise/
│   │   ├── features/
│   │   ├── tracking/
│   │   ├── stereo/
│   │   ├── degeneracy/
│   │   └── keyframes/
│   │
│   ├── slam/
│   │   ├── landmarks/
│   │   ├── local_map/
│   │   ├── bundle_adjustment/
│   │   ├── pose_graph/
│   │   └── loop_closure/
│   │
│   ├── collaboration/
│   │   ├── retrieval/
│   │   ├── candidate_selection/
│   │   ├── protocol/
│   │   ├── pcm/
│   │   ├── gnc/
│   │   ├── broker/
│   │   └── reference_frames/
│   │
│   ├── reconstruction/
│   │   ├── pointcloud/
│   │   ├── gaussian/
│   │   ├── submap/
│   │   ├── fusion/
│   │   └── rendering/
│   │
│   ├── planning/
│   │   ├── frontier/
│   │   ├── nbv/
│   │   ├── risk/
│   │   ├── assignment/
│   │   ├── route/
│   │   └── rendezvous/
│   │
│   └── runtime/
│       ├── agent/
│       ├── supervisor/
│       ├── metrics/
│       └── logging/
│
├── python/
│   ├── training/
│   ├── dataset_tools/
│   ├── analysis/
│   ├── visualization/
│   └── operator_console/
│
├── ros2/
│   ├── ariadne_interfaces/
│   ├── ariadne_sim/
│   ├── ariadne_bridge/
│   ├── ariadne_agent/
│   └── ariadne_operator/
│
├── simulator/
│   ├── worlds/
│   ├── models/
│   ├── sensors/
│   ├── illumination_plugin/
│   ├── dust_plugin/
│   └── comms_plugin/
│
├── evaluation/
│   ├── datasets/
│   ├── scenarios/
│   ├── metrics/
│   ├── baselines/
│   ├── ablations/
│   └── reports/
│
├── infra/
│   ├── docker/
│   ├── aws/
│   │   ├── terraform/
│   │   ├── ecs/
│   │   ├── ec2/
│   │   ├── s3/
│   │   └── ecr/
│   └── scripts/
│
└── tests/
    ├── unit/
    ├── numerical/
    ├── integration/
    ├── protocol/
    ├── simulation/
    ├── regression/
    └── end_to_end/
```

### Dependency rule

```text
core
 ↑
perception / slam / collaboration / reconstruction / planning
 ↑
runtime
 ↑
ROS2 adapters / simulator / AWS services
```

Never let a simulator message type leak into the SLAM core.

Never let an AWS SDK type leak into the algorithm core.

This keeps:

- local deterministic testing easy,
- replay independent of ROS,
- CPU unit tests independent of GPUs,
- AWS deployment interchangeable,
- algorithm benchmarking possible from files alone.

---

# 4. Canonical data model

## 4.1 Frame

```cpp
struct Frame {
    AgentId agent;
    FrameId id;
    Timestamp t;

    cv::Mat left;
    cv::Mat right;

    ImuWindow imu_window;

    PhotometricState photo;
    IlluminationState illumination;

    GroundTruthOptional gt;
};
```

## 4.2 Keyframe

```cpp
struct Keyframe {
    KeyframeId id;
    AgentId agent;
    Timestamp t;

    SE3 T_world_cam;
    Matrix6d pose_cov;

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    StereoDepth depth;
    PointCloud local_cloud;

    GlobalDescriptor global_descriptor;

    std::vector<LandmarkId> landmark_ids;

    double motion_cost;
    double energy_cost;
    double timestamp_age;

    UncertaintySummary uncertainty;
};
```

## 4.3 Loop closure edge

```cpp
struct LoopEdge {
    KeyframeId a;
    KeyframeId b;

    Sim3 or SE3 measurement;
    MatrixNxN information;

    MatchStatistics matches;
    GeometryStatistics geometry;

    EdgeStatus status;

    double retrieval_score;
    double pcm_score;
    double gnc_weight;

    bool degeneracy_flag;
};
```

## 4.4 Communication accounting

Every message crosses exactly one serializer.

```cpp
enum class MessageClass {
    HEARTBEAT,
    GLOBAL_DESCRIPTOR,
    LOCAL_DESCRIPTOR,
    MATCH_REQUEST,
    MATCH_RESPONSE,
    POSE_EDGE,
    POSE_GRAPH,
    SUBMAP_SUMMARY,
    SUBMAP_PATCH,
    TASK_BID,
    TASK_CONSENSUS,
    HUMAN_REQUEST,
    ACK,
};
```

Accounting:

```cpp
struct ByteRecord {
    Timestamp t;
    AgentId sender;
    AgentId receiver;
    MessageClass type;

    uint64_t payload_bytes;
    uint64_t wire_bytes;
    uint64_t cumulative_bytes;
};
```

The byte metric must count bytes at serialization, not "logical payload size."

---

# 5. Simulation architecture

## 5.1 Simulator choice

### Chosen stack

**ROS 2 Jazzy + Gazebo Harmonic.**

This is the clean current pairing for a modern ROS 2 simulation stack; the official ROS 2 documentation states that Jazzy's recommended Gazebo release is Harmonic.

Sources:
- https://docs.ros.org/en/jazzy/Tutorials/Advanced/Simulators/Gazebo/Simulation-Gazebo.html
- https://docs.ros.org/en/iron/Releases/Release-Jazzy-Jalisco.html

### Do not use PX4 for the baseline simulation

Ariadne does not need flight dynamics, motor models, actuator simulation or autopilot fidelity for its research question. PX4 adds integration surface without increasing the quality of the vision/coordination experiment.

Agents are abstract camera-bearing mobile platforms.

Represent motion as:

```text
state = SE(3) pose + velocity + energy + communication state
```

The simulator should command trajectories directly.

A future physical deployment adapter may map the same waypoints to a real robot.

---

## 5.2 Why Gazebo rather than an elaborate robotics stack

Gazebo provides:

- controllable cameras,
- stereo rendering,
- configurable lighting,
- sensor timestamps,
- ROS 2 integration,
- deterministic worlds,
- headless operation,
- straightforward AWS deployment.

The project should use Gazebo primarily as a **sensor and world-state generator**.

---

## 5.3 World generation

Generate parametrically rather than hand-building one showcase world.

World primitives:

- intact rooms,
- collapsed rooms,
- partially collapsed corridors,
- rubble piles,
- beams,
- concrete slabs,
- pipes,
- furniture,
- repeated structural members,
- voids/tunnels,
- dead ends,
- narrow passages.

Each scene stores a hidden ground-truth layer:

```text
mesh
occupancy
signed distance
traversability
clearance
risk
surface normals
material IDs
semantic zones
```

---

## 5.4 Visual degradation model

### Darkness

Use physically motivated co-mounted illumination.

For a point-light approximation:

```text
E(x) ∝ P / r(x)^2
```

with directional/angle effects added only if evidence shows they matter.

NFL-BA explicitly studies co-located near-field lighting and derives an inverse-distance-based formulation for its effects.

Source:
- https://arxiv.org/abs/2412.13176

### Sensor response

Model:

```text
I_sensor = gain * Poisson(I_photon)
           + Gaussian(read_noise)
           + exposure effects
           + lens/vignetting
```

Use a Poisson-Gaussian noise model.

### Lens / flat-field

Store:

```text
K(x) = vignetting + pixel response field
```

and calibrate it offline.

### Exposure

Each simulated camera has:

- exposure time,
- analog gain,
- digital gain,
- clipping threshold.

An adaptive exposure controller becomes part of the mission system, not a rendering cheat.

### Dust

Use a simplified participating-medium model:

```text
I_obs(x) = J(x) * t(x) + A(x) * (1 - t(x))
```

where:

```text
t(x) = exp(-β * depth)
```

augmenting it with a spatially varying local particulate field.

Do not claim full physical atmospheric simulation. The objective is to create controlled, parameterized visual degradations.

---

## 5.5 Communication simulation

Each agent has a communication graph `G(t)`.

For pair `(i,j)`:

```text
bandwidth(t) = B_ij
latency(t)    = L_ij
packet_loss   = P_ij
connected     = C_ij
```

Make communication depend on:

- Euclidean range,
- wall/rubble penetration count,
- structural material,
- randomized shadow zones,
- explicit partitions.

Provide reproducible scenario seeds.

Modes:

```text
ideal
range-limited
attenuated
partitioned
intermittent
store-and-forward
```

---

## 5.6 Agent motion

No physics-accurate locomotion is required.

Represent each agent as:

```text
SE3 kinematic body
camera rig
illumination rig
collision envelope
battery state
communication transceiver
```

Motion commands:

```text
goto(position, orientation)
follow(path)
orbit(target)
rendezvous(peer)
return_to_safe_zone()
hold()
```

Obstacle avoidance is solved over a geometric configuration space rather than actuator simulation.

---

# 6. Per-agent perception

## 6.1 Pipeline

```text
raw stereo frames
   |
timestamp synchronization
   |
radiometric normalization
   |
light-field normalization
   |
quality estimation
   |
XFeat/ALIKED
   |
sparse tracking
   |
stereo correspondence
   |
depth + uncertainty
   |
pose estimation
```

---

## 6.2 Radiometric normalization

### Stage 1 — flat-field

Correct vignetting and fixed spatial gain.

### Stage 2 — exposure normalization

Estimate:

```text
exposure scalar
gain
saturation fraction
shadow fraction
```

and produce a normalized working image.

### Stage 3 — light falloff

Given calibrated lamp pose and scene estimate:

```text
I_corrected(x) =
    f(I(x), r(x), cos(theta), exposure_state)
```

For initial implementation use inverse-square attenuation.

### Stage 4 — avoid aggressive enhancement

The algorithm catalogue correctly identifies a critical empirical risk: enhancement can improve image appearance and even increase matching counts while hurting actual loop closure / reconstruction.

Therefore every restoration stage is tested on geometry, not only visual quality.

Metrics:

- feature repeatability,
- match precision,
- match recall,
- relative pose error,
- track length,
- ATE,
- RPE,
- loop closure precision,
- reconstruction error.

---

## 6.3 Low-light denoising

Start with classical methods using OpenCV.

Primary:

```text
variance-stabilizing transform
 -> denoise
 -> inverse transform
```

Implement generalized Anscombe handling for the Poisson-Gaussian model.

OpenCV operations used around it:

- `cv::GaussianBlur`,
- `cv::fastNlMeansDenoising`,
- `cv::medianBlur`,
- `cv::normalize`,
- `cv::SVD`.

Do not make the transform mandatory if the raw image is already above a noise threshold. The decision is driven by measured noise statistics.

---

# 7. Feature extraction

## 7.1 Feature interface

```cpp
class LocalFeatureEngine {
public:
    virtual FeatureSet detectAndCompute(const cv::Mat& image) = 0;
};
```

Implement:

```text
XFeatEngine
ALIKEDEngine
DISKEngine
SIFT_Engine
ORBEngine
```

All implement the same contract.

---

## 7.2 Primary configuration

### Online frame tracking

- tracked XFeat keypoints,
- pyramidal optical flow,
- forward-backward consistency,
- image-gradient quality,
- spatial uniformity filter.

### Keyframe creation

Run full XFeat extraction.

### Hard inter-agent candidate

Run:

```text
ALIKED
    +
LightGlue
```

and/or XFeat semi-dense refinement.

---

## 7.3 Keypoint selection

Do not keep the top N points globally.

Use spatial bins:

```text
image -> MxN cells

per cell:
    retain top-k by score/reliability
```

This reduces catastrophic concentration in one textured patch.

Also track:

```text
gradient energy
cornerness
descriptor reliability
distance-from-lamp
saturation
dust confidence
```

Feed these into feature weights.

---

# 8. Frame-to-frame tracking

## 8.1 Primary tracker

Use `calcOpticalFlowPyrLK` for inexpensive inter-frame tracking.

OpenCV is responsible for:

- pyramid generation,
- LK flow,
- image operations.

Ariadne owns:

- correspondence filtering,
- robust motion estimation,
- track lifecycle,
- state covariance,
- keyframe insertion.

---

## 8.2 Tracking update

For tracked points:

```text
p_t-1 -> p_t
```

perform:

1. forward-backward check,
2. image-bound check,
3. residual threshold,
4. gradient direction consistency,
5. stereo consistency if available,
6. RANSAC/USAC relative motion,
7. covariance estimate.

If track support collapses:

```text
state = DEGRADED
```

and immediately increase the active-perception value of a recovery observation.

---

## 8.3 Keyframe policy

Insert a keyframe when a weighted combination of:

```text
translation
rotation
time
tracked-feature loss
parallax
new scene coverage
uncertainty
```

exceeds threshold.

A practical score:

```text
S =
    w_t * normalized_translation
  + w_r * normalized_rotation
  + w_p * parallax
  + w_f * feature_loss
  + w_u * uncertainty_growth
```

---

# 9. Stereo geometry

## 9.1 Sensor

Primary simulated camera:

```text
global-shutter stereo pair
baseline: configurable, default 8–12 cm
resolution: 1280x720
frame rate: 20–30 Hz
```

Use a baseline large enough for metre-scale rooms while retaining overlap in narrow voids.

Do not assume a single optimal baseline. Evaluate:

```text
4 cm
8 cm
12 cm
20 cm
```

against range/depth error.

---

## 9.2 Calibration

Use OpenCV:

- `calibrateCamera`,
- `stereoCalibrate`,
- `stereoRectify`,
- `initUndistortRectifyMap`.

Store calibration immutable per sensor rig.

---

## 9.3 Disparity

Primary:

```text
StereoSGBM
```

OpenCV 5 provides stereo correspondence APIs.

Use:

- uniqueness ratio,
- speckle filtering,
- left-right consistency,
- confidence map.

Source:
- https://docs.opencv.org/5.0/main_modules/ptcloud_rgbd.html

---

## 9.4 Depth uncertainty

Approximate stereo depth:

```text
Z = fB / d
```

and propagate disparity uncertainty:

```text
σ_Z ≈ |∂Z/∂d| σ_d
     = fB/d² * σ_d
```

Use this uncertainty later in:

- landmark weights,
- point-cloud fusion,
- active perception,
- traversability confidence.

---

# 10. Local map and SLAM backend

## 10.1 State

Per agent:

```text
poses:
    T_W_Ck ∈ SE(3)

landmarks:
    X_j ∈ R³

extrinsics:
    fixed

camera intrinsics:
    fixed

optional light parameters:
    fixed/calibrated

optional exposure:
    per-keyframe
```

---

## 10.2 Local landmark creation

Triangulate matched observations.

Reject if:

- insufficient parallax,
- negative depth,
- depth outside sensor interval,
- excessive reprojection residual,
- inconsistent stereo depth,
- high uncertainty.

---

## 10.3 Bundle adjustment

Implement from scratch.

Residual:

```text
r_ij = π(T_i X_j) - z_ij
```

State update:

```text
T_i <- exp(δξ_i) T_i
```

Use **one perturbation convention globally**.

Recommended:

```text
left multiplicative SE(3) perturbation
```

Do not mix left/right updates between Jacobian routines.

---

## 10.4 Jacobians

Analytic Jacobians.

Use OpenCV `projectPoints` as a derivative/check oracle when possible, but keep Ariadne's own implementation authoritative.

Validate numerical Jacobians with finite differences:

```text
relative error < configured tolerance
```

Test:

- translation,
- rotation,
- point coordinates,
- near-zero depth,
- wide-FOV rays.

---

## 10.5 Schur complement

Partition normal equations:

```text
H_cc H_cl
H_lc H_ll
```

Eliminate landmarks:

```text
S = H_cc - H_cl H_ll^-1 H_lc
```

Solve camera system, back-substitute landmarks.

This is the core performance mechanism for windowed BA.

---

## 10.6 Solver

Use Levenberg-Marquardt:

```text
(H + λD) δ = -g
```

with trust/radius-like update logic.

Use Eigen sparse Cholesky internally.

Reference-check with:

- Ceres or GTSAM for small graphs,
- not runtime.

---

# 11. Local pose graph

## 11.1 Edge types

```text
ODOMETRY
STEREO_ODOMETRY
LOCAL_LOOP
INTER_AGENT_LOOP
ANCHOR
GRAVITY_PRIOR
```

Only loop-type edges are robustified by default.

---

## 11.2 Residual

For edge `i -> j`:

```text
r_ij = Log( Z_ij^-1 * X_i^-1 * X_j )
```

Use a 6-vector tangent residual.

---

# 12. Rotation initialization

Do **not** throw a cold nonlinear optimizer at the whole pose graph.

Pipeline:

```text
1. rotation synchronization / chordal relaxation
2. translation initialization
3. joint SE(3) refinement
4. GNC loop-edge robustification
```

The rationale follows the catalogue's PGO analysis: rotation synchronization provides a more stable initialization.

---

# 13. Robust loop closure stack

## 13.1 Stage 0 — candidate retrieval

Global descriptor search.

Return top K:

```text
K = 10–50
```

depending on fleet size.

Apply temporal exclusion windows.

---

## 13.2 Stage 1 — 2D correspondence

For candidate:

```text
XFeat BF/ANN
```

or hard candidate:

```text
ALIKED + LightGlue
```

Metrics:

- mutual match ratio,
- spatial spread,
- descriptor distance,
- confidence distribution.

---

## 13.3 Stage 2 — geometric verification

Use OpenCV:

```text
findEssentialMat
recoverPose
findFundamentalMat
solvePnPRansac
```

Use current OpenCV robust estimation infrastructure.

Reject:

- low inlier count,
- low parallax,
- unstable pose,
- physically impossible scale,
- poor cheirality,
- spatially concentrated inliers.

---

## 13.4 Stage 3 — degeneracy detector

Compute geometry conditioning:

```text
Hessian eigenvalues
triangulation angle distribution
inlier spatial covariance
rotation/translation observability
corridor score
```

Define:

```text
degenerate =
    λ_min / λ_max < τ_condition
    OR
    insufficient parallax
    OR
    near-collinear inlier distribution
```

Ariadne does not merely reject a false closure; it can **decline to attempt the closure** when the scene does not provide enough information.

This is specifically motivated by the SubT-derived observation in `ALGORITHMS(1).md` that robust backends cannot rescue an invalid/degenerate hypothesis indefinitely.

---

## 13.5 Stage 4 — PCM

Build a pairwise consistency matrix for candidate loop closures.

For closures `e_i`, `e_j`:

```text
compose(
    e_i,
    odometry path,
    inverse(e_j)
)
```

and test against a covariance-aware threshold.

Do not rely on exact maximum clique for large candidate sets.

Use:

```text
greedy maximal consistent subset
```

or branch-and-bound only for very small sets.

---

## 13.6 Stage 5 — GNC

Run GNC over surviving candidate loop edges.

Track and expose:

```text
initial residual
final residual
weight trajectory
edge acceptance
edge rejection
```

This creates an excellent judge-visible trace.

---

## 13.7 Optional Stage 6 — DCS / switchable constraints

Implement DCS as a lightweight additional robust-loss experiment.

Do not run every robust mechanism simultaneously by default.

Benchmark:

```text
Huber
Cauchy
DCS
Switchable Constraints
GNC
GNC + PCM
GNC + degeneracy gate
```

---

# 14. Distributed collaboration protocol

## 14.1 Principle

Never continuously stream maps.

Transmit:

```text
small global descriptors
      ↓
candidate request
      ↓
local descriptors / 2D features
      ↓
verified pose constraint
      ↓
small pose graph
```

This is the key communication reduction.

---

## 14.2 Message classes

### Heartbeat

```text
agent_id
timestamp
pose summary
battery
link state
keyframe watermark
```

### Global descriptor

```text
keyframe_id
timestamp
descriptor
thumbnail hash
quality
```

### Candidate request

```text
request_id
query_keyframe
candidate_keyframe
requested_feature_type
```

### Feature response

```text
keypoints
descriptors
camera metadata
optional 3D landmarks
```

### Pose edge

```text
keyframe_i
keyframe_j
SE3/Sim3
information
verification stats
```

### Pose graph batch

Compressed graph:

```text
nodes
edges
covariances
frame IDs
reference anchor
```

---

## 14.3 Communication bookkeeping

Maintain per-peer:

```text
sent_global_descriptor_set
sent_local_feature_set
sent_pose_edge_set
sent_submap_set
```

Every message is idempotent.

Duplicate messages are ignored.

Out-of-order messages are stored until prerequisites arrive.

---

# 15. Spectral communication prioritization

This is the **flagship collaboration algorithm**.

## 15.1 Candidate graph

Construct a hypothetical graph using:

```text
existing odometry graph
+
already fixed inter-agent edges
+
candidate edges weighted by retrieval similarity
```

For each budget-limited candidate set:

```text
maximize λ₂(L)
subject to number_of_selected_edges <= B
```

where `L` is the weighted graph Laplacian.

---

## 15.2 Algorithm

Start with a connectivity-enforcing greedy set.

Then iteratively:

```text
compute Fiedler vector v2

for each candidate e:
    gain_e = v2^T L_e v2

select highest gain / communication_cost

update graph
recompute v2
```

For small candidate sets, compare with exact combinatorial enumeration.

For larger sets, use greedy/local refinement.

---

## 15.3 Communication budget

Budget can be:

```text
bytes
messages
energy
time
```

Primary benchmark:

```text
bytes
```

because this is the cleanest headline resource.

Secondary:

```text
bytes + latency
```

---

# 16. Place-recognition stack

## 16.1 Global retrieval

Primary:

```text
EigenPlaces R50, 256-D
```

via ONNX/OpenCV DNN.

Index:

```text
cv::ANNIndex
```

Sources:
- https://arxiv.org/abs/2308.10832
- https://docs.opencv.org/5.0/main_modules/features_annoy.html

---

## 16.2 Compression experiment

Compare:

```text
fp32 256-D
fp16 256-D
int8 256-D
PCA->128 fp16
PCA->256 int8
```

Plot:

```text
recall@1 / recall@5
vs
bytes/keyframe
```

This is an Ariadne-specific contribution because representation quality and communication cost are jointly optimized.

---

## 16.3 Aliasability diagnostics

Measure descriptor distributions for:

```text
same-place pairs
different-place pairs
```

Estimate overlap.

Report:

```text
ROC
precision-recall
false-closure probability
```

Do not hide the building's perceptual ambiguity behind a single aggregate score.

---

# 17. Dense reconstruction / 3D Gaussian representation

## 17.1 Canonical data flow

```text
validated keyframe poses
        +
stereo depth
        +
RGB
        |
        v
local point cloud
        |
        v
Gaussian initialization
        |
        v
submap optimization
        |
        v
submap-level merge
        |
        v
global Gaussian scene
```

---

## 17.2 Use `gsplat` as the implementation base

The algorithm catalogue found `gsplat` to be a clean Apache-2.0 implementation path, while many 3DGS repos pull restrictive Inria rasterizers transitively.

Source:
- https://github.com/nerfstudio-project/gsplat

For Ariadne:

- geometry/state ownership remains Ariadne,
- Gaussian kernels/rasterization can be provided by a dedicated reconstruction package,
- all camera poses supplied by the validated geometric backend.

---

## 17.3 Gaussian state

```cpp
struct Gaussian {
    Vec3 mean;
    Mat3 covariance;
    Vec3 color;
    float opacity;

    float stability;
    float uncertainty;
    KeyframeId source;
    SubmapId submap;
};
```

---

## 17.4 Stable / unstable Gaussian lifecycle

Inspired by RTG-SLAM:

```text
NEW
  ↓
UNSTABLE
  ↓
SUPPORTED
  ↓
STABLE
```

Delete or merge Gaussians if:

- unsupported,
- repeatedly inconsistent,
- extremely transparent,
- redundant with another Gaussian,
- inconsistent under new geometry.

Source:
- https://arxiv.org/abs/2404.19706

---

## 17.5 Occupancy-aware fusion

For overlapping submaps:

1. transform into global coordinates,
2. voxelize occupancy,
3. group overlapping Gaussians,
4. compare geometric support,
5. compare photometric support,
6. merge if compatible,
7. retain competing hypotheses if uncertainty remains high.

This follows the broad direction demonstrated by MAGS-SLAM for geometry/appearance-aware collaborative Gaussian fusion.

Source:
- https://arxiv.org/abs/2605.10760

Do not allow photometric appearance to overwrite geometrically inconsistent structure.

---

# 18. Photometric reconstruction under co-moving illumination

## 18.1 Illumination model

For a Gaussian/surface sample at `x` and light position `l`:

```text
r = ||x - l||

E ∝ 1/r²
```

optionally:

```text
E ∝ cos(theta) / r²
```

Use the measured light calibration to determine whether the cosine term materially improves fit.

---

## 18.2 NFL-aware loss

Implement a practical loss:

```text
L_total =
    λ_photo L_photo
  + λ_depth L_depth
  + λ_normal L_normal
  + λ_light L_light
  + λ_reg L_reg
```

where `L_light` explains intensity variation induced by the changing light-surface distance.

Source:
- https://arxiv.org/abs/2412.13176

---

## 18.3 Specular robustness

GLOW specifically identifies dynamic specular highlights as a problem for co-located flashlight captures and introduces a surface-angle-weighted radiometric loss.

Source:
- https://arxiv.org/abs/2511.22857

Implement the underlying idea:

```text
weight_specular =
    f(view_direction, light_direction, surface_normal)
```

downweight pixels likely to be highlight-dominated.

Do not attempt a full neural inverse renderer for the baseline.

---

# 19. Active perception / next-best-view

This is the major addition that was still missing as a completed domain in `ALGORITHMS(1).md`.

## 19.1 Design principle

A candidate action is valuable when it improves:

- geometric uncertainty,
- coverage,
- map connectivity,
- place-recognition ambiguity,
- reconstruction quality,
- egress validity,

minus:

- travel cost,
- expected energy cost,
- communication cost,
- mission time,
- structural risk.

---

## 19.2 Candidate generation

Generate candidate viewpoints from:

### A. Frontier surfaces

From occupancy/observability volume.

### B. Uncertainty hotspots

Regions with:

```text
high depth variance
high pose influence
low observation count
high route relevance
```

### C. Connectivity targets

Candidate views that can bridge two disconnected map regions.

### D. Alias resolution targets

Locations where visually similar regions need a discriminative observation.

### E. Egress uncertainty

Views that can validate a route's:

- clearance,
- obstruction,
- structural confidence.

---

## 19.3 Information gain approximation

Do not calculate an expensive full posterior over the entire scene.

Approximate:

```text
IG(action) =
    Δcoverage
  + α Δentropy
  + β Δpose_connectivity
  + γ Δegress_confidence
  + δ Δloop_disambiguation
```

Normalize each term.

---

## 19.4 Risk-aware utility

Use:

```text
U(a) =
    G(a)
  - λd C_motion
  - λb C_bandwidth
  - λe C_energy
  - λt C_time
  - λr C_risk
```

This preserves the original research hypothesis in the proposal.

---

## 19.5 TARE-inspired multi-resolution exploration

TARE uses dense local planning with sparse global representation.

Source:
- https://www.cmu-exploration.com/tare-planner
- https://github.com/caochao39/tare_planner

Ariadne adaptation:

```text
global layer:
    coarse occupancy / topology graph
    low-resolution uncertainty

local layer:
    dense voxel map
    detailed candidate viewpoints
    collision / clearance evaluation
```

This is a strong fit because Ariadne's world can be large while local camera planning needs precision.

---

## 19.6 FAR-style route validation

FAR Planner uses a dynamically updated visibility graph and rapid replanning.

Source:
- https://arxiv.org/abs/2110.09460
- https://github.com/MichaelFYang/far_planner

Adaptation:

```text
surface/clearance observations
 -> free-space boundary extraction
 -> dynamic visibility graph
 -> fast route query
```

Use this for operational route planning, not as the only exploration policy.

---

## 19.7 Multi-agent assignment

For a small team:

1. each agent computes local candidate utility,
2. tasks are spatially clustered,
3. agents bid on tasks,
4. consensus resolves duplicate ownership,
5. assignments update on new information.

CBBA is a strong baseline because it has a mature decentralized auction/consensus formulation.

Source:
- https://pure.kaist.ac.kr/en/publications/consensus-based-decentralized-auctions-for-robust-task-allocation/

For Ariadne's communication-constrained scenario, the architecture should be event-triggered rather than consensus-synchronized every frame.

A current 2026 line of work also motivates group-level bidding when tasks are spatially correlated; see the recent grouping auction-consensus work:
- https://arxiv.org/abs/2608.15884

Treat that as an experimental extension, not a hard dependency.

---

## 19.8 Best practical planner

The production planner should be a hybrid:

```text
candidate generation:
    uncertainty/frontier/connectivity

candidate scoring:
    EIG approximation + mission costs

task grouping:
    spatial clustering

assignment:
    auction/consensus

within-task path:
    FAR-style dynamic visibility graph / A* fallback

global exploration scheduling:
    TARE-style coarse-to-fine representation
```

This is more defensible than a pure learned policy because every decision is inspectable.

---

# 20. Route / egress subsystem

## 20.1 Representation

Construct a spatial cost field:

```text
C(x) =
    w_clearance C_clearance
  + w_uncertainty C_uncertainty
  + w_risk C_structural
  + w_traversability C_traversability
```

---

## 20.2 Route confidence

For a route `R`:

```text
confidence(R) =
    aggregation of local support/confidence along R
```

Use a conservative aggregation such as a low quantile or harmonic-type combination rather than a simple mean.

Reason: one catastrophic low-confidence segment should matter more than twenty excellent ones.

---

## 20.3 Dynamic replanning

Replan whenever:

```text
new obstacle discovered
route confidence falls
structural risk estimate changes
agent becomes unavailable
communication changes
```

D* Lite is an academically well-founded incremental replanning baseline.

Source:
- https://aaai.org/papers/00476-aaai02-072-d-lite/

Operationally, compare:

```text
D* Lite
FAR-style visibility graph
```

and choose the faster stable route query for the final system.

---

# 21. Human authority / risk gate

Autonomy must remain bounded.

A proposed action has:

```text
risk_score
expected_information_gain
distance
energy
confidence
reason
```

If:

```text
risk_score > autonomous_threshold
```

generate:

```text
HUMAN_APPROVAL_REQUEST
```

with:

- map snapshot,
- agent position,
- intended action,
- evidence,
- uncertainty,
- predicted benefit,
- risk reason,
- alternative action.

If communication to command is absent:

```text
approved envelope -> continue
outside envelope  -> hold/defer/backtrack
```

---

# 22. Failure supervisor

## 22.1 State machine

Each component has:

```text
HEALTHY
DEGRADED
FAILED
RECOVERING
```

Monitor:

- tracker inlier count,
- BA condition number,
- pose covariance,
- CPU time,
- GPU memory,
- message queues,
- heartbeat freshness,
- battery,
- Gaussian count,
- map growth rate.

---

## 22.2 Last-safe-state checkpointing

Persist:

```text
last valid keyframe
last valid local pose graph
reference transform
task state
communication watermarks
```

On failure:

```text
stop component
restore checkpoint
discard in-flight state
replay deterministic inputs
restart
```

This follows the failure-recovery lessons captured from SubT deployments in the algorithm catalogue.

---

# 23. Agentic orchestration layer

## 23.1 Keep LLM outside the numerical core

LLM/agent reasoning should never own:

- camera pose estimation,
- Jacobian computation,
- PGO numerical solve,
- raw image correspondence,
- safety thresholds without a deterministic rule layer.

The agentic layer owns:

- task selection,
- mission-level negotiation,
- explanation,
- exception handling,
- invocation of tools,
- human escalation.

---

## 23.2 MCP tool surface

Expose:

```text
observe_image
estimate_pose
compute_depth
query_uncertainty
retrieve_place_candidates
verify_loop
request_peer_observation
allocate_task
request_rendezvous
query_route
request_human_approval
```

The runtime trace must show:

```text
vision output
    ->
agent decision
    ->
tool call/action
```

This directly satisfies the competition's agentic behavior requirement.

Competition rules:
- OpenCV 5 and AWS are hard requirements.
- Agentic Vision requires vision output to influence a subsequent plan/tool/action/approval.
- The project must provide a trace and evaluation of failure handling/human control.

Source:
- competition file / challenge information captured in `opencv-ai-competition-2026(1).md`
- official challenge: https://opencv26.devpost.com/

---

## 23.3 Classical planner ablation

Run exactly the same environment with:

```text
A. deterministic Ariadne planner
B. LLM/MCP orchestrated planner
```

If B wins:

```text
quantify why
```

If B does not:

```text
say so
```

The competition rewards credible engineering, not decorative agent terminology.

---

# 24. AWS architecture

## 24.1 Split CPU and GPU workloads

The architecture should explicitly separate:

### Graviton / COOL

Run:

- per-frame OpenCV processing,
- radiometric processing,
- filtering,
- feature preprocessing,
- XFeat/ALIKED where supported,
- geometric verification,
- evaluation benchmarks.

### x86 GPU

Run:

- Gazebo rendering when GPU acceleration is useful,
- Gaussian optimization/rendering,
- expensive feed-forward model experiments,
- batch evaluation.

This is the correct use of the competition's heterogeneous cloud environment.

---

## 24.2 Primary AWS topology

```text
                           +----------------------+
                           |        S3            |
                           | datasets / artifacts |
                           +----------+-----------+
                                      |
                +---------------------+----------------------+
                |                                            |
                v                                            v
     +----------------------+                       +----------------------+
     | Graviton4 EC2        |                       | GPU EC2              |
     | m8g.4xlarge          |                       | g5/g6 family         |
     | COOL + OpenCV 5      |                       | Gazebo + 3DGS        |
     +----------+-----------+                       +----------+-----------+
                |                                               |
                +-------------------+---------------------------+
                                    |
                                    v
                         +----------------------+
                         | Agent orchestrator   |
                         | ECS/EC2 service      |
                         +----------+-----------+
                                    |
                    +---------------+----------------+
                    |                                |
                    v                                v
              Operator API                    Evaluation jobs
                    |                                |
                    v                                v
             Web operator UI                   S3 metrics
```

---

## 24.3 COOL node

AWS Marketplace currently documents COOL on AWS Graviton4 and recommends m8g.4xlarge or larger for performance.

Source:
- https://aws.amazon.com/marketplace/pp/prodview-fdvbfiewzuehs
- https://aws.amazon.com/about-aws/whats-new/2024/09/amazon-ec2-c8g-m8g-instances/
- https://docs.aws.amazon.com/ec2/latest/instancetypes/gp.html

Implementation:

```text
Amazon Linux/Ubuntu ARM64
+
COOL AMI
+
Ariadne ARM64 container
```

The final benchmark must compare:

```text
stock OpenCV 5 on same Graviton instance
vs
COOL OpenCV on same Graviton instance
```

Do not compare a Graviton CPU against a desktop GPU and call it a COOL speedup.

---

## 24.4 GPU node

Use:

```text
g5.xlarge
```

or a later AWS GPU family available at submission time.

Workloads:

- Gazebo rendering,
- Gaussian mapping,
- larger ONNX models,
- training/fine-tuning.

Make the GPU node disposable.

---

# 25. Containerization

## 25.1 Images

Build at least four images:

```text
ariadne-core-amd64
ariadne-core-arm64-cool
ariadne-sim-amd64
ariadne-gpu-amd64
```

---

## 25.2 Base design

### CPU image

```dockerfile
FROM ubuntu:24.04

C++17
CMake
Eigen
OpenCV 5
ONNX models
Ariadne core
```

OpenCV 5 requires C++17 according to the current OpenCV 5 documentation.

Source:
- https://github.com/opencv/opencv/wiki/OpenCV-5

### ARM image

Same source tree, linked against `/opt/cool` OpenCV environment.

### GPU image

NVIDIA runtime with:

```text
CUDA
PyTorch
gsplat
OpenCV 5
Ariadne reconstruction
```

---

# 26. AWS storage and orchestration

## 26.1 S3 layout

```text
s3://ariadne/
    datasets/
    scenes/
    checkpoints/
    runs/
        YYYY/MM/DD/run_id/
            config.yaml
            manifest.json
            metrics.parquet
            trajectories/
            maps/
            logs/
            video/
```

---

## 26.2 Run manifest

Every run records:

```yaml
git_commit:
opencv_version:
cool_version:
cuda_version:
scene:
seed:
agent_count:
camera_config:
illumination_config:
communication_config:
planner_config:
feature_engine:
place_recognition:
loop_backend:
reconstruction_backend:
```

This is critical for reproducibility.

---

## 26.3 Event transport

For the hackathon prototype, avoid an elaborate distributed cloud bus.

Use:

```text
ROS2 / local runtime
+
gRPC or WebSocket for service boundaries
+
S3 for artifacts
```

Optionally use:

```text
SQS
```

only for asynchronous cloud job orchestration.

Do not put every camera frame through SQS.

---

# 27. Operator console

## 27.1 Main view

```text
3D reconstruction
+
agent trajectories
+
current tasks
+
communication links
+
uncertainty field
+
egress route
```

---

## 27.2 Incident timeline

Show:

```text
22:41 Agent 2 enters low-confidence corridor
22:42 retrieval finds 12 candidates
22:42 spectral selector selects 2
22:42 PCM rejects 1
22:42 GNC rejects 1 false closure
22:43 Agent 2 requests peer observation
22:44 peer rendezvous succeeds
22:44 route confidence rises
```

This is much more compelling than simply displaying a 3D viewer.

---

# 28. Evaluation harness

## 28.1 Ground-truth evaluation

Every simulation provides:

```text
GT pose
GT mesh
GT depth
GT traversability
GT route
GT communication graph
```

---

## 28.2 Core metrics

### Localization

```text
ATE RMSE
RPE
rotation error
translation error
```

### Registration

```text
inter-agent relative pose error
accepted/rejected closure precision
false closure rate
```

### Mapping

```text
Chamfer distance
completeness
coverage
held-out-view PSNR
SSIM
LPIPS
```

### Communication

```text
total bytes
bytes/keyframe
bytes/verified loop
bytes/map-area-reconstructed
descriptor bytes
local feature bytes
pose graph bytes
```

### Mission

```text
time to first usable map
time to first valid egress route
route validity
route confidence
distance travelled
energy proxy
```

---

# 29. Ablation matrix

Run a formal matrix.

## 29.1 Collaboration

```text
solo
independent multi-agent
greedy similarity
greedy coverage
Ariadne spectral
Ariadne spectral + vertex-cover exchange
```

## 29.2 Robustness

```text
no robustification
Huber
DCS
Switchable Constraints
GNC
PCM + GNC
degeneracy gate + PCM + GNC
```

## 29.3 Features

```text
ORB
SIFT
XFeat
ALIKED
XFeat + LightGlue
ALIKED + LightGlue
```

## 29.4 Place recognition

```text
DBoW-style
EigenPlaces-128
EigenPlaces-256
MegaLoc reference
AnyLoc reference
```

## 29.5 Illumination

```text
ambient normal
dark
dark + fixed light
dark + co-moving light
dark + co-moving light + radiometric correction
dark + co-moving light + NFL-aware reconstruction
```

## 29.6 Planner

```text
random
greedy frontier
coverage utility
TARE-inspired
risk-aware information utility
Ariadne multi-agent planner
```

---

# 30. SOTA/reference comparison table

| Function | Primary Ariadne | Reference/comparator | Why |
|---|---|---|---|
| Local features | XFeat | ALIKED, DISK, ORB | XFeat gives compact efficient features |
| Hard matching | ALIKED + LightGlue | BF | contextual adaptive matching |
| Frame tracking | LK + geometric filtering | SVO-style | cheap repeated tracking |
| Stereo | StereoSGBM | learned stereo | CPU/OpenCV-native metric geometry |
| Local BA | custom Schur LM | Ceres/GTSAM oracle | technical depth |
| PGO | custom GNC | GTSAM | robust multi-agent backend |
| Retrieval | EigenPlaces 256-D | MegaLoc/AnyLoc | balance accuracy/bandwidth |
| C-SLAM | spectral + sparse exchange | Swarm-SLAM | direct communication objective |
| Loop robustness | degeneracy + PCM + GNC | PCM only | avoids degenerate hypotheses |
| Reconstruction | 3DGS / gsplat | Photo-SLAM / RTG-SLAM ideas | decoupled robust geometry + photorealism |
| Multi-agent 3DGS | local Gaussian submaps | MAGS-SLAM | current monocular/multi-agent frontier |
| Exploration | TARE-inspired | TARE | multiresolution |
| Route | FAR-style | D* Lite | fast dynamic replanning |
| Allocation | event-triggered auction/consensus | CBBA | decentralized conflict resolution |
| Illumination | calibrated co-moving-light model | NFL-BA / GLOW | physically appropriate |
| Low light | classical photometric normalization | learned enhancement | geometry-first |
| Certification | SE-Sync-style certificate | no certificate | independent optimizer validation |

---

# 31. Scoped development program

The development process is intentionally **cross-cutting and synchronized**. Do not build one giant subsystem for weeks and only then discover that its assumptions do not interface with the others.

Each iteration adds a vertical slice through the whole system.

---

## Iteration 0 — repository and deterministic infrastructure

### Implement

- monorepo
- CMake
- Python analysis env
- config schema
- serialization layer
- timestamp types
- SE(3) math
- logging
- run manifests
- deterministic seeds
- test harness
- CI

### Tests

- SE(3) composition/inverse
- quaternion normalization
- Lie exponential/log maps
- covariance propagation sanity
- serialization round trip
- deterministic replay

### Exit condition

A synthetic trajectory can be generated, serialized, replayed and scored exactly.

---

# Iteration 1 — end-to-end "thin Ariadne"

Goal: the whole system runs before sophisticated algorithms exist.

### Simulator

- Gazebo world
- 2 agents
- stereo camera
- IMU
- direct kinematic motion

### Perception

- grayscale
- ORB/SIFT baseline
- LK tracker

### SLAM

- pose chain
- simple triangulation
- basic BA

### Collaboration

- exchange poses
- exchange descriptors
- one valid inter-agent loop closure

### Reconstruction

- point cloud
- simple Gaussian initialization
- offline splat render

### Planning

- frontier candidates
- nearest frontier action
- basic A* route

### Evaluation

- trajectory
- map
- bytes
- route

### Exit condition

Two agents independently explore a simple world, produce local maps, merge them, render a basic reconstruction and compute an exit route.

---

# Iteration 2 — proper OpenCV 5 perception

Replace placeholders with:

- XFeat
- stereo calibration
- StereoSGBM
- OpenCV 5 geometry
- ANNIndex
- LightGlue path
- proper quality masks

### Cross-system test

Run the same scene through:

```text
ORB pipeline
XFeat pipeline
```

and verify:

- runtime
- feature count
- relative pose
- bytes
- map quality.

---

# Iteration 3 — real local SLAM

Implement:

- keyframe policy
- landmark lifecycle
- Schur BA
- pose covariance
- local graph
- loop closure

### Oracle testing

For randomly generated pose graphs:

```text
Ariadne solver
vs
GTSAM/Ceres
```

Require agreement within numerical tolerances.

Do not proceed to multi-agent sophistication until this is stable.

---

# Iteration 4 — robust single-agent SLAM

Add:

- GNC
- DCS
- switchable constraints
- corridor degeneracy
- gravity prior
- bad measurement injection

### Regression suite

Inject:

```text
10%
25%
50%
75%
90%
```

false constraints into synthetic pose graphs.

Plot:

```text
success probability
ATE
runtime
```

This tests robustness before communication complicates the problem.

---

# Iteration 5 — multi-agent sparse collaboration

Add:

- global descriptor broadcast
- candidate retrieval
- local descriptor requests
- bookkeeping
- heartbeats
- store-and-forward
- inter-agent loop closures
- anchor propagation

### Network fault injection

```text
packet loss
latency
partitions
reordering
agent disconnection
```

Every run must terminate with:

```text
consistent or explicitly incomplete
```

never silent corruption.

---

# Iteration 6 — spectral communication optimization

Add:

- candidate Laplacian
- λ₂
- Fiedler vector
- budgeted selection
- exact small-case oracle
- greedy baseline

### Core plot

```text
ATE
 ^
 |       Ariadne
 |      /
 |     /
 |----/------ greedy
 |
 +--------------------> cumulative bytes
```

This is one of the most important figures in the entire project.

---

# Iteration 7 — perceptual aliasing laboratory

Construct adversarial scenes:

- repeated walls,
- repeated beams,
- repeated windows,
- feature-poor concrete,
- identical-looking intersections.

Add:

- descriptor-space aliasing diagnostics,
- degeneracy detector,
- PCM,
- GNC.

Measure:

```text
candidate count
false closure rate
accepted false closure rate
map deformation
recovery latency
```

---

# Iteration 8 — active perception

Add:

- uncertainty volume
- frontier generation
- candidate view scoring
- travel cost
- information gain
- communication value
- agent bidding
- conflict resolution

Compare:

```text
random
coverage greedy
uncertainty-only
Ariadne
```

Core plot:

```text
reconstruction quality
vs
mission budget
```

---

# Iteration 9 — realistic darkness + co-moving illumination

Add simulator:

- camera-mounted point light
- inverse-square falloff
- saturation
- low-light shot noise
- exposure variation
- cast shadows

Add perception:

- flat-field correction
- exposure state
- variance stabilization
- denoise
- light normalization

Add reconstruction:

- NFL-aware photometric term.

Datasets to use for development/validation should prioritize OIVIO and other co-moving-light sources identified in the algorithm catalogue.

OIVIO:
- https://arpg.colorado.edu/oivio/

The OIVIO dataset contains dark-environment stereo + IMU sequences with calibrated onboard illumination and photometric/light calibration.

---

# Iteration 10 — dust and confined spaces

Add:

- particulate attenuation
- backscatter
- lens contamination
- narrow-void stereo
- corridor degeneracy.

Validate separately:

```text
clear
low dust
medium dust
high dust
```

and:

```text
wide space
corridor
void
```

---

# Iteration 11 — dense reconstruction becomes first-class

Implement:

- Gaussian initialization
- Gaussian optimization
- stable/unstable lifecycle
- submaps
- overlap fusion
- uncertainty
- held-out render evaluation.

Use a geometry-first rule:

```text
no validated pose -> no authoritative Gaussian
```

---

# Iteration 12 — risk-aware egress

Build:

- traversability field
- clearance field
- uncertainty field
- risk map
- route cost
- dynamic rerouting
- confidence.

Add:

```text
obstacle discovered after route generation
```

and verify that the thread reroutes.

---

# Iteration 13 — agentic orchestration

Expose MCP tools.

Implement an orchestrator that receives:

```text
coverage
uncertainty
agent status
communication
candidate actions
```

and selects:

```text
move
observe
rendezvous
request peer data
return
human approval
```

Every decision is logged.

---

# Iteration 14 — COOL path

Port the per-frame image pipeline to the COOL Graviton node.

Benchmark:

```text
stock OpenCV 5 on Graviton
vs
COOL OpenCV on same Graviton
```

Measure:

- latency/frame,
- FPS,
- CPU utilization,
- cost/hour,
- cost/frame,
- multi-agent throughput.

The official COOL marketplace documentation recommends m8g.4xlarge or larger and documents custom C++ applications using its optimized OpenCV installation.

Source:
- https://aws.amazon.com/marketplace/pp/prodview-fdvbfiewzuehs

---

# Iteration 15 — full system synchronization

Run:

```text
3–5 agents
dark
dust
communication attenuation
false loop closures
agent failure
active perception
3DGS
egress
```

Collect complete run artifact:

```text
config
logs
trace
metrics
video
map
trajectory
communication accounting
failure events
```

---

# Iteration 16 — adversarial / failure campaign

Mandatory scenarios:

### Failure A

Agent disappears.

Expected:

```text
tasks reassigned
map remains valid
route remains available
```

### Failure B

False loop closure.

Expected:

```text
candidate rejected or GNC downweighted
```

### Failure C

Comms partition.

Expected:

```text
local autonomy continues
data queued
reconnection merges state
```

### Failure D

Tracking collapse.

Expected:

```text
confidence falls
agent chooses recovery action
```

### Failure E

Overexposure.

Expected:

```text
quality detector catches it
exposure/trajectory action changes
```

### Failure F

Dust spike.

Expected:

```text
visual confidence drops
affected observations downweighted
planner changes action
```

### Failure G

Unsafe route.

Expected:

```text
confidence falls
route withdrawn
alternate route or human approval
```

---

# 32. Dataset program

## 32.1 Primary geometry / SLAM validation

Use several independent datasets from the catalogue:

- TUM RGB-D
- EuRoC
- SubT / subterranean datasets where available
- ScanNet / Matterport-type indoor sets for reconstruction

Purpose:

```text
validate estimator independently from simulator
```

---

## 32.2 Co-moving-light validation

### OIVIO

Particularly valuable because it contains:

- stereo,
- IMU,
- onboard illumination,
- photometric calibration,
- light calibration,
- dark environments.

Source:
- https://arpg.colorado.edu/oivio/

### VAROS

Use for synthetic moving illumination / scattering with dense ground truth where applicable.

### C3VDv2

Useful for near-field illumination physics and dense pose/depth supervision despite being an endoscopy domain.

The catalogue identifies its v2 form as substantially larger and notes detailed per-frame geometry.

Use it for **physics/algorithm validation**, not application-domain claims.

---

# 33. Benchmark protocol

Every algorithm comparison must use the same:

```text
scene
trajectory seed
agent start poses
image stream
communication trace
compute budget
mission time
```

Do not change data across baselines.

---

## 33.1 Main benchmark

For each strategy:

```text
independent
coverage greedy
similarity greedy
spectral Ariadne
```

sweep:

```text
B = [small, medium, large communication budget]
```

and report:

```text
quality(B)
ATE(B)
route_validity(B)
time_to_route(B)
```

---

# 34. Reproducibility requirements

A benchmark directory must be sufficient to reproduce a figure.

Example:

```text
evaluation/runs/fig_07/
    config.yaml
    seed.txt
    manifest.json
    command.sh
    raw_metrics.parquet
    processed_metrics.parquet
    plot.py
    README.md
```

Never manually edit a final plot.

---

# 35. Numerical testing requirements

## 35.1 Geometry

Test:

- SO(3) exp/log
- SE(3) exp/log
- adjoint
- covariance transform
- Jacobians
- triangulation
- PnP
- reprojection.

## 35.2 Optimization

Test with synthetic ground truth.

Require:

```text
residual decreases
gradient norm decreases
solution stable under small perturbations
```

## 35.3 Robust estimation

Randomly sample outlier rates.

Report:

```text
breakpoint
```

rather than only best-case performance.

---

# 36. Performance engineering

## 36.1 CPU hot path

Optimize:

- image conversion,
- resize/pyramid,
- LK flow,
- feature tensors,
- descriptor serialization.

This is the primary COOL workload.

---

## 36.2 GPU hot path

Keep on GPU:

- Gaussian rasterization,
- batch mapping,
- large learned model inference,
- optional rendering.

Do not repeatedly move tensors CPU↔GPU.

---

## 36.3 Thread model

Per agent:

```text
sensor thread
tracking thread
mapping thread
loop-closure thread
planner thread
communication thread
```

Global:

```text
broker optimizer
reconstruction worker
operator telemetry
```

Use bounded queues.

Backpressure is explicit.

A slow Gaussian renderer must never stall camera tracking.

---

# 37. Recommended frequencies

Starting point:

```text
sensor:                20–30 Hz
tracking:              20–30 Hz
keyframe decisions:    5–10 Hz
global retrieval:      1–5 Hz / keyframe
loop verification:     event-driven
PGO:                   event-driven
active planning:       1–2 Hz
task consensus:        event-driven
egress replanning:     event-driven + 1 Hz heartbeat
operator telemetry:    2–10 Hz
Gaussian update:       asynchronous
```

Tune from profiling rather than hardcoding the final values.

---

# 38. Resource budgets

## Per agent conceptual budget

```text
image memory      bounded ring buffer
keyframes         bounded local window + archived compact metadata
landmarks         capped / culled
global descriptors compact
local descriptors on demand
```

## GPU

Target:

```text
< 8 GB/agent for normal reconstruction path
```

when practical.

Avoid architecture that requires tens of GB per agent.

---

# 39. Logging schema

Events:

```text
FRAME
KEYFRAME
TRACK_LOSS
LOOP_CANDIDATE
LOOP_REJECT
LOOP_ACCEPT
GNC_UPDATE
PGO
MESSAGE_SENT
MESSAGE_DROPPED
AGENT_DISCONNECT
AGENT_RECONNECT
TASK_CREATED
TASK_BID
TASK_ASSIGNED
ROUTE_CREATED
ROUTE_INVALIDATED
HUMAN_REQUEST
FAILURE
RECOVERY
```

Every event:

```text
timestamp
agent
run_id
sequence_id
state_hash
```

---

# 40. What should be considered "done"

## Core technical demo

Ariadne is demo-ready when all of the following happen in one deterministic run:

1. 3+ agents enter a dark collapsed-style world.
2. Agents use their own illumination.
3. Each agent builds a local metric visual map.
4. Agents discover cross-agent candidate relationships.
5. Communication is explicitly budgeted.
6. Candidate exchanges are selected by the communication policy.
7. At least one false closure is generated intentionally.
8. PCM/GNC prevents it from corrupting the global map.
9. Agents dynamically choose new observations.
10. A global reconstruction emerges.
11. A traversability/uncertainty-aware route is produced.
12. The route changes after new evidence.
13. At least one agent experiences communication loss.
14. It continues locally and reconciles after reconnection.
15. Operator UI displays the state.
16. All bytes and timing are measured.
17. A complete trace can be replayed offline.

---

# 41. Winning demonstration sequence

The final five-minute demo should be built from a **single coherent mission**.

### 0:00–0:25 — Problem

Show collapsed structure.

Text:

```text
GPS unavailable
radio degraded
light unavailable
humans cannot safely enter
```

### 0:25–1:05 — Fleet enters

Show:

- three agents,
- beams from co-mounted lights,
- distinct exploration branches.

### 1:05–1:45 — Collaboration

Visualize:

```text
descriptors
candidate edges
communication budget
spectral selection
```

### 1:45–2:20 — Failure

Intentionally insert false visual similarity.

Show:

```text
candidate
 -> geometry
 -> consistency
 -> GNC weight -> 0
```

Then display:

```text
map remains consistent
```

### 2:20–3:05 — Agentic active perception

Show:

```text
uncertainty hotspot
 -> agent reasoning
 -> task reassignment
 -> rendezvous
 -> new observation
 -> uncertainty reduction
```

### 3:05–3:45 — Reconstruction

Reveal:

```text
multi-agent Gaussian scene
```

with independent ground-truth comparison.

### 3:45–4:20 — Thread

Show risk-weighted path through reconstructed space.

Invalidate one segment.

Show automatic reroute.

### 4:20–4:45 — Quantitative evidence

Three plots:

```text
ATE vs bytes
coverage vs mission budget
route validity vs mission time
```

### 4:45–5:00 — COOL / AWS

One clean screen:

```text
same workload
same Graviton4
stock OpenCV 5 : X ms/frame
COOL             : Y ms/frame
cost/frame       : ...
```

---

# 42. Research claims Ariadne can responsibly make

Strong claims:

> Ariadne evaluates whether communication-aware active perception can improve collaborative 3D reconstruction under a fixed mission budget.

> Ariadne uses OpenCV 5 as a load-bearing computer-vision substrate across feature extraction, matching, geometry, stereo and point-cloud processing.

> Ariadne explicitly models false inter-agent loop closures and evaluates robust geometric containment.

> Ariadne treats communication as a quantified resource and reports reconstruction quality as a function of exchanged bytes.

> Ariadne produces an uncertainty-aware operational route through the recovered scene.

Avoid claiming:

> real disaster deployment readiness

> certified structural safety

> survivor detection accuracy from proxy datasets

> universal robustness to dust

> globally optimal pose graph solution

unless the experiments actually establish those properties.

---

# 43. Recommended final algorithm stack

## Perception

```text
OpenCV 5
    |
    +-- radiometric calibration
    +-- Poisson-Gaussian normalization
    +-- XFeat
    +-- ALIKED
    +-- LightGlue
    +-- calcOpticalFlowPyrLK
    +-- StereoSGBM
    +-- geometry
    +-- calib
    +-- ptcloud
```

## SLAM

```text
keyframe SLAM
local landmark map
Schur LM bundle adjustment
SE(3) pose graph
rotation synchronization
GNC robust PGO
SE-Sync-style certificate
```

## Collaboration

```text
EigenPlaces
cv::ANNIndex
two-stage exchange
spectral λ₂ selection
vertex-cover exchange
PCM
GNC
elected broker
anchor propagation
store-and-forward
```

## Reconstruction

```text
stereo point clouds
Gaussian submaps
stable/unstable Gaussian lifecycle
occupancy-aware fusion
near-field photometric correction
gsplat renderer
```

## Active perception

```text
frontier extraction
uncertainty hotspots
EIG approximation
connectivity value
TARE-style global/local representation
FAR-style route graph
risk-aware utility
auction/consensus assignment
```

## Operational route

```text
traversability
clearance
structural-risk proxy
uncertainty
dynamic graph
D* Lite/FAR-style replanning
```

## Agentic layer

```text
MCP
tool invocation
task negotiation
exception handling
human approval
audit trace
```

## Deployment

```text
Gazebo Harmonic + ROS2 Jazzy
       |
       +-- GPU AWS node
       |
       +-- Graviton4 COOL OpenCV node
       |
       +-- orchestration
       |
       +-- S3 artifacts
       |
       +-- operator console
```

---

# 44. Priority ranking by technical payoff

### Tier A — absolutely central

1. OpenCV 5 perception stack
2. local metric SLAM
3. windowed BA
4. inter-agent visual registration
5. GNC
6. sparse communication protocol
7. byte accounting
8. active perception
9. global map consistency
10. egress route

### Tier B — major differentiators

11. spectral λ₂ communication prioritization
12. degeneracy-before-loop-closure policy
13. EigenPlaces descriptor/communication co-design
14. co-moving-light model
15. risk-aware task assignment
16. Gaussian submaps
17. failure supervisor
18. agentic MCP loop

### Tier C — depth multipliers

19. SE-Sync certificate
20. DCS / switchable constraints comparison
21. descriptor aliasing diagnostics
22. NFL-aware photometric reconstruction
23. stable/unstable Gaussian lifecycle
24. COOL cost-per-frame benchmark

### Tier D — stretch

25. thermal victim witness
26. event camera experiments
27. GLOW-style advanced photometric effects
28. feed-forward dense SLAM comparator
29. learned NBV
30. learned shape prediction

---

# 45. Final architecture rationale

The entire system should be thought of as **three coupled optimization problems** rather than one giant AI system:

### 1. State estimation

> What world configuration best explains the observations?

Solved by:

```text
stereo geometry
SLAM
BA
pose graph
GNC
```

### 2. Information acquisition

> Given finite resources, what observation should happen next?

Solved by:

```text
uncertainty
information gain
connectivity
risk
communication
multi-agent allocation
```

### 3. World-model delivery

> What information must cross the network to make the shared estimate better?

Solved by:

```text
retrieval
spectral edge selection
sparse feature transfer
pose graph synchronization
submap fusion
```

The egress layer is the operational consumer of all three.

That separation is the architecture's strongest property: **perception estimates; collaboration chooses what information is worth sharing; planning chooses what information is worth acquiring; reconstruction visualizes the validated result; the route consumes the resulting uncertainty.**

---

# 46. Primary sources

1. OpenCV 5 overview  
   https://github.com/opencv/opencv/wiki/OpenCV-5

2. OpenCV feature framework  
   https://docs.opencv.org/5.0/main_modules/features_main.html

3. OpenCV LightGlue  
   https://docs.opencv.org/doc/doxygen/html/d8/dc3/classcv_1_1LightGlueMatcher.html

4. OpenCV ANNIndex  
   https://docs.opencv.org/5.0/main_modules/features_annoy.html

5. OpenCV point-cloud / volumetric reconstruction  
   https://docs.opencv.org/5.0/main_modules/ptcloud.html

6. XFeat  
   https://arxiv.org/abs/2404.19174

7. LightGlue  
   https://arxiv.org/abs/2306.13643

8. EigenPlaces  
   https://arxiv.org/abs/2308.10832

9. Swarm-SLAM  
   https://arxiv.org/abs/2301.06230

10. GNC  
    https://arxiv.org/abs/1909.08605

11. Photo-SLAM  
    https://arxiv.org/abs/2311.16728

12. RTG-SLAM  
    https://arxiv.org/abs/2404.19706

13. MASt3R-SLAM  
    https://arxiv.org/abs/2412.12392

14. VGGT-SLAM 2.0  
    https://arxiv.org/abs/2601.19887

15. EC3R-SLAM  
    https://arxiv.org/abs/2510.02080

16. MAGS-SLAM  
    https://arxiv.org/abs/2605.10760

17. NFL-BA  
    https://arxiv.org/abs/2412.13176

18. GLOW  
    https://arxiv.org/abs/2511.22857

19. Burst Imaging for Light-Constrained SfM  
    https://arxiv.org/abs/2108.09895

20. OIVIO  
    https://arpg.colorado.edu/oivio/

21. TARE  
    https://www.cmu-exploration.com/tare-planner

22. FAR Planner  
    https://arxiv.org/abs/2110.09460

23. CBBA  
    https://pure.kaist.ac.kr/en/publications/consensus-based-decentralized-auctions-for-robust-task-allocation/

24. Multi-Agent Risk-Aware NBV, 2026  
    https://arxiv.org/abs/2606.04158

25. GenNBV  
    https://openaccess.thecvf.com/content/CVPR2024/html/Chen_GenNBV_Generalizable_Next-Best-View_Policy_for_Active_3D_Reconstruction_CVPR_2024_paper.html

26. Active Neural Mapping at Scale  
    https://arxiv.org/abs/2409.20276

27. D* Lite  
    https://aaai.org/papers/00476-aaai02-072-d-lite/

28. ROS 2 Jazzy / Gazebo Harmonic  
    https://docs.ros.org/en/jazzy/Tutorials/Advanced/Simulators/Gazebo/Simulation-Gazebo.html

29. AWS Graviton4 M8g  
    https://aws.amazon.com/about-aws/whats-new/2024/09/amazon-ec2-c8g-m8g-instances/

30. AWS COOL for Graviton4  
    https://aws.amazon.com/marketplace/pp/prodview-fdvbfiewzuehs

31. gsplat  
    https://github.com/nerfstudio-project/gsplat

---

# 47. Source-of-truth relationship to `ALGORITHMS(1).md`

This implementation document should be treated as the execution layer above `ALGORITHMS(1).md`.

The catalogue remains the detailed algorithm research record.

This document makes the architecture-level decisions:

```text
ALGORITHMS
    |
    +-- literature evidence
    +-- benchmark evidence
    +-- failure evidence
    +-- implementation difficulty
    |
    v
IMPLEMENTATION.md
    |
    +-- chosen algorithms
    +-- interfaces
    +-- data contracts
    +-- test contracts
    +-- integration order
    +-- deployment topology
```

When an algorithm changes, update:

1. the catalogue entry,
2. this implementation decision,
3. the ablation matrix,
4. the benchmark configuration,
5. the run manifest schema.

Never silently replace an algorithm in runtime code without recording which paper/mechanism was substituted and why.
