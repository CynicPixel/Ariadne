# Ariadne

Autonomous multi-agent visual SLAM and 3D reconstruction for GPS-denied, communication-constrained disaster environments.

## The problem

After a structural collapse, the spaces most important to search are often the least accessible to people: unstable, unlit, obstructed by rubble, and disconnected from positioning and communications infrastructure.

Ariadne uses a fleet of autonomous visual agents to explore these spaces, build a shared 3D model, and determine how responders can safely reach points of interest and return.

The difficulty is not simply mapping the environment. Agents must decide what to observe, what information is worth transmitting, which observations belong to the same physical location, and when exploration should give way to consolidation.

Concrete and rubble create low texture and strong perceptual aliasing. Self-carried illumination makes appearance dependent on camera pose. Communication, energy, motion, and mission time are finite resources.

## The question

How should multiple autonomous visual agents allocate sensing, motion, and communication while maintaining a consistent world model under a shared mission budget?

Ariadne evaluates three principles:

1. Allocate sensing and motion to uncertainty that matters to the shared model, rather than simply maximizing local coverage.
2. Spend communication on relationships that improve global estimation, rather than transmitting redundant observations.
3. Treat false inter-agent correspondences as an expected operating condition and contain them before they corrupt the global map.

## The thread

A map is useful only when it supports a decision.

Ariadne therefore treats responder egress as a first-class output. The reconstructed environment becomes a spatial cost field incorporating traversability, clearance, reconstruction uncertainty, and structural risk. Routes are continuously recomputed as the world model changes.

A newly observed obstruction can invalidate a route. Better reconstruction can increase route confidence. Poorly observed space remains expensive to traverse.

Routes are evaluated at responder clearance, not agent clearance. A void a robot can enter is not automatically a viable human route.

## Approach

Ariadne uses stereo RGB cameras, an IMU, and a co-mounted controllable light. No LiDAR or depth sensor is required.

**Perception.** OpenCV 5 handles image restoration, feature extraction, matching, optical flow, stereo, calibration, and geometric estimation. Frame-rate tracking uses sparse optical flow, while learned features are evaluated at keyframes and for candidate inter-agent registrations.

**Local SLAM.** Each agent maintains an independent keyframe map with landmark management, triangulation, windowed bundle adjustment, and a local pose graph. Bundle adjustment uses analytic Jacobians, manifold SE(3) updates, and Schur elimination. The estimator is implemented around OpenCV 5 primitives with a single global perturbation convention.

**Collaboration.** Agents exchange compact global descriptors first. Expensive local descriptors and geometric evidence are requested only for selected candidates. Candidate selection is budgeted according to the contribution of each potential relationship to the connectivity of the joint pose graph. Communication is measured at the serialization boundary, including framing and compression overhead.

**Consensus.** Candidate relationships pass through retrieval, two-view geometry, degeneracy checks, pairwise consistency filtering, and robust pose-graph optimization. Graduated non-convexity allows the global optimizer to reject severe correspondence contamination without requiring a clean initial solution.

**Reconstruction.** Validated keyframes anchor incremental 3D Gaussian submaps. Camera geometry and inter-agent registration remain authoritative; the photorealistic representation is downstream of the geometric solution.

**Orchestration.** Deterministic estimators produce pose, depth, registration, uncertainty, and coverage information. The coordination layer uses these outputs to allocate observations, negotiate resources, request peer interactions, and escalate high-risk actions to human command. Every decision is traceable to the visual evidence that triggered it.

## OpenCV 5 in the stack

OpenCV 5 is a core part of the system rather than a preprocessing dependency.

| Stage        | OpenCV 5                     | Role                                                    |
| ------------ | ---------------------------- | ------------------------------------------------------- |
| Restoration  | `photo`, `imgproc`, `xphoto` | denoising, illumination compensation, contrast recovery |
| Tracking     | `video`                      | pyramidal optical flow                                  |
| Features     | `features`                   | ALIKED, DISK, XFeat                                     |
| Matching     | `features`                   | LightGlue, ANNIndex                                     |
| Geometry     | `geometry`                   | essential matrix, PnP, triangulation                    |
| Calibration  | `calib`                      | camera and multi-camera calibration                     |
| Stereo       | `stereo`, `ximgproc`         | disparity and filtering                                 |
| Point clouds | `ptcloud`                    | TSDF, HashTSDF, point-cloud processing                  |
| Evaluation   | `quality`                    | image-quality metrics                                   |

The systems above these primitives are implemented in Ariadne: keyframes, local maps, bundle adjustment, pose graphs, robust consensus, inter-agent communication, task allocation, byte accounting, reconstruction management, and egress planning.

## Deterministic evaluation

Simulation is separated from perception and estimation so every baseline consumes identical observations.

Gazebo generates a scene, trajectory, and seed once and records the result as MCAP. Every subsequent run consumes that recording. Image degradation is applied at replay time through a seeded deterministic operator keyed by seed, frame, and processing site.

This allows restoration, SLAM, communication, and planning changes to be compared on exactly the same observations rather than on independently rendered simulations.

Closed-loop active perception is evaluated over multiple seeds rather than a single run.

## Layout

```text
cpp/core            SE(3), uncertainty, serialization, RNG, timing
cpp/perception      restoration, features, tracking, stereo, exposure
cpp/slam            IMU, landmarks, local map, BA, pose graph, loop closure
cpp/collaboration   retrieval, communication, candidate selection, PCM, GNC
cpp/reconstruction  point clouds, Gaussian submaps, fusion, rendering
cpp/planning        frontier detection, NBV, risk, assignment, routes
python/              reconstruction services, orchestration, world generation
ros2/                 agent, simulator, and operator nodes
evaluation/           datasets, scenarios, metrics, baselines, ablations
infra/                containers, deployment, AWS infrastructure
tests/                unit, numerical, integration, protocol, regression, E2E
```

Dependency direction is enforced. Algorithm libraries do not depend on ROS 2, simulator types, or AWS services. Replay and evaluation operate entirely from recorded data.

## Build

Requirements:

* C++17
* CMake 3.22+
* Ninja
* OpenCV 5 and contrib
* Eigen
* ROS 2 Jazzy
* Gazebo Harmonic

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build
```

OpenCV is built from a pinned source revision so the exact 5.x module set and configuration are reproducible.

## Validation

Validation starts with public datasets containing known trajectories and geometry, then moves to controlled disaster environments in simulation and real co-moving-light sequences.

The evaluation compares independent exploration, coverage-driven coordination, and Ariadne's uncertainty and connectivity-aware policy using:

* trajectory and registration error
* reconstruction quality and completeness
* shared-map consistency
* communication volume by category
* mission time and energy proxies
* time to first valid route
* route validity against ground truth
* robustness under communication loss and visual corruption

Ablations remove restoration, active allocation, communication prioritization, and robust consensus independently so each component is evaluated on measured contribution rather than assumed value.

## Responsible use

Ariadne is designed for search and rescue in environments unsafe for human entry.

The system represents geometry, traversability, uncertainty, and routes rather than persistent identity. High-consequence actions remain subject to explicit human approval, and uncertainty is exposed throughout the planning stack.

Ariadne is decision support for incident command. It does not replace professional structural assessment or human authority.

## Project

Built for the OpenCV AI Competition 2026.
