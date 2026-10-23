# Early-coverage-rate mechanisms in published exploration planners (primary-source check)

Date: 2026-10-24. Sources read locally (pdftotext of arXiv PDFs; GitHub source at a pinned commit).
Copies of the read PDFs and code files were kept in A1's session scratchpad (not committed); every quote below names its source URL or commit.

Our context (from the brief): U = G - lambda_path(t)*path/p90 - risk - bw; kinodynamic-free mover,
turns IN PLACE at omega_max 0.44 rad/s; 44-46% of time turning, 28-31% moving; ~half of turn time is
"turn to face waypoint -> drive -> turn back to the view's final yaw" pairs; new goal every 5 s barrier.

---

## 1. FUEL (Zhou, Zhang, Chen, Shen, RA-L 2021, arXiv 2010.11561) -- time + yaw cost

### 1.1 Paper (arXiv 2010.11561v2 PDF, 8 pp.; section numbers as printed, page = PDF page)

**Eq. (1), Sec. IV-C "Viewpoint Generation and Cost Update" (p.4):** time lower bound between two viewpoints
x = (p, xi), xi = yaw:

    t_lb(x_k1,j1, x_k2,j2) = max( length(P(p_k1,j1, p_k2,j2)) / v_max ,
                                  min(|xi_k1,j1 - xi_k2,j2|, 2*pi - |xi_k1,j1 - xi_k2,j2|) / xidot_max )

Quote: "where P(p_k1,j1, p_k2,j2) denote a collision-free path ... found by a path searching algorithm,
v_max and xidot_max are the limits of velocity and angular rate of yaw. For each pair (F_k1, F_k2), we
select the viewpoints with highest coverage and estimate the cost as t_lb(x_k1,1, x_k2,1), in which
P(...) is searched on the voxel grid map using the A* algorithm."

**Eq. (2), Sec. V-A (p.4):** M_tsp(k1,k2) = M_tsp(k2,k1) = t_lb(x_k1,1, x_k2,1) (symmetric cluster block).

**Eq. (3)-(4), Sec. V-A (p.4):** first row (from current state x0 = (p0, xi0)):

    M_tsp(0,k) = t_lb(x0, x_k,1) + w_c * c_c(x_k,1)
    c_c(x_k,j) = acos( (p_k,j - p0) . v0 / (||p_k,j - p0|| ||v0||) ),  v0 = current velocity

Quote: "In some cases, multiple tours have comparable time lower bound, so back-and-forth maneuvers may be
generated in successive planning steps and slow down the progress. We eliminate this inconsistency with
c_c(x_k,1), which penalizes large changes in flight direction."

**Eq. (5):** M_tsp(k,0) = 0 (open-loop tour reduced to ATSP). **Eq. (6), Sec. V-B (p.5):** local refinement
over the truncated tour, Dijkstra on a DAG of each cluster's viewpoints, cost
c_rf(Xi) = t_lb(x0,x_1,j1) + w_c*c_c(x_1,j1) + sum_k t_lb(x_k,jk, x_k+1,jk+1) + t_lb(x_Nrf,jNrf, x_Nrf+1,1).
Quote (p.5): "it is straight forward to incorporate information gain [4, 12] to Equ.6, however, evaluating
information gain for numerous viewpoints is expensive. Practically, we find that simply adopting viewpoints
based on their coverages is much faster and leads to consistently satisfactory results."
=> FUEL's goal selection is **pure time cost over a tour**; gain only enters as a viewpoint-admission
threshold + per-cluster ordering by coverage (Sec. IV-C: "viewpoints with coverage higher than a threshold
are reserved and sorted in descending order of coverage. We reserve at most N_view viewpoints").

**Parameters, Sec. VI-A (p.6):** "We set w_c = 1.5 in Equ.3 and 6. ... the ATSP is solved using a
Lin-Kernighan-Helsgaun heuristic solver ... we reserve at most N_view = 15 viewpoints in each FIS and truncate
the global tour within radius R_rf = 5.0." Sim benchmark (Sec. VI-B): "v_max = 2.0 m/s and xidot_max = 0.9
rad/s for all methods", FOV 80x60 deg, range 4.5 m. Field (Sec. VI-C): v_max 1.5 m/s, a_max 0.8, xidot_max 0.9.
Result claim (Sec. VI-B-2, maze): "our method completes the exploration 4+ times faster on average ... it
produces a much shorter coverage path and an approximately linear exploration rate (Fig.8)."

### 1.2 Released code (HKUST-Aerial-Robotics/FUEL, commit `662dd23c7b52b258d3c4a0155ff6632118e8984f`, 2024-11-19)

`fuel_planner/active_perception/src/graph_node.cpp`, `ViewNode::computeCost(p1,p2,y1,y2,v1,yd1,path)`:

```cpp
double pos_cost = ViewNode::searchPath(p1, p2, path) / vm_;     // A* length (straight line if free; 1000 if no path)
if (v1.norm() > 1e-3) {                                          // motion consistency
  Vector3d dir = (p2 - p1).normalized(); Vector3d vdir = v1.normalized();
  double diff = acos(vdir.dot(dir));
  pos_cost += w_dir_ * diff;
}
double diff = fabs(y2 - y1); diff = min(diff, 2 * M_PI - diff);
double yaw_cost = diff / yd_;
return max(pos_cost, yaw_cost);
```

**Code vs paper discrepancy:** the code adds the consistency term *inside* the max
(max(L/v + w_dir*angle, dyaw/yd)), whereas paper Eq. (3) adds it *outside* (max(L/v, dyaw/yd) + w_c*c_c).
Unreachable pairs get pos_cost = 1000 / vm (A* failure, `searchPath` returns 1000).

Where computeCost is called (verified, same commit) -- consistency applies to the first leg only, matching Eq. 3-5:
- `frontier_finder.cpp` `FrontierFinder::updateFrontierCostMatrix()` line 296: cluster-to-cluster block uses
  `computeCost(vui.pos_, vuj.pos_, vui.yaw_, vuj.yaw_, Vector3d(0, 0, 0), 0, path_ij)` -- **zero velocity**, so no
  consistency term (Eq. 2); the same cost is pushed to both clusters (symmetric).
- `FrontierFinder::getFullCostMatrix(cur_pos, cur_vel, cur_yaw, mat)`, ATSP branch: line 581
  `mat.leftCols<1>().setZero();` (= Eq. 5, M(k,0) = 0); line 588 row 0 =
  `computeCost(cur_pos, vj.pos_, cur_yaw[0], vj.yaw_, cur_vel, cur_yaw[1], path)` -- the **only** row given the
  current velocity (Eq. 3). (A disabled `if (false)` symmetric-TSP branch also exists, line 552.)
- `fast_exploration_manager.cpp` `refineLocalTour(...)` line 442: `first->vel_ = cur_vel;` only the start node of
  the Dijkstra graph carries velocity (Eq. 6's single w_c term); other ViewNodes default `vel_.setZero()`.

`fuel_planner/exploration_manager/launch/algorithm.xml` (same commit):
`exploration/vm = 1.0*max_vel`, `exploration/am = 1.0*max_acc`, **`exploration/yd = 60 deg/s` (1.047 rad/s)**,
`exploration/ydd = 90 deg/s^2`, **`exploration/w_dir = 1.5`**, `exploration/relax_time = 1.0`,
`exploration/refined_num = 7`, `exploration/refined_radius = 5.0`, `exploration/top_view_num = 15`,
`exploration/max_decay = 0.8`; frontier viewpoint sampling `frontier/candidate_rmin = 1.5`,
`candidate_rmax = 2.5`, `candidate_rnum = 3`, `candidate_dphi = 15 deg`, `min_candidate_dist = 0.75`,
`min_visib_num = 15`. (Note yd = 60 deg/s in code vs 0.9 rad/s = 51.6 deg/s in the paper's sim limits.)

`fast_exploration_manager.cpp`, `FastExplorationManager::planExploreMotion`: goal = first refined viewpoint
(`next_pos = ed_->refined_points_[0]; next_yaw = refined_yaws[0];`), i.e. the head of a tour, never a greedy
argmax. The yaw-change time also sets a **trajectory duration lower bound**:
`double time_lb = min(diff, 2*M_PI - diff) / ViewNode::yd_;` passed to `planExploreTraj(..., time_lb)` /
`kinodynamicReplan(..., time_lb)`, then `planYawExplore(yaw, next_yaw, true, ep_->relax_time_)`.
Paths longer than `radius_far = 5.0` m are truncated to a 5 m intermediate goal; `< radius_close = 1.5` m uses a
waypoint-based optimisation instead of kinodynamic search.

### 1.3 What this means for an in-place-turning mover (analysis, not from the paper)

- FUEL's `max(translation_time, yaw_time)` is valid **only because a quadrotor yaws while translating**
  (position and yaw are separate B-splines, Eq. (7) optimises x,y,z,xi jointly; yaw constraint
  P(xidot_max, |xidot_c,i|) in Eq. (13)). For a mover that turns in place and drives along its heading the
  correct time lower bound is **additive**:
  t = |wrap(bearing_0 - psi0)|/omega + L/v + sum of in-path turn angles/omega + |wrap(psi_goal - bearing_last)|/omega.
  [DESIGN inference] Using max() or a distance-only cost for such a mover under-prices near goals with large
  heading changes -- consistent with our 44-46% turning share.
- FUEL does **not** let the view keep the arrival heading: each viewpoint has a fixed yaw chosen for coverage,
  and the yaw difference between consecutive views is charged in t_lb. It does not model the
  "turn to path heading" leg at all (holonomic assumption). See Sec. 1.4 for how FUEL picks view yaw.

### 1.4 How FUEL picks the viewpoint yaw (code, same commit)

`frontier_finder.cpp`, `FrontierFinder::sampleViewpoints(Frontier&)`: candidates on circles around the cluster
centroid, `sample_pos = frontier.average_ + rc*(cos phi, sin phi, 0)`, rc in [candidate_rmin 1.5, candidate_rmax
2.5] m step (rmax-rmin)/rnum, phi every candidate_dphi = 15 deg; rejected if outside box, inflated-occupied, or
`isNearUnknown`. **Yaw = mean bearing from sample_pos to the cluster's filtered cells**
(`avg_yaw = avg_yaw / cells.size() + atan2(ref_dir[1], ref_dir[0])`), i.e. the view faces its target -- exactly
our sampling pattern. A view is kept if `countVisibleCells(sample_pos, avg_yaw, cells) > min_visib_num` (15);
the commented-out `findMaxGainYaw` shows a yaw-optimisation variant was not used.

`FrontierFinder::getViewpointsInfo(...)`: for local refinement, each of the next K tour clusters offers up to
`top_view_num` = 15 viewpoints whose `visib_num_ > max_decay (0.8) * best visib_num_`, skipping views closer than
`min_candidate_dist` = 0.75 m to the robot. Dijkstra over Eq. (6) then picks the **time-cheapest** of these
near-equal-coverage views. So FUEL's implicit answer to "final-yaw turn" is: offer many positions around the
target (each with its own target-facing yaw) and let the yaw term of t_lb choose the one whose yaw is closest
to the current/previous yaw, accepting up to 20% less coverage. It never sets the view yaw = arrival heading.

---

## 3a. FUEL goal commitment / replan triggers (code, same commit)

`fast_exploration_fsm.cpp`, `FastExplorationFSM::FSMCallback`, state `EXEC_TRAJ` (timer 0.01 s); params from
`algorithm.xml`: `fsm/thresh_replan1 = 0.5`, `fsm/thresh_replan2 = 0.5`, `fsm/thresh_replan3 = 1.5`,
`fsm/replan_time = 0.200`.

```cpp
double time_to_end = info->duration_ - t_cur;
if (time_to_end < fp_->replan_thresh1_) { transitState(PLAN_TRAJ, "FSM"); ... }        // traj almost done (<0.5 s left)
if (t_cur > fp_->replan_thresh2_ && expl_manager_->frontier_finder_->isFrontierCovered()) // target cluster covered
   { transitState(PLAN_TRAJ, "FSM"); ... }
if (t_cur > fp_->replan_thresh3_ && !classic_) { transitState(PLAN_TRAJ, "FSM"); }     // periodic, every 1.5 s
```

`isFrontierCovered()` returns true when, inside the map's last updated box, a frontier has
`>= min_view_finish_fraction (0.2) * cells.size()` cells that are no longer known-free-next-to-unknown.
Frontier detection runs on `frontier_timer_` at 0.5 s.

Replanning is **from the predicted state 0.2 s ahead on the current trajectory** (`t_r = now - start + replan_time`;
start pos/vel/acc/yaw/yawdot evaluated on the B-splines). So FUEL re-decides every <=1.5 s, but continuity
comes from (a) the motion-consistency term w_dir=1.5 * angle(v0, p_goal - p0) (seconds per radian) in the
first leg, (b) planning from the current velocity, and (c) the whole-tour (ATSP) objective, which changes little
between replans. There is **no explicit hysteresis/commitment** threshold on the goal identity. [Verified in code;
paper Sec. I only says "it is desirable to replan new motions immediately whenever new information of the
environment is available".]

Relevance [DESIGN inference]: our planner re-decides every 5 s (slower than FUEL) but has neither a heading-
consistency term nor a tour objective, so successive greedy argmaxes can reverse direction; FUEL's authors name
exactly this failure ("back-and-forth maneuvers ... slow down the progress", Sec. V-A).

---

## 2a. Cieslewski, Kaufmann, Scaramuzza, IROS 2017 ("Rapid exploration with multi-rotors ...")

PDF: rpg.ifi.uzh.ch/docs/IROS17_Cieslewski.pdf (8 pp.). DOI 10.1109/IROS.2017.8206030.

- Diagnosis (Sec. IV, p.3): "the goal frontier often lies behind the robot. A multi-rotor flying at a non-zero
  velocity would thus need to reverse its flight direction and re-visit ground that it has already covered. ...
  we find that choosing a direction and maintaining it, and only later returning, can result in faster coverage
  than going back and forth in a breadth-first-search manner, especially when flying at high speeds."
- Rule (p.3): restrict candidates to frontiers **in the current field of view**, F_v subset F_g, then accessible
  ones F_a (straight segment clear by d_safe, Eqs. 6-7). "From F_v we select the frontier x*_i with the lowest
  cost c_i = |v_i - v_curr|, the norm of the difference between the current velocity and the desired velocity
  at the frontier", with **Eq. (5)** v(x_i) = (x_i - x_r) * v_max / r_sensor. "The control loop runs every 20ms."
  No information-gain term at all.
- Fallback / switching (p.4): "As soon as F_a = empty, the algorithm switches to a classical frontier selection
  method" (Dijkstra shortest path to any frontier); tracking of that path "is aborted if ... w_m has been reached
  or is unreachable ... Alternatively, if at any point of the waypoint tracking F_a != empty, the control switches
  back to the rapid frontier selection method."
- Time-axis metric (Sec. V, p.6, **Eq. (9)**): expected voxel discovery time
  t_exp = (1/|V*_free|) * sum_{x in V*_free} min{t : x in V_free(t)} -- i.e. the mean time-to-discovery, which is
  the area above the coverage-vs-time curve; exactly an "early coverage" metric.
- Result (Sec. VI-A, p.6): "our approach consistently outperforms classical Frontier-based exploration and
  next-best-view exploration across all scenarios with respect to t_max. The improvement is even more
  significant for t_exp, as in our approach there is often a significant amount of time at the end used only for
  re-visiting frontiers that have been previously skipped." Abstract (p.1, verbatim): "The goal frontier is
  selected in a way that minimizes the change in velocity necessary to reach it. While this approach can increase
  the total path length, it significantly reduces the exploration time, since the multi-rotor can fly at
  consistently higher speeds." Fig. 1 caption (p.1): "While our approach sometimes ends up traveling a larger
  distance, it is able to complete the exploration task faster, as it is capable of flying at higher speeds".
  Table IV (p.7) gives d_max/t_max/t_exp per scenario/v_max.
- Relevance [DESIGN inference]: the in-FOV restriction is the multirotor analogue of "keep the arrival heading":
  for an in-place turner, a goal inside the current FOV costs ~0 turn time. Note their win is largest on t_exp
  (early coverage), not just total time.

---

## 3b. TARE (Cao, Zhu, Choset, Zhang, RSS 2021) -- paper + released code

PDF: frc.ri.cmu.edu/~zhangji/publications/RSS_2021.pdf (9 pp.; DOI 10.15607/RSS.2021.XVII.018).
Code: github.com/caochao39/tare_planner, commit `44500592b86138257273e0cab264e6a847ccefc7` (2024-06-01).

**Paper.**
- Local path cost **Eq. (4)** (Sec. IV-B, p.3): c'_smooth = sum_{j=1..n~} l_j + p*(n~ - 1), l_j = length of segment j,
  n~ = number of segments. Quote: "The vehicle stops at the end of each segment before moving on to the next
  segment in a different direction. This requires that the vehicle can turn in one place. ... We discourage
  stopping too frequently by applying a penalty p at each stop." Fig. 3 caption: "The vehicle makes a stop at
  each break-point before moving on to the next path segment." The numeric value of p is **not given in the
  paper** [UNVERIFIED value; I did not locate it in the code files read].
  => TARE's local cost is length + a **per-stop (turn-in-place) penalty**; viewpoints carry no yaw (360 deg lidar).
- Viewpoint selection (Alg. 1, p.3): greedy submodular coverage of uncovered surface within the local horizon,
  randomised K times, keep the set whose TSP path cost c'_smooth is lowest; i.e. **coverage is a constraint
  (cover S_H), path cost is the objective** -- no gain/cost weighting.
- "Our method re-plans at 1Hz" (Sec. VI, p.7). Metric: "exploration efficiency, defined as the average explored
  volume per second through a run" (Abstract; Table I) -- 80% higher than NBVP/GBP/MBP; Fig. 7(b)/8(b) plot
  explored volume vs time.

**Code -- what the robot is actually sent (goal = lookahead point, with direction hysteresis).**
`src/tare_planner/src/sensor_coverage_planner/sensor_coverage_planner_ground.cpp`:
- `execution_timer_ = nh.createTimer(ros::Duration(1.0), &SensorCoveragePlanner3D::execute, this);` (1 Hz).
- `GetLookAheadPoint(...)`: walks the (cyclic) exploration path both **forward and backward** from the robot node
  and stops at the first node that is a LOCAL_VIEWPOINT / local-path start/end, or where cumulative length
  `> kLookAheadDistance`, or where the next node leaves line of sight (`kUseLineOfSightLookAheadPoint`).
  Each candidate is scored by `angle_score = dx*diff.x() + dy*diff.y()` = cosine between the bearing to it and the
  **previous lookahead direction** (`pd_.lookahead_point_direction_`). Rule:
  - keep the previous lookahead point if `lookahead_angle_score > 0 && dist_robot_to_lookahead > kLookAheadDistance/2
    && InLocalPlanningHorizon(...)` -- an explicit **commitment/hysteresis** rule;
  - otherwise take forward vs backward by larger angle score (i.e. continue in the current direction of travel);
  - in relocation mode, take the shorter one unless momentum is active, in which case the better-aligned one.
- `CountDirectionChange()`: if the displacement since last cycle (>0.5 m) reverses (`dot < 0`) more than
  `kDirectionChangeCounterThr` times, set `use_momentum_ = true`; reset after `kDirectionNoChangeCounterThr`
  non-reversals. (`kUseMomentum: false` in indoor.yaml, so this is off indoors.)
- `PublishWaypoint()`: the published waypoint is the lookahead bearing **extended** to `kExtendWayPointDistanceBig`
  (in LOS) or `kExtendWayPointDistanceSmall` if closer (`if (r < extend_dist && pp_.kExtendWayPoint) { dx = dx/r*extend_dist; ... }`),
  so the robot never decelerates to stop at a near goal; the 1 Hz replan re-targets it before arrival.
- `config/indoor.yaml`: `kLookAheadDistance: 8`, `kExtendWayPoint: true`, `kExtendWayPointDistanceBig: 8.0`,
  `kExtendWayPointDistanceSmall: 3.5`, `kUseLineOfSightLookAheadPoint: true`, `kUseMomentum: false`,
  `kDirectionChangeCounterThr: 6`, `kDirectionNoChangeCounterThr: 5` (code defaults 5.0 / 8.0 / 3.0 / 4 / 5).

Relevance [DESIGN inference]: a "TARE-style = first view of a local NN+2-opt tour" baseline omits three
TARE mechanisms that matter for early rate: (1) tour **direction** chosen by alignment with the previous heading,
(2) keep-previous-goal hysteresis (cos>0 and still > L/2 away), (3) waypoint extension so motion never stops at a
near goal. Our measured "near goals, few m/s" symptom is what (2)+(3) are designed to prevent.

---

## 2. Gain-vs-cost combination forms

### 2b. NBVP / RH-NBVP (Bircher et al., ICRA 2016; extended Auton. Robots 42(2) 2018)

- Papers: ICRA 2016 PDF not obtained; the Auton. Robots 2018 PDF at research-collection.ethz.ch returned a
  "Rate Limited" HTML page. **Paper equations [UNVERIFIED from primary PDF].**
- Released code (ethz-asl/nbvplanner, commit `5c07657af18f967d7049d187db0607024746a42c`, 2017-07-26),
  `nbvplanner/src/rrt.cpp`, tree expansion:
  `newNode->distance_ = newParent->distance_ + direction.norm();`
  `newNode->gain_ = newParent->gain_ + gain(newNode->state_) * exp(-params_.degressiveCoeff_ * newNode->distance_);`
  i.e. branch value = sum over nodes of gain * exp(-lambda * cumulative **path length**); `degressiveCoeff_`
  default 0.25 (`nbvp.hpp`: "Default is 0.25"). Yaw is sampled uniformly at random per node
  (`newState[3] = 2.0 * M_PI * (rand()/RAND_MAX - 0.5)`) and does **not** enter the cost; execution
  discretisation uses `min(dt*v_max/|d|, dt*dyaw_max/|dyaw|)` (concurrent yaw+translate; defaults
  v_max 0.25 m/s, dyaw_max 0.5 rad/s). Only the first edge of the best branch is executed (receding horizon).
- Secondary confirmation of the form: AEP (Selin et al.) Eq. (1) p.2 writes it as s(x) = g(x) exp(-lambda c(x)),
  node score s(x) + s(p(x)) "[2]"; Schmid et al. Eq. (7) p.4 as v_exp(V_i) = v(parent(V_i)) + g(V_i) exp(-lambda c(V_i)).
- Observed failure modes (third-party, Schmid et al. Sec. VII-A, p.5): "RH-NBVP suffers from several issues,
  including: long computation times when stuck in a dead end, **spending a lot of time turning on the spot, and
  frequent changes of the exploration site.** Consequently RH-NBVP travels only 210 m, covering 60% of the
  environment. By contrast AEP and our method keep constantly moving, covering a distance of 297 m and 330 m."

### 2c. AEP (Selin, Tiger, Duberg, Heintz, Jensfelt, RA-L 2019)

PDF: ida.liu.se/divisions/aiics/publications/IRAL-2019-Efficient-Autonomous-Exploration.pdf (8 pp.).
- Utility = NBVP's exponential form, Eq. (1) p.2: "s(x) = g(x) exp(-lambda c(x)), where ... c(x) is the cost of
  going to x and lambda is a coefficient to control how much distance should be penalized." Interpretation
  (p.2): "A high lambda penalizes movement hard and makes the agent perform careful exploration nearby before
  moving into the next region, while a low lambda will have the effect that the agent will move faster towards
  completely unexplored space, missing details that it has to go back to later and fill in."
- Diagnosis (Sec. V, p.3): "when ... the nearest frontier is far away, this frontier will typically have a very low
  score ... caused by the exponential decay ... Either small scores are ignored and the exploration will be
  incomplete or all scores above zero will be included, but then the exploration will focus on very small
  information gains nearby that could be skipped." Remedy: local RH-NBVP + **global frontier planner over cached
  high-gain nodes** (mode switch when local gain < g_zero).
- **Yaw (Sec. V-A, Eq. (2), p.3):** g(x) = max over phi of g_phi(x, phi): "let phi be the value that maximizes
  g_phi(x, phi) ... ray-tracing 360 deg around the agent ... sliding-window summation to find the information gain
  for every yaw angle." "Calculating the best yaw instead of sampling yaw angles reduces the sampling space ...
  As the experiments show, this leads to more efficient exploration." The yaw is gain-optimal, **not
  cost-aware** (no turn time in c(x)).
- Time-axis evidence (Sec. VII-B, p.6, Fig. 6): with frontier (global) layer "AEP manages to explore the first half
  completely in 240 seconds and reaches the other side of the maze in 400 seconds. The AEP-F gets stuck in the
  first half. For high lambda it never gets out ... With a low lambda it eventually ... reaches the second half
  after 600 seconds. Lower lambda will make exploration go faster forward but will be less careful along the way."
  Parameters (Fig. 6 legend, p.6): AEP lambda = 0.75, g_zero = 1, N = 100; AEP-F lambda = 0.25 (g_zero 1 or 2) and 0.15 (g_zero 2).

### 2d. Schmid, Pantic, Khanna, Ott, Siegwart, Nieto, RA-L 2020 (arXiv 1909.09548) -- the direct comparison

- Cost = **expected execution time**, Eq. (6) p.4: c(V_i) = t_end(tau(V_i)) - t_start(tau(V_i)); Sec. V p.4:
  "We use a constant acceleration model for position and yaw to estimate the execution time"; yaw chosen per node
  by 30-deg-sector gain maximisation ("The yaw maximizing the total gain of a node is then greedily selected").
- Forms compared (Sec. IV-B, p.4): Eq. (7) v_exp = v(parent) + g exp(-lambda c); Eq. (8) v_lin = v(parent) + g - alpha c;
  **Eq. (9) global normalization** v_GN(V_i) = max over V_k in subtree(V_i) of [ sum_{V_l in R(V_k)} g(V_l) ] /
  [ sum_{V_l in R(V_k)} c(V_l) ] (R = nodes from root to V_k) -- i.e. **accumulated gain per accumulated time**
  along the best branch through V_i. Quote: "v_exp is strictly increasing, favoring long sub-trees even when
  subsequently executed nodes might be sub-optimal. Furthermore, both require careful tuning of the parameters
  lambda and alpha. To address these two issues, we use the notion of efficiency, i.e. the accumulated gain per
  cost, as a central idea for the value."
- Result, **Table V (p.7)**, city building, gain fixed to voxel impact, alpha = 3.0, lambda = 0.5:
  exploration at 30 min: linear penalty 76.4 +/- 6.5 %, exponential 96.6 +/- 5.3 %, global-norm 99.8 +/- 0.1 %;
  reconstruction error 8.9 / 8.0 / 6.4 cm. Quote (p.7): "By construction both linear and exponential penalties
  favor suboptimal viewpoints that have low cost". (Note: the exp variant used accumulated *path length* as cost.)
  Exploration-vs-time curves: Fig. 5a/5b p.6 (maze, building); maze vs RH-NBVP/AEP: full observable volume at
  25 min "in all runs".
- Parameters (Table I p.5): v_max 1 m/s, yaw rate pi/2 rad/s, a_max 1 m/s^2, FOV 90x73.7 deg, voxel 0.1 m.

Relevance [DESIGN inference]: our U = G - lambda_path(t)*path/p90 - ... is Schmid's **v_lin** (Eq. 8), the worst
form in their Table V on both coverage-at-30-min and error, and their stated mechanism ("favor suboptimal
viewpoints that have low cost") matches our "median goal 0.8-1.7 m" observation. Their remedy is a ratio of accumulated
gain to accumulated execution time along a branch (not per-candidate), which is also what FUEL's "time" axis and TARE's "volume per second" metric reward.

### 2e. Cieslewski 2017 (see 2a): no gain term; cost = |v_desired - v_current| over in-FOV frontiers.

### 2f. Dharmadhikari et al., ICRA 2020 (motion primitives, fast exploration)

- Main ICRA 2020 paper PDF **not obtained** (no arXiv copy found). Its utility formula is **[UNVERIFIED]**.
- Read only the companion arXiv 2012.03228 ("Appendix for the Motion Primitives-based Path Planning ...",
  Dharmadhikari, Dang, Alexis, 6 pp.). Relevant content: robot state includes velocity and heading
  (Sec. III, p.2: "the robot's configuration at time t be defined as the combination of 3D position, linear
  velocity and heading xi_t = [x, y, z, v_x, v_y, v_z, psi_t]"); a bifurcated local planner (motion primitives,
  maximise "the volume which is expected to be mapped when the robot traverses along the path") and a global
  planner "triggered" on "local completion" to re-position toward frontiers. No utility equation or time-axis
  coverage comparison in the parts read.

---

## 4. Does mixing refinement terms with coverage slow early coverage? Standard remedies

Direct evidence found:
- **Schmid et al. 2020** use ONE gain on a common per-voxel scale, Eq. (5) p.4: g_rec(xi) = sum over m in Vis(xi) of
  { 1 if m in U (unknown voxel near an observed surface); (eta(m) - eta_min)/(1 - eta_min) if m in S (observed
  surface) and eta(m) > eta_min; 0 otherwise }, with impact factor Eq. (4) eta(m) = 1 / (1 + z^2 w(m)/N_rays(m)).
  A known voxel is therefore worth strictly less than an unknown one and its worth decays as it is re-observed, so
  the planner is frontier-first automatically: "Our gain formulation initially focuses on global surface
  exploration, before focusing on areas of low map weight, i.e. low expected reconstruction quality, when no more
  unobserved areas are present. This behavior is apparent after 10 minutes, when exploration slows down, the error
  is further reduced." (Sec. VII-B, p.6). Table IV (p.7), time to 95% exploration: unknown volume 15.0 +/- 1.9 min,
  surface frontiers 16.2 +/- 1.5, voxel confidence 15.6 +/- 1.8, voxel impact 14.7 +/- 1.7 -- "While most gain
  formulations trade off exploration speed for increased quality, our gain quickly explores the building and then
  further refines areas of high uncertainty" (p.7). So mixing *on a common scale where unknown >= known* did not
  slow coverage; the other quality-aware gains were slower to 95%.
- **AEP** (p.2, quoted in 2c): a high distance penalty makes the agent "perform careful exploration nearby before
  moving into the next region", and (p.3) "the exploration will focus on very small information gains nearby that
  could be skipped" -- small local gains crowding out progress; remedy = a separate global frontier layer.
- **FUEL** removes information gain from goal choice entirely (Sec. V-B p.5 quote in 1.1): coverage only filters
  and orders viewpoints; the decision is pure time.
- **Cieslewski 2017**: no gain at all in the reactive mode; explicit rapid-mode / classic-mode switch (p.4).
- **TARE**: coverage of the local uncovered surface is a constraint of the viewpoint set; the objective is path
  length + stop penalty (Eq. 4); no refinement term.

Remedy patterns seen across these (all primary-source): (a) single common-unit gain with unknown dominating known
(Schmid Eq. 5); (b) gain as a filter/threshold, time as objective (FUEL, TARE); (c) explicit mode switching
local/global or rapid/classic (AEP, Cieslewski, Dharmadhikari appendix "local completion" -> global planner).
I found **no** paper among those read that normalises each information term by its own percentile over the
candidate set; [DESIGN inference] per-term p90 normalisation gives a refinement term (entropy of known voxels)
equal weight to delta-coverage regardless of absolute magnitude, which is the opposite of (a). Time-scheduled
phase weights lambda(t) were not found in these papers [not searched beyond them].

---

## 5. Bottom line for Ariadne (inferences, tagged)

1. [PAPER: FUEL Eq. 1 + code] Use a **time** cost; for an in-place turner make it additive, not FUEL's max():
   t = |dpsi_depart|/omega + L/v + |dpsi_arrive|/omega (+ in-path turns). With omega = 0.44 rad/s a 90 deg
   final-yaw turn costs 3.6 s = 2.1 m of driving at 0.6 m/s.
2. [CODE: FUEL getViewpointsInfo] Offer several near-equal-coverage views per target (FUEL: up to 15 with
   >= 0.8x best coverage) and let the yaw term pick; or [DESIGN] let a view keep arrival heading if its coverage
   at that heading is within a tolerance (no surveyed paper does exactly this; AEP/Schmid choose gain-optimal yaw).
3. [PAPER: Schmid Eq. 9 + Table V; recommendation is DESIGN inference] Replace the linear penalty with an
   efficiency (gain/time) form -- but **computed over a multi-step horizon/tour** (accumulated gain / accumulated
   time along a branch, maximised over the subtree, as v_GN is), not a per-candidate g/t rescored every 5 s, which
   would reward tiny-cost candidates and worsen the near-goal pathology. Schmid's own text: "Instead of greedily
   maximizing the efficiency of each node, we account for the global context of each node" (p.4). Caveat: Table V is
   coverage at 30 min in the building-reconstruction scenario (exp variant used path length, not time) -- it is
   **not** an early-rate result.
   Early-rate metrics: among the papers read, the only explicit early-coverage-on-the-time-axis metric is
   Cieslewski's t_exp (Eq. 9, mean voxel discovery time). TARE's "explored volume per second" is a whole-run
   average; FUEL and Schmid report completion times and coverage-vs-time curves (FUEL Fig. 8, Schmid Fig. 5).
4. [CODE: TARE GetLookAheadPoint; PAPER: FUEL Eq. 3-4, Cieslewski Eq. 5] Add heading consistency / commitment:
   keep the previous goal if still ahead (cos > 0) and > L/2 away (TARE); penalise direction change w_dir = 1.5 s/rad
   (FUEL code); extend near waypoints so the mover doesn't stop (TARE kExtendWayPoint 8.0/3.5 m).
5. [PAPER: Schmid Eq. 5] Put refinement on the same per-voxel scale as coverage (unknown = 1 > known) instead of
   p90-normalising each term.
