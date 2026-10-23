# Primary-text literature check: Ariadne NBV/egress planner fixes 1-3

Date: 2026-10-23. Every quote below was read in the primary source in this session, from text extracted with `pdftotext` or from raw source files. "PDF p. N" is the page index inside the PDF file, not the printed journal page. Code is cited at the commit that was HEAD when I read it. Anything I could not read in the primary source is tagged **[UNVERIFIED]**.

---

## FIX 1: plan space = known free only (`trav_agent AND clearance >= rho_agent`, unknown counted as occupied in the ESDF)

### 1(a) FUEL (Zhou et al., RA-L 2021, arXiv 2010.11561)

**Verdict: SUPPORTS.** The paper text is ambiguous. The released code is explicit.

**Paper** (read at https://arxiv.org/pdf/2010.11561):
- §IV-C "Viewpoint Generation and Cost Update", PDF p. 4: *"VPi are found by uniformly sampling points in the cylindrical coordinate system whose origin locates at the cluster's center ... For each of the sampled points p lying within the free space, the yaw angle ξ is determined as the one maximizing sensor coverage to the cluster"*
- Same section: *"P(pk1,j1, pk2,j2) denote a collision-free path between pk1,j1 and pk2,j2 found by a path searching algorithm"*. The path is from A*, per the text that follows.
- §IV, PDF p. 3: *"frontiers are defined as known-free voxels right adjacent to unknown voxels"*
- §V-C, Eq. 12: the trajectory collision penalty uses *"d(pc,i) is the distance of point pc,i to the nearest obstacle, which can be obtained from the Euclidean signed distance field (ESDF)"*. §VI-A sets d_min = 0.4.
- The paper never says whether "free space" means known-free, and never says how the ESDF treats unknown.

**Code** (HKUST-Aerial-Robotics/FUEL @ `662dd23c7b52`, raw.githubusercontent.com/.../main/fuel_planner/...):
- `active_perception/src/frontier_finder.cpp`, `sampleViewpoints`:
  `// Qualified viewpoint is in bounding box and in safe region` / `if (!...isInBox(sample_pos) || ...getInflateOccupancy(sample_pos) == 1 || isNearUnknown(sample_pos)) continue;`
- `isNearUnknown()` returns true if any voxel in a ±`min_candidate_clearance` xy box (z ±1 voxel) `== SDFMap::UNKNOWN`. In `exploration_manager/launch/algorithm.xml`, `frontier/min_candidate_clearance` is `0.21`, and the candidate radius runs from `candidate_rmin` 1.5 to `candidate_rmax` 2.5 m around the cluster centre.
- `path_searching/src/astar2.cpp` (the A* behind the tour costs): `if (...getInflateOccupancy(nbr_pos) == 1 || ...getOccupancy(nbr_pos) == SDFMap::UNKNOWN) continue;`. The same test is applied at 0.1 m steps along each edge.
- `plan_env/src/sdf_map.cpp`, `updateESDF3d()`: if `optimistic_` is true, only inflated occupied cells seed distance 0. Otherwise a cell seeds 0 when `(occupancy_buffer_inflate_[adr] == 1 || occupancy_buffer_[adr] < clamp_min_log_ - 1e-3)`. Unknown cells are initialised to `clamp_min_log_ - unknown_flag_` (with `unknown_flag_ = 0.01`), so they fall below that threshold. In the non-optimistic mode, **unknown is an obstacle for the ESDF**. This is exactly Ariadne's clearance definition.
- The exploration config `exploration_manager/launch/algorithm.xml` sets `sdf_map/optimistic = false` and `search/optimistic = false`.

### 1(b) NBVP (Bircher et al.), GBPlanner (Dang et al.), GBPlanner2 (Kulkarni et al.)

**NBVP verdict: SUPPORTS.**
- ICRA 2016 conference text: **[UNVERIFIED]**. I could not get the PDF (IEEE only).
- Journal extension, Bircher et al., "Receding horizon path planning for 3D exploration and surface inspection", Auton. Robots 42:291-306 (2018). Read at http://web.archive.org/web/20240412220004id_/https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/127860/s10514-016-9610-0.pdf:
  - §1, PDF p. 4: *"next best views are randomly sampled in the known free space ... the views are sampled as nodes in a random tree, the edges of which directly provide a collision free path."*
  - §3, PDF p. 5: *"Paths are only planned through known free volume V_free, thus providing collision free navigation."*
  - Caveat, so this is not misread as optimistic traversal. PDF p. 7, surface-inspection gain: *"For the gain computation the unmapped voxels are considered free."* That applies **only to the line-of-sight gain ray**. It does not apply to motion.
- Code, ethz-asl/nbvplanner @ `5c07657af18f`, `nbvplanner/src/rrt.cpp`. A tree edge is added only if `CellStatus::kFree == manager_->getLineStatusBoundingBox(origin, direction + origin + direction.normalized()*dOvershoot_, boundingBox_)`.
  - I read the helper in ethz-asl/volumetric_mapping @ `2e25b6da4c4e`, `octomap_world/src/octomap_world.cc`. `getLineStatus()` returns `kOccupied` for a missing (unknown) node when `treat_unknown_as_occupied` is set, and `kUnknown` otherwise. `getLineStatusBoundingBox()` returns the first non-`kFree` status it meets.
  - So unknown fails the edge check **whichever way that option is configured**.
- Related to Fix 3: PDF p. 12 says *"The exploration starts at t_tot = 0, after an initialization motion to allow the computation of a first collision free path segment."* A known-free planner cannot start without some bootstrap free space.

**GBPlanner1 verdict: SILENT on known-free in the text I could read.**
- JFR 2020 paper: **[UNVERIFIED]**. Wiley and ETH did not serve the PDF.
- Instead I read the GBPlanner description in Dang et al., "Field-hardened Robotic Autonomy for Subterranean Exploration" (FSR 2019, EasyChair preprint 1452), at https://easychair.org/publications/preprint/FBld/open.
  - §5, Problem 1, PDF p. 8: *"Given an occupancy map M, find a collision-free path σ* ..."*
  - "Collision-free" is never defined against unknown.

**GBPlanner2 verdict: SUPPORTS, explicitly.** Read at https://arxiv.org/pdf/2111.06482.
- §IV-A, PDF p. 3: *"the method randomly samples vertices in the known free space within a local bounding box ... For a flying robot, admissible edge connections between two vertices are simply collision-free straight line segments."*
- For ground robots it additionally checks that *"ground exists and is mapped"*. Vertices without mapped ground are "hanging", and *"the edges to them are not allowed to be commanded to the robot."*
- Code, ntnu-arl/gbplanner_ros branch `gbplanner2` @ `7301b5350915`, `gbplanner/src/rrg.cpp`:
  - `sampleVertex()` requires `kFree == map_manager_->getBoxStatus(center, robot_box_size_, true)`.
  - Edges require `kFree == map_manager_->getPathStatus(start, end, robot_box_size_, true)`.
  - The third argument is `stop_at_unknown_voxel` (from `planner_common/map_manager.h`).
  - So the **whole robot box** has to be known-free.

### 1(c) Does anyone plan optimistically through unknown? Under what conditions?

**Verdict: YES, but only for point-goal navigation with continuous replanning. None of the exploration planners I read do this.**
- **D\* Lite.** Koenig & Likhachev, "Fast Replanning for Navigation in Unknown Terrain", IEEE T-RO 2005 preprint, read at http://idm-lab.org/bib/abstracts/papers/tro05.pdf. §I, PDF p. 1: *"The robot always plans a shortest path from its current coordinates to the goal coordinates under the assumption that unknown terrain is traversable ... If it observes obstacles as it follows this path, it enters them into its map and then repeats the procedure"*. The index terms name it "Planning with the Freespace Assumption".
- **FAR Planner.** Yang et al., arXiv 2110.09460, read at https://arxiv.org/pdf/2110.09460. §III-C, PDF p. 4: *"we provide the option of searching in the combined space (free and unknown) for attemptable planning or in free space only for non-attemptable planning."* The setting is navigation to a *given goal* (§IV-A, "the vehicle is set to navigate through a series of points"), with the v-graph updated as obstacles appear.
- **Nav2.** Raw docs at ros-navigation/docs.nav2.org @ `64a93c252eb4`:
  - `configuring_navfn.md`: `allow_unknown`, *Default: true*, *"Whether to allow planning in unknown space."*
  - `smac_2d/configuring_smac_2d.md`: *Default: true*, *"Whether to allow traversing/search in unknown space."*
  - `costmap_2d/index.md`: `track_unknown_space`, *Default: false*, *"If false, treats unknown space as free space"*.
  - All of these are point-goal planners.
- **Cleanest evidence of the condition, from one codebase.** FUEL's point-goal launch files `plan_manage/launch/kino_algorithm.xml` (line 76) and `topo_algorithm.xml` (line 71) set `sdf_map/optimistic = true`. FUEL's *exploration* config sets it to `false` (see 1(a)).
- **Soft middle ground.** STEP (Fan et al., arXiv 2103.02828, read at https://arxiv.org/pdf/2103.02828), §V-B, PDF p. 8: higher CVaR α results in *"closing narrow openings, assigning high cost to unknown regions, and increasing the size of obstacles."* Unknown is penalised, not forbidden.
- **What this means for Ariadne.** Optimistic planning is used when (i) the goal is externally given, and (ii) a reactive local layer re-plans as the map fills in. For *exploration goals* (viewpoints), every system I read requires known-free space. Fix 1 matches SOTA.

---

## FIX 2: goal risk = max bottleneck (widest-path clearance) over a 0.5 m disc, mapped as (rho_r - b)/(rho_r - rho_a)

### 2(a) Is maximin clearance (widest / bottleneck path) an established route-quality measure?

**Verdict: SUPPORTS as a geometric clearance measure. The SubT risk planners I read aggregate risk differently.**
- **Agarwal, Fox & Salzman**, "An Efficient Algorithm for Computing High-Quality Paths amid Polygonal Obstacles", arXiv 1706.02939, read at https://arxiv.org/pdf/1706.02939. Related work, PDF p. 2: *"if maximizing the distance from the obstacles is the optimization criteria, then the path can be computed by constructing a maximum spanning tree in the Voronoi diagram of the obstacles (see Ó'Dúnlaing and Yap [14])."* A maximum spanning tree gives exactly the bottleneck (maximin) path. Ó'Dúnlaing & Yap 1985 itself is **[UNVERIFIED]** (not read).
- **Geraerts & Overmars**, "Creating High-quality Paths for Motion Planning", IJRR 26 (2007), read at https://webspace.science.uu.nl/~gerae101/pdf/hqpijr.pdf. §1.2 "Path Clearance", PDF p. 3: *"paths on the GVD have appealing properties such as large clearance from obstacles."* Also: *"Kim et al. (2003) use an augmented version of Dijkstra's algorithm ... The minimum clearance along the path is maximized"*.
- **Huang, Su & Au**, "Passage-traversing optimal path planning", arXiv 2506.23614v2, read at https://arxiv.org/pdf/2506.23614. §4, PDF p. 8, Eq. 13: *"Since the narrowest passage gauges the most constrained region the path should avoid, the minimum passage width passed by σ is an important metric"*, with `f_p(σ) = min ||P_i-j||`. The same page warns that mixing length with min-width (Eq. 14) is *"not order-preserving"*. Ties in a pure maximin need a deterministic secondary key, such as path length.
- **STEP**, §III-A, PDF p. 3, Eq. 4: path risk is *"the dynamic, time-consistent risk metric given by compounding the one-step risk metrics"*. This is a CVaR sum along the path, so it is a **different aggregation**.
  - STEP also has a per-state hard limit. §IV-E, PDF p. 6: *"we locate areas on the map where the risk ρ is greater than the maximum allowable risk. These areas are marked as obstacles"*. A per-state threshold is a bottleneck-type feasibility test.
  - "Collision" is one risk factor, *"quantified by the distance to the closest obstacle point"* (§IV-B, PDF p. 4).
- **PLGRIM** (Kim et al., ICAPS 2021, arXiv 2102.05633, read at https://arxiv.org/pdf/2102.05633). PDF p. 6, Eq. 12: `C = k_d d_ij + k_ρ ρ_ij + k_μ μ_ij(q)`. Edge risk is **additive**, a different aggregation.
- **NeBula** (Agha et al., arXiv 2103.11470) uses Riskmap/IRM edge risks. I found no maximin formulation in it.
- **Net.** "The minimum clearance along the route is at least rho_r" is the textbook test for whether a disc of radius rho_r can pass (the Voronoi / maximum-spanning-tree result). That is a sound basis for a *human-passability* claim. The SubT risk planners rank paths by summed risk. If Ariadne needs to rank goals, maximin plus a length tiebreak is defensible but is a **[DESIGN]** choice.

### 2(b) How do SOTA planners treat clearance/risk of a goal at the frontier? Neighbourhood or standoff?

**Verdict: SILENT on taking the MAX over a neighbourhood. SOTA uses a standoff plus a whole-body (all-cells) check at the pose actually commanded.**
- **FUEL.** Viewpoints are sampled 1.5-2.5 m from the frontier-cluster centre and rejected if any unknown voxel lies within 0.21 m (`isNearUnknown`, see 1(a)). This is a standoff plus a min-over-neighbourhood test.
- **NBVP.** AuRo 2018, §4 frontier baseline, PDF p. 8: *"To allow collision-free placement of the vehicle, voxels up to half the collision box diagonal from the frontier cells are considered as well."* The goal is moved back from the frontier.
- **GBPlanner2.** The robot box at a vertex must be `kFree` with `stop_at_unknown_voxel=true` (see 1(b)).
- **EAAE.** Elskamp et al., ICUAS 2026, arXiv 2603.15604, read at https://arxiv.org/pdf/2603.15604. §III, PDF p. 3: *"we check reachability around each cluster centroid by examining neighbouring voxels and rejecting candidates whose surrounding space contains known occupied voxels ... sample multiple candidate viewpoints around each feasible cluster centroid and retain only those located in free space."*
- **Gap in the current fix.** Max-over-disc gives the goal the *best* bottleneck of any cell within 0.5 m. The agent might be commanded to a cell that never achieves that value. That is a consistency problem, and it touches the egress honesty rule.
- **SOTA-aligned alternative [DESIGN].** Snap the goal to the argmax-bottleneck known-free cell inside the disc, command that cell, and compute risk at the commanded cell. This keeps the "goals sit at the frontier" insight and matches the FUEL/NBVP standoff pattern.

---

## FIX 3 (pending): return distance counts the agent's swept footprint along its flown track as free unless now observed occupied

### 3(a) GBPlanner homing: how is the path computed, and what is the rule?

**Verdict: SUPPORTS using the traversed track as graph support.**
- **FSR 2019 GBPlanner**, §5.2 "Global Planner", PDF p. 10: *"the global planner utilizes the incrementally sampled local graphs {G_L} alongside the actual robot odometry to build the global graph G_G. At every iteration, G_G is utilized to identify a return-to-home path σ_RH ... This homing path is engaged when the robot endurance is such that continuation of the exploration mission and return to home is not possible."*
- **GBPlanner2**, §IV-A, PDF p. 3: the global planner is triggered *"to ensure that the robot returns home within its endurance limits."*
- **GBPlanner2**, §IV-B (COHORT), PDF p. 4: subgraphs are built from free samples *"alongside sampled points over the trajectory of the robot."*
- **GBPlanner2 code**, `rrg.cpp` @ `7301b5350915`, `getBestPath()`:
  - `time_remaining = std::min(time_budget_remaining, current_battery_time_remaining_)`
  - `time_to_home = homing_len / planning_params_.v_homing_max`
  - `const double kTimeDelta = 20; if (time_to_home > time_remaining - kTimeDelta) { ... "REACHED TIME LIMIT: HOMING ENGAGED." }`
  - This is an **additive 20 s margin**, not Ariadne's ×1.3 factor.
  - The *commanded* best path is added to the global graph (`addRefPathToGraph(global_graph_, ref_vertices)`). That path was checked kFree with the robot box when it was planned. The comment there reads *"The whole path is collision free already"*. A link of up to 0.5 m may be added *"Blindly"*.
  - So GBPlanner2 certifies these cells by **observation at plan time**. FSR 2019 says "actual robot odometry" and PLGRIM uses pose-graph breadcrumbs, but neither says how those cells are validated.
  - Fix 3 would certify cells **because the agent physically passed through them, not because they were observed**. That gap is why limiting the swept cells to rho_a and d_ret only is load-bearing.
- **Direct precedent for "occupied but never seen" (the bootstrap case, which is exactly Ariadne's unobserved entrance):**
  - **GBPlanner2** `Rrg::setState()`: `// Clear free space based on current voxel size.` and `map_manager_->augmentFreeBox(current pos, robot_box_size_)` for the first `augment_free_voxels_time` (default 5, params.cpp) planner triggers. This certifies the robot's own box as free even though it was never observed.
  - **NBVP** AuRo 2018 PDF p. 12 needs an *"initialization motion to allow the computation of a first collision free path segment."*
  - Both systems apply this only at start-up. Fix 3 extends the same idea along the whole flown track.
- **CERBERUS** (Tranzatto et al., arXiv 2207.04914, read at https://arxiv.org/pdf/2207.04914):
  - §3.3.1, PDF p. 18: *"at each local planning iteration the global planner calculates a path from the current robot location to the home location and commands the same to the robot if the remaining endurance is not enough to perform one more local exploration planning step and return home."* This is a one-step lookahead.
  - §3.3.3, PDF p. 20: the Watchdog *"triggers a backtracking of the traversed path, called 'Recovery Homing'"*. The WiFi Safety block *"commands the robot to backtrack its path until the connection is reestablished."*
- **MBPlanner** (Dharmadhikari et al., arXiv 2012.03228), PDF p. 2: *"Feasible paths ... must take into account the remaining endurance of the robot ... or the battery limits are approaching, find a collision-free path σ_H to return the robot to its home location."*
- **PLGRIM**, PDF p. 5: *"Breadcrumb nodes are sampled directly from the Pose Graph, and thus capture the covered traversable space"*.
- **NeBula**, §11.2, PDF p. 42: *"RTB (Return to Base) - Locates the agent within the current IRM and finds the shortest path to the Base IRM node."*
- **Net.** Every SubT homing system I read routes home over a graph that contains the robot's own trajectory, as odometry samples, breadcrumbs or the executed path. Fix 3's idea matches this.
- GBPlanner1 JFR 2020 homing text: **[UNVERIFIED]**.

### 3(b) Footprint clearing in costmap_2d / Nav2

**Verdict: EXISTS, but its meaning differs from Fix 3.**
- Nav2 docs (raw `.md` @ docs.nav2.org `64a93c252eb4`, `costmap_plugins/obstacle.md` and `voxel.md`): `footprint_clearing_enabled`, *bool, Default: true*, *"Clear any occupied cells under robot footprint."*
- In `static.md` the same parameter defaults to *false*, and `restore_cleared_footprint` *"Restore map after clearing the area the footprint occupied."*
- Code: navigation2 @ `94bace6f7bcf`, `nav2_costmap_2d/plugins/obstacle_layer.cpp`: `if (footprint_clearing_enabled_) { setConvexPolygonCost(transformed_footprint_, nav2_costmap_2d::FREE_SPACE); }`.
  - It clears the **current** footprint on each update and overrides even occupied cells. Whether cleared cells then persist as a swept history is **[ASSUMPTION]**; I did not read the layer's update/reset path.
  - It is not an explicit "swept history, unless now occupied" rule.
- **GBPlanner2 analogue:** bootstrap-only `augmentFreeBox`, see 3(a).
- ROS1 costmap_2d wiki text: **[UNVERIFIED]** (not fetched).

### 3(c) Unreachable home (no known path): what do SOTA systems do?

**Verdict: no deliberate policy in the sources read.**
- GBPlanner2 marks the case as a TODO and falls through to exploration. CERBERUS handles related failures (stuck, comms loss) by backtracking the traversed path.
- `rrg.cpp`: `} else { // @TODO Issue with global graph, cannot find a homing path. ROS_WARN(... "Can not find a path to return home from here."); }`.
  - The authors flag this as unresolved, not as a design choice.
  - The code then falls through to the normal local best path. There is no hold and no separate event.
  - Hard stops happen only when `time_budget_remaining <= 0` or `current_battery_time_remaining_ <= 0` (`"RAN OUT OF ... --> STOP HERE."`).
- Because the trajectory is in the global graph (3(a)), this case is rare by construction.
- CERBERUS handles the comms-loss and stuck cases by backtracking the traversed path (3(a)).
- **[DESIGN] implication for Ariadne (Ariadne's own choice, not taken from the literature).**
  - Fix 3 removes the main cause of d_ret = ∞. This is the bootstrap precedent in 3(a), applied along the whole track.
  - Keep a distinct `home_unreachable` event in the decision trace rather than silently treating ∞ as "reserve violated". GBPlanner's TODO fall-through would hide the condition.

### 3(d) Other energy- or return-aware exploration work

- **EAAE** (arXiv 2603.15604): energy is the cost for selecting the next frontier. In what I read, I found no return-reserve rule. **SILENT** on the reserve.
- **RACER** (Zhou et al., arXiv 2209.08533): related work only, PDF p. 2-3, *"Cesare et al. [46] proposed a collaboration scheme that allows robots with low battery to take on the role of a relay"*. **SILENT.**
- **NeBula**, PDF p. 9: team behaviour *"Return to Base, when a battery swap is needed, optimal, and possible at base"*.
- Cesare et al. 2015 and Rappaport et al. 2016 (the AAU host did not respond): **[UNVERIFIED]**.
- **Summary.** I found no ×1.3 multiplicative reserve in any primary source. GBPlanner2 uses an additive 20 s margin over `min(time budget, battery time)`. CERBERUS uses a one-local-step lookahead. Ariadne's 1.3 factor is **[DESIGN]**.

### Project constraint to flag

Swept-track "free" cells are validated only at agent width (rho_a), and only by the fact that the agent physically passed through them. They may feed d_ret and the agent reserve rule only. They must **never** enter rho_responder egress claims (CLAUDE.md invariant 4).

Also, odometry/SLAM drift can shift the swept footprint relative to the map [ASSUMPTION]. Gate the swept layer on map-occupied observations (as proposed), and consider inflating the swept footprint inward (eroding it) by the expected drift.

---

## URLs actually read
- https://arxiv.org/pdf/2010.11561 (FUEL); FUEL code @662dd23c7b52: `frontier_finder.cpp`, `astar2.cpp`, `sdf_map.cpp`, `sdf_map.h`, `exploration_manager/launch/algorithm.xml`, `plan_manage/launch/kino_algorithm.xml`, `topo_algorithm.xml`
- http://web.archive.org/web/20240412220004id_/https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/127860/s10514-016-9610-0.pdf (NBVP AuRo 2018); https://raw.githubusercontent.com/ethz-asl/nbvplanner/master/nbvplanner/src/rrt.cpp @5c07657af18f
- https://easychair.org/publications/preprint/FBld/open (GBPlanner, FSR 2019)
- https://arxiv.org/pdf/2111.06482 (GBPlanner2); gbplanner_ros `gbplanner2` @7301b5350915: `rrg.cpp`, `params.cpp`, `map_manager.h`
- https://arxiv.org/pdf/2207.04914 (CERBERUS); https://arxiv.org/pdf/2012.03228 (MBPlanner appendix)
- https://arxiv.org/pdf/2110.09460 (FAR); http://idm-lab.org/bib/abstracts/papers/tro05.pdf (D* Lite)
- docs.nav2.org raw `.md` @64a93c252eb4 (`obstacle.md`, `static.md`, `voxel.md`, `configuring_navfn.md`, `configuring_smac_2d.md`, `costmap_2d/index.md`); navigation2 `obstacle_layer.cpp` @94bace6f7bcf
- https://arxiv.org/pdf/2103.02828 (STEP); https://arxiv.org/pdf/2103.11470 (NeBula); https://arxiv.org/pdf/2102.05633 (PLGRIM)
- https://arxiv.org/pdf/1706.02939; https://webspace.science.uu.nl/~gerae101/pdf/hqpijr.pdf; https://arxiv.org/pdf/2506.23614
- https://arxiv.org/pdf/2603.15604 (EAAE); https://arxiv.org/pdf/2209.08533 (RACER)

## [UNVERIFIED] list
- NBVP ICRA 2016 conference text
- GBPlanner1 JFR 2020 text
- Ó'Dúnlaing & Yap 1985
- Cesare et al. 2015
- Rappaport et al. 2016
- ROS1 costmap_2d wiki
- Nav2 obstacle-layer persistence of cleared cells ([ASSUMPTION], see 3(b))

Not re-read here: TARE (Cao et al., RSS 2021). The caller had already read §III.
