# 4.6 TARE-style coarse/fine scheduling and a FAR-style visibility graph for agent motion (A1)

Status: **reviewed** (advisor 2026-10-23, §6), before any code. Decision: D116. Arch §21.5, §21.6, §22.3 (the split),
§31.6 (planner rows and the route-query comparator). ALGORITHMS §22.2 (TARE, FAR, read in primary text).

## 0. What 4.6 owes

- **§31.6 planner row "TARE-inspired":** a policy that schedules exploration with a sparse global layer and a dense
  local one. It is a row beside random, greedy frontier, coverage utility, risk-aware utility and Ariadne.
- **§31.6 route-query comparator, D\* Lite vs a FAR-style visibility graph.** §22.3 states what is measured:
  > the cost of the split — replan latency and path quality for each, on the same scenarios.

  Neither replaces the other. D\* Lite stays the egress planner (D113); FAR is the exploration-time router.
- **Plan 4.6:** "TARE-style coarse/fine scheduling and FAR-style route graph for agent motion". It runs at ρ_agent:
  agent motion, never an egress claim (invariant 4).

## 1. TARE-style scheduling (arch §21.5; TARE, Cao et al. RSS 2021)

**TARE, as read:**
- **Local horizon:** sample viewpoints that cover the uncovered surfaces, choose a small covering set, and run a
  TSP through it.
- **Global layer:** subspaces marked unexplored / exploring / explored, with a TSP over the exploring ones.
- The two paths join at the horizon boundary.

**Ariadne's adaptation, on structures that already exist:**
- **Global layer:** D110's 0.5 m coarse layer, grouped into subspaces of S × S coarse cells.
  - S is set so a subspace is one horizon wide: S = ⌈2·depth gate / 0.5 m⌉ = ⌈5.68 / 0.5⌉ = 12 cells (6 m).
    It's derived from the rig, not tuned.
  - **Exploring:** the subspace holds a 4.3 target (frontier, uncertainty or egress).
  - **Explored:** observed cells, no target.
  - **Unexplored:** no observed cell.
- **Local layer:** the 4.3 views whose goal lies inside the agent's local horizon, a square of half-width = the
  depth gate (2.84 m) around the agent.
- **Scheduling:**
  1. If the local horizon holds views: take the best view per local target, by D114 utility, then the open tour
     from the agent over those views.
     - **No set cover.** In 4.3 each view belongs to exactly one target, so covering the local targets reduces to
       one view per target (advisor review).
     - **Solver:** nearest-neighbour + 2-opt, always. Exact enumeration is the test oracle only, with the gap
       reported. There is no size threshold to tune.
     - The goal is the first view on that tour.
  2. Otherwise, take the global tour over the exploring subspaces' centres from the agent (the same TSP), and go to
     the best view inside the first subspace on it.
- **Distances** are ρ_agent geodesics (`GridDistances` on the plan space), never Euclidean, so a tour never
  assumes a wall away.
- It plugs in as `NbvPolicy::kTare`. The generators, views and IG are shared, so the §31.6 rows differ only in
  their scheduling.

## 2. FAR-style visibility graph (arch §21.6; FAR, Yang et al. IROS 2022)

**FAR, as read:** polygons from the edge points around obstacles, and a dynamically updated two-layer visibility
graph: a local layer merged into the global one. Routes may be "attemptable" through unknown space, with fast
replanning.

**Ariadne's adaptation**, on the 2D plan space at ρ_agent (§22.6 OpenCV adjacency):
- **Obstacles:** connected components of not-traversable cells, by `cv::findContours` (`RETR_CCOMP`, so holes
  count), simplified by `cv::approxPolyDP` with ε = 1 cell.
- **Graph vertices:** the polygon vertices that bulge into free space, i.e. corners a shortest path can bend around.
  Corners that open toward free space never lie on a shortest path.
  - **Orientation (advisor review):** `findContours` with `RETR_CCOMP` returns outer contours and holes with
    opposite winding. The free-space enclosure is a hole, so a single cross-product sign would keep exactly the
    wrong corners there.
  - Each vertex is therefore classified by its interior angle on the **blocked** side: kept iff that angle is
    < 180°, with the sign test flipped for hole contours.
  - Each kept vertex is offset by one cell into free space, along the bisector.
- **Edges:** pairs of vertices with line of sight, by 4-connected `cv::LineIterator` on the plan space, with the
  same no-corner-cut rule as `Route`. The edge cost is Euclidean length.
- **Dynamic update:**
  - each replan re-extracts the polygons only inside the boxes where the plan space changed (the diff of
    consecutive plan spaces);
  - it removes the vertices and edges touching those boxes and re-tests the line of sight of edges that cross
    them.
  - That is the FAR property the comparison is about: an update cost that scales with the change, not the map.
- **"Attemptable":** a second edge set where unknown cells count as free (`slab == kUnknown` added to the plan).
  It's used only for exploration routes, flagged per route, and never for egress (invariant 4).
- **Query:** connect start and goal to their visible vertices, then A\* over the graph with a Euclidean heuristic.
- **Not taken from FAR:** its terrain and 3D handling (we are 2D at ρ_agent by construction), and its
  free-space-boundary sensing pipeline (we have D110's map).

## 3. The §31.6 comparator: replan latency and path quality

**Same scenarios:** the evidence spool's replans, and the 4.5 exit-gate scenario test (obstacle after route).

| Row | Structure | Query |
|---|---|---|
| D\* Lite, **shortest** (all weights 0, EGRESS_4.5 §2's ablation row) | D113's ρ_responder lattice | target → exits, incremental |
| FAR-style graph | the same ρ_responder traversable set, as polygons (this row only, so the two answer one question) | target → exits |
| A\* on the lattice | the same lattice | fresh search (the oracle) |
| D\* Lite, weighted (production, w_u = w_c = 1) | the same lattice | reported separately: it minimises cost, not length, so it is never graded on length (advisor review) |

- **Latency counts only each structure's own update and query.** ComputeTraversability + BodySupport (≈ 0.5 s)
  are shared by every row and excluded, or the split's cost would be buried under them (advisor review).

- **Columns:**
  - update ms per replan, and query ms;
  - path length vs the A\* optimum on the same lattice;
  - validity (every segment's cells traversable, re-checked);
  - graph size.
- **Expectation, stated before measuring:** FAR's paths are near-optimal any-angle paths. Shorter than 8-connected
  lattice paths is possible, since octile paths overestimate the Euclidean optimum by up to ≈ 8 %. Its updates
  should beat a fresh A\* when changes are local.
- **Path quality is compared as Euclidean length of the executed polyline.** A lattice path is string-pulled (the
  `Route` rule) before comparison, so any-angle and lattice are measured alike.

## 4. Tests

1. **Visibility graph = exact shortest any-angle path on polygonal maps:** random rectangles, holes included,
   with start and goal.
   - The graph route's length equals a reference visibility graph over **all** polygon vertices (no convexity
     filter, every pair tested by brute force), within ε. So a wrong orientation test fails it.
   - It is never longer than the string-pulled A\* path + ε.
   - ε per bend = 2δ, with δ = offset·√2 cells, the node's distance from its true corner (along the diagonal
     bisector). By the triangle inequality, passing through the node instead of the corner adds at most 2δ.
     - That's 2√2 cells per bend at offset 1.
     - Derived, after a first guess of 1 cell failed on one random map (a 64.17 vs 63.20 route).
2. **Incremental = full rebuild:** after random local changes, the updated graph's route lengths equal a
   from-scratch build's.
3. **No edge crosses a blocked cell:** densely sample every edge on random maps.
4. **Attemptable edges only when asked, and flagged.**
5. **TARE scheduling:**
   - with local views present, the goal is the first view of the NN + 2-opt tour. On small instances, the tour's
     length is compared with exact enumeration and the gap reported;
   - with none, the goal lies in the first subspace of the global tour;
   - deterministic.
6. **Subspace states on a hand map:** exploring / explored / unexplored.

## 5. Evaluation

- **Replay (now):** the §3 comparator rows on the evidence spool's replans, and TARE's decision log beside 4.3's
  (machinery evidence, as §7 of NBV_4.3).
- **Gate:** the TARE row in the lockstep L1 ablation (NBV_4.3 §5's pre-registered axes).

## 6. Review (advisor, 2026-10-23)

Blocking points, all adopted:
1. The comparator grades on the shortest-path D\* Lite row and times only each structure's own work (§3).
2. No degenerate set cover; NN + 2-opt always, with exact enumeration as the oracle (§1).
3. Vertex selection is orientation-aware; the reference uses all vertices; ε is defined per bend (§2, §4).

Non-blocking points:
- **Attemptable FAR routes:** kept to the comparator and exploration logging until B confirms the mover accepts
  segments through unknown space. `Route` never emits one before that.
- **Gate cost:** the TARE row makes 6 policies × 2 scenes × 3 seeds = 36 lockstep runs, ≈ 14 h on llvmpipe
  against the ≈ 10 h first stated. Added to NBV_4.3 §5, and B and the maintainer are told.
- **ci_tiny is about one subspace** at S = 6 m, so its TARE row is nearly local-only; a null result there is not
  over-read.

## 7. Facts from B (2026-10-23)

- **The mover accepts segments through unknown cells.** Its check is against the scene's ground-truth ESDF, not
  the estimated map. A refusal holds the whole command, and the runtime then excludes goals within
  `refused_radius_m`.
- So FAR's attemptable edges are executable. A refusal is the sim's ground-truth physical check standing in for
  collision ("hit unseen geometry"); it is not perception, and is stated as such.
- **`linked` is always true in L1 today.** D102's comms model isn't wired into L1, and the trace says so. CBBA's
  disconnected regimes therefore come from the synthetic tests only.
