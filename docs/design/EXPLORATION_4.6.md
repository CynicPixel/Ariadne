# 4.6 TARE-style coarse/fine scheduling and a FAR-style visibility graph for agent motion (A1)

Status: **draft for review** (advisor), before any code. Decision: D116. Arch §21.5, §21.6, §22.3 (the split),
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
  1. If the local horizon holds views, cover them: the greedy set-cover over their targets, then the shortest
     open tour from the agent (exact enumeration up to 8 views, else nearest-neighbour + 2-opt). The goal is the
     first view on that tour.
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
- **Graph vertices:** the convex polygon vertices, i.e. corners a shortest path can bend around (reflex vertices
  never lie on a shortest path). Each is offset by one cell into free space.
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
| D\* Lite (production egress) | D113's ρ_responder lattice | target → exits, incremental |
| FAR-style graph | plan space at ρ_responder (for this row only, so the two answer the same question) | target → exits |
| A\* on the lattice | the same lattice | fresh search (the oracle) |

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

1. **Visibility graph = exact shortest any-angle path on polygonal maps:** random rectangles, start and goal; the
   graph route's length equals a reference visibility-graph shortest path built by brute force (every vertex pair
   tested) within ε. It is never longer than the string-pulled A\* path + ε.
2. **Incremental = full rebuild:** after random local changes, the updated graph's route lengths equal a
   from-scratch build's.
3. **No edge crosses a blocked cell:** densely sample every edge on random maps.
4. **Attemptable edges only when asked, and flagged.**
5. **TARE scheduling:**
   - with local views present, the goal is the first view of the exact shortest open tour on small instances;
   - with none, the goal lies in the first subspace of the global tour;
   - deterministic.
6. **Subspace states on a hand map:** exploring / explored / unexplored.

## 5. Evaluation

- **Replay (now):** the §3 comparator rows on the evidence spool's replans, and TARE's decision log beside 4.3's
  (machinery evidence, as §7 of NBV_4.3).
- **Gate:** the TARE row in the lockstep L1 ablation (NBV_4.3 §5's pre-registered axes).

## 6. Open points for review

1. S = 12 coarse cells and a horizon half-width of 2.84 m, both derived from the depth gate.
2. Convex vertices offset by one cell, and ε = 1 cell for `approxPolyDP`.
3. Running the FAR comparator row at ρ_responder so it answers the same query as D\* Lite.
