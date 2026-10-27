# Exploration scheduling (TARE-style) and visibility-graph motion (FAR-style)

Status: **reviewed** (design review 2026-10-23, §6), before any code. Decision: D116. Arch §21.5, §21.6, §22.3 (the split),
§31.6 (planner rows and the route-query comparator). algorithms.md §22.2 (TARE, FAR, read in primary text).

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
       one view per target (design review).
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
  - **Orientation (design review):** `findContours` with `RETR_CCOMP` returns outer contours and holes with
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
| D\* Lite, **shortest** (all weights 0, egress.md §2's ablation row) | D113's ρ_responder lattice | target → exits, incremental |
| FAR-style graph | the same ρ_responder traversable set, as polygons (this row only, so the two answer one question) | target → exits |
| A\* on the lattice | the same lattice | fresh search (the oracle) |
| D\* Lite, weighted (production, w_u = w_c = 1) | the same lattice | reported separately: it minimises cost, not length, so it is never graded on length (design review) |

- **Latency counts only each structure's own update and query.** ComputeTraversability + BodySupport (≈ 0.5 s)
  are shared by every row and excluded, or the split's cost would be buried under them (design review).

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
  (machinery evidence, as §7 of next-best-view.md).
- **Gate:** the TARE row in the lockstep L1 ablation (next-best-view.md §5's pre-registered axes).

## 6. Design review (2026-10-23)

Blocking points, all adopted:
1. The comparator grades on the shortest-path D\* Lite row and times only each structure's own work (§3).
2. No degenerate set cover; NN + 2-opt always, with exact enumeration as the oracle (§1).
3. Vertex selection is orientation-aware; the reference uses all vertices; ε is defined per bend (§2, §4).

Non-blocking points:
- **Attemptable FAR routes:** kept to the comparator and exploration logging until B confirms the mover accepts
  segments through unknown space. `Route` never emits one before that.
- **Gate cost:** the TARE row makes 6 policies × 2 scenes × 3 seeds = 36 lockstep runs, ≈ 14 h on llvmpipe
  against the ≈ 10 h first stated. Added to next-best-view.md §5.
- **ci_tiny is about one subspace** at S = 6 m, so its TARE row is nearly local-only; a null result there is not
  over-read.

## 7. Facts from the runtime (2026-10-23)

- **The mover accepts segments through unknown cells.** Its check is against the scene's ground-truth ESDF, not
  the estimated map. A refusal holds the whole command, and the runtime then excludes goals within
  `refused_radius_m`.
- So FAR's attemptable edges are executable. A refusal is the sim's ground-truth physical check standing in for
  collision ("hit unseen geometry"); it is not perception, and is stated as such.
- **`linked` is always true in L1 today.** D102's comms model isn't wired into L1, and the trace says so. CBBA's
  disconnected regimes therefore come from the synthetic tests only.

## 8. As built: TARE scheduling (2026-10-23)

- **Policy:** `NbvPolicy::kTare` (`cpp/planning/mission/nbv.cpp`, `TareSchedule`).
  - Views are ranked as Ariadne's.
  - **Local:** the best view per target inside the depth-gate box, then an NN + 2-opt open tour on geodesics within
    that box.
  - **Global otherwise:** the best view per exploring 6 m subspace, and a tour on the full plan space's geodesics.
  - The scheduled view moves to the front; `NbvPlan.tare` records "local k" or "global m".
- **Tour quality, measured against exact enumeration** (200 random instances, ≤ 8 nodes): NN + 2-opt is at worst
  1/0.811 ≈ 1.23× the optimum.
  - Reported, not tuned.
  - Or-opt or exact local tours are possible refinements, each needing its own row.
- **Allocation for baseline rows:** only the Ariadne row allocates by auction. The auction's values are Ariadne's,
  so baseline policies fed to it would be re-scored as Ariadne and the §31.6 rows would collapse. Baselines,
  including TARE, coordinate by the exactly-once greedy pick over their own ranking.

## 9. §31.6 comparator, as measured (2026-10-23; `evaluation/runs/planning/route_compare/`)

Evidence spool, 40 replans. A route exists from 145 s, so 12 replans route; all three structures find the same
ones, and every route is valid (each segment re-checked by symmetric line of sight).

| Row | Length (12 routed replans) | Own time per replan |
|---|---|---|
| D\* Lite, shortest (incremental) | lattice 4.441 m, string-pulled 4.403 m | mean 0.46 ms, max 1.36 |
| FAR-style graph | **4.403 m** (any-angle) | update mean 0.7 ms (max 3.9), query 0.01 ms; ≤ 31 nodes; mean 65 line-of-sight tests |
| A\* fresh (oracle) | lattice 4.441 m, string-pulled 4.403 m | mean 0.04 ms |

- **FAR's any-angle length equals the string-pulled lattice paths**, 0.9 % shorter than the raw 8-connected
  lattice.
- **On this spool the split costs nothing measurable.** Every structure takes milliseconds.
- **A fresh A\* is the fastest:** the route is short (≈ 90 cells) and nearly straight, so incremental bookkeeping
  doesn't pay here. That is stated, not over-read.
- The comparison only discriminates on larger or more cluttered scenes (maze_short, the hazard seeds), and on
  frequent local changes. Those rows come with their spools.

## 10. Hazard seeds, and an open FAR defect (2026-10-23; `evaluation/runs/egress/hazard_replan/s*_w0/route_compare.jsonl`)

- **Seeds 1, 2:** FAR's length equals the string-pulled A\* on every routed replan (9 / 9, 11 / 11), all valid.
- **Seed 0:** FAR is shorter on 11 of 13 replans, but **longer on the last two, 11.115 m vs 4.424 m**. FAR misses a
  passage the lattice path uses, which violates §4's derived bound.
  - Suspected (`[UNVERIFIED]`): a corner node at a narrow gap whose offset cell is blocked gets dropped, leaving no
    visibility through the gap.
  - **Open bug, to fix with a regression test made from this map** (the run's `trav_responder.png`).
  - The gate rows are unaffected: `Route` uses A\* plus string-pulling, not FAR.

## 11. Literature review of the post-design fixes (2026-10-23; the literature notes `phase4_fixes_route_cbba` (archive/dev-history branch))

**Amended 2026-10-24 (D123, D116): "FAR ≤ string-pulled lattice" is FALSE in general.** It holds on the 12
evidence-spool routes above. On the unit test's random maps, once their draws are sequenced (D123), trial #9 gives
37.40 cells against 33.24 (+12.5 %). That is the same defect class as hazard seed 0 below. The test is quarantined
and OPEN, and no report claims FAR ≤ lattice.

**The lattice join (`749187f`): our claim about FAR was wrong; the join is `[DESIGN]`.**
- 749187f said the join works "as FAR joins start/goal through its local planner". FAR §III-C (p. 3) says
  otherwise: "The planner first add p_robot and p_goal as two vertices … and connect them to the vertices … with
  non-blocking visibility edges. Then, a breath-first search is run". The "local planner [34]" is the waypoint
  follower that runs after FAR.
- FAR's code (`graph_planner.cpp` @2799b69) handles the hard cases differently:
  - goal edges are checked only against observed obstacles, so unknown space is passable ("attemptable");
  - a goal inside an obstacle is moved to the nearest free cell by grid BFS (`ReEvaluateGoalPosition`);
  - with no path, it replays the last path, sends the raw goal in attemptable mode, or reports failure.
- Our comparator runs on the known ρ_responder lattice and cannot attempt through unknown space, so the lattice join
  is our own substitute. It keeps the comparator's routes valid on the known map.
- §9–10's comparator numbers stand: they measured our graph, labelled FAR-style.
- **Two further deviations, recorded:**
  - FAR prunes long edges that point into polygons (Fig. 3); ours does not;
  - FAR inflates obstacles by the vehicle size; we offset nodes 1 cell instead.
- **The gate is unaffected:** `Route` and every NBV row use A\* plus string-pulling, not this graph.

**Line of sight checked in both directions (`cdbc6b1`): conservative and safe, `[DESIGN]`.**
- Amanatides & Woo (1987) visit exactly the cells a ray crosses. Theta\* (JAIR 2010, App. A) defines line of sight
  geometrically, which makes it symmetric. Neither is what `cv::LineIterator` 4-connected does.
- Port of OpenCV 5 `LineIterator` (`[ASSUMPTION]`: it matches the pinned tag), over 624 centre-to-centre segments:
  - the two directions differ on 392 segments;
  - one direction alone misses a crossed cell on 272 segments, so it is unsafe;
  - the union of both directions misses none, but over-blocks (includes untouched cells) on 272.
- Kept for now: it is safe, and it affects `StringPull` identically in every gate row.
- An exact order-free traversal with Theta\*'s corner rule would remove the over-blocking. That is a post-gate change.

**D\* Lite's change trigger (4.5): supported.**
- D\* Lite assumes nothing about whether costs "go up or down" (AAAI 2002, p. 4).
- Its main loop is "for all directed edges (u, v) with changed edge costs … UpdateVertex(u)", and detecting the
  change is left to the caller.
- `DStarLite::Update` applies every changed cell on every replan. `new_obstacle` and the other triggers are trace
  labels only.

## 12. Gate result: the TARE row (2026-10-24; `evaluation/runs/planning/nbv_gate/`, next-best-view.md §14.1)

These are 6 TARE runs plus 6 runs of the 1.8 nearest-frontier row, on the same rig and commit as the 4.3 rows: AWS
c7i with llvmpipe, RTF 0 with D168, a1p `d59ee3d`, every job exit 0.

**TARE-inspired scheduling (D116) is level with Ariadne on AUC and not ahead of every row.**

| | ci_tiny AUC | maze_short AUC |
|---|---|---|
| tare | **0.895** | 0.443 |
| ariadne | 0.884 | 0.440 |
| coverage_greedy | 0.892 | **0.495** |
| nearest_frontier_1.8 | 0.789 | 0.237 |

- **ci_tiny** (about one 6 m subspace, so nearly local-only, as pre-registered in §6): TARE has the best AUC. It is
  the best row at T = 60 s (0.901), below Ariadne from T = 80 s, and 0.960 against 0.962 at 200 s. This scene does not
  test the global tour; it is a null result for TARE's global layer, not evidence for or against it.
- **maze_short:** TARE is above Ariadne at T = 20 s and T = 80–160 s, and below it at 180–200 s (0.681 vs 0.757). It
  is below coverage_greedy at every T.
- **Per metre flown:** on ci_tiny, TARE and Ariadne are within 0.02 of each other at every point from 5 to 30 m. On
  maze_short they are within 0.01 up to 20 m, then Ariadne is ahead (0.556 vs 0.535 at 25 m, 0.641 vs 0.592 at 30 m).
- **The 1.8 row** (the D45 placeholder) is the fastest early on ci_tiny (0.522 at 20 s), then plateaus at 0.82. It is
  last on both scenes by AUC.

Neither 4.6 row changes the 4.3 verdict, which stays false.
