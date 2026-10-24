"""Beat 7: sharing the work. Rooms become tasks; robots bid (value of what they'd learn minus how far they'd travel);
a conflict is resolved by the higher bid; ownership spreads the team out. CBBA pseudo-code runs alongside.

Mechanism is illustrative on maze_demo_0's real geometry: tasks are the scene's rooms (scene.json topology), travel
is the geodesic on the agent-width GT lattice (scene_gt.npz trav_agent), task values are set by hand so that the
casualty's room is worth most. With one task per robot (L_t = 1) the result equals sequential greedy, which is what
CBBA converges to under D115's score (RESULTS_INDEX §7). Measured table: RESULTS_INDEX §7 (D115, replay):
CBBA/optimum mean 0.994, worst 0.983; greedy pick 0.985 / 0.900; published guarantee ≥ 0.5 (CBBA Theorem 2).
The recorded top-view footage (B's capture) is a placeholder until it lands.

  python/video/render.sh scenes/s07_cbba.py SharingWork
"""
from __future__ import annotations

import heapq
import json
import sys
from pathlib import Path

import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, AnimationGroup, Create, DashedVMobject, FadeIn, FadeOut, LaggedStart,
                   Rectangle, RoundedRectangle, Succession, Text, VGroup)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ariadne_video.data import load_scene  # noqa: E402
from ariadne_video.mobjects import CodePanel, agent_marker, maze_frame, maze_walls, polyline  # noqa: E402
from ariadne_video.style import AGENT, DATA, DIM, FONT, TARGET, TEXT, THREAD, CaptionedScene, tag  # noqa: E402

# RESULTS_INDEX §7 (D115 replay; CBBA Theorem 2)
CBBA_MEAN, CBBA_WORST, GREEDY_MEAN, GREEDY_WORST, GUARANTEE = 0.994, 0.983, 0.985, 0.900, 0.5
VALUE = {"0,0": 8.9, "0,1": 9.0, "2,0": 8.9, "2,1": 14.0, "2,2": 7.3}  # illustrative; casualty room worth most
PER_M = 0.5  # illustrative travel cost per metre
NB = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
      (-1, -1, 2 ** .5), (-1, 1, 2 ** .5), (1, -1, 2 ** .5), (1, 1, 2 ** .5)]


def dijkstra(trav: np.ndarray, src: tuple[int, int]) -> np.ndarray:
    R, C = trav.shape
    d = np.full((R, C), np.inf)
    d[src] = 0.0
    h = [(0.0, *src)]
    while h:
        dd, r, c = heapq.heappop(h)
        if dd > d[r, c]:
            continue
        for dr, dc, w in NB:
            rr, cc = r + dr, c + dc
            if 0 <= rr < R and 0 <= cc < C and trav[rr, cc] and (dr == 0 or dc == 0 or (trav[r + dr, c] and trav[r, c + dc])):
                if dd + w < d[rr, cc]:
                    d[rr, cc] = dd + w
                    heapq.heappush(h, (dd + w, rr, cc))
    return d


def descend(d: np.ndarray, start: tuple[int, int]) -> list[tuple[int, int]]:
    path, (r, c) = [start], start
    while d[r, c] > 0:
        r, c = min(((r + dr, c + dc) for dr, dc, _ in NB), key=lambda q: d[q])
        path.append((r, c))
    return path


class SharingWork(CaptionedScene):
    def construct(self):
        # placeholder for the recorded top-view panel (B's capture of the demo recording)
        ph = Rectangle(width=9.0, height=4.8, stroke_color=DIM, stroke_width=2).shift(UP * 0.5)
        ph = VGroup(DashedVMobject(ph, num_dashes=80),
                    Text("recorded run: top view (pending B)", font=FONT, font_size=26, color=DIM).move_to(ph))
        self.say("With one shared map, the robots can share the work too.", run_with=[FadeIn(ph)])
        self.play(FadeOut(ph), run_time=0.5)

        sc = load_scene(DATA / "maze_demo_0")
        topo = json.loads((DATA / "maze_demo_0" / "scene.json").read_text())["topology"]["rooms"]
        z = np.load(DATA / "maze_demo_0" / "scene_gt.npz")
        ta = z["trav_agent"]
        ox, oy = sc.origin_xy_m
        cell = lambda x, y: (int((y - oy) / sc.cell_m), int((x - ox) / sc.cell_m))  # noqa: E731
        tasks = {k: ((v[0] + v[2]) / 2, (v[1] + v[3]) / 2) for k, v in topo.items()}
        names = sorted(tasks)
        starts = [tuple(e["start_xy_m"]) for e in sc.starts]
        fields = [dijkstra(ta, cell(*s)) for s in starts]
        dist = {(a, k): fields[a][cell(*tasks[k])] * sc.cell_m for a in range(3) for k in names}
        score = {(a, k): VALUE[k] - PER_M * dist[a, k] for a in range(3) for k in names}

        f = maze_frame(sc, height=5.7, center=(-3.25, 0.6, 0))
        walls = maze_walls(sc, f, opacity=0.8)
        robots = VGroup(*[agent_marker(e["agent"], f, *e["start_xy_m"], np.deg2rad(e["start_yaw_deg"]))
                          for e in sc.starts])
        chip = tag("illustrative", "on the real maze").to_edge(UP, buff=0.2).to_edge(LEFT, buff=0.3)
        code = CodePanel("CBBA · consensus-based bundle algorithm", [
            "each robot: add the task with the best bid",
            "share winning bids with neighbours",
            "conflict? the higher bid keeps the task",
            "the loser drops it, and bids again",
            "repeat until every robot agrees",
        ], font_size=16).to_edge(RIGHT, buff=0.3).shift(UP * 0.7)
        self.play(FadeIn(walls), FadeIn(robots), FadeIn(chip), FadeIn(code), run_time=0.8)

        cards = {}
        for k in names:
            x, y = tasks[k]
            box = RoundedRectangle(corner_radius=0.05, width=0.42, height=0.42, stroke_color=TEXT, stroke_width=2,
                                   fill_color="#151a22", fill_opacity=0.9).move_to(f.p(x, y))
            val = Rectangle(width=0.1, height=0.32 * VALUE[k] / 14.0, stroke_width=0, fill_color=THREAD,
                            fill_opacity=0.9).move_to(box).align_to(box, DOWN).shift(UP * 0.05)
            cards[k] = VGroup(box, val)
        self.say("Every unexplored doorway becomes a task,",
                 run_with=[LaggedStart(*[FadeIn(cards[k], scale=1.3) for k in names], lag_ratio=0.2, run_time=1.4)])

        # round 1: each robot's best task, drawn as its route on the agent-width lattice
        first = {a: max(names, key=lambda k: score[a, k]) for a in range(3)}

        def route(a, k, opacity=0.9):
            path = descend(fields[a], cell(*tasks[k]))[::-1]
            return polyline(f, sc.cell_to_xy(np.array(path)), AGENT[a], width=4, opacity=opacity)

        bids = {a: route(a, first[a]) for a in range(3)}
        self.say("and the robots bid for them,",
                 run_with=[code.focus(0), AnimationGroup(*[Create(bids[a]) for a in range(3)], run_time=1.6)])
        rule = VGroup(Text("bid = what it would learn − how far it would travel", font=FONT, font_size=18, color=TEXT),
                      tag("illustrative")).arrange(DOWN, buff=0.12, aligned_edge=RIGHT)
        rule.next_to(code, DOWN, 0.3).align_to(code, RIGHT)
        self.say("each weighing what it would learn against how far it would have to travel.",
                 run_with=[FadeIn(rule), code.focus(1)])

        # conflict: two robots want the same room; the higher bid keeps it, the other bids again
        owner, taken = {}, set()
        for a, k in sorted(((a, first[a]) for a in range(3)), key=lambda p: -score[p]):
            if k not in taken:
                owner[a], _ = k, taken.add(k)
        losers = [a for a in range(3) if a not in owner]
        for a in losers:
            owner[a] = max((k for k in names if k not in taken), key=lambda k: score[a, k])
            taken.add(owner[a])
        clash = [k for k in names if sum(first[a] == k for a in range(3)) > 1]
        flash = [cards[k][0].animate(run_time=0.4).set_stroke(TARGET, width=4) for k in clash]
        rebids = [route(a, owner[a]) for a in losers]
        own = [cards[owner[a]][0].animate(run_time=0.6).set_stroke(AGENT[a], width=5) for a in range(3)]
        self.say("So they spread out instead of crowding into the same corridor.",
                 run_with=[Succession(AnimationGroup(*flash, code.focus(2)),
                                      AnimationGroup(*[bids[a].animate.set_stroke(opacity=0.15) for a in losers],
                                                     code.focus(3), run_time=0.6),
                                      AnimationGroup(*[Create(r) for r in rebids], run_time=1.2),
                                      AnimationGroup(*own, code.focus(4)))])
        self.clear_caption()

        tbl = VGroup(
            Text("allocation quality vs the exact optimum (replay)", font=FONT, font_size=19, color=TEXT),
            VGroup(Text("CBBA", font=FONT, font_size=19, color=THREAD),
                   Text(f"mean {CBBA_MEAN:.3f} · worst {CBBA_WORST:.3f}", font=FONT, font_size=19, color=THREAD)),
            VGroup(Text("greedy pick", font=FONT, font_size=19, color=DIM),
                   Text(f"mean {GREEDY_MEAN:.3f} · worst {GREEDY_WORST:.3f}", font=FONT, font_size=19, color=DIM)),
            VGroup(Text("CBBA's guarantee", font=FONT, font_size=19, color=DIM),
                   Text(f"≥ {GUARANTEE:.1f}", font=FONT, font_size=19, color=DIM)),
        )
        for r in tbl[1:]:
            r.arrange(RIGHT, buff=0.3)
        tbl.arrange(DOWN, aligned_edge=LEFT, buff=0.14)
        chips = VGroup(tag("measured", "D115, replay"), tag("published", "CBBA Theorem 2")).arrange(RIGHT, buff=0.15)
        block = VGroup(tbl, chips).arrange(DOWN, aligned_edge=LEFT, buff=0.2).next_to(code, DOWN, 0.35).align_to(
            code, LEFT)
        self.play(FadeOut(rule), FadeIn(block, shift=UP * 0.1), run_time=0.7)
        self.wait(3.0)
