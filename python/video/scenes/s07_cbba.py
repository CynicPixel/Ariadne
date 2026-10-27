"""Beat 7 (≈ 44 s): sharing the work. The project's CBBA (cpp/planning/mission/allocation.cpp RunCbba, unchanged, driven
by tools/cbba_trace.cpp) run on maze_demo_0: three robots, two rooms per robot, an always-connected link.

Instance (tools/cbba_replay.py, the way phase4_planner.cpp BuildAuction builds one): tasks are the scene's five rooms;
value = floor area / its p90 over rooms (a stand-in for information gain); distance = geodesic on the agent-width
lattice; bid = value · exp(-distance / p90 distance), the marginal of the path score. Every round below replays the
recorded snapshots (y, z after each robot's build): build (bids added in bundle order, drawn at their best insertion),
broadcast, resolve (the higher bid keeps; Table I on an all-to-all link), release (eq. 6). Round 1: rooms A and C are
claimed twice; round 2: both losers bid for D (0.4220 vs 0.4234); round 3: no change. The converged allocation equals
the brute-force optimum on this instance (derived). Measured row: docs/results/index.md §7, the recorded-mission replay
(2 robots, one room each); guarantee: Choi, Brunet & How 2009, Theorem 2.

  python3 python/video/tools/cbba_replay.py   # regenerates data/cbba/trace.json
  python/video/render.sh scenes/s07_cbba.py SharingWork
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, AnimationGroup, Create, DashedVMobject, Dot, FadeIn, FadeOut,
                   GrowFromCenter, Indicate, LaggedStart, Line, MoveAlongPath, MoveToTarget, Rectangle,
                   RoundedRectangle, Text, Triangle, Uncreate, VGroup, rush_from, smooth)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ariadne_video.data import load_scene  # noqa: E402
from ariadne_video.mobjects import agent_marker, maze_frame, maze_walls, polyline  # noqa: E402
from ariadne_video.slides import chip_top  # noqa: E402
from ariadne_video.style import AGENT, BG, DATA, DIM, FONT, TARGET, TEXT, THREAD, CaptionedScene, tag  # noqa: E402

TR = json.loads((DATA / "cbba" / "trace.json").read_text())
ROOM = "ABCDE"
CHIP_BOTTOM, CAP2 = 3.4, -2.93  # bottom of the top chip; top of a two-line subtitle
GAP, STRIP_H, MH = 0.3, 0.62, 4.5  # strip-to-content gap; strip height; maze drawing height
GAPX, RW = 0.8, 5.2  # maze-to-column gap; the right column's width
SW, SH = 1.6, 0.46  # a bundle slot
# docs/results/index.md §7, the recorded-mission replay: CBBA = central greedy (SGA) 40/40; CBBA / exact optimum mean, worst
REPLAY_EQ, REPLAY_N, CBBA_MEAN, CBBA_WORST = 40, 40, 0.994, 0.983
STEPS = [("build", "add the best bid while there is room"), ("broadcast", "winning bids and winners"),
         ("resolve", "higher bid keeps · loser releases")]


def held(snap: dict, a: int) -> list[int]:
    """The rooms robot a claims in its own snapshot, in the order added (warped bids never increase along a bundle)."""
    return sorted((j for j, z in enumerate(snap["z"]) if z == a), key=lambda j: -snap["y"][j])


def score(a: int, path: list[int]) -> float:
    """S_i(p) = sum over the path of value · exp(-rate · distance along the path)."""
    s, d, prev = 0.0, 0.0, None
    for j in path:
        d += TR["d0_m"][a][j] if prev is None else TR["leg_m"][prev][j]
        s += TR["value"][j] * math.exp(-TR["rate"] * d)
        prev = j
    return s


def insert(a: int, path: list[int], j: int) -> list[int]:
    """Room j at its best position in the path (BuildBundle's insertion)."""
    return max((path[:k] + [j] + path[k:] for k in range(len(path) + 1)), key=lambda p: score(a, p))


def rounds() -> list[dict]:
    """Per round: what each robot adds (in order), the contested rooms with their (robot, bid) claims, the losers'
    released rooms, and the bundles kept after the round."""
    U, out, keep = len(TR["rounds"][0]), [], [[] for _ in TR["rounds"][0]]
    for snaps in TR["rounds"]:
        bund = [held(snaps[a], a) for a in range(U)]
        assert all(b[:len(k)] == k for b, k in zip(bund, keep)), "a kept bundle changed during a build"
        added = [b[len(k):] for b, k in zip(bund, keep)]
        claims: dict[int, list[tuple[int, float]]] = {}
        for a in range(U):
            for j in bund[a]:
                claims.setdefault(j, []).append((a, snaps[a]["y"][j]))
        contest = {j: c for j, c in claims.items() if len(c) > 1}
        win = {j: max(c, key=lambda ab: (ab[1], -ab[0]))[0] for j, c in contest.items()}  # bid, then the lower id
        lost = [[j for j in bund[a] if j in win and win[j] != a] for a in range(U)]
        release = [b[b.index(ls[0]):] if ls else [] for b, ls in zip(bund, lost)]  # eq. 6: the lost room and after
        keep = [[j for j in b if j not in r] for b, r in zip(bund, release)]
        out.append(dict(added=added, contest=contest, win=win, release=release, keep=keep))
    return out


def cluster(content_h: float, cap_top: float) -> tuple[float, float]:
    """The step strip and the content as one block, centred between the top chip and the subtitle:
    returns (strip centre y, content top y)."""
    top = (CHIP_BOTTOM + cap_top) / 2 + (STRIP_H + GAP + content_h) / 2
    return top - STRIP_H / 2, top - STRIP_H - GAP


class SharingWork(CaptionedScene):
    def construct(self):
        sc = load_scene(DATA / "maze_demo_0")
        rects = json.loads((DATA / "maze_demo_0" / "scene.json").read_text())["topology"]["rooms"]
        R = rounds()
        assert [sorted(r["contest"]) for r in R] == [[0, 2], [3], []], "the narration describes this trace"
        paths = [[] for _ in range(3)]
        for r in R:  # replay the insertions; they must land where the C++ put them
            for a in range(3):
                for j in r["added"][a]:
                    paths[a] = insert(a, paths[a], j)
                paths[a] = [j for j in paths[a] if j not in r["release"][a]]
        assert paths == TR["paths"], (paths, TR["paths"])

        # ---- layout: the step strip over (maze | formula + bundles), one block centred between chip and subtitle ----
        strip_y, ctop = cluster(MH, CAP2)
        f = maze_frame(sc, height=MH, center=(0, ctop - MH / 2, 0))
        w = np.array(sc.walls)
        mw = f.length(np.ptp(w[:, [0, 2]]) + 2.0)  # the maze's drawn width (maze_frame's 1 m margins)
        left = -(mw + GAPX + RW) / 2  # (maze | column) centred horizontally too
        MX, RX = left + mw / 2, left + mw + GAPX + RW / 2
        f = maze_frame(sc, height=MH, center=(MX, ctop - MH / 2, 0))
        walls = maze_walls(sc, f, width=2.5, opacity=0.85)
        robots = VGroup(*[agent_marker(e["agent"], f, *e["start_xy_m"], np.deg2rad(e["start_yaw_deg"]), size=0.12)
                          for e in sc.starts]).set_z_index(4)
        chip = chip_top("illustrative", "the project's CBBA code, run on the generated maze")

        pills = VGroup()
        for i, (name, sub) in enumerate(STEPS):
            b = RoundedRectangle(corner_radius=0.08, width=2.75, height=STRIP_H, stroke_color=DIM, stroke_width=1.5)
            pills.add(VGroup(b, VGroup(Text(f"{i + 1} · {name}", font=FONT, font_size=14, color=TEXT),
                                       Text(sub, font=FONT, font_size=10, color=DIM)).arrange(DOWN, buff=0.05).move_to(b)))
        rbox = RoundedRectangle(corner_radius=0.08, width=2.1, height=STRIP_H, stroke_color=THREAD, stroke_width=1.5)

        def round_label(t: int, sub: str = "repeat until nothing changes", color=DIM) -> VGroup:
            return VGroup(Text(f"round {t}", font=FONT, font_size=14, color=THREAD),
                          Text(sub, font=FONT, font_size=10, color=color)).arrange(DOWN, buff=0.05).move_to(rbox)

        strip = VGroup(*pills, rbox).arrange(RIGHT, buff=0.32).move_to([0, strip_y, 0])
        rlab = round_label(1).move_to(rbox)
        arrows = VGroup(*[Line(strip[i].get_right(), strip[i + 1].get_left(), color=DIM, stroke_width=1.5)
                          for i in range(3)])

        def light(i: int) -> list:
            return [p[0].animate(run_time=0.3).set_stroke(THREAD if k == i else DIM, 3 if k == i else 1.5)
                    for k, p in enumerate(pills)]

        def new_round(t: int, sub: str = "repeat until nothing changes", color=DIM) -> list:
            nonlocal rlab
            old, rlab = rlab, round_label(t, sub, color)
            return [FadeOut(old, shift=UP * 0.15, run_time=0.35), FadeIn(rlab, shift=UP * 0.15, run_time=0.35)]

        # rooms: unexplored floor (tinted) and a badge, over the routes: the letter and the floor area (its value)
        rooms, labels = VGroup(), VGroup()
        for j, k in enumerate(TR["rooms"]):
            x0, y0, x1, y1 = rects[k]
            rect = Rectangle(width=f.length(x1 - x0), height=f.length(y1 - y0), stroke_width=0, fill_color=DIM,
                             fill_opacity=0.22).move_to(f.p((x0 + x1) / 2, (y0 + y1) / 2))
            rooms.add(rect)
            lab = VGroup(Text(ROOM[j], font=FONT, font_size=17, color=TEXT, weight="BOLD"),
                         Text(f"{TR['area_m2'][j]:.0f} m²", font=FONT, font_size=10, color=TEXT)).arrange(DOWN, buff=0.03)
            bg = RoundedRectangle(corner_radius=0.06, width=lab.width + 0.14, height=lab.height + 0.1, stroke_width=0,
                                  fill_color=BG, fill_opacity=0.82)
            labels.add(VGroup(bg, lab.move_to(bg)).move_to(rect))
        rooms.set_z_index(-1)
        labels.set_z_index(6)

        # right column: the bid formula over the bundles every robot holds
        base = Text("bid  =  value × e", font=FONT, font_size=21, color=TEXT)
        expo = Text("−distance / d₉₀", font=FONT, font_size=13, color=TEXT).next_to(base, RIGHT, 0.04).align_to(base, UP)
        expo.shift(UP * 0.1)
        legend = Text("value: floor area it would map · distance: driving distance\n"
                      "d₉₀: the 90th-percentile distance (a common scale)",
                      font=FONT, font_size=12, color=DIM, line_spacing=0.8)
        formula = VGroup(VGroup(base, expo), legend).arrange(DOWN, buff=0.16)
        formula.move_to([RX, 0, 0]).align_to([0, ctop, 0], UP)

        head = Text("what each robot holds", font=FONT, font_size=15, color=DIM)
        rows, slots = VGroup(), []
        for a in range(3):
            mark = Triangle(color=AGENT[a], fill_opacity=1, stroke_width=0).scale(0.11).rotate(-np.pi / 2)
            ss = [DashedVMobject(RoundedRectangle(corner_radius=0.06, width=SW, height=SH, stroke_color=DIM,
                                                  stroke_width=1.2), num_dashes=40) for _ in range(2)]
            slots.append(ss)
            rows.add(VGroup(mark, *ss).arrange(RIGHT, buff=0.22))
        rows.arrange(DOWN, buff=0.14, aligned_edge=LEFT)
        table = VGroup(head, rows).arrange(DOWN, buff=0.2)
        table.next_to(formula, DOWN, 0.4)
        table.set_x(RX)

        # ---- 1 · the map, the robots, the rooms --------------------------------------------------------------------
        self.say_steps("With one shared map, the robots can now split up the work.", [
            [FadeIn(chip), FadeIn(walls, run_time=1.0), FadeIn(strip[:3], run_time=1.0), FadeIn(rbox), FadeIn(rlab),
             FadeIn(arrows)],
            [LaggedStart(*[GrowFromCenter(r) for r in robots], lag_ratio=0.3, run_time=1.2)]])
        self.say_steps("Every unexplored room becomes a task, worth its floor area.", [
            [LaggedStart(*[AnimationGroup(FadeIn(rooms[j]), FadeIn(labels[j], scale=1.4)) for j in range(5)],
                         lag_ratio=0.18, run_time=1.6), FadeIn(table, run_time=1.0)]])

        # ---- 2 · round 1: build ----------------------------------------------------------------------------------
        seg: dict[tuple[int, int], object] = {}
        chips: dict[tuple[int, int], VGroup] = {}
        bundle: list[list[int]] = [[], [], []]

        def bid_of(t: int, a: int, j: int) -> float:
            return TR["rounds"][t][a]["y"][j]

        def make_chip(t: int, a: int, j: int) -> VGroup:
            box = RoundedRectangle(corner_radius=0.06, width=SW, height=SH, stroke_color=AGENT[a], stroke_width=2.2,
                                   fill_color="#151a22", fill_opacity=1)
            return VGroup(box, Text(f"{ROOM[j]}   {bid_of(t, a, j):.3f}", font="Noto Mono", font_size=15,
                                    color=TEXT).move_to(box)).set_z_index(6)

        def build(t: int, slot: int) -> tuple[list, list]:
            """The slot-th room each robot adds in round t: its route segment draws, its bid appears at the room;
            then (second list) the bid flies into the robot's bundle."""
            draw, fly = [], []
            for a in range(3):
                add = R[t]["added"][a]
                if slot >= len(add):
                    continue
                j = add[slot]
                prev = bundle[a][-1] if bundle[a] else None
                xy = TR["routes"][a][j] if prev is None else TR["legs"][prev][j]
                s = polyline(f, np.array(xy), AGENT[a], width=3.5, opacity=0.9).set_z_index(2)
                seg[a, j] = s
                c = make_chip(t, a, j).scale(0.55).move_to(rooms[j].get_center() + DOWN * 0.02)
                chips[a, j] = c
                n = len(bundle[a])
                bundle[a].append(j)
                c.generate_target()
                c.target.scale(1 / 0.55).move_to(slots[a][n])
                draw += [Create(s, run_time=1.3, rate_func=smooth),
                         FadeIn(c, scale=0.6, run_time=0.5, rate_func=rush_from)]
                fly.append(c)
            return draw, [MoveToTarget(c, path_arc=-0.5, run_time=0.8) for c in fly]

        d1, f1 = build(0, 0)
        d2, f2 = build(0, 1)
        self.say_steps("Each robot bids for its best two rooms: the value, discounted by how far it must drive.", [
            [*light(0), FadeIn(formula, shift=LEFT * 0.2, run_time=0.8)],
            d1, f1, d2, f2])

        # ---- 3 · round 1: broadcast, two rooms claimed twice; resolve --------------------------------------------
        def packets() -> tuple[list, list]:
            """Every robot's (y, z) snapshot flies to the other two (beat 5's radio packets); then they vanish."""
            dots, moves = [], []
            for i in range(3):
                for k in range(3):
                    if i != k:
                        d = Dot(robots[i].get_center(), radius=0.06, color=AGENT[i]).set_z_index(5)
                        dots.append(d)
                        moves.append(d.animate(run_time=1.0, rate_func=smooth).move_to(robots[k].get_center()))
            return [LaggedStart(*moves, lag_ratio=0.12)], [FadeOut(d, run_time=0.25) for d in dots]

        strikes: list[Line] = []

        def contest(t: int) -> list:
            """Anticipation: the contested rooms flush red on the map, and the competing bids light up in the bundles."""
            out = []
            for j, cl in R[t]["contest"].items():
                out.append(rooms[j].animate(run_time=0.5).set_fill(TARGET, 0.4))
                out += [chips[a, j][0].animate(run_time=0.5).set_stroke(TARGET, 3.5) for a, _ in cl]
            return out

        def verdict(t: int) -> list:
            """The higher bid is indicated; the lower is struck through."""
            out = []
            for j, cl in R[t]["contest"].items():
                for a, _ in cl:
                    c = chips[a, j]
                    if a == R[t]["win"][j]:
                        out += [Indicate(c, scale_factor=1.12, color=AGENT[a], run_time=0.9),
                                c[0].animate(run_time=0.9).set_stroke(AGENT[a], 2.2)]
                    else:
                        strikes.append(Line(c.get_left() + RIGHT * 0.1, c.get_right() + LEFT * 0.1, color=TARGET,
                                            stroke_width=3).set_z_index(8))
                        out += [Create(strikes[-1], run_time=0.5), c.animate(run_time=0.5).set_opacity(0.4)]
            return out

        def release(t: int) -> list:
            """Follow-through: the loser's bid drops out of its bundle and its route segment retracts (eq. 6); the
            room settles back to unexplored."""
            out = [FadeOut(s, shift=DOWN * 0.35, run_time=0.6) for s in strikes]
            strikes.clear()
            for a in range(3):
                for j in R[t]["release"][a]:
                    out += [FadeOut(chips.pop((a, j)), shift=DOWN * 0.35, run_time=0.6),
                            Uncreate(seg.pop((a, j)), run_time=0.8, rate_func=smooth)]
                    bundle[a].remove(j)
            out += [rooms[j].animate(run_time=0.6).set_fill(DIM, 0.22) for j in R[t]["contest"]]
            return out

        pk1, gone1 = packets()
        self.say_steps("They broadcast their bids, and two rooms are claimed twice.", [
            lambda: light(1), pk1, lambda: [*gone1, *contest(0)]])

        nxt: dict[str, list] = {}

        def rebuild() -> list:  # built when it runs: the losers' bundles are only short after the release
            nxt["draw"], nxt["fly"] = build(1, 0)
            return [*light(0), *new_round(2), *nxt["draw"]]

        self.say_steps("The higher bid keeps each room; each loser releases it, and bids again.", [
            lambda: light(2), lambda: verdict(0), lambda: release(0), rebuild, lambda: nxt["fly"]])

        # ---- 4 · round 2: both losers want room D -------------------------------------------------------------------
        pk2, gone2 = packets()
        self.say_steps("Now both want room D, and the bids differ by a thousandth.", [
            lambda: light(1), pk2, lambda: [*gone2, *contest(1)], 0.5, lambda: light(2), lambda: verdict(1), lambda: release(1)])

        # ---- 5 · round 3: nothing changes; ownership; the robots set off -------------------------------------------
        pk3, gone3 = packets()
        drive = []
        for a in range(3):
            pts = []
            for n, j in enumerate(TR["paths"][a]):
                xy = TR["routes"][a][j] if n == 0 else TR["legs"][TR["paths"][a][n - 1]][j]
                pts += xy if n == 0 else xy[1:]
            pts = np.array(pts)
            run = np.r_[0, np.cumsum(np.hypot(*np.diff(pts, axis=0).T))]
            pts = pts[run <= run[-1] - 1.4]  # stop at the last room's doorway, clear of its label
            metres = float(run[len(pts) - 1])
            drive.append(MoveAlongPath(robots[a], polyline(f, pts, AGENT[a]), run_time=1.0 + 0.12 * metres,
                                       rate_func=smooth))
        own = [rooms[j].animate(run_time=0.8).set_fill(AGENT[a], 0.3) for a in range(3) for j in TR["paths"][a]]
        self.say_steps("In round three nothing changes: they agree, with no central planner.", [
            lambda: [*light(1), *new_round(3)], pk3, gone3,
            lambda: [*[p[0].animate(run_time=0.3).set_stroke(DIM, 1.5) for p in pills],
                     *new_round(3, "no change · agreed ✓", THREAD)], own, drive])

        # ---- 6 · the results: this instance, the recorded replay, the published bound ------------------------------
        def row(head: str, kind: str, src: str, body: str = "") -> VGroup:
            first = VGroup(Text(head, font=FONT, font_size=14, color=TEXT), tag(kind, src).scale(0.75)).arrange(
                RIGHT, buff=0.15)
            if not body:
                return first
            return VGroup(first, Text(body, font=FONT, font_size=13, color=DIM, line_spacing=0.8)).arrange(
                DOWN, buff=0.08, aligned_edge=LEFT)

        res = VGroup(
            row("this maze: the exact optimum", "derived", "brute force",
                f"{TR['n_rounds']} rounds, {TR['messages']} messages to agree"),
            row("a recorded mission, replayed", "measured", "replay",
                f"two robots, one room each: the central planner's choice\nin {REPLAY_EQ} of {REPLAY_N} replans; "
                f"{CBBA_MEAN:.3f} of the optimum on average,\n{CBBA_WORST:.3f} at worst"),
            row("never below half the optimum", "published", "Choi et al. 2009"),
        ).arrange(DOWN, buff=0.22, aligned_edge=LEFT)
        assert res.width <= RW, [r.width for r in res]
        lift = ctop - table.get_top()[1]
        res.next_to(table.copy().shift(UP * lift), DOWN, 0.4).align_to(rows, LEFT)
        self.say_steps("So they spread out, instead of crowding into the same corridor.", [
            [FadeOut(formula, shift=UP * 0.2, run_time=0.6),
             VGroup(table, *chips.values()).animate(run_time=0.8, rate_func=smooth).shift(UP * lift)],
            [LaggedStart(*[FadeIn(r, shift=UP * 0.1) for r in res], lag_ratio=0.35, run_time=1.4)]])
        self.wait(2.5)
        self.clear_caption()
        self.play(*[FadeOut(m) for m in self.mobjects], run_time=0.8)
