"""Beat 6 (replaces 6a/6b/6c): trusting a loop closure, in ≤ 30 s. A strip across the top shows every check a candidate
meets, in order; four slides show what passes it and what stops it.

Facts and numbers: 2026-10-25, each checked against its file (configs/collab/collab_live_v1.yaml; cpp/slam/robust/*).
1. The alias (python/video/data/collab60d/collab_edges.csv, agent0 kf96 <-> agent1 kf52; frames f880/f900 of the pack):
   95 matches, 55 essential inliers, 35 PnP inliers; ground truth: wrong by 14.9 m, 138 deg. Read here.
2. Gate off, recorded run collab60d: the verify reasons over all its candidates (read here; header row duplicated once in
   the file, dropped), 32 of 32 geometry-passing loops false; robot 2's estimate vs ground truth from the pack's paths,
   the merge at t = 40 s from merge_events.csv (read here).
3. The known-start test: d2 = r' Sigma^-1 r, r = Log(prior^-1 * implied), cut chi2_6(0.999) = 22.46. Separation derived
   here by joining evaluation/runs/collaboration/known_entry_gate/logged_4ec3f6/*/entry_gate.csv to edges.csv (true = pose error <= 10 deg):
   560 true <= 1.1, 1,164 false >= 6,175. Pre-registered sweep result: 3.78 m -> <= 1.5 cm, 0/560 true lost, 0/7 worse.
4. Gate on, recorded run collab60e (data/gate_report_collab60e.json): 8 tested, 8 refused, 0 kept; every robot <= 7.5 cm.
   Not shown as an A/B with 2 (scene/seed match unconfirmed); "the 8 were false" is NOT claimed (no per-edge GT in file).
Corner chips: PCM -> GNC-TLS to 90 % outliers (synthetic ramp); certificate on 7 public graphs, SE-Sync's cost, offline.

  python/video/render.sh scenes/s06_trust.py Trust
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, Circle, Create, Cross, DashedLine, Dot, Ellipse, FadeIn, FadeOut,
                   GrowFromCenter, Group, ImageMobject, Line, Rectangle, RoundedRectangle, Square, Text, ValueTracker, VGroup,
                   always_redraw)
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ariadne_video.data import load_scene  # noqa: E402
from ariadne_video.evidence import REPO  # noqa: E402
from ariadne_video.mobjects import maze_frame, maze_walls, polyline  # noqa: E402
from ariadne_video.slides import chip_top, note  # noqa: E402
from ariadne_video.style import AGENT, DATA, DIM, FONT, GIF, TARGET, TEXT, THREAD, CaptionedScene, tag  # noqa: E402

GOOD = "#56d364"
PACK = DATA / "pack_collab60d"
SWEEP = REPO / "evaluation/runs/collaboration/known_entry_gate/logged_4ec3f6"
CUT = 22.46
SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
CHIP_BOTTOM, CAP2, CAP1 = 3.4, -2.93, -3.09  # bottom of the top chip; top of a two-line / one-line subtitle
GAP, STRIP_H = 0.3, 0.62  # strip-to-content gap; the check strip's height


STAGES = [("retrieval", "cosine ≥ 0.8"), ("LightGlue", "≥ 40 matches"), ("essential", "≥ 30 inliers, 1 px"),
          ("PnP", "≥ 25 inliers, 2 px"), ("degeneracy", "1σ ≤ 10 cm, 1.1°"), ("known start", "χ² ≤ 22.46"),
          ("PCM", "max clique"), ("GNC-TLS", "weights → 0 / 1")]


def cluster(content_h: float, cap_top: float) -> tuple[float, float]:
    """The check strip and a slide's content as one block, centred between the top chip and the subtitle:
    returns (strip centre y, content top y)."""
    total = STRIP_H + GAP + content_h
    top = (CHIP_BOTTOM + cap_top) / 2 + total / 2
    return top - STRIP_H / 2, top - STRIP_H - GAP


def paths(kind: str) -> dict[int, np.ndarray]:
    return {a: np.array([[float(r[k]) for k in ("t_s", "x_m", "y_m")]
                         for r in csv.DictReader(open(PACK / "paths" / f"agent{a}_{kind}.csv"))]) for a in range(3)}


def err_at(est: np.ndarray, gt: np.ndarray, t: float) -> float:
    k = min(len(est) - 1, int(np.searchsorted(est[:, 0], t)))
    gx, gy = np.interp(est[k, 0], gt[:, 0], gt[:, 1]), np.interp(est[k, 0], gt[:, 0], gt[:, 2])
    return float(np.hypot(est[k, 1] - gx, est[k, 2] - gy))


def point_along(xy: np.ndarray, frac: float) -> np.ndarray:
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    return np.array([np.interp(frac * d[-1], d, xy[:, 0]), np.interp(frac * d[-1], d, xy[:, 1])])


def head(xy: np.ndarray, frac: float) -> np.ndarray:
    """The first `frac` of a polyline by arc length: each robot's map so far."""
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    return xy[d <= frac * d[-1]]


def tls_weights(r2: np.ndarray, mu: float, c2: float) -> np.ndarray:
    """GNC-TLS weight update (Yang et al., RA-L 2020): 1 inside, 0 outside, a ramp in between."""
    lo, hi = mu / (mu + 1) * c2, (mu + 1) / mu * c2
    w = np.sqrt(c2 * mu * (mu + 1) / np.maximum(r2, 1e-12)) - mu
    return np.where(r2 <= lo, 1.0, np.where(r2 >= hi, 0.0, np.clip(w, 0, 1)))


def separation() -> tuple[np.ndarray, np.ndarray]:
    """Entry-gate d² of every logged sweep candidate, split by ground truth (true = pose error ≤ 10°)."""
    true, false = [], []
    for g in sorted(glob.glob(str(SWEEP / "*" / "entry_gate.csv"))):
        e = {(r["agent_a"], r["kf_a"], r["agent_b"], r["kf_b"]): r
             for r in csv.DictReader(open(os.path.join(os.path.dirname(g), "edges.csv")))}
        for r in csv.DictReader(open(g)):
            m = e.get((r["agent_a"], r["kf_a"], r["agent_b"], r["kf_b"]))
            if m and m["pose_err_deg"] not in ("", "nan"):
                (true if float(m["pose_err_deg"]) <= 10 else false).append(float(r["last_d2"]))
    return np.array(true), np.array(false)


class Trust(CaptionedScene):
    def construct(self):
        rows = [r for r in csv.DictReader(open(DATA / "collab60d" / "collab_edges.csv")) if r["reason"] != "reason"]
        alias = next(r for r in rows if (r["agent_a"], r["kf_a"], r["agent_b"], r["kf_b"]) == ("0", "96", "1", "52"))
        reasons = {}
        for r in rows:
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
        passed = [r for r in rows if r["reason"] == "verified"]
        report = json.loads((DATA / "gate_report_collab60e.json").read_text())

        # ---- the check strip, persistent across the four slides ------------------------------------------------------
        strip = VGroup()
        for name, thr in STAGES:
            b = RoundedRectangle(corner_radius=0.08, width=1.48, height=0.62, stroke_color=DIM, stroke_width=1.5)
            strip.add(VGroup(b, VGroup(Text(name, font=FONT, font_size=13, color=TEXT),
                                       Text(thr, font=FONT, font_size=10, color=DIM)).arrange(DOWN, buff=0.04).move_to(b)))
        H, MH = 4.2, 2.5  # slide 1's two panels; the maze drawing's height
        strip_y1, ctop1 = cluster(H, CAP2)
        strip.arrange(RIGHT, buff=0.08).move_to([0, strip_y1, 0])
        arrows = VGroup(*[Line(strip[i].get_right(), strip[i + 1].get_left(), color=DIM, stroke_width=1.5)
                          for i in range(len(STAGES) - 1)])

        def light(idx, color, width=3):
            return [strip[i][0].animate(run_time=0.25).set_stroke(color, width) for i in idx]

        # ---- 1 · the problem, one slide: the real look-alike pair (left) and what trusting it did (right) -----------
        top = ctop1
        mid1 = top - H / 2
        boxes = VGroup(*[RoundedRectangle(corner_radius=0.12, width=6.5, height=H, stroke_color=DIM, stroke_width=1.2,
                                          fill_color="#151a22", fill_opacity=0.9).move_to([x, mid1, 0]) for x in (-3.35, 3.35)])
        if GIF:  # the standalone GIFs have no panels
            boxes.set_stroke(opacity=0).set_fill(opacity=0)

        def img(a, f):
            return ImageMobject(np.asarray(Image.open(PACK / "frames" / f"agent{a}" / f"agent{a}_f{f}.png")
                                           .convert("RGB"))).set_width(2.3)

        ia, ib = img(0, 880), img(1, 900)
        pair = Group(ia, ib).arrange(DOWN, buff=0.32)
        la = Text("robot 1", font=FONT, font_size=12, color=AGENT[0]).next_to(ia, RIGHT, 0.12)
        lb = Text("robot 2", font=FONT, font_size=12, color=AGENT[1]).next_to(ib, RIGHT, 0.12)
        eq = Text("=?", font=FONT, font_size=18, color=THREAD, weight="BOLD").move_to(pair)
        counts = VGroup(*[Text(t_, font=FONT, font_size=11, color=GOOD) for t_ in (
            f"{alias['matches']} matches ✓", f"{alias['essential_inliers']} essential inliers ✓",
            f"{alias['pnp_inliers']} PnP inliers ✓")]).arrange(RIGHT, buff=0.25)
        truth = Text(f"ground truth: wrong by {float(alias['gt_err_t_m']):.1f} m, {float(alias['gt_err_r_deg']):.0f}°",
                     font=FONT, font_size=14, color=TARGET, weight="BOLD")
        left = Group(Group(pair, la, lb, eq), counts, truth).arrange(DOWN, buff=0.16).move_to([-3.35, mid1, 0])

        refused = len(rows) - len(passed)
        fun = VGroup(*[Text(s_, font=FONT, font_size=sz, color=col) for s_, sz, col in (
            (f"recorded run, gate off: {len(rows)} candidates · {refused} refused by geometry", 11, DIM),
            (f"{len(passed)} pass geometry, and {len(passed)} of {len(passed)} are false "
             f"({min(float(r['gt_err_t_m']) for r in passed):.1f}–{max(float(r['gt_err_t_m']) for r in passed):.1f} m, "
             f"{min(float(r['gt_err_r_deg']) for r in passed):.0f}–{max(float(r['gt_err_r_deg']) for r in passed):.0f}° "
             f"wrong)", 11, TARGET),
            (f"one false merge left robot 2 {err_at(*[paths(k)[1] for k in ('est', 'gt')], 1e9):.1f} m off", 12, TARGET))
        ]).arrange(DOWN, buff=0.07)
        off = Cross(strip[5], stroke_color=TARGET, stroke_width=3)

        # right column: the false loop and the known-start test, at the original deliberate pace (illustrative)
        sc = load_scene(DATA / "maze_demo_0")
        f = maze_frame(sc, height=MH, center=(3.35, top - 0.45 - MH / 2, 0))
        walls = maze_walls(sc, f, width=1.5, opacity=0.6)
        wx = np.array(sc.walls)
        inside = lambda xy: xy[(xy[:, 0] >= wx[:, [0, 2]].min()) & (xy[:, 0] <= wx[:, [0, 2]].max()) &  # noqa: E731
                               (xy[:, 1] >= wx[:, [1, 3]].min()) & (xy[:, 1] <= wx[:, [1, 3]].max())]
        r0, r1 = inside(head(sc.gt_routes_responder[0], 0.5)), inside(head(sc.gt_routes_responder[1], 0.5))
        q0, q1 = polyline(f, r0, AGENT[0], width=4), polyline(f, r1, AGENT[1], width=4)
        a_, b_ = point_along(r0, 0.7), point_along(r1, 0.7)
        look = VGroup(*[Square(0.26, color=c, stroke_width=3).move_to(f.p(*q)) for q, c in ((a_, AGENT[0]),
                                                                                             (b_, AGENT[1]))])
        alias = DashedLine(f.p(*b_), f.p(*a_), color=TARGET, stroke_width=4)
        alias2 = alias.copy()  # the same match, offered again once the gate is in place
        drag = f.p(*a_) - f.p(*b_)
        ell = Ellipse(width=f.length(6.0), height=f.length(4.0), color=AGENT[1], stroke_width=3,
                      fill_opacity=0.12).move_to(f.p(*b_))
        entry = Dot(f.p(*r1[0]), color=AGENT[1], radius=0.07)
        x_ = Cross(scale_factor=0.16, stroke_color=TARGET, stroke_width=5).move_to(alias.get_center())
        key = Text("illustrative: robot 2's map, a look-alike match, its known entrance", font=FONT, font_size=11,
                   color=DIM).move_to([3.35, top - 0.25, 0])
        fun.next_to([3.35, f.center[1] - MH / 2, 0], DOWN, 0.12)
        c1 = chip_top("measured", "recorded run · one real look-alike pair · gate off")
        self.play(FadeIn(strip), FadeIn(arrows), FadeIn(boxes), FadeIn(pair), FadeIn(la), FadeIn(lb), FadeIn(c1),
                  FadeIn(walls), FadeIn(key), run_time=0.5)
        self.say_steps("But corridors can look alike:", [[FadeIn(eq), Create(q0, run_time=0.8), Create(q1, run_time=0.8)]])
        self.say_steps("this pair passes every geometric check, though the robots stood in different places.", [
            [*light(range(5), GOOD)], [FadeIn(counts), GrowFromCenter(look, run_time=0.4)], [FadeIn(truth, scale=1.2)]])
        self.say_steps("Trust a match like that, and the merged map tears: one robot is teleported across the building.", [
            [FadeIn(off), Create(alias, run_time=0.4)], [VGroup(q1, look[1]).animate(run_time=1.0).shift(drag)], 0.2,
            [FadeIn(fun, lag_ratio=0.3, run_time=1.0)]])
        self.say_steps("So Ariadne remembers where each robot entered,", [
            lambda: [VGroup(q1, look[1]).animate(run_time=0.6).shift(-drag), FadeOut(alias, run_time=0.6),
                     FadeOut(off, run_time=0.6), *light([5], THREAD, 4)],
            [FadeIn(entry, scale=2, run_time=0.4), GrowFromCenter(ell, run_time=0.9)]])
        self.say_steps("and refuses any loop that would teleport a robot.", [
            [Create(alias2, run_time=0.5)], [GrowFromCenter(x_, run_time=0.4), alias2.animate(run_time=0.4)
                                             .set_opacity(0.25)], 0.3])
        self.play(FadeOut(Group(boxes, left, fun, walls, key, c1, q0, q1, look, ell, entry, alias2, x_)), run_time=0.3)

        # ---- 2 · what else stands guard, shown on three small panels: the real separation, PCM, GNC-TLS; what it all
        # achieves stays on the slide as result rows
        pw, ph, py = 3.95, 2.9, 0.45
        cx = [-4.15, 0.0, 4.15]

        def pane(x, head_):
            box = RoundedRectangle(corner_radius=0.1, width=pw, height=ph, stroke_color=DIM, stroke_width=1.2,
                                   fill_color="#151a22", fill_opacity=0.9).move_to([x, py, 0])
            head = Text(head_, font=FONT, font_size=12, color=DIM).next_to(box.get_corner(UP + LEFT), DOWN + RIGHT, 0.1)
            if GIF:  # the standalone GIFs have no panels
                box.set_stroke(opacity=0).set_fill(opacity=0)
                head.set_opacity(0)
            return VGroup(box, head)

        panes = VGroup(pane(cx[0], "known-start d² per loop · logged sweep"), pane(cx[1], "PCM · illustrative"),
                       pane(cx[2], "GNC-TLS · illustrative"))
        foot = VGroup(*[note(t_, [x, py - ph / 2 + 0.27, 0], 10, DIM) for t_, x in zip(
            (f"each dot is one candidate loop, scored against the\nknown starts: true ≤ 1.1, false ≥ 6,175, cut {CUT}",
             "two loops agree if the cycle through both closes;\nkeep the largest group where every pair agrees",
             "each loop gets a weight; the cost starts convex and\ntightens each round until outliers weigh 0"), cx)])

        # the real separation (derived from the logged sweep)
        tr, fl = separation()
        o = np.array([cx[0] - 1.75, py - 0.75, 0])
        W_, lo, hi = 3.5, -3.0, 4.5

        def X(d2):
            return o + RIGHT * (np.log10(max(d2, 10 ** lo)) - lo) / (hi - lo) * W_

        rng = np.random.default_rng(1)
        dots = VGroup(*[Dot(X(d) + UP * (0.15 + rng.uniform(0, 1.3)), radius=0.014, color=GOOD, fill_opacity=0.7)
                        for d in tr],
                      *[Dot(X(d) + UP * (0.15 + rng.uniform(0, 1.3)), radius=0.014, color=TARGET, fill_opacity=0.7)
                        for d in fl])
        axis = VGroup(Line(o, o + RIGHT * W_, color=DIM, stroke_width=1.5),
                      *[Text("10" + str(k).translate(SUP), font=FONT, font_size=9, color=DIM).next_to(X(10 ** k), DOWN,
                                                                                                    0.04)
                        for k in (-2, 0, 2, 4)])
        cut = DashedLine(X(CUT), X(CUT) + UP * 1.55, color=THREAD, stroke_width=2)

        # PCM: an irregular consistency graph. Six true loops all agree pairwise (the maximum clique); three false loops
        # agree with each other (a self-consistent wrong group: PCM alone would not catch it if it were bigger), one
        # false loop agrees with a single true one by chance, one agrees with nothing
        cc = np.array([cx[1], py + 0.2, 0])
        p_rng = np.random.default_rng(11)
        tp = [cc + np.array([x, y, 0]) for x, y in zip(p_rng.uniform(-1.6, 0.2, 6), p_rng.uniform(-0.75, 0.85, 6))]
        fp_ = [cc + np.array([x, y, 0]) for x, y in ((0.95, 0.65), (1.55, 0.35), (1.2, -0.05), (0.75, -0.6),
                                                       (1.65, -0.7))]
        n_t = VGroup(*[Dot(p_, radius=0.07, color=GOOD) for p_ in tp])
        n_f = VGroup(*[Dot(p_, radius=0.07, color=TARGET) for p_ in fp_])
        l_t = VGroup(*[Line(p_, q_, stroke_width=1.2, color=GOOD, stroke_opacity=0.6) for i, p_ in enumerate(tp)
                       for q_ in tp[i + 1:]])
        l_f = VGroup(*[Line(fp_[i], fp_[j], stroke_width=1.2, color=TARGET, stroke_opacity=0.6)
                       for i, j in ((0, 1), (1, 2), (0, 2))],
                     Line(fp_[3], tp[int(np.argmax([q[0] for q in tp]))], stroke_width=1.2, color=DIM, stroke_opacity=0.6))
        ok_pair = VGroup(Line(tp[0], tp[3], color=THREAD, stroke_width=3),
                         Text("closes ✓", font=FONT, font_size=11, color=THREAD).move_to((tp[0] + tp[3]) / 2 + UP * 0.15))
        bad_pair = VGroup(DashedLine(tp[1], fp_[4], color=TARGET, stroke_width=2.5),
                          Text("✗", font=FONT, font_size=14, color=TARGET).move_to((tp[1] + fp_[4]) / 2 + UP * 0.15))
        clique = VGroup(*[Circle(radius=0.13, color=GOOD, stroke_width=2.5).move_to(p_) for p_ in tp])

        # GNC-TLS on a toy line fit: the outliers' weights anneal to zero (Yang et al. 2020)
        g_rng = np.random.default_rng(3)
        xs = np.linspace(-3, 3, 14)
        ys = 0.45 * xs + 0.3 + g_rng.normal(0, 0.12, xs.size)
        out = np.array([2, 6, 11])
        ys[out] += np.array([1.6, -1.5, 1.7])
        g_ofs = np.array([cx[2], py + 0.45, 0]) - panes[2][0].get_center()  # anchored to its panel, so it moves with it

        def G(x, y):
            return panes[2][0].get_center() + g_ofs + np.array([x * 0.55, y * 0.24, 0])

        def wls(w):
            A = np.stack([xs, np.ones_like(xs)], 1) * np.sqrt(w)[:, None]
            return np.linalg.lstsq(A, ys * np.sqrt(w), rcond=None)[0]

        sched = []
        w = np.ones(xs.size)
        mu = 0.05
        for _ in range(14):
            m, c = wls(w)
            sched.append((w.copy(), (m, c)))
            w = tls_weights((ys - (m * xs + c)) ** 2, mu, 0.35 ** 2)
            mu *= 1.4
        k_t = ValueTracker(0)
        gnc = always_redraw(lambda: VGroup(
            *[Dot(G(x, y), radius=0.05, color=TARGET if i in out else TEXT,
                  fill_opacity=0.15 + 0.85 * sched[int(k_t.get_value())][0][i]) for i, (x, y) in enumerate(zip(xs, ys))],
            Line(G(-3.3, sched[int(k_t.get_value())][1][0] * -3.3 + sched[int(k_t.get_value())][1][1]),
                 G(3.3, sched[int(k_t.get_value())][1][0] * 3.3 + sched[int(k_t.get_value())][1][1]),
                 color=THREAD, stroke_width=2.5),
            # one weight bar per loop under the fit, and the round: the outliers' bars drain to zero
            *[Rectangle(width=0.16, height=max(0.01, 0.45 * sched[int(k_t.get_value())][0][i]), stroke_width=0,
                        fill_color=TARGET if i in out else GOOD, fill_opacity=0.85)
              .move_to([G(x, 0)[0], G(0, -4.1)[1], 0], aligned_edge=DOWN) for i, x in enumerate(xs)],
            Text(f"round {int(k_t.get_value()) + 1} · weights", font=FONT, font_size=10, color=DIM)
            .move_to(G(-3.0, -4.95), aligned_edge=LEFT)))

        eg, sf = report["entry_gate_final"], report["shared_frame_vs_gt"]
        worst = max(v["max_err_m"] for v in sf.values())

        def row(text, *chip):
            return VGroup(Text(text, font=FONT, font_size=14, color=TEXT), tag(*chip).scale(0.7)).arrange(RIGHT, buff=0.18)

        res = VGroup(row("simulation sweep: merged map 3.78 m → ≤ 1.5 cm · 0 of 560 true loops lost · 0 of 7 control "
                         "cells worse", "measured", "3 seeds, pre-registered"),
                     row(f"a second recorded run, gate on: {eg['rejected']} of {eg['tested']} refused · every robot within "
                         f"{worst * 100:.1f} cm", "measured", "recorded run"),
                     row("PCM → GNC-TLS hold up to 90 % outlier loops", "measured", "synthetic"),
                     row("certified globally optimal on 7 public benchmark graphs (SE-Sync's cost)", "measured", "offline")
                     ).arrange(DOWN, buff=0.1).next_to([0, py - ph / 2, 0], DOWN, 0.35)
        block = Group(panes, foot, dots, axis, cut, n_t, n_f, l_t, l_f, ok_pair, bad_pair, clique, res)
        strip_yf, ctopf = cluster(block.height, CAP1)  # the final state plays over a one-line subtitle
        block.shift(UP * (ctopf - block.get_top()[1]))
        upper = Group(*[m for m in block if m is not res and m is not ok_pair and m is not bad_pair])
        strip_yp, ctopp = cluster(upper.height, CAP2)  # until the result rows arrive, the panels alone
        lift = ctopp - upper.get_top()[1]
        upper.shift(UP * lift)
        VGroup(ok_pair, bad_pair).shift(UP * lift)  # they fade before the slide moves, so they stay out of `upper`
        upper.add(gnc)  # moved as one group later, so its panel's fill never ends up drawn over the GNC drawing
        gnc.update()  # redraw once at the panels' final place, so its FadeIn starts there (not where it was built)
        c3 = chip_top("illustrative", "mechanism panels · the d² panel and result rows are measured")
        bar = Group(strip, arrows)
        self.play(FadeIn(c3), FadeIn(panes), bar.animate.shift(UP * (strip_yp - strip.get_y())), run_time=0.4)
        self.say_steps("True and false loops land thousands of times apart;", [
            [FadeIn(dots, lag_ratio=0.0005, run_time=1.0), FadeIn(axis)], [Create(cut), FadeIn(foot[0])]])
        self.say_steps("What survives must agree pairwise: PCM keeps the largest group of loops that all close "
                       "consistently,", [
                           [*light([6], THREAD, 4), FadeIn(n_t), FadeIn(n_f), Create(l_t, run_time=0.8),
                            Create(l_f, run_time=0.8), FadeIn(foot[1])],
                           [Create(ok_pair[0], run_time=0.5), FadeIn(ok_pair[1])],
                           [Create(bad_pair[0], run_time=0.5), FadeIn(bad_pair[1])],
                           [FadeOut(VGroup(ok_pair, bad_pair), run_time=0.3), Create(clique, run_time=0.6),
                            VGroup(n_f, l_f).animate(run_time=0.6).set_opacity(0.25)]])
        self.say_steps("and a graduated non-convex solver weighs every loop, letting the last outliers fade to zero.", [
            [*light([7], THREAD, 4), FadeIn(gnc), FadeIn(foot[2])],
            [k_t.animate(run_time=2.6).set_value(len(sched) - 1)]])
        self.say_steps("Together, the map holds.", [
            light(range(5, 8), GOOD),
            lambda: [upper.animate(run_time=0.6).shift(DOWN * lift), bar.animate(run_time=0.6).shift(UP * (strip_yf - strip_yp)),
                     FadeIn(res, lag_ratio=0.2, run_time=1.0)], 2.5])
        self.clear_caption()
        self.play(FadeOut(Group(strip, arrows, c3, upper, res)), run_time=0.4)  # not ok_pair/bad_pair: already gone
