"""Beat 4: knowing where you are. An overview (cameras + IMU -> front end -> back end -> pose graph), the front end one
step per slide in two panels (how it works | on our recorded frames), the back end in its own labelled panels (the
window's poses settling under bundle adjustment, the Schur complement, keyframes leaving into the pose graph), and the
results: EuRoC vs ORB-SLAM3's published numbers, and the pose-graph solver vs GTSAM. (Replaces beat 4b.)

Data. Right-hand front-end crops: the collab60d pack's agent-0 frames, with the coarse-to-fine LK estimates and the
forward-backward and epipolar rejects from an OFFLINE replay of the front end's per-frame loop with OpenCV 5
(tools/track_replay.py -> data/tracks/): illustrative. The diagrams are illustrative. Facts: 2026-10-24 (front end:
FAST + ORB, pyramidal LK 21 px / 4 levels, forward-backward 0.5 px, F-RANSAC 1 px, left-right LK triangulation with an
SGBM 3-sigma check; back end: 10-keyframe window with Forster IMU preintegration, LM with the Schur complement as in
docs/design/local-slam.md §3: invert each landmark's 3x3 block, S = H_cc - sum H_cl H_ll^-1 H_lc, dense LDLT for
the keyframes, back-substitute each landmark; drop plus odometry edge on exit). The demo runs the front end + PnP +
2-keyframe BA; the IMU window is the full stack, measured on EuRoC, and is drawn only in its own panel, never over demo
frames. Results: EuRoC wins counted from evidence.euroc() (D87) at ORB-SLAM3's published 3-dp precision, no mean
(by design); pose-graph solver vs GTSAM 100/100 (D52). Internal IDs stay off screen.

  python/video/render.sh scenes/s04a_ba.py BundleAdjust
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, ArcBetweenPoints, Arrow, Circle, Create, CurvedArrow, DashedLine,
                   DashedVMobject, Dot, FadeIn, FadeOut, GrowArrow, Group, Line, ManimColor, Polygon, Rectangle,
                   ReplacementTransform, RoundedRectangle, Square, Text, Transform, ValueTracker, VGroup, VMobject,
                   always_redraw, interpolate_color)
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ariadne_video.evidence import euroc  # noqa: E402
from ariadne_video.style import AGENT, DIM, FONT, TARGET, TEXT, THREAD, CaptionedScene, tag  # noqa: E402
from ariadne_video.slides import LC, RC, chip_top, note, panel, title  # noqa: E402
from ariadne_video.tracks import TRACKS, Tile, id_color, image, load_tracks  # noqa: E402

GOOD = "#56d364"
O3 = np.zeros(3)


def zigzag(a, b, amp=0.06, n=7) -> VMobject:
    pts = [a + (b - a) * t + DOWN * (amp if k % 2 else -amp) * (0 < k < n - 1) for k, t in enumerate(np.linspace(0, 1, n))]
    return VMobject(stroke_color=THREAD, stroke_width=2).set_points_as_corners(pts)


def frustum_at(c, s=1.0, op=0.3) -> Polygon:
    return Polygon(c, c + LEFT * 0.1 * s + UP * 0.22 * s, c + RIGHT * 0.1 * s + UP * 0.22 * s, color=AGENT[0],
                   fill_opacity=op, stroke_width=2)


def grow(arrow):
    return GrowArrow(arrow, run_time=0.7)


class BundleAdjust(CaptionedScene):
    def construct(self):
        tr, arrs = load_tracks()

        # ---- overview: four boxes, each with a picture of what it does --------------------------------------------
        bw, bh = 2.5, 2.1
        heads = [("stereo + IMU", AGENT[0]), ("front end", TEXT), ("back end", TEXT), ("pose graph", THREAD)]
        subs = ["images + motion", "tracks features", "optimises poses", "the robot's path"]
        boxes = VGroup()
        for (h, colr), sub in zip(heads, subs):
            b = RoundedRectangle(corner_radius=0.12, width=bw, height=bh, stroke_color=colr, stroke_width=2)
            ht = Text(h, font=FONT, font_size=17, color=colr).next_to(b.get_top(), DOWN, 0.12)
            st = Text(sub, font=FONT, font_size=14, color=DIM).next_to(b.get_bottom(), UP, 0.12)
            boxes.add(VGroup(b, ht, st))
        boxes.arrange(RIGHT, buff=0.45).move_to([0, 0.35, 0])
        arrows = VGroup(*[Arrow(boxes[i].get_right(), boxes[i + 1].get_left(), buff=0.06, color=DIM, stroke_width=3)
                          for i in range(3)])
        c0, c1, c2, c3 = (b[0].get_center() + DOWN * 0.05 for b in boxes)
        rig = VGroup(Rectangle(width=1.0, height=0.42, color=TEXT, stroke_width=2).move_to(c0 + LEFT * 0.35),
                     *[Circle(0.1, color=AGENT[0], stroke_width=2).move_to(c0 + LEFT * 0.35 + RIGHT * dx)
                       for dx in (-0.25, 0.25)],
                     Square(0.36, color=THREAD, stroke_width=2).move_to(c0 + RIGHT * 0.65),
                     Text("IMU", font=FONT, font_size=11, color=THREAD).move_to(c0 + RIGHT * 0.65))
        k6 = 6
        thumb = image(arrs[k6], 2.1).move_to(c1)
        s_ = thumb.width / tr["size"][0]
        tl = thumb.get_corner(UP + LEFT)
        f6 = tr["frames"][k6]
        tdots = VGroup(*[Dot(tl + RIGHT * x * s_ + DOWN * y * s_, radius=0.014, color=id_color(i))
                         for i, (x, y) in zip(f6["ids"], f6["pts"])])
        mini = VGroup(*[frustum_at(c2 + LEFT * 0.8 + RIGHT * 0.4 * n + DOWN * 0.2, 0.8) for n in range(5)],
                      *[zigzag(c2 + LEFT * 0.8 + RIGHT * 0.4 * n + DOWN * 0.36, c2 + LEFT * 0.4 + RIGHT * 0.4 * n
                               + DOWN * 0.36, 0.04, 5) for n in range(4)],
                      *[Dot(c2 + v, radius=0.035, color=THREAD) for v in (LEFT * 0.6 + UP * 0.35, UP * 0.42,
                                                                          RIGHT * 0.55 + UP * 0.3)])
        chain_pts = [c3 + LEFT * 0.85 + RIGHT * 0.42 * n + UP * 0.12 * np.sin(n * 1.3) for n in range(5)]
        chain = VGroup(*[Line(chain_pts[n], chain_pts[n + 1], color=TEXT, stroke_width=2.5) for n in range(4)],
                       *[Dot(p, radius=0.05, color=AGENT[0]) for p in chain_pts])
        pics = Group(rig, Group(thumb, tdots), mini, chain)
        self.say_steps("The tracking across frames is done by the front end; the back end turns it into a position.", [
                           [FadeIn(boxes[0]), FadeIn(pics[0]), FadeIn(boxes[1], run_time=0.8), FadeIn(pics[1]),
                            Create(arrows[0])],
                           [FadeIn(boxes[2]), FadeIn(pics[2]), Create(arrows[1]), FadeIn(boxes[3]), FadeIn(pics[3]),
                            Create(arrows[2])],
                           [boxes[1][0].animate.set_stroke(THREAD, 4)]])
        self.play(FadeOut(Group(boxes, arrows, pics)), run_time=0.4)

        # ---- front end, step by step: diagram (left) | on our recorded frames (right) --------------------------------
        left, right = panel(LC, "how it works"), panel(RC, "on our recorded frames")
        chip = chip_top("illustrative", "real frames · tracks replayed offline")
        hi = tr["highlight"]
        p0, c2f = hi["p0"], hi["coarse_to_fine"]
        col = id_color(hi["id"])

        # step 1: pyramidal Lucas-Kanade
        t1 = title("Front end · 1 / 4 · pyramidal Lucas–Kanade tracking")
        layers, larrs, llabs = VGroup(), VGroup(), VGroup()
        shift = np.array([0.10, 0.07])
        for n, (w, y) in enumerate(zip((0.9, 1.5, 2.3, 3.3), (1.55, 1.05, 0.35, -0.55))):
            h = w * 0.3
            r = Rectangle(width=w, height=h, stroke_color=DIM, stroke_width=1.5, fill_color=TEXT,
                          fill_opacity=0.06).move_to(LC + LEFT * 0.35 + UP * y)
            c = r.get_center() + RIGHT * 0.18 * w + UP * 0.05 * h
            frac = (0.82, 0.95, 0.99, 1.0)[n]  # the shift estimate converging, coarse to fine (schematic)
            layers.add(VGroup(r, Circle(radius=0.04, color=TEXT, stroke_width=2).move_to(c)))
            larrs.add(Arrow(c, c + np.r_[shift * w * frac, 0], buff=0, color=col, stroke_width=4,
                            max_tip_length_to_length_ratio=0.4))
            llabs.add(Text(("1/8", "1/4", "1/2", "full")[n], font=FONT, font_size=14, color=DIM).next_to(r, RIGHT, 0.15))
        lk_txt = note("estimate the shift on the coarsest level,\nthen refine it level by level", LC + DOWN * 1.55, 16)
        pyr = [np.asarray(Image.open(TRACKS / f"pyr_{lvl}.png").convert("RGB")) for lvl in range(4)]
        tiles, tarrs = Group(), VGroup()
        for n, c in enumerate(c2f):
            lvl = c["level"]
            centre = RC + RIGHT * (-1.1 + (n % 2) * 2.2) + UP * (1.0 - (n // 2) * 1.45)
            t = Tile(pyr[lvl], p0[0], p0[1], 48 >> lvl, 1.3, centre, scale=2 ** lvl)
            tiles.add(t.group, Circle(radius=0.06, color=TEXT, stroke_width=2).move_to(t.p(*p0)),
                      Text("full" if not lvl else f"1/{2 ** lvl}", font=FONT, font_size=13, color=DIM)
                      .next_to(t.frame, LEFT, 0.1))
            tarrs.add(Arrow(t.p(*p0), t.p(*c["est"]), buff=0, color=col, stroke_width=4,
                            max_tip_length_to_length_ratio=0.35))
        d = np.subtract(c2f[-1]["est"], p0)
        lk_real = note(f"one real track moves ({d[0]:+.1f}, {d[1]:+.1f}) px", RC + DOWN * 1.35, 15)

        self.play(FadeIn(t1), FadeIn(left), FadeIn(right), FadeIn(chip), run_time=0.5)
        self.say_steps("Frame to frame, pyramidal Lucas–Kanade follows each corner, from coarse to fine.", [
            [FadeIn(layers, lag_ratio=0.2, run_time=0.8), FadeIn(llabs), FadeIn(tiles, run_time=0.8)],
            *[[grow(larrs[n]), grow(tarrs[n])] for n in range(4)], [FadeIn(lk_txt), FadeIn(lk_real)], 1.2])
        step1 = Group(t1, layers, larrs, llabs, lk_txt, tiles, tarrs, lk_real)

        # step 2: forward-backward check
        t2 = title("Front end · 2 / 4 · forward–backward check")
        fa = Rectangle(width=2.1, height=1.5, stroke_color=DIM, stroke_width=1.5).move_to(LC + LEFT * 1.25 + UP * 0.55)
        fb_ = Rectangle(width=2.1, height=1.5, stroke_color=DIM, stroke_width=1.5).move_to(LC + RIGHT * 1.25 + UP * 0.55)
        q0, q1 = fa.get_center() + LEFT * 0.3 + DOWN * 0.1, fb_.get_center() + RIGHT * 0.1 + UP * 0.2
        good = Dot(q0 + RIGHT * 0.05 + UP * 0.03, radius=0.05, color=GOOD)
        fwd_d = CurvedArrow(q0, q1, angle=-0.7, color=GOOD, stroke_width=3)
        bwd_d = CurvedArrow(q1, good.get_center(), angle=-0.7, color=GOOD, stroke_width=3)
        diag2 = VGroup(fa, fb_, Text("frame t−1", font=FONT, font_size=14, color=DIM).next_to(fa, DOWN, 0.08),
                       Text("frame t", font=FONT, font_size=14, color=DIM).next_to(fb_, DOWN, 0.08),
                       Dot(q0, radius=0.05, color=TEXT), Dot(q1, radius=0.05, color=TEXT),
                       Circle(radius=0.16, color=GOOD, stroke_width=1.5).move_to(q0))
        fb_txt = note("track forward, then back again:\nlands within 0.5 px of the start → keep\nmisses → drop",
                      LC + DOWN * 1.2, 16)
        fb = tr["fb_example"]
        mid = np.mean([fb["p0"], fb["p1"], fb["back"]], axis=0)
        ta = Tile(arrs[fb["t"] - 1], mid[0], mid[1], 80, 2.3, RC + LEFT * 1.3 + UP * 0.35)
        tb = Tile(arrs[fb["t"]], mid[0], mid[1], 80, 2.3, RC + RIGHT * 1.3 + UP * 0.35)
        fwd = CurvedArrow(ta.p(*fb["p0"]), tb.p(*fb["p1"]), angle=-0.6, color=GOOD, stroke_width=3)
        bwd = CurvedArrow(tb.p(*fb["p1"]), ta.p(*fb["back"]), angle=-0.6, color=TARGET, stroke_width=3)
        real2 = Group(ta.group, tb.group,
                      Text("frame t−1", font=FONT, font_size=14, color=DIM).next_to(ta.frame, DOWN, 0.08),
                      Text("frame t", font=FONT, font_size=14, color=DIM).next_to(tb.frame, DOWN, 0.08),
                      Circle(radius=0.07, color=TEXT, stroke_width=2).move_to(ta.p(*fb["p0"])),
                      Dot(tb.p(*fb["p1"]), radius=0.05, color=GOOD))
        miss = VGroup(Text("×", font=FONT, font_size=26, color=TARGET).move_to(ta.p(*fb["back"])),
                      note(f"a real track: the round trip misses\nby {fb['err_px']:.1f} px → dropped", RC + DOWN * 1.35,
                           15, TARGET))

        self.play(FadeOut(step1), run_time=0.4)
        self.say_steps("Each track is then run backward, and dropped if it doesn't land where it started.", [
            [FadeIn(t2), FadeIn(diag2), FadeIn(real2)], [Create(fwd_d), Create(fwd)], [Create(bwd_d), Create(bwd)],
            [FadeIn(good), FadeIn(fb_txt), FadeIn(miss)], 1.2])
        step2 = Group(t2, diag2, fwd_d, bwd_d, good, fb_txt, real2, fwd, bwd, miss)

        # step 3: RANSAC epipolar fit
        t3 = title("Front end · 3 / 4 · RANSAC epipolar fit")
        ea = Rectangle(width=2.1, height=1.5, stroke_color=DIM, stroke_width=1.5).move_to(LC + LEFT * 1.25 + UP * 0.55)
        eb = Rectangle(width=2.1, height=1.5, stroke_color=DIM, stroke_width=1.5).move_to(LC + RIGHT * 1.25 + UP * 0.55)
        x1 = ea.get_center() + LEFT * 0.25 + UP * 0.15
        e0, e1 = eb.get_corner(DOWN + LEFT) + RIGHT * 0.15 + UP * 0.2, eb.get_corner(UP + RIGHT) + LEFT * 0.3 + DOWN * 0.15
        epi = Line(e0, e1, color=THREAD, stroke_width=4)
        inl_d = Dot(e0 + (e1 - e0) * 0.45, radius=0.06, color=GOOD)
        outl_d = Dot(e0 + (e1 - e0) * 0.7 + LEFT * 0.35 + UP * 0.12, radius=0.06, color=TARGET)
        diag3 = VGroup(ea, eb, Dot(x1, radius=0.06, color=TEXT),
                       Text("frame t−1", font=FONT, font_size=14, color=DIM).next_to(ea, DOWN, 0.08),
                       Text("frame t", font=FONT, font_size=14, color=DIM).next_to(eb, DOWN, 0.08),
                       CurvedArrow(x1 + RIGHT * 0.1, epi.point_from_proportion(0.25) + UP * 0.08, angle=-0.5, color=DIM,
                                   stroke_width=2))
        inl = VGroup(inl_d, Text("✓", font=FONT, font_size=18, color=GOOD).next_to(inl_d, RIGHT, 0.08))
        outl = VGroup(outl_d, Text("×", font=FONT, font_size=20, color=TARGET).next_to(outl_d, LEFT, 0.08))
        ep_txt = note("a point in frame t−1 maps to a line in frame t\n(x′ᵀ F x = 0); RANSAC fits F to all tracks:\n"
                      "a match more than 1 px off its line → outlier", LC + DOWN * 1.3, 15)
        ep = tr["epipolar_example"]
        te = Tile(arrs[ep["t"]], ep["p1"][0], ep["p1"][1], 18, 2.6, RC + UP * 0.35)
        a, b, c = ep["line"]
        lo_x, lo_y, span = te.x0, te.y0, 36  # clip a x + b y + c = 0 to the tile's pixel square
        ends = [(x, -(a * x + c) / b) for x in (lo_x, lo_x + span)] + [(-(b * y + c) / a, y) for y in (lo_y, lo_y + span)]
        ends = sorted({(round(x, 6), round(y, 6)) for x, y in ends
                       if lo_x - 1e-6 <= x <= lo_x + span + 1e-6 and lo_y - 1e-6 <= y <= lo_y + span + 1e-6})
        foot = np.array(ep["p1"]) - (a * ep["p1"][0] + b * ep["p1"][1] + c) / (a * a + b * b) * np.array([a, b])
        real3 = Group(te.group, Line(te.p(*ends[0]), te.p(*ends[-1]), color=THREAD, stroke_width=4))
        out3 = VGroup(Dot(te.p(*ep["p1"]), radius=0.07, color=TARGET),
                      DashedLine(te.p(*ep["p1"]), te.p(*foot), color=TARGET, stroke_width=2),
                      note(f"a real match {ep['dist_px']:.1f} px off its epipolar line → outlier", RC + DOWN * 1.4, 15,
                           TARGET))

        self.play(FadeOut(step2), run_time=0.4)
        self.say_steps("The survivors must also agree with the geometry of the two views, which a RANSAC epipolar fit "
                       "enforces.", [
                           [FadeIn(t3), FadeIn(diag3), FadeIn(real3)], [Create(epi, run_time=0.8)],
                           [FadeIn(inl), FadeIn(outl)], [FadeIn(ep_txt), FadeIn(out3)], 1.2])
        step3 = Group(t3, diag3, epi, inl, outl, ep_txt, real3, out3)

        # step 4: stereo triangulation at keyframes (both panels schematic: the pack has no right-camera frames)
        t4 = title("Front end · 4 / 4 · stereo depth at keyframes")
        L = Rectangle(width=2.1, height=1.4, stroke_color=AGENT[0], stroke_width=2).move_to(LC + LEFT * 1.25 + UP * 0.6)
        R = Rectangle(width=2.1, height=1.4, stroke_color=AGENT[0], stroke_width=2).move_to(LC + RIGHT * 1.25 + UP * 0.6)
        xl, xr = L.get_center() + RIGHT * 0.35 + UP * 0.15, R.get_center() + LEFT * 0.25 + UP * 0.15
        diag4 = VGroup(L, R, DashedLine(L.get_left() + UP * 0.15, R.get_right() + UP * 0.15, color=DIM, stroke_width=1.5),
                       Dot(xl, radius=0.06, color=col), Dot(xr, radius=0.06, color=col),
                       Text("left", font=FONT, font_size=14, color=DIM).next_to(L, DOWN, 0.08),
                       Text("right", font=FONT, font_size=14, color=DIM).next_to(R, DOWN, 0.08))
        st_txt = note("find the match on the same rectified row\n(|Δy| ≤ 1 px) with Lucas–Kanade", LC + DOWN * 1.2, 16)
        cl, cr = RC + LEFT * 0.6 + DOWN * 0.45, RC + RIGHT * 0.6 + DOWN * 0.45
        Xs = RC + UP * 1.45
        base_l = DashedLine(cl, cr, color=DIM)
        tri = VGroup(Dot(cl, color=AGENT[0]), Dot(cr, color=AGENT[0]), Line(cl, Xs, color=DIM), Line(cr, Xs, color=DIM),
                     Dot(Xs, radius=0.08, color=col), base_l,
                     Text("B", font=FONT, font_size=15, color=DIM).next_to(base_l, DOWN, 0.08),
                     Text("Z", font=FONT, font_size=15, color=DIM).move_to(RC + LEFT * 0.55 + UP * 0.6))
        tri_txt = note("Z = f · B / disparity\ncheck against SGBM depth: within 3σ ✓", RC + DOWN * 1.35, 15)
        right4 = panel(RC, "triangulation")

        self.play(FadeOut(step3), run_time=0.4)
        self.say_steps("And at keyframes, a left–right stereo match triangulates each track, cross-checked against SGBM.", [
            [FadeIn(t4), FadeIn(diag4), Transform(right[1], right4[1]), FadeOut(chip)], [FadeIn(st_txt)],
            [Create(tri, run_time=1.0)], [FadeIn(tri_txt)], 1.0])
        self.play(FadeOut(Group(t4, diag4, st_txt, tri, tri_txt, left, right)), run_time=0.5)

        # ---- back end: its own labelled panels (no demo frames under it) ---------------------------------------------
        tb_ = title("Back end · sliding-window visual–inertial bundle adjustment")
        chip_b = chip_top("illustrative", "full stack, as benchmarked on EuRoC")
        lp, rp = panel(LC, "the window"), panel(RC, "the solve")
        rng = np.random.default_rng(7)
        n_nodes, w0, WIN = 16, 4, 10
        xs = np.linspace(LC[0] - 2.45, LC[0] + 2.45, n_nodes)
        base = np.c_[xs, LC[1] - 0.75 + 0.1 * np.sin(xs * 1.3), np.zeros(n_nodes)]
        lm = np.c_[rng.uniform(xs[0] - 0.1, xs[-1] + 0.1, 22), rng.uniform(LC[1] + 0.2, LC[1] + 1.5, 22), np.zeros(22)]
        seen = {j: [n for n in range(n_nodes) if abs(xs[n] - lm[j, 0]) < 0.75 and rng.random() < 0.8]
                for j in range(len(lm))}
        off = np.c_[rng.normal(0, 0.12, (n_nodes, 2)), np.zeros(n_nodes)]  # the initial guess, before the solve
        win = list(range(w0, w0 + WIN))
        off[win[0]] = 0  # the oldest keyframe in the window is the fixed gauge

        frs = {n: frustum_at(base[n]) for n in range(n_nodes)}
        node = {n: Dot(base[n], radius=0.06, color=AGENT[0]) for n in range(n_nodes)}
        obs = {n: VGroup(*[Line(base[n] + UP * 0.22, lm[j], color=DIM, stroke_width=1, stroke_opacity=0.4)
                           for j in seen if n in seen[j]]) for n in range(n_nodes)}
        imu = {n: zigzag(base[n] + DOWN * 0.2, base[n + 1] + DOWN * 0.2) for n in range(n_nodes - 1)}
        odo = {n: Line(base[n], base[n + 1], color=TEXT, stroke_width=3) for n in range(n_nodes - 1)}
        lms = VGroup(*[Dot(p, radius=0.045, color=THREAD) for p in lm])

        # the window while it is being solved: poses start off the converged ones, observation rays carry the error
        conv = ValueTracker(0.0)

        def solving():
            a_ = conv.get_value()
            pos = {n: base[n] + off[n] * (1 - a_) for n in win}
            g = VGroup()
            for n in win:
                for j in seen:
                    if n in seen[j]:
                        g.add(Line(pos[n] + UP * 0.22, lm[j], stroke_width=1.0 + 0.6 * (1 - a_),
                                   color=interpolate_color(ManimColor(TARGET), ManimColor(DIM), a_),
                                   stroke_opacity=0.4 + 0.2 * (1 - a_)))
            g.add(*[zigzag(pos[n] + DOWN * 0.2, pos[n + 1] + DOWN * 0.2) for n in win[:-1]])
            g.add(*[frustum_at(pos[n]) for n in win])
            return g

        def bracket(start):
            g = VGroup(*[frs[n] for n in range(start, start + WIN)])
            return DashedVMobject(RoundedRectangle(corner_radius=0.1, width=g.width + 0.25, height=0.8,
                                                   stroke_color=AGENT[0], stroke_width=1.5).move_to(g)
                                  .shift(DOWN * 0.08), num_dashes=50)

        graph = VGroup(*[node[n] for n in range(w0)], *[odo[n] for n in range(w0)])
        brk = bracket(w0)
        legend = VGroup(
            VGroup(frustum_at(O3, 0.8), Text("keyframe", font=FONT, font_size=13, color=TEXT)).arrange(RIGHT, buff=0.1),
            VGroup(Dot(radius=0.045, color=THREAD), Text("landmark", font=FONT, font_size=13, color=TEXT))
            .arrange(RIGHT, buff=0.1),
            VGroup(zigzag(O3, O3 + RIGHT * 0.3, 0.05), Text("IMU factor", font=FONT, font_size=13, color=TEXT))
            .arrange(RIGHT, buff=0.1),
            VGroup(Line(O3, O3 + RIGHT * 0.3, color=TEXT, stroke_width=3),
                   Text("pose graph", font=FONT, font_size=13, color=TEXT)).arrange(RIGHT, buff=0.1),
        ).arrange(RIGHT, buff=0.3).move_to(LC + DOWN * 1.7)
        err_lbl = note("red rays: reprojection error", LC + UP * 1.75 + RIGHT * 1.2, 13, TARGET)

        # the solve (local-slam.md §3): H in block form -> invert each landmark's tiny block -> fold them into the
        # keyframes (Schur complement) -> solve the small dense keyframe system -> back-substitute each landmark
        lw = [j for j in seen if any(n in win for n in seen[j])]
        nt, nx = WIN, len(lw)
        side = 2.3
        cs = side / (nt + nx)
        m0 = RC + UP * 1.55 + LEFT * side / 2 + RIGHT * 0.35  # top-left corner of H

        def cell(r, c, colr, op=0.9):
            return Square(cs * 0.9, stroke_width=0, fill_color=colr, fill_opacity=op).move_to(
                m0 + RIGHT * (c + 0.5) * cs + DOWN * (r + 0.5) * cs)

        Htt, Htx, Hxx, fill, pairs = VGroup(), VGroup(), VGroup(), VGroup(), set()
        for r in range(nt):
            for c in range(nt):
                if abs(r - c) <= 1:
                    Htt.add(cell(r, c, AGENT[0]))
        for k, j in enumerate(lw):
            obs_t = [win.index(n) for n in seen[j] if n in win]
            Hxx.add(cell(nt + k, nt + k, THREAD))
            for t in obs_t:
                Htx.add(cell(t, nt + k, DIM, 0.8), cell(nt + k, t, DIM, 0.8))
            for t1 in obs_t:
                for t2 in obs_t:
                    if abs(t1 - t2) > 1:
                        fill.add(cell(t1, t2, AGENT[0], 0.55))
                        pairs.add((min(t1, t2), max(t1, t2)))
        frame_H = Rectangle(width=side, height=side, stroke_color=DIM, stroke_width=1).move_to(
            m0 + RIGHT * side / 2 + DOWN * side / 2)
        axis = VGroup(Text("keyframes", font=FONT, font_size=12, color=AGENT[0])
                      .next_to(m0 + DOWN * cs * nt / 2, LEFT, 0.1),
                      Text("landmarks", font=FONT, font_size=12, color=THREAD)
                      .next_to(m0 + DOWN * cs * (nt + nx / 2), LEFT, 0.1))
        hx = Rectangle(width=cs * nx, height=cs * nx, stroke_color=THREAD, stroke_width=2.5).move_to(
            m0 + RIGHT * cs * (nt + nx / 2) + DOWN * cs * (nt + nx / 2))
        solve_y = RC + DOWN * 1.45
        s1 = note("the linear system H δ = −b, one row per unknown;\na grey cell = a keyframe sees a landmark", solve_y, 14)
        s2 = note("each landmark's block is tiny and independent:\ninvert them one by one", solve_y, 14, THREAD)
        s3 = note("fold them into the keyframes (Schur complement):\nkeyframes that share a landmark become linked",
                  solve_y, 14)
        s4 = note("solve the small dense keyframe system,\nthen back-substitute each landmark", solve_y, 14)
        links = VGroup(*[ArcBetweenPoints(base[win[a_]] + UP * 0.24, base[win[b_]] + UP * 0.24, angle=-0.9,
                                          color=AGENT[0], stroke_width=1.5, stroke_opacity=0.8)
                         for a_, b_ in sorted(pairs)])

        self.play(FadeIn(tb_), FadeIn(chip_b), FadeIn(lp), FadeIn(rp), run_time=0.5)
        dyn = always_redraw(solving)
        self.say_steps("Those keyframes feed the back end: the last ten, tied together by preintegrated IMU factors,", [
            [FadeIn(graph), FadeIn(lms), FadeIn(legend)], [FadeIn(dyn, run_time=1.0), FadeIn(brk)], [FadeIn(err_lbl)],
            0.6])
        self.say_steps("are solved jointly by Levenberg–Marquardt bundle adjustment,", [
            [FadeIn(VGroup(frame_H, Htt, Htx, Hxx), run_time=0.6), FadeIn(axis), FadeIn(s1)],
            [conv.animate(run_time=0.9).set_value(0.55)], [conv.animate(run_time=0.7).set_value(0.85)],
            [conv.animate(run_time=0.6).set_value(1.0), FadeOut(err_lbl)], 0.4])
        self.remove(dyn)
        self.add(*[obs[n] for n in win], *[imu[n] for n in win[:-1]], *[frs[n] for n in win])
        self.say_steps("with the landmarks eliminated by a Schur complement.", [
            [Create(hx, run_time=0.5), FadeOut(s1), FadeIn(s2)], 1.0,
            [ReplacementTransform(Htx, fill, run_time=1.2),
             VGroup(Hxx, hx).animate(run_time=1.2).scale(0.25).move_to(Htt.get_center()).set_opacity(0),
             FadeOut(s2), FadeIn(s3), Create(links, run_time=1.2)], 1.2,
            [VGroup(Htt, fill).animate(run_time=0.8).scale((nt + nx) / nt, about_point=m0), FadeOut(frame_H),
             FadeOut(axis[1]), FadeOut(links), FadeOut(s3), FadeIn(s4)], 1.0])

        def slide(start):  # keyframe `start` leaves the window: it becomes a pose-graph node with an odometry edge
            new = start + WIN
            return [Transform(frs[start], node[start]), FadeOut(obs[start]), FadeOut(imu[start]),
                    Create(odo[start]), FadeIn(frs[new]), Create(obs[new]), Create(imu[new - 1]),
                    Transform(brk, bracket(start + 1))]

        self.say_steps("As a keyframe leaves the window, it becomes an odometry edge in the robot's pose graph.", [
            slide(w0), 0.5, slide(w0 + 1), 0.6])

        # ---- results -----------------------------------------------------------------------------------------------
        rows = euroc()
        wins = sum(round(r["ate_m"], 3) <= r["orb_m"] for r in rows)  # at or better at ORB-SLAM3's published precision
        res = VGroup(
            VGroup(Text(f"matches or beats ORB-SLAM3 on {wins} of {len(rows)} EuRoC flights", font=FONT, font_size=14,
                        color=TEXT), tag("measured", "vs ORB-SLAM3's published results").scale(0.8)).arrange(RIGHT, buff=0.2),
            VGroup(Text("pose-graph solver matches GTSAM on 100 of 100 benchmark graphs", font=FONT, font_size=14,
                        color=TEXT), tag("measured").scale(0.8)).arrange(RIGHT, buff=0.2),
        ).arrange(DOWN, buff=0.08).move_to([0, -2.4, 0])  # midway between the panels and the subtitles
        words = "zero one two three four five six seven eight nine ten eleven".split()
        self.say_steps(f"Tested on EuRoC's real drone flights, it matches or beats ORB-SLAM3 on {words[wins]} of "
                       f"{words[len(rows)]}.", [[FadeIn(res, run_time=0.6)], 1.5])
        self.clear_caption()
        self.play(FadeOut(Group(tb_, chip_b, lp, rp, graph, lms, legend, brk, *frs.values(), *obs.values(),
                                *imu.values(), *odo.values(), Htt, fill, s4, axis[0], res)), run_time=0.5)
