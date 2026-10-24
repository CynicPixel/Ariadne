"""Beat 5: sharing maps over a weak radio. The problem (each robot's map in its own frame), then four two-panel slides
(how it works | our data): stage 1, a global descriptor per keyframe; stage 2, details on demand; one recorded run end
to end; spectral (λ₂) selection under a byte budget.

Data and tags (M, 2026-10-24; every number checked against its file that day):
- each robot's own-frame path: the collab60d pack's estimated tracks, re-expressed in each robot's start frame;
- frames: collab60d pack (true-colour PNGs, read as-is). Stage 2's pair is NOT a true same-place pair (collab60d has
  none in its stored frames): labelled "illustrative pair"; the matches drawn on it are illustrative;
- EuRoC retrieval, 297 queries, 11 flights, matched baselines (RESULTS_INDEX §4, D62): R@1 EigenPlaces 752 px 0.677,
  376 px 0.626, mean-bit 0.444, random 0.167; compression (D93): fp16 and int8 change no query's top hit, 297/297;
- detail stage on EuRoC, 4815 candidate pairs (D63): ALIKED + LightGlue pose AUC@10° 0.607 vs ORB 0.284, better on
  11/11 flights; AP_geom 0.807 vs 0.303; OpenCV-dnn exports match upstream (keypoints within 9.3e-5 px);
- one gated run (evaluation/runs/collab_d178/gated_e31421/spectral_s0_p25_seed1, read here): the funnel, the 128 verify
  calls, bytes per class (1 = global descriptor, 4 = match response), merged-map RMSE;
- λ₂ from that run's graph (odometry chains of the released keyframes + its verified edges in file order, κ = 1),
  computed here: derived; the toy graph's λ₂ is computed live (illustrative);
- selection objective on 60 synthetic graphs vs the exact optimum (D170): spectral 0.970, greedy 0.936; per-message
  cost model ±0.4 % (D180). Do not show: class-total byte table, spectral-beats-greedy in sim, vertex-cover saving.

  python/video/render.sh scenes/s05_sharing.py Sharing
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, Arc, Arrow, Circle, Create, DashedLine, Dot, FadeIn, FadeOut, Group,
                   ImageMobject, LaggedStart, Line, Rectangle, RoundedRectangle, Text, VGroup, VMobject)
from manim.constants import RESAMPLING_ALGORITHMS
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ariadne_video.evidence import REPO  # noqa: E402
from ariadne_video.slides import LC, RC, chip_top, note, panel, title  # noqa: E402
from ariadne_video.style import AGENT, DATA, DIM, FONT, TARGET, TEXT, THREAD, CaptionedScene, tag  # noqa: E402

GOOD = "#56d364"
ORB_C = "#7d8fa8"
PACK = DATA / "pack_collab60d"
RUN = REPO / "evaluation/runs/collab_d178/gated_e31421/spectral_s0_p25_seed1"
RECALL = [("EigenPlaces, 752 px", 0.677), ("EigenPlaces, 376 px", 0.626), ("mean-bit baseline", 0.444),
          ("random", 0.167)]
AUC, AP = (0.607, 0.284), (0.807, 0.303)
SPECTRAL, GREEDY = 0.970, 0.936


def frame(agent: int, f: int, width: float) -> ImageMobject:
    return ImageMobject(np.asarray(Image.open(PACK / "frames" / f"agent{agent}" / f"agent{agent}_f{f}.png")
                                   .convert("RGB"))).set_width(width)


def fingerprint(vals: np.ndarray, color: str, width: float, height: float) -> ImageMobject:
    """256 bars as one image: column i filled from the bottom to vals[i] (0..1)."""
    rows = 40
    rgba = np.zeros((rows, len(vals), 4), np.uint8)
    rgb = [int(color[k:k + 2], 16) for k in (1, 3, 5)]
    for i, v in enumerate(vals):
        h = max(1, int(round(v * rows)))
        rgba[rows - h:, i, :3] = rgb
        rgba[rows - h:, i, 3] = 230
    m = ImageMobject(rgba)
    m.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    return m.stretch_to_fit_width(width).stretch_to_fit_height(height)


def lam2(L: np.ndarray) -> float:
    return float(np.linalg.eigvalsh(L)[1])


def laplacian(n: int, edges) -> np.ndarray:
    L = np.zeros((n, n))
    for a, b in edges:
        L[a, a] += 1
        L[b, b] += 1
        L[a, b] -= 1
        L[b, a] -= 1
    return L


def pose_chain(rng, n: int, box, start, heading) -> list[np.ndarray]:
    """A wandering pose chain: n poses 0.45 apart inside box (x0, x1, y0, y1), never closer than 0.3 to an earlier one."""
    p, h = [np.array(start, float)], heading
    for _ in range(n - 1):
        for _ in range(50):
            h2 = h + rng.uniform(-0.9, 0.9)
            q = p[-1] + 0.45 * np.array([np.cos(h2), np.sin(h2)])
            if box[0] <= q[0] <= box[1] and box[2] <= q[1] <= box[3] and all(np.linalg.norm(q - r) > 0.3 for r in p):
                break
        p.append(np.clip(q, [box[0], box[2]], [box[1], box[3]]))
        h = h2
    return p


def bars(rows, scale: float, top_left, colors, size=14, gap=0.62, label_w=2.0) -> VGroup:
    """Horizontal bars: label, bar of value*scale, value."""
    g = VGroup()
    for n, ((lab, v), col) in enumerate(zip(rows, colors)):
        y = top_left + DOWN * n * gap
        t = Text(lab, font=FONT, font_size=size, color=TEXT).move_to(y + RIGHT * label_w, aligned_edge=RIGHT)
        r = Rectangle(width=max(v * scale, 0.02), height=0.3, stroke_width=0, fill_color=col, fill_opacity=0.9)
        r.move_to(y + RIGHT * (label_w + 0.15), aligned_edge=LEFT)
        val = Text(f"{v:.3f}", font=FONT, font_size=size, color=col).next_to(r, RIGHT, 0.1)
        g.add(VGroup(t, r, val))
    return g


def radio_waves(c, color=DIM) -> VGroup:
    return VGroup(*[Arc(radius=0.12 + 0.12 * k, start_angle=-0.6, angle=1.2, color=color, stroke_width=2).move_arc_center_to(c)
                    for k in range(3)])


class Sharing(CaptionedScene):
    def construct(self):
        self.exchange()
        self.stage1()
        self.stage2()
        self.one_run()
        self.spectral()

    # ---- three robots, one radio budget: fingerprints cost little, details cost a lot (illustrative) ------------
    def exchange(self):
        t0 = title("Sharing · two stages on one radio budget")
        chip = chip_top("illustrative", "recorded camera frames · packets and budget schematic")
        nodes = [np.array([-4.3, -0.35, 0]), np.array([-0.6, 1.5, 0]), np.array([3.1, -0.35, 0])]
        cams = Group()
        for a, c in enumerate(nodes):
            img = frame(a, 600, 2.4).move_to(c)
            cams.add(Group(img, Rectangle(width=img.width, height=img.height, stroke_color=AGENT[a],
                                          stroke_width=4).move_to(c),
                           Text(f"robot {a + 1}", font=FONT, font_size=14, color=AGENT[a]).next_to(img, DOWN, 0.08)))
        mf = RoundedRectangle(corner_radius=0.06, width=0.42, height=3.0, stroke_color=DIM, stroke_width=1.5)
        mf.move_to([5.6, 0.35, 0])
        fill = Rectangle(width=0.3, height=2.86, stroke_width=0, fill_color=GOOD, fill_opacity=0.8)
        fill.move_to(mf).align_to(mf, DOWN).shift(UP * 0.07)
        mlab = Text("radio\nbudget", font=FONT, font_size=14, color=DIM, line_spacing=0.8).next_to(mf, DOWN, 0.12)

        def drain(frac, rt):
            return fill.animate(run_time=rt).stretch_to_fit_height(2.86 * frac).align_to(mf, DOWN).shift(UP * 0.07)

        def flight(i, j, color, r, rt):  # a round packet from robot i to robot j
            dot = Dot(nodes[i], radius=r, color=color, fill_opacity=0.95)
            return dot, dot.animate(run_time=rt).move_to(nodes[j])

        fps = [flight(i, j, AGENT[i], 0.08, 2.4) for i in range(3) for j in range(3) if i != j]
        fp_lbl = note("fingerprints: every keyframe, to everyone", [-0.6, -1.75, 0], 15, TEXT)
        link = DashedLine(nodes[0], nodes[1], color=THREAD, stroke_width=3)
        req = flight(0, 1, AGENT[0], 0.11, 1.0)
        resp = flight(1, 0, AGENT[1], 0.32, 2.0)
        dt_lbl = note("details: only for a likely match", [-0.6, -2.1, 0], 15, THREAD)

        self.play(FadeIn(t0), FadeIn(chip), FadeIn(cams, lag_ratio=0.2), FadeIn(mf), FadeIn(fill), FadeIn(mlab),
                  run_time=0.5)
        self.say_steps("To fuse their maps, the robots must find places they've both seen, over one shared radio budget.",
                       [[*[FadeIn(b_, run_time=0.1) for b_, _ in fps]],
                        [LaggedStart(*[m for _, m in fps], lag_ratio=0.18, run_time=4.6), drain(0.88, 4.6), FadeIn(fp_lbl)],
                        [*[FadeOut(b_, run_time=0.2) for b_, _ in fps]]])
        self.say_steps("Fingerprints are cheap; the details of a match are not.", [
            [Create(link, run_time=0.4), FadeIn(req[0], run_time=0.1)], [req[1]],
            [FadeOut(req[0], run_time=0.1), FadeIn(resp[0], run_time=0.1)], [resp[1], drain(0.42, 2.0), FadeIn(dt_lbl)],
            [FadeOut(resp[0], run_time=0.2)]])
        self.play(FadeOut(Group(t0, chip, cams, mf, fill, mlab, fp_lbl, link, dt_lbl)), run_time=0.3)

    # ---- stage 1: a 256-D EigenPlaces fingerprint per keyframe ---------------------------------------------------
    def stage1(self):
        t = title("Sharing · 1 / 4 · stage 1: a fingerprint per keyframe")
        left, right = panel(LC, "how it works"), panel(RC, "on EuRoC: right place ranked first (Recall@1)")
        rng = np.random.default_rng(5)
        img = frame(0, 600, 2.0).move_to(LC + UP * 1.1 + LEFT * 1.35)
        net = VGroup(RoundedRectangle(corner_radius=0.1, width=2.0, height=0.75, stroke_color=AGENT[0], stroke_width=2),
                     Text("EigenPlaces CNN\nvia OpenCV DNN", font=FONT, font_size=13, color=TEXT, line_spacing=0.8))
        net[1].move_to(net[0])
        net.move_to(LC + UP * 1.1 + RIGHT * 1.35)
        fp = fingerprint(rng.random(256) ** 1.5, AGENT[0], 4.6, 0.42).move_to(LC + DOWN * 0.05)
        a1 = Arrow(img.get_right(), net.get_left(), buff=0.08, color=DIM, stroke_width=2.5)
        a2 = Arrow(net.get_bottom(), fp.get_top() + RIGHT * 1.35, buff=0.08, color=DIM, stroke_width=2.5)
        fp_lbl = Text("256 numbers", font=FONT, font_size=13, color=DIM).next_to(fp, UP, 0.06).align_to(fp, LEFT)
        wire = note("half precision on the wire: ≈ 670 bytes per keyframe", LC + DOWN * 0.6, 14)
        tx = LC + DOWN * 1.35 + LEFT * 2.2
        bcast = VGroup(Dot(tx, radius=0.07, color=AGENT[0]), radio_waves(tx + RIGHT * 0.05),
                       *[Dot(tx + RIGHT * (0.95 + 0.35 * k), radius=0.07, color=AGENT[k + 1]) for k in range(2)])
        bcast.add(Text("broadcast to every robot", font=FONT, font_size=13, color=DIM).next_to(bcast, RIGHT, 0.25))
        rec = bars(RECALL, 3.2, RC + UP * 1.2 + LEFT * 2.6, [AGENT[0], AGENT[0], ORB_C, DIM], size=13, label_w=1.75)
        rec_tag = tag("measured", "297 queries, 11 flights").scale(0.8).next_to(rec, DOWN, 0.15).align_to(rec, LEFT)
        comp = note("fp16 or int8 compression: no query's top hit changes (297 of 297)", RC + DOWN * 1.5, 13)

        self.play(FadeIn(t), FadeIn(left), FadeIn(right), run_time=0.5)
        self.say_steps("Ariadne runs a two-stage exchange.", [[FadeIn(img)]])
        self.say_steps("Every keyframe is broadcast as a 256-dimensional EigenPlaces descriptor,", [
            [Create(a1), FadeIn(net), FadeIn(rec[0]), FadeIn(rec[1])],
            [Create(a2), FadeIn(fp), FadeIn(fp_lbl), FadeIn(rec[2]), FadeIn(rec[3]), FadeIn(rec_tag)]])
        self.say_steps("computed with OpenCV's DNN module: about 670 bytes on the wire.", [
            [FadeIn(wire)], [FadeIn(bcast), FadeIn(comp)], 0.8])
        self.play(FadeOut(Group(t, left, right, img, net, fp, a1, a2, fp_lbl, wire, bcast, rec, rec_tag, comp)),
                  run_time=0.4)

    # ---- stage 2: details only on a likely match --------------------------------------------------------------------
    def stage2(self):
        t = title("Sharing · 2 / 4 · stage 2: details on demand")
        left, right = panel(LC, "how it works (illustrative pair)"), panel(RC, "on EuRoC: 4,815 candidate pairs")
        rng = np.random.default_rng(9)
        ia, ib = frame(0, 600, 2.3).move_to(LC + UP * 1.05 + LEFT * 1.3), frame(1, 600, 2.3).move_to(LC + UP * 1.05 + RIGHT * 1.3)
        base = rng.random(256) ** 1.5
        fa = fingerprint(np.clip(base + rng.normal(0, 0.08, 256), 0.02, 1), AGENT[0], 2.3, 0.3).next_to(ia, DOWN, 0.1)
        fb = fingerprint(np.clip(base + rng.normal(0, 0.08, 256), 0.02, 1), AGENT[1], 2.3, 0.3).next_to(ib, DOWN, 0.1)
        sim = note("cosine similarity 0.83 ≥ 0.8 → the broker requests details", LC + DOWN * 0.45, 14, THREAD)
        pa = [ia.get_corner(DOWN + LEFT) + RIGHT * rng.uniform(0.1, 2.2) + UP * rng.uniform(0.1, 1.2) for _ in range(12)]
        pb = [ib.get_corner(DOWN + LEFT) + RIGHT * rng.uniform(0.1, 2.2) + UP * rng.uniform(0.1, 1.2) for _ in range(12)]
        matches = VGroup(*[Line(p, q, color=GOOD, stroke_width=1.5, stroke_opacity=0.85) for p, q in zip(pa, pb)],
                         *[Dot(p, radius=0.03, color=GOOD) for p in pa + pb])
        steps = VGroup(*[Text(s, font=FONT, font_size=14, color=c) for s, c in (
            ("ALIKED keypoints + descriptors + 3-D points", TEXT), ("LightGlue matching", TEXT),
            ("essential matrix ✓   PnP ✓   → a loop closure", GOOD))]).arrange(DOWN, buff=0.1).move_to(LC + DOWN * 1.25)
        auc = bars([("ALIKED + LightGlue", AUC[0]), ("ORB", AUC[1])], 3.0, RC + UP * 1.05 + LEFT * 2.6,
                   [AGENT[0], ORB_C], size=13, gap=0.5, label_w=1.6)
        ap = bars([("ALIKED + LightGlue", AP[0]), ("ORB", AP[1])], 3.0, RC + DOWN * 0.35 + LEFT * 2.6,
                  [AGENT[0], ORB_C], size=13, gap=0.5, label_w=1.6)
        h1 = Text("pose accuracy, AUC at 10°", font=FONT, font_size=13, color=DIM).next_to(auc, UP, 0.1).align_to(auc, LEFT)
        h2 = Text("geometric precision, AP", font=FONT, font_size=13, color=DIM).next_to(ap, UP, 0.1).align_to(ap, LEFT)
        foot = VGroup(Text("better on 11 of 11 flights", font=FONT, font_size=14, color=TEXT),
                      tag("measured", "matched baseline").scale(0.8)).arrange(RIGHT, buff=0.2).move_to(RC + DOWN * 1.45)

        self.play(FadeIn(t), FadeIn(left), FadeIn(right), run_time=0.5)
        self.say_steps("Only when two descriptors clear a cosine similarity of 0.8", [
            [FadeIn(ia), FadeIn(ib), FadeIn(fa), FadeIn(fb)], [FadeIn(sim)]])
        self.say_steps("does the broker request the expensive part: ALIKED keypoints with 3-D points,", [
            [FadeIn(steps[0]), FadeIn(h1), FadeIn(auc)]])
        self.say_steps("matched by LightGlue, and checked with an essential-matrix and PnP test.", [
            [Create(matches, run_time=1.0), FadeIn(steps[1]), FadeIn(h2), FadeIn(ap)], [FadeIn(steps[2]), FadeIn(foot)],
            0.8])
        self.play(FadeOut(Group(t, left, right, ia, ib, fa, fb, sim, matches, steps, auc, ap, h1, h2, foot)),
                  run_time=0.4)

    # ---- one recorded run, end to end ---------------------------------------------------------------------------
    def one_run(self):
        res = (RUN / "results.yaml").read_text()
        proto = dict((k, int(v)) for k, v in re.findall(r"(\w+): (\d+)", re.search(r"^protocol: \{(.*)\}", res, re.M)
                                                          .group(1)))
        reasons = dict((k, int(v)) for k, v in re.findall(r"(\w+): (\d+)", re.search(r"reasons: \{(.*?)\}", res).group(1)))
        calls = int(re.search(r"^verify: \{calls: (\d+)", res, re.M).group(1))
        cls = {int(c): (int(m), int(w)) for c, m, w in
               re.findall(r"\{class: (\d+), messages: (\d+), wire_first: (\d+)", res)}
        ate_mm = float(re.search(r"ate_merged: \{rmse_m: ([0-9.]+)", res).group(1)) * 1000
        edges = list(csv.DictReader(open(RUN / "edges.csv")))
        assert len(edges) == calls

        t = title("Sharing · 3 / 4 · one recorded run, end to end")
        left, right = panel(LC, "the funnel"), panel(RC, "all 128 verifications")
        rows = [(proto["candidates"], "candidate pairs (similarity ≥ 0.8)", TEXT),
                (proto["selected"], "fetched under the byte budget", TEXT),
                (calls, "verified geometrically", TEXT), (proto["edges_verified"], "loop closures accepted ✓", GOOD)]
        fun = VGroup()
        for n, (v, txt, col) in enumerate(rows):
            y = LC + UP * (1.35 - n * 0.72) + LEFT * 2.45
            num = Text(f"{v:,}", font=FONT, font_size=22, color=col, weight="BOLD").move_to(y, aligned_edge=LEFT)
            lab = Text(txt, font=FONT, font_size=14, color=col).next_to(num, RIGHT, 0.2)
            bar = Rectangle(width=0.15 + 4.75 * np.sqrt(v / rows[0][0]), height=0.1, stroke_width=0, fill_color=col,
                            fill_opacity=0.7).move_to(y + DOWN * 0.3, aligned_edge=LEFT)
            fun.add(VGroup(num, lab, bar))
        rej = note(f"rejected: essential matrix {reasons['essential_gate']} · too few 3-D points {reasons['few_3d']}\n"
                   f"too few matches {reasons['few_matches']} · PnP {reasons['pnp_gate']}", LC + DOWN * 1.55, 12, TARGET)

        sc = np.array([float(e["score"]) for e in edges])
        mt = np.array([float(e["matches"]) for e in edges])
        ok = np.array([e["reason"] == "verified" for e in edges])
        x0, x1 = 0.8, np.ceil(sc.max() * 100) / 100
        y1 = np.ceil(mt.max() / 50) * 50
        o = RC + LEFT * 2.1 + DOWN * 0.75
        W_, H_ = 4.3, 1.9

        def P(x, y):
            return o + RIGHT * (x - x0) / (x1 - x0) * W_ + UP * y / y1 * H_

        axes = VGroup(Line(o, o + RIGHT * W_, color=DIM), Line(o, o + UP * H_, color=DIM),
                      Text(f"{x0:.2f}", font=FONT, font_size=11, color=DIM).next_to(o, DOWN, 0.06),
                      Text(f"{x1:.2f}", font=FONT, font_size=11, color=DIM).next_to(o + RIGHT * W_, DOWN, 0.06),
                      Text("cosine similarity", font=FONT, font_size=12, color=DIM).next_to(o + RIGHT * W_ / 2, DOWN, 0.06),
                      Text(f"{int(y1)}", font=FONT, font_size=11, color=DIM).next_to(o + UP * H_, LEFT, 0.06),
                      Text("matches", font=FONT, font_size=12, color=DIM).rotate(np.pi / 2).next_to(o + UP * H_ / 2,
                                                                                                    LEFT, 0.1))
        pts = VGroup(*[Dot(P(x, y), radius=0.035, color=GOOD if k else TARGET, fill_opacity=0.85)
                       for x, y, k in zip(sc, mt, ok)])
        legend = VGroup(VGroup(Dot(radius=0.04, color=GOOD), Text("accepted", font=FONT, font_size=12, color=TEXT))
                        .arrange(RIGHT, buff=0.08),
                        VGroup(Dot(radius=0.04, color=TARGET), Text("rejected", font=FONT, font_size=12, color=TEXT))
                        .arrange(RIGHT, buff=0.08)).arrange(RIGHT, buff=0.3).move_to(RC + UP * 1.5 + RIGHT * 1.3)
        kb = cls[1][1] / 1000
        mb = cls[4][1] / 1e6
        foot = VGroup(note(f"fingerprints {kb:.0f} kB in {cls[1][0]} messages · details {mb:.2f} MB in {cls[4][0]} "
                           f"messages", [0, 0, 0], 12),
                      VGroup(Text(f"merged map error {ate_mm:.1f} mm", font=FONT, font_size=14, color=THREAD,
                                  weight="BOLD"), tag("measured", "one run").scale(0.8)).arrange(RIGHT, buff=0.2)
                      ).arrange(DOWN, buff=0.08).move_to(RC + DOWN * 1.62)

        self.play(FadeIn(t), FadeIn(left), FadeIn(right), run_time=0.5)
        self.say_steps(f"In one recorded run: {proto['candidates']:,} candidate pairs, {proto['selected']} fetched under "
                       f"the budget,", [[FadeIn(fun[0]), FadeIn(axes)], [FadeIn(fun[1]), FadeIn(fun[2])],
                                        [FadeIn(pts, lag_ratio=0.01, run_time=1.2), FadeIn(legend)]])
        self.say_steps(f"{proto['edges_verified']} verified, and a merged map within {round(ate_mm)} millimetres of "
                       f"ground truth.", [[FadeIn(fun[3]), FadeIn(rej)], [FadeIn(foot)], 1.0])
        self.play(FadeOut(Group(t, left, right, fun, rej, axes, pts, legend, foot)), run_time=0.4)

    # ---- spectral selection under a byte budget ------------------------------------------------------------------
    def spectral(self):
        t = title("Sharing · 4 / 4 · which links are worth the bytes: λ₂")
        left, right = panel(LC, "how it works"), panel(RC, "from the run: λ₂ per loop closure")

        # toy: two robots' pose graphs (a wandering chain each, one loop closure each) joined by one inter-robot
        # link; the candidates are picked by their λ₂ gain (illustrative; λ₂ computed live)
        rng = np.random.default_rng(0)
        pa = pose_chain(rng, 7, (-2.4, -0.3, 0.3, 1.75), (-2.2, 0.5), 0.6)
        pb = pose_chain(rng, 6, (0.3, 2.4, 0.3, 1.75), (0.5, 1.5), -0.4)
        pts2 = pa + pb
        na, n = len(pa), len(pa) + len(pb)
        pos = [LC + np.r_[q, 0] for q in pts2]
        base_e = [(i, i + 1) for i in range(na - 1)] + [(na + i, na + i + 1) for i in range(len(pb) - 1)]
        for lo, hi in ((0, na), (na, n)):  # one loop closure per robot: its closest non-neighbouring pair
            base_e.append(min(((i, j) for i in range(lo, hi) for j in range(i + 3, hi)),
                              key=lambda e: np.linalg.norm(pts2[e[0]] - pts2[e[1]])))
        cross = sorted((np.linalg.norm(pts2[i] - pts2[j]), i, j) for i in range(na) for j in range(na, n))
        base_e.append(cross[0][1:])
        l0 = lam2(laplacian(n, base_e))
        gain = lambda e: lam2(laplacian(n, base_e + [e])) / l0 - 1  # noqa: E731
        bridge = max(((i, j) for d, i, j in cross[1:] if d < 2.6 and abs(i - cross[0][1]) >= 3), key=gain)
        inner = min(((i, j) for i in range(na, n) for j in range(i + 2, n) if (i, j) not in base_e
                     and np.linalg.norm(pts2[i] - pts2[j]) < 0.9 and gain((i, j)) > 0.005), key=gain)
        lb, li = lam2(laplacian(n, base_e + [bridge])), lam2(laplacian(n, base_e + [inner]))
        g = VGroup(*[Line(pos[a_], pos[b_], color=DIM, stroke_width=2) for a_, b_ in base_e],
                   *[Dot(p_, radius=0.07, color=AGENT[0] if i < na else AGENT[1]) for i, p_ in enumerate(pos)])
        cand_b = DashedLine(pos[bridge[0]], pos[bridge[1]], color=GOOD, stroke_width=3)
        cand_i = DashedLine(pos[inner[0]], pos[inner[1]], color=TARGET, stroke_width=3)
        lap = note("pose graph as a weighted Laplacian L = D − W;\nλ₂ (the Fiedler value) measures how well it holds "
                   "together", LC + DOWN * 0.75, 13)
        lap.move_to(LC + DOWN * 0.55)
        v0 = note(f"now: λ₂ = {l0:.3f}", LC + DOWN * 1.02, 14)
        vb = note(f"+ bridge: {lb:.3f} (+{(lb / l0 - 1) * 100:.0f} %)", LC + DOWN * 1.36 + LEFT * 1.25, 13, GOOD)
        vi = note(f"+ inside: {li:.3f} (+{(li / l0 - 1) * 100:.0f} %)", LC + DOWN * 1.36 + RIGHT * 1.25, 13, TARGET)
        rule = note("score = Δλ₂ ÷ predicted bytes · greedy pick, then one swap", LC + DOWN * 1.72, 13, THREAD)

        # the run: λ₂ of its graph as its verified edges arrive, in file order (derived)
        res = (RUN / "results.yaml").read_text()
        kf = [int(k) for k in re.findall(r"\{agent: \d+, rmse_m: [0-9.e-]+, keyframes: (\d+)", res)]
        ed = [e for e in csv.DictReader(open(RUN / "edges.csv")) if e["reason"] == "verified"]
        n = sum(kf)
        off = [0, kf[0]]
        chain = [(i, i + 1) for i in range(kf[0] - 1)] + [(kf[0] + i, kf[0] + i + 1) for i in range(kf[1] - 1)]
        L = laplacian(n, chain)
        vals = [lam2(L)]
        for e in ed:
            a, b = off[int(e["agent_a"])] + int(e["kf_a"]), off[int(e["agent_b"])] + int(e["kf_b"])
            L[a, a] += 1
            L[b, b] += 1
            L[a, b] -= 1
            L[b, a] -= 1
            vals.append(lam2(L))
        v = np.array(vals) / max(vals)
        o = RC + LEFT * 2.2 + DOWN * 1.0
        W_, H_ = 4.4, 2.3

        def P(i, y):
            return o + RIGHT * i / (len(v) - 1) * W_ + UP * y * H_

        steps_pts = [P(0, v[0])]
        for i in range(1, len(v)):
            steps_pts += [P(i, v[i - 1]), P(i, v[i])]
        curve = VMobject(stroke_color=THREAD, stroke_width=3).set_points_as_corners(steps_pts)
        axes = VGroup(Line(o, o + RIGHT * W_, color=DIM), Line(o, o + UP * H_, color=DIM),
                      Text("loop closures accepted →", font=FONT, font_size=12, color=DIM).next_to(o + RIGHT * W_ / 2,
                                                                                                   DOWN, 0.06),
                      Text("λ₂ (relative)", font=FONT, font_size=12, color=DIM).next_to(o + UP * H_, UP, 0.06))
        va = np.array(vals)
        k = int(np.argmax(np.diff(va)[1:])) + 1  # the biggest single gain after the joining edge
        same = 0  # the edges right after the joining one, in the same small area: robot 2's keyframe unchanged and
        while same + 1 < len(ed) and ed[same + 1]["kf_b"] == ed[0]["kf_b"] and \
                abs(int(ed[same + 1]["kf_a"]) - int(ed[same]["kf_a"])) <= 3:  # robot 1's keyframes neighbouring
            same += 1
        ann = VGroup(note("first link: the two maps join", P(1, v[1]) + UP * 0.62, 12, TEXT).align_to(P(1.5, 0), LEFT),
                     note(f"next {same} at one place: +{(va[same + 1] / va[1] - 1) * 100:.0f} %",
                          P(1, v[same]) + UP * 0.3, 12, TARGET).align_to(P(1.5, 0), LEFT),
                     note(f"one in a new region: +{(va[k + 1] / va[k] - 1) * 100:.0f} %",
                          P(k + 1, v[k + 1]) + LEFT * 1.35 + UP * 0.12, 12, GOOD))
        derived = chip_top("derived", "λ₂ computed from the run's accepted edges")
        foot = VGroup(VGroup(Text(f"selection: {SPECTRAL:.3f} of the optimum (greedy {GREEDY:.3f})", font=FONT,
                                  font_size=11, color=TEXT), tag("measured", "synthetic graphs").scale(0.62))
                      .arrange(RIGHT, buff=0.15),
                      VGroup(Text("per-message byte cost predicted within ±0.4 %", font=FONT, font_size=11,
                                  color=TEXT), tag("measured").scale(0.62)).arrange(RIGHT, buff=0.15)
                      ).arrange(DOWN, buff=0.06).move_to(RC + DOWN * 1.6)

        self.play(FadeIn(t), FadeIn(left), FadeIn(right), run_time=0.5)
        self.say_steps("Those requests compete for a byte budget.", [[FadeIn(g)]])
        self.say_steps("The broker holds the team's pose graph as a weighted Laplacian,", [
            [FadeIn(lap)], [FadeIn(v0), FadeIn(axes), FadeIn(derived)]])
        self.say_steps("and scores each candidate by the gain in algebraic connectivity, λ₂, per predicted byte.", [
            [Create(cand_b), FadeIn(vb)], [Create(cand_i), FadeIn(vi)], [FadeIn(rule)]])
        self.say_steps("A bridge between weakly joined maps lifts λ₂ sharply;", [
            [Create(curve, run_time=2.0), FadeIn(ann[0])], [FadeIn(ann[2])]])
        self.say_steps("one more edge inside a rigid region barely moves it.", [[FadeIn(ann[1])], [FadeIn(foot)], 1.2])
        self.clear_caption()
        self.play(FadeOut(Group(t, left, right, g, cand_b, cand_i, lap, v0, vb, vi, rule, curve, axes, ann, derived,
                                foot)), run_time=0.5)
