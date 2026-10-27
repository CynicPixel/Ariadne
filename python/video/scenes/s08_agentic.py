"""Beat 8 (≈ 20 s): the agentic loop, one real decision. The final integrated live run (Nova 2 Lite on Amazon Bedrock,
two robots), t = 80 s: the robots' map and camera keyframes sit dimmed; the model's tool calls tick in as recorded, and
each image brightens when the model asks for it (observe_image); it assigns option 0 to robot 1 (risk 0.00) over an
approval-gated option 3 (risk 0.88); the approval rule (design: above the threshold the operator decides; no
approval fired in this run); C++ validation passes, the command goes out, the mover accepts it. Whole-run
counts: 80/80 validation ok, an approval-gated option on offer in 80/80 and picked 0 times, 3 goals refused by the mover
at the 0.20 m collision envelope.

Data: data/agentic_final/ (tools/agentic_extract.py copies it from the run).

  python/video/render.sh scenes/s08_agentic.py HumanInLoop
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from manim import (DOWN, LEFT, RIGHT, UP, Arrow, Create, FadeIn, FadeOut, Group, ImageMobject, Line, RoundedRectangle,
                   SurroundingRectangle, Text, VGroup, smooth)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ariadne_video.slides import chip_top  # noqa: E402
from ariadne_video.style import DATA, DIM, FONT, TARGET, TEXT, THREAD, CaptionedScene, tag  # noqa: E402

SRC = DATA / "agentic_final"
GOOD = "#56d364"
CHIP_BOTTOM, CAP2 = 3.4, -2.93
GAP, STRIP_H = 0.3, 0.62
STAGES = [("cameras", "two robots' keyframes"), ("map", "candidate goals, risk"), ("AI on Bedrock", "tool calls"),
          ("C++ check", "validate the pick"), ("robot", "the mover executes")]


def crop_map(path: Path) -> np.ndarray:
    """The runtime's map image without its empty grey margin."""
    a = np.array(Image.open(path).convert("RGB"))
    used = np.argwhere(np.abs(a.astype(int) - 128).sum(axis=2) > 12)
    (r0, c0), (r1, c1) = used.min(0), used.max(0)
    return a[max(0, r0 - 10):r1 + 10, max(0, c0 - 10):c1 + 10]


def call_text(c: dict) -> str:
    args = c["input"]
    if c["tool"] == "assign":
        return "assign(" + ", ".join(f"agent {a['agent']} → option {a['candidate']}" for a in args["assignments"]) + ")"
    return c["tool"] + "(" + ", ".join(f"{k}={v}" for k, v in args.items()) + ")"


class HumanInLoop(CaptionedScene):
    def construct(self):
        doc = json.loads((SRC / "decision.json").read_text())
        d, tot = doc["decision"], doc["totals"]
        a0 = d["agents"][0]
        pick = next(c for c in a0["candidates"] if c["index"] == a0["chosen"])
        gated = next(c for c in a0["candidates"] if c["needs_approval"])
        reason = next(a["reason"] for a in d["assignments"] if a["agent"] == 0)
        assert not pick["needs_approval"] and a0["validation"] == "ok" and a0["mover"][0]["accepted"]

        # ---- the robots' images (left) and the model's tool log (right) --------------------------------------------
        H = 3.3
        m = ImageMobject(crop_map(SRC / f"t{d['t_s'] * 1000}ms_map.png"))
        m.height = H
        cams = [ImageMobject(str(SRC / f"t{d['t_s'] * 1000}ms_a{a}_kf.png")) for a in (0, 1)]
        for c in cams:
            c.height = (H - 0.15) / 2
        Group(*cams).arrange(DOWN, buff=0.15)
        imgs = Group(m, Group(*cams)).arrange(RIGHT, buff=0.15)
        for im in (m, *cams):
            im.set_opacity(0.35)

        calls = d["tool_calls"]
        log = VGroup(*[Text(call_text(c), font="Noto Mono", font_size=11, color=DIM) for c in calls]).arrange(
            DOWN, aligned_edge=LEFT, buff=0.045)
        foot = Text(f"{len(calls)} tool calls · {d['latency_s']:.1f} s · Nova 2 Lite", font=FONT,
                    font_size=12, color=DIM)
        opt_pick = VGroup(Text(f"option {pick['index']} · risk {pick['risk']:.2f}", font=FONT, font_size=15, color=TEXT),
                          tag("measured", "picked").scale(0.7)).arrange(RIGHT, buff=0.15)
        thr = a0["autonomous_threshold"]
        opt_gate = Text(f"option {gated['index']} · risk {gated['risk']:.2f} > {thr:.2f} · needs a human", font=FONT,
                        font_size=15, color=TARGET)
        quote = Text(f"“{reason}”", font=FONT, font_size=14, color=THREAD, slant="ITALIC")
        opts = VGroup(Text(f"agent 0 · {a0['candidates_total']} options", font=FONT, font_size=13, color=DIM),
                      opt_pick, opt_gate, quote).arrange(DOWN, aligned_edge=LEFT, buff=0.1)
        # the approval rule (design): an option above the threshold waits for the operator's call
        btns = VGroup(*[VGroup(RoundedRectangle(corner_radius=0.06, width=1.0, height=0.32, stroke_color=col,
                                                stroke_width=1.5), Text(t, font=FONT, font_size=12, color=col))
                        for t, col in (("approve", GOOD), ("reject", TARGET))])
        for b in btns:
            b[1].move_to(b[0])
        btns.arrange(DOWN, buff=0.06)
        human = VGroup(Text("operator", font=FONT, font_size=15, color=TEXT, weight="BOLD"), btns).arrange(
            DOWN, buff=0.08)
        hbox = RoundedRectangle(corner_radius=0.1, width=human.width + 0.35, height=human.height + 0.3,
                                stroke_color=TARGET, stroke_width=1.5, fill_color="#151a22", fill_opacity=0.95)
        hcard = VGroup(hbox, human.move_to(hbox))
        htag = tag("design", "the approval rule").scale(0.75).next_to(hcard, DOWN, 0.12)
        wp = a0["command"]["waypoints_xy_m"][0]
        check = Text(f"C++ validation: {a0['validation']}  →  {a0['command']['kind']} ({wp[0]:.2f}, {wp[1]:.2f})  →  "
                     f"mover accepted", font=FONT, font_size=14, color=GOOD)
        decide = VGroup(opts, VGroup(hcard, htag)).arrange(RIGHT, buff=0.9, aligned_edge=UP)
        right = VGroup(VGroup(log, foot).arrange(DOWN, aligned_edge=LEFT, buff=0.14), decide, check).arrange(
            DOWN, aligned_edge=LEFT, buff=0.2)
        body = Group(imgs, right).arrange(RIGHT, buff=0.5, aligned_edge=UP)

        res = Text(f"final live run: {tot['validation_ok']}/{tot['decisions']} picks passed the check · an option needing "
                   f"a human never taken · {tot['mover_refused']} goals refused within 0.20 m of a wall",
                   font=FONT, font_size=13, color=TEXT)
        res_tag = tag("measured", "whole run").scale(0.8)
        foot_row = VGroup(res, res_tag).arrange(RIGHT, buff=0.25)
        assert foot_row.width < 13.4, foot_row.width

        strip = VGroup()
        for name, sub in STAGES:
            b = RoundedRectangle(corner_radius=0.08, width=2.25, height=STRIP_H, stroke_color=DIM, stroke_width=1.5)
            strip.add(VGroup(b, VGroup(Text(name, font=FONT, font_size=14, color=TEXT),
                                       Text(sub, font=FONT, font_size=10, color=DIM)).arrange(DOWN, buff=0.05).move_to(b)))
        strip.arrange(RIGHT, buff=0.3)
        arrows = VGroup(*[Line(strip[i].get_right(), strip[i + 1].get_left(), color=DIM, stroke_width=1.5)
                          for i in range(len(STAGES) - 1)])

        # one block (strip, body, result row) centred between the top chip and the subtitle
        total = STRIP_H + GAP + body.height + 0.3 + foot_row.height
        top = (CHIP_BOTTOM + CAP2) / 2 + total / 2
        Group(strip, arrows).move_to([0, top - STRIP_H / 2, 0])
        body.move_to([0, top - STRIP_H - GAP - body.height / 2, 0])
        foot_row.next_to(body, DOWN, 0.3).set_x(0)
        assert top + 0.1 < CHIP_BOTTOM, (top, body.height)
        hlink = Arrow(opt_gate.get_right(), [hcard.get_left()[0], opt_gate.get_y(), 0], buff=0.1, color=TARGET,
                      stroke_width=2.5, max_tip_length_to_length_ratio=0.15)

        chip = chip_top("measured", f"final live run, t = {d['t_s']} s · Amazon Bedrock")

        def light(*idx, color=THREAD):
            return [strip[i][0].animate(run_time=0.3).set_stroke(color, 3) for i in idx]

        def look(im) -> tuple[list, object]:
            """The image the model asked for brightens under a flash of its frame."""
            box = SurroundingRectangle(im, color=THREAD, buff=0.03, stroke_width=3)
            return [im.animate(run_time=0.6).set_opacity(1.0), Create(box, run_time=0.6)], box

        # ---- 1 · the model reads the robots' map and frames, and picks ---------------------------------------------
        self.play(FadeIn(chip), FadeIn(strip), FadeIn(arrows), FadeIn(imgs), *light(0, 1), run_time=0.6)
        img_of = {"map": m, 0: cams[0], 1: cams[1]}
        steps, boxes = [[*light(2), FadeIn(foot, run_time=0.3)]], []
        for k, c in enumerate(calls):  # one step per recorded call, in order
            row = FadeIn(log[k], shift=RIGHT * 0.08, run_time=0.35)
            if c["tool"] == "observe_image":
                anims, box = look(img_of["map" if c["input"]["what"] == "map" else c["input"]["agent"]])
                boxes.append(box)
                steps.append([row, *anims])
            elif c["tool"] == "assign":
                log[k].set_color(THREAD)
                steps.append([row, *[FadeOut(bx, run_time=0.3) for bx in boxes]])
            else:
                steps.append([row])
        steps.append([FadeIn(opts[0], run_time=0.3), FadeIn(opt_pick, shift=RIGHT * 0.1, run_time=0.4),
                      FadeIn(opt_gate, shift=RIGHT * 0.1, run_time=0.4)])
        steps.append([FadeIn(quote, run_time=0.6)])
        self.say_steps("An AI on Amazon Bedrock reads the robots' map and camera frames, then picks their next "
                       "goals.", steps)

        # ---- 2 · the approval rule: above the threshold, the operator decides ----------------------------------------
        self.say_steps("Anything above the risk threshold is designed to wait for a human, who makes the call;", [
            lambda: [opt_gate.animate(run_time=0.5).scale(1.06)], [Create(hlink, run_time=0.6)],
            [FadeIn(hcard, shift=RIGHT * 0.15, run_time=0.6), FadeIn(htag, run_time=0.6)],
            lambda: [opt_gate.animate(run_time=0.4).scale(1 / 1.06)]])

        # ---- 2 · plain code checks the pick; the mover executes -----------------------------------------------------
        self.say_steps("and plain code checks every pick before anything moves.", [
            light(3, color=GOOD), [FadeIn(check, shift=RIGHT * 0.15, run_time=0.6, rate_func=smooth)],
            light(4, color=GOOD)])
        self.play(FadeIn(foot_row, shift=UP * 0.1), run_time=0.5)
        self.wait(2.2)
        self.clear_caption()
        self.play(*[FadeOut(x) for x in self.mobjects], run_time=0.6)
