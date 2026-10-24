"""Beat 8: a human stays in charge. One real decision from the live Bedrock run, step by step, then the safety envelope.

Real data: evaluation/runs/L1/llm_live_run5_maze_short_90s/trace.jsonl, the t = 5 s record for agent 0 (vision summary,
14 candidate goals with risk and needs_approval, the Nova 2 Lite tool calls and its stated reason, C++ validation, the
command). Totals: RESULTS_INDEX §11 (36/36 answered, 0 fallbacks). The safety envelope (approval hold / link-loss
fallback) is drawn from B's test-configuration trace lines when they exist (data/safety_traces/); until then it is an
illustrative mechanism, and nothing implies it fired in the demo mission.

  python/video/render.sh scenes/s08_agentic.py HumanInLoop
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from manim import (DOWN, LEFT, RIGHT, UP, Arrow, FadeIn, FadeOut, RoundedRectangle, Text, VGroup)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ariadne_video.evidence import REPO  # noqa: E402
from ariadne_video.style import DIM, FONT, TARGET, TEXT, THREAD, CaptionedScene, tag  # noqa: E402

RUN5 = REPO / "evaluation/runs/L1/llm_live_run5_maze_short_90s"
GOOD = "#56d364"
CARD_BG = "#151a22"


def card(title: str, body: VGroup, color=TEXT, width: float = 3.0) -> VGroup:
    head = Text(title, font=FONT, font_size=18, color=color, weight="BOLD")
    inner = VGroup(head, body).arrange(DOWN, aligned_edge=LEFT, buff=0.15)
    box = RoundedRectangle(corner_radius=0.12, width=max(width, inner.width + 0.4), height=inner.height + 0.4,
                           stroke_color=color, stroke_width=1.5, fill_color=CARD_BG, fill_opacity=0.95)
    return VGroup(box, inner.move_to(box))


def lines(*items, size=15, color=TEXT) -> VGroup:
    return VGroup(*[Text(s, font=FONT, font_size=size, color=color) for s in items]).arrange(DOWN, aligned_edge=LEFT, buff=0.07)


class HumanInLoop(CaptionedScene):
    def construct(self):
        recs = [json.loads(l) for l in (RUN5 / "trace.jsonl").read_text().splitlines() if l.strip()]
        r = next(x for x in recs if x["t_s"] == 5 and x["agent"] == 0)
        v, cands, ans = r["vision"], r["candidates"], r["choice"]["answer"]
        orch = ans["orchestrator"]
        pick = next(a for a in ans["assignments"] if a["agent"] == 0)
        thr = r["autonomous_threshold"]
        answered = sum(1 for x in recs if x["choice"].get("validation") == "ok")
        fallbacks = sum(1 for x in recs if isinstance(x["choice"].get("answer"), dict)
                        and x["choice"]["answer"]["orchestrator"].get("fallback_used"))

        c_vis = card("1 · what the cameras found", lines(
            f"{v['free_cells']:,} open cells mapped", f"{v['occupied_cells']:,} wall cells",
            f"{r['candidates_total']} places worth exploring"), color=TEXT)
        opt_rows = [f"option {c['index']}: value {c['utility']:.2f} · risk {c['risk']:.2f}"
                    + ("  ⚠ needs a human" if c["needs_approval"] else "") for c in cands[:4]]
        c_opt = card("2 · options", VGroup(*[Text(s, font=FONT, font_size=15,
                                                   color=TARGET if "⚠" in s else TEXT) for s in opt_rows]
                                           ).arrange(DOWN, aligned_edge=LEFT, buff=0.07))
        tools = [t["tool"] for t in orch["tool_calls"]]
        c_llm = card("3 · AI assistant · Amazon Bedrock", lines(
            "Nova 2 Lite, tools via MCP:", "  " + " → ".join(dict.fromkeys(tools)),
            f"picks option {pick['candidate']}: “{pick['reason']}”"), color=THREAD)
        c_chk = card("4 · checked by plain code", lines(f"C++ validation: {r['choice']['validation']}",
                                                         f"command: {r['command']['kind']} → waypoint"), color=GOOD)
        row1 = VGroup(c_vis, c_opt).arrange(RIGHT, buff=0.6)
        row2 = VGroup(c_llm, c_chk).arrange(RIGHT, buff=0.6)
        grid = VGroup(row1, row2).arrange(DOWN, buff=0.45).shift(UP * 0.55)
        arrows = VGroup(Arrow(c_vis.get_right(), c_opt.get_left(), buff=0.08, color=DIM, stroke_width=3),
                        Arrow(c_opt.get_bottom(), c_llm.get_top() + RIGHT * 1.2, buff=0.08, color=DIM, stroke_width=3),
                        Arrow(c_llm.get_right(), c_chk.get_left(), buff=0.08, color=DIM, stroke_width=3))
        chip = tag("measured", f"live run, t = {r['t_s']} s").to_edge(UP, buff=0.15).to_edge(RIGHT, buff=0.3)

        self.say("Some decisions shouldn't be left to a robot alone.", run_with=[FadeIn(chip), FadeIn(c_vis)])
        self.say("Ariadne has an AI assistant that reads what the robots have seen, proposes the next move, and "
                 "explains why.", run_with=[FadeIn(arrows[0]), FadeIn(c_opt), FadeIn(arrows[1], run_time=1.5),
                                            FadeIn(c_llm, run_time=2.5)])
        self.say("Every proposal is checked by plain code before anything moves.",
                 run_with=[FadeIn(arrows[2]), FadeIn(c_chk)])
        total = Text(f"{answered} of {len(recs)} live decisions answered · {fallbacks} fallbacks",
                     font=FONT, font_size=22, color=GOOD).next_to(grid, DOWN, 0.3)
        self.play(FadeIn(total), run_time=0.6)
        self.wait(1.2)
        self.play(FadeOut(VGroup(grid, arrows, total, chip)), run_time=0.5)

        # the safety envelope: what the code does when a move is too risky or the link drops
        hold = card("too risky", lines(f"risk above {thr:.2f}", "→ hold, ask a human", "→ act only on approval"),
                    color=TARGET, width=3.6)
        lost = card("link to command lost", lines("no answer in time", "→ fall back to the safe", "   classical choice"),
                    color=TARGET, width=3.6)
        env = VGroup(hold, lost).arrange(RIGHT, buff=0.8).shift(UP * 0.6)
        label = tag("illustrative", "mechanism, as built").next_to(env, DOWN, 0.35)
        self.say("And when a move would be too risky, or the link to the command post drops, the robot is built to "
                 "wait for a human, or fall back to the safe choice.",
                 run_with=[FadeIn(hold, shift=UP * 0.2), FadeIn(lost, shift=UP * 0.2, run_time=1.4), FadeIn(label)])
        self.clear_caption()
        self.play(FadeOut(VGroup(env, label)), run_time=0.5)
