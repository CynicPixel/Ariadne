"""Beat 6c: the live before/after on the same maze (recorded runs), then the pose-graph certificate.

Left, gate OFF: recorded run collab60d (B's pack, paths/agent*_est.csv): the robots' shared-frame estimates replayed
over 60 s; robot 2 (agent 1) jumps when the aliased merges land (merge_events.csv) and ends far from its ground truth
(error computed here from est vs gt). Right, gate ON: recorded run collab60e on the same scene. Its numbers come from
B's gate_report_collab60e.json (entry gate tested/rejected, per-agent max error vs GT); its paths are drawn when
data/pack_collab60e/ exists. Certificate: RESULTS_INDEX §5 (D74), scoped to SE-Sync's cost.

  python/video/render.sh scenes/s06c_live.py LiveGate
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from manim import (DOWN, LEFT, RIGHT, UP, Create, DashedLine, FadeIn, FadeOut, Text, ValueTracker, VGroup,
                   always_redraw)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ariadne_video.data import load_scene  # noqa: E402
from ariadne_video.mobjects import agent_marker, maze_frame, maze_walls, polyline  # noqa: E402
from ariadne_video.style import AGENT, DATA, DIM, FONT, TARGET, TEXT, THREAD, CaptionedScene, tag  # noqa: E402

GOOD = "#56d364"
CERTIFIED = ["garage", "tinyGrid3D", "smallGrid3D", "sphere2500", "torus3D", "cubicle", "grid3D"]  # RI §5, D74


def load_paths(pack: Path, kind: str) -> dict[int, np.ndarray]:
    out = {}
    for a in range(3):
        p = pack / "paths" / f"agent{a}_{kind}.csv"
        if p.exists():
            out[a] = np.array([[float(r["t_s"]), float(r["x_m"]), float(r["y_m"]), float(r["yaw_rad"])]
                               for r in csv.DictReader(open(p))])
    return out


def final_error(est: np.ndarray, gt: np.ndarray) -> float:
    gx, gy = np.interp(est[-1, 0], gt[:, 0], gt[:, 1]), np.interp(est[-1, 0], gt[:, 0], gt[:, 2])
    return float(np.hypot(est[-1, 1] - gx, est[-1, 2] - gy))


class LiveGate(CaptionedScene):
    def construct(self):
        sc = load_scene(DATA / "maze_demo_0")
        off_pack, on_pack = DATA / "pack_collab60d", DATA / "pack_collab60e"
        report = json.loads((DATA / "gate_report_collab60e.json").read_text())

        fl = maze_frame(sc, height=4.6, center=LEFT * 3.5 + UP * 0.35)
        fr = maze_frame(sc, height=4.6, center=RIGHT * 3.5 + UP * 0.35)
        wl, wr = maze_walls(sc, fl, width=2), maze_walls(sc, fr, width=2)
        hl = Text("gate off", font=FONT, font_size=26, color=TARGET, weight="BOLD").next_to(wl, UP, 0.2)
        hr = Text("gate on", font=FONT, font_size=26, color=GOOD, weight="BOLD").next_to(wr, UP, 0.2)
        title = Text("Same maze, same look-alike corridors · two recorded runs", font=FONT, font_size=22,
                     color=DIM).to_edge(UP, buff=0.15)
        chip = tag("measured", "recorded runs").to_edge(UP, buff=0.12).to_edge(RIGHT, buff=0.3)

        est_off, gt_off = load_paths(off_pack, "est"), load_paths(off_pack, "gt")
        est_on = load_paths(on_pack, "est") if on_pack.exists() else {}
        t = ValueTracker(0.0)

        def trail(f, paths, a):
            p = paths[a]
            k = max(2, int(np.searchsorted(p[:, 0], t.get_value())))
            return polyline(f, p[:k, 1:3], AGENT[a], width=3)

        def marker(f, paths, a):
            p = paths[a]
            k = min(len(p) - 1, int(np.searchsorted(p[:, 0], t.get_value())))
            return agent_marker(a, f, p[k, 1], p[k, 2], p[k, 3], size=0.08)

        trails = VGroup()
        for f, paths in ((fl, est_off), (fr, est_on)):
            for a in paths:
                trails.add(always_redraw(lambda f=f, paths=paths, a=a: trail(f, paths, a)),
                           always_redraw(lambda f=f, paths=paths, a=a: marker(f, paths, a)))

        # until B's collab60e pack lands, the gate-on panel shows the robots at their entrances (no path drawn)
        starts_on = VGroup(*[agent_marker(e["agent"], fr, *e["start_xy_m"], np.deg2rad(e["start_yaw_deg"]), size=0.08)
                             for e in sc.starts]) if not est_on else VGroup()
        self.play(FadeIn(title), FadeIn(wl), FadeIn(wr), FadeIn(hl), FadeIn(hr), FadeIn(chip), FadeIn(starts_on),
                  run_time=0.8)
        self.add(trails)
        self.say("In our recorded runs, the difference is stark.",
                 run_with=[t.animate(run_time=6.0, rate_func=lambda x: x).set_value(60.0)])
        self.clear_caption()

        err1 = final_error(est_off[1], gt_off[1])
        merges = list(csv.DictReader(open(off_pack / "collab" / "merge_events.csv")))
        jump_t = next(int(r["t_ns"]) / 1e9 for r in merges if r["agent"] == "1")  # robot 2's (agent 1) false merge
        g = gt_off[1]
        truth = np.array([np.interp(60.0, g[:, 0], g[:, 1]), np.interp(60.0, g[:, 0], g[:, 2])])
        wrong = est_off[1][-1, 1:3]
        gap = DashedLine(fl.p(*truth), fl.p(*wrong), color=TARGET, stroke_width=3)
        worst_on = max(v["max_err_m"] for v in report["shared_frame_vs_gt"].values())
        eg = report["entry_gate_final"]
        left_txt = VGroup(Text(f"robot 2 ends {err1:.1f} m from where it really is", font=FONT, font_size=20, color=TARGET),
                          Text(f"merged on false matches at t = {jump_t:.0f} s", font=FONT, font_size=17, color=DIM)
                          ).arrange(DOWN, buff=0.1).next_to(wl, DOWN, 0.25)
        right_txt = VGroup(Text(f"{eg['rejected']} of {eg['tested']} false matches refused", font=FONT, font_size=20, color=GOOD),
                           Text(f"every robot within {worst_on * 100:.1f} cm of ground truth", font=FONT, font_size=17,
                                color=TEXT)).arrange(DOWN, buff=0.1).next_to(wr, DOWN, 0.25)
        self.play(Create(gap), FadeIn(left_txt), FadeIn(right_txt), run_time=1.2)
        self.wait(3.2)
        self.play(FadeOut(VGroup(title, wl, wr, hl, hr, chip, gap, left_txt, right_txt, starts_on)), FadeOut(trails), run_time=0.6)

        # the certificate: on public benchmarks, the solver proves its answer is the best possible for SE-Sync's cost
        head = Text("Certified optimal pose graphs", font=FONT, font_size=30, color=TEXT)
        sub = Text("global optimum proved for SE-Sync's cost · objectives match the published values",
                   font=FONT, font_size=19, color=DIM)
        names = VGroup(*[VGroup(Text("✓", font=FONT, font_size=26, color=GOOD), Text(n, font=FONT, font_size=24, color=TEXT)
                                ).arrange(RIGHT, buff=0.2) for n in CERTIFIED]).arrange_in_grid(rows=2, buff=(0.6, 0.3))
        block = VGroup(head, sub, names).arrange(DOWN, buff=0.35).shift(UP * 0.5)
        chips = VGroup(tag("measured", "D74"), tag("published", "SE-Sync, Table 1")).arrange(RIGHT, buff=0.2)
        chips.next_to(block, DOWN, buff=0.35)
        self.play(FadeIn(head), FadeIn(sub), run_time=0.6)
        self.play(FadeIn(names, lag_ratio=0.15, run_time=1.4), FadeIn(chips))
        self.wait(2.6)
        self.play(FadeOut(VGroup(block, chips)), run_time=0.5)
