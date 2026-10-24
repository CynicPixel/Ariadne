"""Reusable Manim objects drawn from real data: the maze top view, agent markers, polylines in world metres."""
from __future__ import annotations

import numpy as np
from manim import Line, Triangle, VGroup, VMobject

from .data import Scene
from .style import AGENT, WALL, MapFrame


def maze_frame(sc: Scene, height: float = 5.9, center=(0, 0.55, 0), margin_m: float = 1.0) -> MapFrame:
    # height/center leave the bottom ~1.1 units free for two-line subtitles
    """MapFrame fitted to the wall bounding box (+ margin), so the maze fills the frame the same way in every beat."""
    w = np.array(sc.walls)
    return MapFrame(w[:, [0, 2]].min() - margin_m, w[:, [0, 2]].max() + margin_m,
                    w[:, [1, 3]].min() - margin_m, w[:, [1, 3]].max() + margin_m, height=height, center=center)


def maze_walls(sc: Scene, f: MapFrame, width: float = 3, opacity: float = 1.0) -> VGroup:
    return VGroup(*[Line(f.p(x0, y0), f.p(x1, y1), color=WALL, stroke_width=width, stroke_opacity=opacity)
                    for x0, y0, x1, y1 in sc.walls])


def agent_marker(agent: int, f: MapFrame, x: float, y: float, yaw_rad: float, size: float = 0.11) -> Triangle:
    tri = Triangle(color=AGENT[agent], fill_opacity=1, stroke_width=0).scale(size)
    return tri.rotate(yaw_rad - np.pi / 2).move_to(f.p(x, y))


class CodePanel(VGroup):
    """Pseudo-code lines (monospace) with a highlight bar; `focus(i)` animates the bar to line i (a run-through)."""

    def __init__(self, title: str, lines: list[str], font_size: int = 18, width: float | None = None):
        from manim import DOWN, LEFT, RIGHT, RoundedRectangle, Text
        from .style import DIM, FONT, TEXT, THREAD
        head = Text(title, font=FONT, font_size=font_size + 4, color=TEXT, weight="BOLD")
        body = VGroup(*[Text(l if l.strip() else " ", font="Noto Mono", font_size=font_size, color=TEXT)
                        for l in lines]).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        # indent by leading spaces, measured in the monospace advance of one character
        adv = Text("MM", font="Noto Mono", font_size=font_size).width / 2
        for t, l in zip(body, lines):
            t.shift(RIGHT * adv * (len(l) - len(l.lstrip())))
        VGroup(head, body).arrange(DOWN, aligned_edge=LEFT, buff=0.25)
        w = width or max(head.width, body.width) + 0.5
        frame = RoundedRectangle(corner_radius=0.12, width=w, height=head.height + body.height + 0.75,
                                 stroke_color=DIM, stroke_width=1.5, fill_color="#151a22", fill_opacity=0.9)
        frame.move_to(VGroup(head, body))
        self.bar = RoundedRectangle(corner_radius=0.05, width=w - 0.2, height=body[0].height + 0.12, stroke_width=0,
                                    fill_color=THREAD, fill_opacity=0.0).move_to(body[0]).set_x(frame.get_x())
        super().__init__(frame, self.bar, head, body)
        self.lines = body
        self._dim = DIM

    def focus(self, i: int, run_time: float = 0.4):
        return self.bar.animate(run_time=run_time).set_fill(opacity=0.28).set_y(self.lines[i].get_y())


def polyline(f: MapFrame, xy: np.ndarray, color, width: float = 3, opacity: float = 1.0) -> VMobject:
    m = VMobject(stroke_color=color, stroke_width=width, stroke_opacity=opacity)
    m.set_points_as_corners([f.p(x, y) for x, y in xy])
    return m
