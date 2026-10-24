"""Reusable Manim objects drawn from real data: the maze top view, agent markers, polylines in world metres."""
from __future__ import annotations

import numpy as np
from manim import Line, Triangle, VGroup, VMobject

from .data import Scene
from .style import AGENT, WALL, MapFrame


def maze_frame(sc: Scene, height: float = 6.6, center=(0, 0.3, 0), margin_m: float = 1.0) -> MapFrame:
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


def polyline(f: MapFrame, xy: np.ndarray, color, width: float = 3, opacity: float = 1.0) -> VMobject:
    m = VMobject(stroke_color=color, stroke_width=width, stroke_opacity=opacity)
    m.set_points_as_corners([f.p(x, y) for x, y in xy])
    return m
