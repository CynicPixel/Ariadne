"""The two-panel slide layout shared by the technical beats (4, 5): a title, a source chip top right, and two panels
(how it works | our data) that stay clear of the screen edges and of the subtitle band."""
from __future__ import annotations

import numpy as np
from manim import DOWN, LEFT, RIGHT, UP, RoundedRectangle, Text, VGroup

from .style import DIM, FONT, GIF, TEXT, tag

PW, PH = 5.6, 4.2  # each panel
LC, RC = np.array([-3.0, 0.2, 0]), np.array([3.0, 0.2, 0])  # panel centres
TITLE_Y = 2.85


def panel(center, header: str) -> VGroup:
    box = RoundedRectangle(corner_radius=0.12, width=PW, height=PH, stroke_color=DIM, stroke_width=1.5,
                           fill_color="#151a22", fill_opacity=0.9).move_to(center)
    head = Text(header, font=FONT, font_size=16, color=DIM).next_to(box.get_corner(UP + LEFT), DOWN + RIGHT, 0.15)
    if GIF:  # no slide furniture in the standalone GIFs
        box.set_stroke(opacity=0).set_fill(opacity=0)
        head.set_opacity(0)
    return VGroup(box, head)


def title(text: str) -> Text:
    t = Text(text, font=FONT, font_size=26, color=TEXT).move_to([0, TITLE_Y, 0])
    return t.set_opacity(0) if GIF else t


def chip_top(kind: str, source: str = "") -> VGroup:
    return tag(kind, source).to_edge(UP, buff=0.3).to_edge(RIGHT, buff=0.4)


def note(text: str, center, size=17, color=TEXT) -> Text:
    t = Text(text, font=FONT, font_size=size, color=color, line_spacing=0.8).move_to(center)
    return t.set_opacity(0) if GIF else t
