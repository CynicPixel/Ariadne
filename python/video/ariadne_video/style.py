"""Shared look and helpers for every scene: palette, the world-to-frame map, source tags, timed captions.

Captions are the film's only words until the TTS pass (maintainer, 2026-10-24). Each scene records (start_s, dur_s,
text) and writes media/captions/<Scene>.json, so a voice can be timed against the same cue sheet later.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from manim import (DOWN, LEFT, ORIGIN, UP, RIGHT, FadeIn, FadeOut, RoundedRectangle, Scene, Text, VGroup,
                   config)

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("ARIADNE_VIDEO_DATA", ROOT / "data"))

BG = "#0e1116"
WALL = "#d8dee9"
FLOOR = "#1b2028"
DIM = "#5c6370"
TEXT = "#e5e9f0"
AGENT = ["#58a6ff", "#f0883e", "#56d364"]  # agent 0, 1, 2: held for the whole film
TARGET = "#ff5c5c"
THREAD = "#e3b341"  # the Ariadne thread (egress route)
TAG_COLOR = {"measured": "#56d364", "published": "#58a6ff", "design": "#d2a8ff", "illustrative": "#8b949e"}
FONT = "Noto Sans"  # present in the pinned image and on the host

config.background_color = BG


class MapFrame:
    """World metres (x east, y north) -> Manim frame points, fitted to a box and centred on `center`."""

    def __init__(self, xmin: float, xmax: float, ymin: float, ymax: float, height: float = 7.2,
                 center=ORIGIN):
        self.c = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2])
        self.s = min(height / (ymax - ymin), 13.4 / (xmax - xmin))
        self.center = np.array(center, dtype=float)

    def p(self, x: float, y: float, z: float = 0.0) -> np.ndarray:
        return self.center + np.array([(x - self.c[0]) * self.s, (y - self.c[1]) * self.s, z * self.s])

    def length(self, m: float) -> float:
        return m * self.s


def tag(kind: str, source: str = "") -> VGroup:
    """The small on-screen source chip every number carries: measured / published / design / illustrative."""
    label = Text(kind if not source else f"{kind} · {source}", font=FONT, font_size=16, color=TAG_COLOR[kind])
    box = RoundedRectangle(corner_radius=0.08, width=label.width + 0.25, height=label.height + 0.15,
                           stroke_color=TAG_COLOR[kind], stroke_width=1.2)
    return VGroup(box, label.move_to(box))


class CaptionedScene(Scene):
    """Scene with `say()`: show a caption for `dur` seconds and log it to the cue sheet."""

    caption_pos = DOWN * 3.45

    def setup(self):
        self.cues: list[dict] = []
        self._caption = None

    def say(self, text: str, dur: float = 2.5, run_with=None):
        """Caption on screen for `dur` s from now. `run_with` animations play alongside at their own run_time
        (a longer one stretches the cue, which the cue sheet records as the actual duration)."""
        cap = Text(text, font=FONT, font_size=26, color=TEXT).move_to(self.caption_pos)
        anims = [FadeIn(cap, shift=UP * 0.1, run_time=0.5)]
        if self._caption:
            anims.append(FadeOut(self._caption, run_time=0.4))
        t0 = self.renderer.time
        self.play(*anims, *(run_with or []))
        self._caption = cap
        rest = dur - (self.renderer.time - t0)
        if rest > 0.01:
            self.wait(rest)
        self.cues.append({"start_s": round(t0, 3), "dur_s": round(self.renderer.time - t0, 3), "text": text})

    def clear_caption(self):
        if self._caption:
            self.play(FadeOut(self._caption), run_time=0.4)
            self._caption = None

    def tear_down(self):
        out = Path(config.media_dir) / "captions"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{type(self).__name__}.json").write_text(json.dumps(self.cues, indent=1))


__all__ = ["AGENT", "BG", "DATA", "DIM", "FLOOR", "FONT", "MapFrame", "TARGET", "TEXT", "THREAD", "WALL",
           "CaptionedScene", "tag", "LEFT", "RIGHT", "UP", "DOWN"]
