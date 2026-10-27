"""Shared look and helpers for every scene: palette, the world-to-frame map, source tags, timed captions.

Captions are the film's only words until the TTS pass (2026-10-24). Each scene records (start_s, dur_s,
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

GIF = os.environ.get("ARIADNE_GIF") == "1"  # standalone-GIF renders: chips, titles, panels and subtitles drawn invisible
BG = "#0e1116"
WALL = "#d8dee9"
FLOOR = "#1b2028"
DIM = "#5c6370"
TEXT = "#e5e9f0"
AGENT = ["#58a6ff", "#f0883e", "#56d364"]  # agent 0, 1, 2: held for the whole film
TARGET = "#ff5c5c"
THREAD = "#e3b341"  # the Ariadne thread (egress route)
TAG_COLOR = {"measured": "#56d364", "published": "#58a6ff", "design": "#d2a8ff", "illustrative": "#8b949e",
             "derived": "#e3b341"}  # derived: computed here from measured run data
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
    g = VGroup(box, label.move_to(box))
    return g.set_opacity(0) if GIF else g  # same geometry either way, so layouts do not move


WPM = 150  # narration pace the cue timing assumes (the TTS / voice pass reads the same cue sheet)
LINE_CHARS = 52  # subtitle line length; at most two lines per cue


def wrap(text: str, width: int = LINE_CHARS) -> list[str]:
    """Greedy wrap; a two-line result is re-split at the word break nearest the middle (no dangling last word)."""
    lines, cur = [], ""
    for w in text.split():
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    lines += [cur] if cur else []
    if len(lines) == 2:
        words = text.split()
        best = min(range(1, len(words)), key=lambda k: abs(len(" ".join(words[:k])) - len(" ".join(words[k:]))))
        lines = [" ".join(words[:best]), " ".join(words[best:])]
    return lines


def cues_of(text: str) -> list[str]:
    """Split narration into subtitle cues of at most two lines, breaking at sentence ends, then at commas."""
    import re
    out = []
    for sent in re.split(r"(?<=[.!?])\s+", text.strip()):
        lines = wrap(sent)
        if len(lines) <= 2:
            out.append(sent)
            continue
        parts, cur = re.split(r"(?<=[,:;])\s+", sent), ""
        for p in parts:
            if cur and len(wrap(f"{cur} {p}")) > 2:
                out.append(cur)
                cur = p
            else:
                cur = f"{cur} {p}".strip()
        out.append(cur)
    return out


class CaptionedScene(Scene):
    """Scene with `say()`: narration shown as subtitles, timed by word count, logged to the cue sheet."""

    caption_pos = DOWN * 3.35

    def setup(self):
        self.cues: list[dict] = []
        self._caption = None

    def _subtitle(self, text: str) -> VGroup:
        t = Text("\n".join(wrap(text)), font=FONT, font_size=24, color=TEXT, line_spacing=0.8)
        box = RoundedRectangle(corner_radius=0.1, width=t.width + 0.4, height=t.height + 0.22, stroke_width=0,
                               fill_color=BG, fill_opacity=0.72)
        g = VGroup(box, t.move_to(box)).move_to(self.caption_pos)
        if GIF:
            g.set_opacity(0)
        if hasattr(self, "add_fixed_in_frame_mobjects"):  # 3D scenes: subtitles stay on the screen, not in the world
            self.add_fixed_in_frame_mobjects(g)
        return g

    def say(self, text: str, dur: float | None = None, run_with=None):
        """Narrate `text`: one subtitle cue per sentence (or comma clause when a sentence exceeds two lines), each held
        for its word count at WPM (or `dur` split by words). `run_with` plays alongside the first cue at its own
        run_time; a longer animation stretches that cue, and the cue sheet records actual times."""
        parts = cues_of(text)
        words = [len(p.split()) for p in parts]
        total = dur if dur is not None else sum(words) * 60 / WPM
        for i, (p, w) in enumerate(zip(parts, words)):
            want = total * w / sum(words)
            cap = self._subtitle(p)
            anims = [FadeIn(cap, run_time=0.3)]
            if self._caption:
                anims.append(FadeOut(self._caption, run_time=0.25))
            t0 = self.renderer.time
            self.play(*anims, *((run_with or []) if i == 0 else []))
            self._caption = cap
            rest = want - (self.renderer.time - t0)
            if rest > 0.01:
                self.wait(rest)
            self.cues.append({"start_s": round(t0, 3), "dur_s": round(self.renderer.time - t0, 3), "text": p})

    def say_steps(self, text: str, steps: list):
        """Narrate `text` as ONE cue while a sequence of plays runs under it: each step is a list of animations played
        together (or a function returning them, built when the step runs: `.animate` targets are snapshots), or a number
        of seconds to wait. (A Succession inside `say` would add every later mobject to the
        scene at full opacity before its FadeIn starts.) The cue lasts its word count at WPM, or the steps if longer."""
        cap = self._subtitle(text)
        anims = [FadeIn(cap, run_time=0.3)]
        if self._caption:
            anims.append(FadeOut(self._caption, run_time=0.25))
        t0 = self.renderer.time
        first = steps[0] if steps and not isinstance(steps[0], (int, float)) else []
        self.play(*anims, *(first() if callable(first) else first))
        self._caption = cap
        for st in steps[1 if first else 0:]:
            if isinstance(st, (int, float)):
                self.wait(st)
            else:  # a callable step builds its animations now, from the state the earlier steps left
                self.play(*(st() if callable(st) else st))
        rest = len(text.split()) * 60 / WPM - (self.renderer.time - t0)
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
