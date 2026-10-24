"""Beat 3/4 feature-track helpers: the offline LK replay (tools/track_replay.py -> data/tracks/tracks.json) over the
collab60d pack's agent-0 frames, a playback with colour-coded tracks, and square crops for the algorithm panels."""
from __future__ import annotations

import colorsys
import json

import numpy as np
from manim import DOWN, LEFT, RIGHT, UP, Dot, Group, ImageMobject, Rectangle, VGroup, VMobject
from manim.constants import RESAMPLING_ALGORITHMS
from PIL import Image

from .style import DATA, DIM

FRAMES = DATA / "pack_collab60d" / "frames" / "agent0"
TRACKS = DATA / "tracks"


def load_tracks():
    """(tracks.json as a dict, the replayed frames as RGB arrays)."""
    tr = json.loads((TRACKS / "tracks.json").read_text())
    arrs = [np.asarray(Image.open(FRAMES / f"agent0_f{f['frame']}.png").convert("RGB")) for f in tr["frames"]]
    return tr, arrs


def id_color(i: int) -> str:
    r, g, b = colorsys.hsv_to_rgb((i * 0.618034) % 1.0, 0.75, 1.0)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def image(arr: np.ndarray, width: float, nearest: bool = False) -> ImageMobject:
    m = ImageMobject(arr)
    if nearest:
        m.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
    return m.set_width(width)


class TrackPlayer:
    """The replayed frames with every live track as a dot plus a trail over its last `trail` frames, in the track's
    own colour. Fixed objects updated in place (one image, a pool of dots and trails): the renderer fixes its draw
    list at the start of each play, so mobjects swapped in and out by an updater would leave stale copies on screen.
    `group` is what goes on screen; `start()` / `stop()` attach and detach the clock."""

    def __init__(self, tr: dict, arrs: list, img: ImageMobject, per_frame_s: float, trail: int = 3):
        self.frames, self.arrs, self.img, self.per = tr["frames"], arrs, img, per_frame_s
        w = tr["size"][0]
        self.s = img.width / w
        self.pos = [dict(zip(f["ids"], f["pts"])) for f in self.frames]
        self.pix = [ImageMobject(a).pixel_array.copy() for a in arrs]
        m = max(len(f["ids"]) for f in self.frames)
        c = img.get_center()
        self.dots = VGroup(*[Dot(c, radius=0.026) for _ in range(m)])
        self.trails = VGroup(*[VMobject(stroke_width=2.4).set_points_as_corners([c, c]) for _ in range(m)])
        self.overlay = VGroup(self.trails, self.dots)
        self.trail, self.t, self.k = trail, 0.0, 0
        self.show(0, image_too=False)

    def p(self, x, y):
        return self.img.get_corner(UP + LEFT) + RIGHT * x * self.s + DOWN * y * self.s

    def show(self, k: int, image_too: bool = True):
        if image_too:
            self.img.pixel_array = self.pix[k].copy()
        items = list(self.pos[k].items())
        for n, (d, tl) in enumerate(zip(self.dots, self.trails)):
            if n < len(items):
                i, (x, y) = items[n]
                col = id_color(i)
                d.move_to(self.p(x, y)).set_fill(col, 1).set_stroke(width=0)
                hist = [self.pos[j][i] for j in range(max(0, k - self.trail), k + 1) if i in self.pos[j]]
                if len(hist) > 1:
                    tl.set_points_as_corners([self.p(*q) for q in hist]).set_stroke(col, 2.4, 0.9)
                else:
                    tl.set_stroke(opacity=0)
            else:
                d.set_fill(opacity=0)
                tl.set_stroke(opacity=0)

    def _tick(self, m, dt):
        self.t += dt
        k = int(self.t / self.per) % len(self.frames)
        if k != self.k:
            self.k = k
            self.show(k)

    def start(self):
        self.img.add_updater(self._tick)

    def stop(self):
        self.img.remove_updater(self._tick)


class Tile:
    """A square crop of an image around (cx, cy), `half` source pixels each way, shown `width` units wide; p() maps
    full-resolution pixel coordinates (divided by `scale` for a pyramid level) onto it."""

    def __init__(self, arr, cx, cy, half, width, center, scale=1):
        x0, y0 = int(round(cx / scale)) - half, int(round(cy / scale)) - half
        pad = np.pad(arr, ((half, half), (half, half)) + ((0, 0),) * (arr.ndim - 2), mode="edge")
        self.mob = image(pad[y0 + half:y0 + 3 * half, x0 + half:x0 + 3 * half], width, nearest=True).move_to(center)
        self.x0, self.y0, self.k, self.scale, self.half = x0, y0, width / (2 * half), scale, half
        self.frame = Rectangle(width=width, height=width, stroke_color=DIM, stroke_width=1.5).move_to(center)
        self.group = Group(self.mob, self.frame)

    def p(self, x, y):
        tl = self.mob.get_corner(UP + LEFT)
        return tl + RIGHT * (x / self.scale - self.x0) * self.k + DOWN * (y / self.scale - self.y0) * self.k
