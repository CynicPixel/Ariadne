"""Repository assets used by worldgen (read-only): the ChArUco boards drawn by
cpp/runtime/tools/ariadne_charuco_assets.cpp into simulator/assets/charuco/."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

ASSETS = os.environ.get("ARIADNE_ASSETS",
                        os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                                      "simulator", "assets")))


def read_pgm(path) -> np.ndarray:
    """Binary 8-bit PGM (P5, maxval 255), as written by cv::imwrite."""
    data = open(path, "rb").read()
    fields, pos = [], 0
    while len(fields) < 4:                      # magic, width, height, maxval (comments skipped)
        while data[pos:pos + 1].isspace():
            pos += 1
        if data[pos:pos + 1] == b"#":
            pos = data.index(b"\n", pos) + 1
            continue
        end = pos
        while not data[end:end + 1].isspace():
            end += 1
        fields.append(data[pos:end])
        pos = end
    if fields[0] != b"P5" or int(fields[3]) != 255:
        raise ValueError(f"{path}: not an 8-bit binary PGM")
    w, h = int(fields[1]), int(fields[2])
    pos += 1                                    # exactly one whitespace byte before the raster
    px = np.frombuffer(data, np.uint8, count=w * h, offset=pos)
    return px.reshape(h, w).copy()


@dataclass(frozen=True)
class CharucoAssets:
    meta: dict
    images: tuple                               # uint8 HxW per board index

    @property
    def size_m(self) -> tuple[float, float]:
        """(width, height) of the full image incl. the white margin, metres."""
        m = self.meta
        per = m["square_m"] / m["px_per_square"]
        return m["image_wh"][0] * per, m["image_wh"][1] * per

    @property
    def squares_fraction(self) -> tuple[float, float, float, float]:
        """(u0, v0, u1, v1): the squares area inside the image, as fractions (v up)."""
        m = self.meta
        nx = m["squares_xy"][0] + 2 * m["margin_squares"]
        ny = m["squares_xy"][1] + 2 * m["margin_squares"]
        k = m["margin_squares"]
        return k / nx, k / ny, 1 - k / nx, 1 - k / ny


def load_charuco() -> CharucoAssets:
    d = os.path.join(ASSETS, "charuco")
    meta = json.load(open(os.path.join(d, "boards.json")))
    imgs = tuple(read_pgm(os.path.join(d, b["file"])) for b in meta["boards"])
    for img in imgs:
        if (img.shape[1], img.shape[0]) != tuple(meta["image_wh"]):
            raise ValueError("board image size does not match boards.json")
    return CharucoAssets(meta, imgs)
