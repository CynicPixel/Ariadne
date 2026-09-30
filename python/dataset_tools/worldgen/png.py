"""Deterministic PNG writer/reader (8-bit gray or RGB, no metadata chunks).

Only IHDR/IDAT/IEND are written, filter type 0 on every row, zlib level fixed.
The *file* bytes still depend on the zlib build, so golden tests hash the decoded
pixel array (pixel_digest), never the file.
"""
from __future__ import annotations

import hashlib
import struct
import zlib

import numpy as np

_SIG = b"\x89PNG\r\n\x1a\n"
ZLIB_LEVEL = 6


def _chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def write_png(path, img: np.ndarray) -> None:
    if img.dtype != np.uint8:
        raise TypeError("write_png takes uint8 images")
    if img.ndim == 2:
        color_type, channels = 0, 1
    elif img.ndim == 3 and img.shape[2] == 3:
        color_type, channels = 2, 3
    else:
        raise ValueError("write_png takes HxW or HxWx3 images")
    h, w = img.shape[:2]
    rows = np.ascontiguousarray(img).reshape(h, w * channels)
    raw = np.concatenate([np.zeros((h, 1), np.uint8), rows], axis=1).tobytes()
    ihdr = struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0)
    with open(path, "wb") as f:
        f.write(_SIG + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", zlib.compress(raw, ZLIB_LEVEL))
                + _chunk(b"IEND", b""))


def read_png(path) -> np.ndarray:
    """Reader for the files write_png produces (filter 0 only); used by tests."""
    data = open(path, "rb").read()
    if data[:8] != _SIG:
        raise ValueError("not a PNG")
    pos, idat = 8, b""
    w = h = color_type = None
    while pos < len(data):
        (n,) = struct.unpack(">I", data[pos:pos + 4])
        tag, body = data[pos + 4:pos + 8], data[pos + 8:pos + 8 + n]
        pos += 12 + n
        if tag == b"IHDR":
            w, h, depth, color_type, _, _, _ = struct.unpack(">IIBBBBB", body)
            if depth != 8:
                raise ValueError("8-bit only")
        elif tag == b"IDAT":
            idat += body
    ch = {0: 1, 2: 3}[color_type]
    rows = np.frombuffer(zlib.decompress(idat), np.uint8).reshape(h, 1 + w * ch)
    if np.any(rows[:, 0] != 0):
        raise ValueError("only filter type 0 is supported")
    px = rows[:, 1:].reshape(h, w, ch) if ch == 3 else rows[:, 1:].reshape(h, w)
    return px.copy()


def pixel_digest(img: np.ndarray) -> str:
    """sha256 over shape + dtype + raw pixels: the golden value for a texture."""
    h = hashlib.sha256()
    h.update(repr((img.shape, str(img.dtype))).encode())
    h.update(np.ascontiguousarray(img).tobytes())
    return h.hexdigest()
