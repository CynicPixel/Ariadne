"""The live keyframe tee's wire format (reconstruction.md §3; contract agreed with B 2026-10-24, D231). numpy + stdlib only.

Framing = B's LinkFrame (cpp/runtime/live/link_frame.hpp, D150), little-endian:
  frame = u32 magic "ARL1" | u32 topic_len | topic | u32 payload_len | payload
The service LISTENS on a unix stream socket (ARIADNE_RECON_SOCK); ariadne_live connects and tees from deliver_kf. The
runtime side is bounded and drop-oldest, and a no-op when no service listens: the splat never back-pressures VO.

Topics:
  #kf     payload = u32 json_len | JSON | Z | S | RGB
          JSON {"agent", "id", "t_ns", "pose_version" (0 until PGO re-anchors), "T_O_C": 12 row-major (3x4),
                "K": [fx, fy, cx, cy] full-res (the RGB's), "Kz": [...] spool scale (Z/S's), "w", "h", "zw", "zh",
                "training_image": "rgb"}
          Z, S  float16 LE, zh x zw each, 0 = invalid (HalfAsSpool(MaskedDepth32(...)), the spool convention)
          RGB   rgb8, h x w x 3: the rectified left image deliver_kf holds
  #t_w_o  JSON {"agent", "t_s", "xyzquat": [x, y, z, qx, qy, qz, qw]} (qw last): agent k's odometry frame in W, sent
          once per agent at start (the briefing constant) and again on every change
  #pose_version  (later) {"agent", "version", "poses": {"<kf_id>": 12 row-major}}

Training resolution: the replay trains on the spool (scale 0.5: RGB by INTER_AREA, Z/S nearest, Kz). Live does the
same: half_rgb() is the exact 2x2 area average OpenCV's INTER_AREA computes for an integer factor of 2
((a + b + c + d + 2) >> 2, resizeAreaFast), so a live keyframe trains on the same pixels its spool would hold.
"""
from __future__ import annotations

import json
import socket
import struct

import numpy as np

MAGIC = 0x314C5241          # bytes "ARL1"
MAX_TOPIC, MAX_PAYLOAD = 256, 64 << 20


class LinkError(RuntimeError):
    pass


def _read_exact(sock: socket.socket, n: int, eof_ok: bool) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            if not buf and eof_ok:
                return None
            raise LinkError("connection closed mid-frame")
        buf += chunk
    return bytes(buf)


def read_frame(sock: socket.socket) -> tuple[str, bytes] | None:
    """(topic, payload), or None on a clean EOF between frames. Raises LinkError on a malformed frame."""
    h = _read_exact(sock, 8, eof_ok=True)
    if h is None:
        return None
    magic, tl = struct.unpack("<II", h)
    if magic != MAGIC:
        raise LinkError("bad magic")
    if tl > MAX_TOPIC:
        raise LinkError(f"topic length {tl}")
    topic = _read_exact(sock, tl, False).decode()
    (n,) = struct.unpack("<I", _read_exact(sock, 4, False))
    if n > MAX_PAYLOAD:
        raise LinkError(f"payload length {n}")
    return topic, _read_exact(sock, n, False) if n else b""


def frame_bytes(topic: str, payload: bytes) -> bytes:
    t = topic.encode()
    return struct.pack("<II", MAGIC, len(t)) + t + struct.pack("<I", len(payload)) + payload


def decode_kf(payload: bytes) -> dict:
    try:   # any header defect is a malformed record (dropped and counted), never a crash of the service
        (jl,) = struct.unpack_from("<I", payload, 0)
        hdr = json.loads(payload[4:4 + jl])
        w, h, zw, zh = int(hdr["w"]), int(hdr["h"]), int(hdr["zw"]), int(hdr["zh"])
        missing = [k for k in ("agent", "id", "t_ns", "pose_version", "T_O_C", "K", "Kz", "training_image")
                   if k not in hdr]
        if missing:
            raise KeyError(", ".join(missing))
    except (struct.error, ValueError, KeyError, TypeError) as e:
        raise LinkError(f"#kf header: {type(e).__name__} {e}") from None
    nz, nrgb = zw * zh * 2, w * h * 3
    if len(payload) != 4 + jl + 2 * nz + nrgb:
        raise LinkError(f"#kf agent {hdr.get('agent')} id {hdr.get('id')}: payload {len(payload)} B, expected "
                        f"{4 + jl + 2 * nz + nrgb}")
    o = 4 + jl
    Z = np.frombuffer(payload, "<f2", zw * zh, o).reshape(zh, zw).astype(np.float32)
    S = np.frombuffer(payload, "<f2", zw * zh, o + nz).reshape(zh, zw).astype(np.float32)
    rgb = np.frombuffer(payload, np.uint8, nrgb, o + 2 * nz).reshape(h, w, 3)
    return {**hdr, "Z": Z, "S": S, "rgb": rgb}


def encode_kf(hdr: dict, Z: np.ndarray, S: np.ndarray, rgb: np.ndarray) -> bytes:
    """The runtime side's encoding (the mock and the tests use it; B's C++ tee writes the same bytes)."""
    j = json.dumps(hdr).encode()
    return (struct.pack("<I", len(j)) + j + np.asarray(Z, "<f2").tobytes() + np.asarray(S, "<f2").tobytes()
            + np.ascontiguousarray(rgb, np.uint8).tobytes())


def pose12(v) -> np.ndarray:
    T = np.eye(4)
    T[:3, :] = np.asarray(v, float).reshape(3, 4)
    return T


def decode_t_w_o(payload: bytes) -> tuple[int, np.ndarray]:
    m = json.loads(payload)
    x, y, z, qx, qy, qz, qw = (float(v) for v in m["xyzquat"])
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if not 0.99 < n < 1.01:
        raise LinkError(f"#t_w_o agent {m['agent']}: quaternion norm {n}")
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    T = np.eye(4)
    T[:3, :3] = [[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                 [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                 [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]]
    T[:3, 3] = [x, y, z]
    return int(m["agent"]), T


def half_rgb(rgb: np.ndarray) -> np.ndarray:
    """uint8 HxWx3 -> (H/2)x(W/2)x3, bit-exact with cv2.resize(..., 0.5, 0.5, INTER_AREA) for even H, W."""
    h, w = rgb.shape[:2]
    if h % 2 or w % 2:
        raise LinkError(f"RGB {w}x{h} is not even: the 2x area average is undefined")
    s = rgb.reshape(h // 2, 2, w // 2, 2, 3).astype(np.uint16).sum(axis=(1, 3))
    return ((s + 2) >> 2).astype(np.uint8)


def K3(k) -> np.ndarray:
    fx, fy, cx, cy = (float(v) for v in k)
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
