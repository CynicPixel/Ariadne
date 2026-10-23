"""A stand-in for B's runtime tee (link.py's contract, D231): plays per-agent spools into the service's socket.

  python -m ariadne_recon.live.mock_tee --sock SOCK --spool DIR [DIR ...] [--scene DIR] [--pace 1]

Sends what ariadne_live's deliver_kf tee sends: #t_w_o per agent first (the briefing constant from --scene, else
identity), then every keyframe as #kf in mission-time order. The spool holds RGB at spool scale, the tee sends it at
full res: the mock upsamples 2x nearest, which half_rgb() maps back to the spool's pixels exactly, so a mocked live
run trains on the same pixels as the replay. K (full res) follows B's formula from the spool camera Kz.
For bring-up and tests only; the live demo uses B's C++ tee.
"""
from __future__ import annotations

import argparse
import json
import socket
import time

import numpy as np

from ..spool import load
from .link import encode_kf, frame_bytes
from .placement import briefing_T_W_O


def xyzquat(T: np.ndarray) -> list[float]:
    """[x, y, z, qx, qy, qz, qw]. Shepperd's method: divide by the largest of 4w^2, 4x^2, 4y^2, 4z^2, so a rotation
    near 180 deg (trace -1, e.g. an entrance facing west) stays exact."""
    R = T[:3, :3]
    tr = np.trace(R)
    i = int(np.argmax([tr, R[0, 0], R[1, 1], R[2, 2]]))
    if i == 0:
        r = np.sqrt(1 + tr) * 2
        q = [(R[2, 1] - R[1, 2]) / r, (R[0, 2] - R[2, 0]) / r, (R[1, 0] - R[0, 1]) / r, r / 4]
    elif i == 1:
        r = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [r / 4, (R[0, 1] + R[1, 0]) / r, (R[0, 2] + R[2, 0]) / r, (R[2, 1] - R[1, 2]) / r]
    elif i == 2:
        r = np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 1] + R[1, 0]) / r, r / 4, (R[1, 2] + R[2, 1]) / r, (R[0, 2] - R[2, 0]) / r]
    else:
        r = np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[0, 2] + R[2, 0]) / r, (R[1, 2] + R[2, 1]) / r, r / 4, (R[1, 0] - R[0, 1]) / r]
    return [float(v) for v in (*T[:3, 3], *q)]


def kf_payload(b) -> bytes:
    fx, fy, cx, cy = b.K[0, 0], b.K[1, 1], b.K[0, 2], b.K[1, 2]
    Z, S = b.depth()
    Z, S = np.nan_to_num(Z, nan=0.0), np.nan_to_num(S, nan=0.0)       # the tee's convention: 0 = invalid
    rgb = np.repeat(np.repeat(b.rgb(), 2, axis=0), 2, axis=1)
    hdr = {"agent": b.agent, "id": b.kf_id, "t_ns": b.t_ns, "pose_version": 0,
           "T_O_C": [float(x) for x in b.T_O_C[:3, :].ravel()],
           "K": [2 * fx, 2 * fy, 2 * (cx + 0.5) - 0.5, 2 * (cy + 0.5) - 0.5], "Kz": [fx, fy, cx, cy],
           "w": rgb.shape[1], "h": rgb.shape[0], "zw": Z.shape[1], "zh": Z.shape[0], "training_image": "rgb"}
    return encode_kf(hdr, Z, S, rgb)


def send(sock_path: str, spool_dirs: list[str], scene: str | None, pace: float = 0.0, connect_s: float = 10.0) -> int:
    spools = [load(d) for d in spool_dirs]
    kfs = sorted((b for sp in spools for b in sp.keyframes()), key=lambda b: (b.t_ns, b.agent))
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    t_end = time.monotonic() + connect_s
    while True:
        try:
            s.connect(sock_path)
            break
        except (FileNotFoundError, ConnectionRefusedError):
            if time.monotonic() > t_end:
                raise
            time.sleep(0.05)
    with s:
        for k in sorted({b.agent for b in kfs}):
            T = briefing_T_W_O(scene, k) if scene else np.eye(4)
            s.sendall(frame_bytes("#t_w_o", json.dumps({"agent": k, "t_s": 0.0, "xyzquat": xyzquat(T)}).encode()))
        wall0 = time.monotonic()
        for b in kfs:
            if pace > 0:
                wait = (b.t_ns - kfs[0].t_ns) * 1e-9 / pace - (time.monotonic() - wall0)
                if wait > 0:
                    time.sleep(wait)
            s.sendall(frame_bytes("#kf", kf_payload(b)))
    return len(kfs)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sock", required=True)
    ap.add_argument("--spool", required=True, nargs="+")
    ap.add_argument("--scene", default=None)
    ap.add_argument("--pace", type=float, default=0.0)
    a = ap.parse_args(argv)
    print(f"mock_tee: sent {send(a.sock, a.spool, a.scene, a.pace)} keyframes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
