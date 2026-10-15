"""Read and validate the reconstruction spool (design RECON_1.7, arch §18.2).

The spool is written by C++ (`cpp/reconstruction/spool/keyframe_spool.hpp`). Every
§18.2 field must be present; a missing or malformed field is an error, never a
default. Poses are rectified-left cameras in each agent's odometry frame O_k.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

SCHEMA_VERSION = 1
BUNDLE_FIELDS = ("role", "agent", "kf_id", "t_ns", "frame_id_of_pose", "T_WiC", "K", "width", "height",
                 "baseline_m", "files", "rgb_kind", "exposure_state", "lamp_state", "mask")


class SpoolError(ValueError):
    pass


def pose_from_row_major(v) -> np.ndarray:
    """12 numbers [R | t] row-major -> 4x4 homogeneous."""
    a = np.asarray(v, dtype=np.float64)
    if a.shape != (12,) or not np.all(np.isfinite(a)):
        raise SpoolError(f"pose must be 12 finite numbers, got {v!r}")
    T = np.eye(4)
    T[:3, :] = a.reshape(3, 4)
    if not np.allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-6):
        raise SpoolError("pose rotation is not orthonormal")
    return T


@dataclass(frozen=True)
class Bundle:
    role: str            # "keyframe" | "heldout"
    agent: int
    kf_id: int
    t_ns: int
    frame_id_of_pose: int
    T_O_C: np.ndarray    # 4x4, camera -> agent odometry frame
    K: np.ndarray        # 3x3 at spool scale
    width: int
    height: int
    stem: Path

    def rgb(self) -> np.ndarray:
        """HxWx3 uint8, RGB order."""
        bgr = _imread(self.stem.with_name(self.stem.name + "_rgb.png"), cv2.IMREAD_COLOR)
        if bgr.shape != (self.height, self.width, 3):
            raise SpoolError(f"{self.stem}: rgb shape {bgr.shape} != {(self.height, self.width, 3)}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def depth(self) -> tuple[np.ndarray, np.ndarray]:
        """(Z, sigma_Z), float32 metres, NaN = no depth. Keyframes only."""
        if self.role != "keyframe":
            raise SpoolError(f"{self.stem}: held-out frames carry no depth")
        return (_half_png(self.stem.with_name(self.stem.name + "_depth.png"), self.height, self.width),
                _half_png(self.stem.with_name(self.stem.name + "_sigma.png"), self.height, self.width))


@dataclass(frozen=True)
class Spool:
    root: Path
    run_id: str
    scene_id: str
    merge_ok: bool
    T_O0_O1: np.ndarray
    merge_edge: str
    bundles: tuple[Bundle, ...]

    def keyframes(self) -> list[Bundle]:
        return [b for b in self.bundles if b.role == "keyframe"]

    def heldout(self) -> list[Bundle]:
        return [b for b in self.bundles if b.role == "heldout"]

    def T_O0_C(self, b: Bundle) -> np.ndarray:
        """A bundle's camera in agent 0's odometry frame (the 1.6 merge applied to agent 1)."""
        if b.agent == 0:
            return b.T_O_C
        if b.agent == 1:
            if not self.merge_ok:
                raise SpoolError("agent 1 bundle but the spool has no merge")
            return self.T_O0_O1 @ b.T_O_C
        raise SpoolError(f"agent {b.agent}: 1.7 handles two agents")


def _imread(path: Path, flags: int) -> np.ndarray:
    img = cv2.imread(str(path), flags)
    if img is None:
        raise SpoolError(f"cannot read {path}")
    return img


def _half_png(path: Path, h: int, w: int) -> np.ndarray:
    bits = _imread(path, cv2.IMREAD_UNCHANGED)
    if bits.dtype != np.uint16 or bits.shape != (h, w):
        raise SpoolError(f"{path}: expected uint16 {h}x{w}, got {bits.dtype} {bits.shape}")
    return np.ascontiguousarray(bits).view(np.float16).astype(np.float32)


def load(root: str | Path) -> Spool:
    root = Path(root)
    y = yaml.safe_load((root / "spool.yaml").read_text())
    if not isinstance(y, dict):
        raise SpoolError("spool.yaml is not a mapping")
    if y.get("schema_version") != SCHEMA_VERSION:
        raise SpoolError(f"schema_version {y.get('schema_version')!r} != {SCHEMA_VERSION}")
    for k in ("run_id", "scene_id", "merge", "bundles"):
        if k not in y:
            raise SpoolError(f"spool.yaml: missing {k}")
    m = y["merge"]
    bundles = []
    for i, e in enumerate(y["bundles"]):
        missing = [f for f in BUNDLE_FIELDS if f not in e]
        if missing:
            raise SpoolError(f"bundle {i}: missing {missing}")
        if e["role"] not in ("keyframe", "heldout"):
            raise SpoolError(f"bundle {i}: role {e['role']!r}")
        fx, fy, cx, cy = (float(v) for v in e["K"])
        bundles.append(Bundle(role=e["role"], agent=int(e["agent"]), kf_id=int(e["kf_id"]), t_ns=int(e["t_ns"]),
                              frame_id_of_pose=int(e["frame_id_of_pose"]), T_O_C=pose_from_row_major(e["T_WiC"]),
                              K=np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]), width=int(e["width"]),
                              height=int(e["height"]), stem=root / e["files"]))
    if not bundles:
        raise SpoolError("spool has no bundles")
    return Spool(root=root, run_id=str(y["run_id"]), scene_id=str(y["scene_id"]), merge_ok=bool(m["ok"]),
                 T_O0_O1=pose_from_row_major(m["T_O0_O1"]), merge_edge=str(m["edge"]), bundles=tuple(bundles))
