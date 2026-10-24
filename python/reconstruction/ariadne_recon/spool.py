"""Read and validate the reconstruction spool (design RECON_1.7, arch §18.2).

The spool is written by C++ (`cpp/reconstruction/spool/keyframe_spool.hpp`). Every
§18.2 field must be present; a missing or malformed field is an error, never a
default. Poses are rectified-left cameras in each agent's odometry frame O_k.

Schema 2 (6.1, D195) adds the §19.1 albedo proxy per keyframe (`rgb_kind: albedo_proxy`, `_albedo.png`), the APPLIED
photometric state per bundle (exposure x gain, lamp power, controller decision), and the run's lamp (its calibration
copied into the spool, the render intensity, R_rect_raw). Schema 1 spools load unchanged.

Schema 3 (agreed with A1 and M, 2026-10-23) = schema 2's fields plus a `merges:` list REPLACING the single `merge:`
block, for N agents: {agent: k, ok, T_O0_O: 12 row-major floats (agent k's odometry frame in agent 0's), edge}, one
entry per non-anchor agent; agent 0 anchors. Schema < 3 reads `merge:` (agent 1 into agent 0); schema 3 requires
`merges:` and refuses a `merge:` block. N = 1 + the largest bundle agent id.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

SCHEMA_VERSIONS = (1, 2, 3)
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
class Lamp:
    """Schema 2: the lamp the albedo proxy was normalised with (C++ SpoolLamp)."""
    calibration: Path      # the copied calibration YAML (dataset_tools.lamp_calib format)
    source: str            # where it was copied from (provenance only)
    name: str
    rig: str
    intensity: float       # the render's lamp intensity; / the calibration's intensity_ref = its power
    R_rect_raw: np.ndarray  # 3x3, rectified-left <- raw-left (the calibration's camera frame)


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
    rgb_kind: str = "raw_radiance"          # the TRAINING image: raw_radiance (rgb) | albedo_proxy (albedo)
    exposure_gain: float | None = None      # schema 2: applied exposure x gain; None = not modelled
    lamp_rel: float | None = None           # schema 2: applied lamp power (D); None = inert
    decision_id: int | None = None

    def albedo(self) -> np.ndarray:
        """HxW float32 reflectance in calibration units, NaN where the proxy is not defined (schema 2 keyframes)."""
        if self.rgb_kind != "albedo_proxy":
            raise SpoolError(f"{self.stem}: rgb_kind {self.rgb_kind!r} has no albedo proxy")
        return _half_png(self.stem.with_name(self.stem.name + "_albedo.png"), self.height, self.width)

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
    schema_version: int
    lamp: Lamp | None
    run_id: str
    scene_id: str
    merge_ok: bool
    T_O0_O1: np.ndarray
    merge_edge: str
    bundles: tuple[Bundle, ...]
    merges: dict | None = None      # schema 3: agent k -> (ok, T_O0_Ok 4x4, edge)

    def keyframes(self) -> list[Bundle]:
        return [b for b in self.bundles if b.role == "keyframe"]

    def heldout(self) -> list[Bundle]:
        return [b for b in self.bundles if b.role == "heldout"]

    def T_O0_C(self, b: Bundle) -> np.ndarray:
        """A bundle's camera in agent 0's odometry frame (the 1.6 merge applied to agent 1; schema 3: agent k's)."""
        if b.agent == 0:
            return b.T_O_C
        if self.merges is not None:
            if b.agent not in self.merges or not self.merges[b.agent][0]:
                raise SpoolError(f"agent {b.agent} bundle but the spool has no ok merge for it")
            return self.merges[b.agent][1] @ b.T_O_C
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


def e_row(merges: dict, k: int) -> list:
    """Agent k's merge as the 12 row-major numbers of schema < 3 (identity when absent)."""
    T = merges[k][1] if k in merges else np.eye(4)
    return [float(x) for x in T[:3, :].ravel()]


def load(root: str | Path) -> Spool:
    root = Path(root)
    y = yaml.safe_load((root / "spool.yaml").read_text())
    if not isinstance(y, dict):
        raise SpoolError("spool.yaml is not a mapping")
    version = y.get("schema_version")
    if version not in SCHEMA_VERSIONS:
        raise SpoolError(f"schema_version {version!r} not in {SCHEMA_VERSIONS}")
    for k in ("run_id", "scene_id", "merges" if version == 3 else "merge", "bundles"):
        if k not in y:
            raise SpoolError(f"spool.yaml: missing {k}")
    merges = None
    if version == 3:
        if "merge" in y:
            raise SpoolError("schema 3 carries merges:, not merge:")
        merges = {}
        for e in y["merges"]:
            for k in ("agent", "ok", "T_O0_O", "edge"):
                if k not in e:
                    raise SpoolError(f"merges entry: missing {k}")
            merges[int(e["agent"])] = (bool(e["ok"]), pose_from_row_major(e["T_O0_O"]), str(e["edge"]))
        m = {"ok": bool(merges.get(1, (False,))[0]), "T_O0_O1": e_row(merges, 1), "edge": merges.get(1, (0, 0, "none"))[2]}
    else:
        m = y["merge"]
    lamp = None
    if "lamp" in y:
        if version < 2:
            raise SpoolError("a lamp record needs schema_version 2")
        lm = y["lamp"]
        for k in ("calibration", "calibration_source", "calibration_name", "rig", "intensity", "R_rect_raw"):
            if k not in lm:
                raise SpoolError(f"spool.yaml lamp: missing {k}")
        R = np.asarray(lm["R_rect_raw"], dtype=np.float64)
        if R.shape != (9,) or not np.allclose(R.reshape(3, 3) @ R.reshape(3, 3).T, np.eye(3), atol=1e-6):
            raise SpoolError("lamp R_rect_raw must be a row-major rotation")
        cal = root / lm["calibration"]
        if not cal.is_file():
            raise SpoolError(f"lamp calibration {cal} missing from the spool")
        lamp = Lamp(calibration=cal, source=str(lm["calibration_source"]), name=str(lm["calibration_name"]),
                    rig=str(lm["rig"]), intensity=float(lm["intensity"]), R_rect_raw=R.reshape(3, 3))
    bundles = []
    for i, e in enumerate(y["bundles"]):
        missing = [f for f in BUNDLE_FIELDS if f not in e]
        if missing:
            raise SpoolError(f"bundle {i}: missing {missing}")
        if e["role"] not in ("keyframe", "heldout"):
            raise SpoolError(f"bundle {i}: role {e['role']!r}")
        fx, fy, cx, cy = (float(v) for v in e["K"])
        kind = str(e["rgb_kind"])
        if kind not in ("raw_radiance", "albedo_proxy") or (kind == "albedo_proxy" and e["role"] != "keyframe"):
            raise SpoolError(f"bundle {i}: rgb_kind {kind!r} on a {e['role']}")
        if kind == "albedo_proxy" and lamp is None:
            raise SpoolError(f"bundle {i}: an albedo proxy without the spool's lamp record")
        ex, ls = e["exposure_state"], e["lamp_state"]
        photo = isinstance(ex, dict)
        if photo != isinstance(ls, dict) or (not photo and (ex, ls) != ("not_modeled", "inert")):
            raise SpoolError(f"bundle {i}: exposure_state/lamp_state {ex!r}/{ls!r}")
        bundles.append(Bundle(role=e["role"], agent=int(e["agent"]), kf_id=int(e["kf_id"]), t_ns=int(e["t_ns"]),
                              frame_id_of_pose=int(e["frame_id_of_pose"]), T_O_C=pose_from_row_major(e["T_WiC"]),
                              K=np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]), width=int(e["width"]),
                              height=int(e["height"]), stem=root / e["files"], rgb_kind=kind,
                              exposure_gain=float(ex["exposure_gain"]) if photo else None,
                              lamp_rel=float(ls["lamp_rel"]) if photo else None,
                              decision_id=int(ex["decision_id"]) if photo else None))
    if not bundles:
        raise SpoolError("spool has no bundles")
    return Spool(root=root, schema_version=int(version), lamp=lamp, run_id=str(y["run_id"]), scene_id=str(y["scene_id"]), merge_ok=bool(m["ok"]),
                 T_O0_O1=pose_from_row_major(m["T_O0_O1"]), merge_edge=str(m["edge"]), bundles=tuple(bundles),
                 merges=merges)
