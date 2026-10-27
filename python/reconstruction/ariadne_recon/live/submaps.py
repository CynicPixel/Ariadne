"""Keyframe-anchored submaps (arch §18.6; reconstruction.md §2, D190).

Each submap is anchored to its FIRST keyframe. Its Gaussians live in that keyframe's camera frame ("anchor frame")
and are never rewritten by a pose update: a pose-graph update changes only T_O_anchor, so the whole submap moves
rigidly. The console places a submap at T_W_O[agent] @ T_O_anchor (the runtime tee sends T_W_O per agent), so a PGO update
or an O -> W change is one 4x4 per submap: no retraining, no re-download (exit-gate bullet 2).

Intra-submap drift is deliberately not corrected (arch §18.6); submaps are kept short instead: a submap closes after
`max_keyframes` keyframes (arch: 20-40; reconstruction.md uses 30) or when told a loop closure bent the trajectory.

Pure numpy: no torch, no GPU. The trainer holds the Gaussian tensors; this module holds the frames.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def inv_se3(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    o = np.eye(4)
    o[:3, :3] = R.T
    o[:3, 3] = -R.T @ t
    return o


@dataclass
class Submap:
    sid: int
    agent: int
    anchor_kf: int
    T_O_anchor: np.ndarray                 # 4x4: the anchor keyframe's camera in the agent's odometry frame
    pose_version: int                      # the graph version T_O_anchor was taken at
    keyframes: list[int] = field(default_factory=list)
    T_anchor_C: dict[int, np.ndarray] = field(default_factory=dict)   # each member keyframe in the anchor frame
    closed: bool = False

    def placement(self, T_W_O: np.ndarray) -> np.ndarray:
        """Where the console draws this submap: anchor frame -> mission world W."""
        return T_W_O @ self.T_O_anchor


class SubmapManager:
    def __init__(self, max_keyframes: int = 30):
        if max_keyframes < 1:
            raise ValueError("max_keyframes >= 1")
        self.max_keyframes = max_keyframes
        self.submaps: list[Submap] = []
        self._active: dict[int, Submap] = {}   # agent -> its open submap

    def active(self, agent: int) -> Submap | None:
        return self._active.get(agent)

    def add_keyframe(self, agent: int, kf_id: int, T_O_C: np.ndarray, pose_version: int) -> tuple[Submap, bool]:
        """Assign a keyframe; returns (its submap, whether that submap was opened by it)."""
        sm = self._active.get(agent)
        opened = False
        if sm is None or len(sm.keyframes) >= self.max_keyframes:
            if sm is not None:
                sm.closed = True
            sm = Submap(sid=len(self.submaps), agent=agent, anchor_kf=kf_id, T_O_anchor=T_O_C.copy(),
                        pose_version=pose_version)
            self.submaps.append(sm)
            self._active[agent] = sm
            opened = True
        sm.keyframes.append(kf_id)
        sm.T_anchor_C[kf_id] = inv_se3(sm.T_O_anchor) @ T_O_C
        return sm, opened

    def close(self, agent: int) -> None:
        """Close the agent's open submap now (a loop closure bent the local trajectory, arch §18.6)."""
        sm = self._active.pop(agent, None)
        if sm is not None:
            sm.closed = True

    def reanchor(self, agent: int, poses: dict[int, np.ndarray], pose_version: int) -> list[int]:
        """A pose-graph update: new T_O_C for (some of) the agent's keyframes. Only each submap's ANCHOR pose is used;
        member poses inside a submap keep their anchor-relative values (no intra-submap correction). Returns the ids of
        the submaps whose placement changed."""
        moved = []
        for sm in self.submaps:
            if sm.agent == agent and sm.anchor_kf in poses:
                if pose_version < sm.pose_version:
                    raise ValueError(f"submap {sm.sid}: pose version {pose_version} older than {sm.pose_version}")
                sm.T_O_anchor = poses[sm.anchor_kf].copy()
                sm.pose_version = pose_version
                moved.append(sm.sid)
        return moved


class FrameStore:
    """The trainer's per-keyframe training data for the ACTIVE submaps, keyed by (submap id, keyframe id).

    Every agent numbers its own keyframes, so a bare keyframe id collides across agents. A submap belongs to one
    agent, so (sid, kf) is unique. Keying by kf alone crashed the 626abe g4dn run (KeyError 999): one agent's submap
    closing deleted another agent's frame. Before that, colliding ids silently overwrote training images across
    agents. Pure Python: the values are whatever the trainer stores (GPU tensors there, anything in tests).
    """

    def __init__(self) -> None:
        self._d: dict[tuple[int, int], object] = {}

    def put(self, sid: int, kf: int, data) -> None:
        self._d[(sid, kf)] = data

    def get(self, sid: int, kf: int):
        return self._d[(sid, kf)]

    def drop_submap(self, sid: int) -> int:
        """A closed submap's frames are released (bounded memory); returns how many."""
        keys = [k for k in self._d if k[0] == sid]
        for k in keys:
            del self._d[k]
        return len(keys)

    def __len__(self) -> int:
        return len(self._d)

    def __contains__(self, key: tuple[int, int]) -> bool:
        return key in self._d
