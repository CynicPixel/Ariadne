"""Illustrative feature tracks for beat 4: the front end's per-frame loop replayed offline with OpenCV 5 on recorded
frames, so the film's colours really follow the same feature from frame to frame.

Input: the collab60d pack's agent-0 frames (every 20th recorded frame, i.e. 1 s apart; the live tracker runs on every
frame at 20 Hz, so these tracks are illustrative, not the run's own). Parameters are configs/perception/sim_v1.yaml's
(M, 2026-10-24): ORB/FAST detection (8 levels, scale 1.2), 500 tracks at >= 25 px spacing, LK window 21 px, 4 pyramid
levels, forward-backward 0.5 px, fundamental-matrix RANSAC 1 px at 0.99.
Detection asks ORB for 20x the target at FAST threshold 7 (not sim_v1's), so refill gets near the 500 target on
these frames; the 12x7 grid is not reproduced.

Output (out_dir): tracks.json (per frame: track ids, points, ages, the filters' counts, the fitted F; plus one real
forward-backward reject, one real epipolar reject, and a highlighted track's coarse-to-fine LK estimates) and
pyr_<level>.png (the LK pyramid of the highlighted transition's previous frame).

Runs in the OpenCV 5 analysis image (cv2 5.0.0):
  docker run --rm --user $(id -u):$(id -g) -v $PWD/data/pack_collab60d/frames/agent0:/f:ro -v $PWD/data/tracks:/o \
      -v $PWD/tools:/t:ro --entrypoint python3 <ariadne-core analysis image> -I /t/track_replay.py /f /o
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

FRAMES = range(120, 361, 20)
WIN, LEVELS, FB_PX, F_PX, F_CONF = 21, 3, 0.5, 1.0, 0.99  # LEVELS = max_level (4 pyramid levels, 0-3)
TARGET, SPACING, REQUEST = 500, 25.0, 4
CRIT = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)


def lk(prev, cur, p, level=LEVELS):
    q, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p.reshape(-1, 1, 2), None, winSize=(WIN, WIN), maxLevel=level,
                                        criteria=CRIT)
    return q.reshape(-1, 2), st.ravel() == 1


def main(src: str, out: str) -> int:
    o = Path(out)
    o.mkdir(parents=True, exist_ok=True)
    cv2.setRNGSeed(0)
    orb = cv2.ORB_create(nfeatures=TARGET * REQUEST * 5, scaleFactor=1.2, nlevels=8, fastThreshold=7)
    imgs = [cv2.imread(f"{src}/agent0_f{k}.png", cv2.IMREAD_GRAYSCALE) for k in FRAMES]
    h, w = imgs[0].shape
    ids, pts, ages, next_id = np.zeros(0, int), np.zeros((0, 2), np.float32), np.zeros(0, int), 0
    frames, fb_ex, epi_ex = [], None, None

    def replenish(img, ids, pts, ages, next_id):
        mask = np.full((h, w), 255, np.uint8)
        for x, y in pts:
            cv2.circle(mask, (int(x), int(y)), int(SPACING), 0, -1)
        kps = sorted(orb.detect(img, None), key=lambda k: -k.response)
        new = []
        for k in kps:
            if len(pts) + len(new) >= TARGET:
                break
            x, y = k.pt
            if mask[int(y), int(x)]:
                new.append((x, y))
                cv2.circle(mask, (int(x), int(y)), int(SPACING), 0, -1)
        n = len(new)
        if n:
            pts = np.vstack([pts, np.float32(new)])
            ids = np.r_[ids, np.arange(next_id, next_id + n)]
            ages = np.r_[ages, np.zeros(n, int)]
        return ids, pts, ages, next_id + n, n

    for t, img in enumerate(imgs):
        st = dict(attempted=len(ids), lk_lost=0, fb_rejected=0, border_rejected=0, epipolar_rejected=0,
                  crowded_out=0, replenished=0)
        F = None
        if t and len(ids):
            prev = imgs[t - 1]
            q, ok_f = lk(prev, img, pts)
            b, ok_b = lk(img, prev, q)
            fb = np.linalg.norm(b - pts, axis=1)
            lost = ~(ok_f & ok_b)
            fbr = ~lost & (fb >= FB_PX)
            inb = (q[:, 0] >= 1) & (q[:, 1] >= 1) & (q[:, 0] < w - 1) & (q[:, 1] < h - 1)
            bord = ~lost & ~fbr & ~inb
            keep = ~lost & ~fbr & ~bord
            st.update(lk_lost=int(lost.sum()), fb_rejected=int(fbr.sum()), border_rejected=int(bord.sum()))
            if fb_ex is None and fbr.any():  # a visible failure: the largest round-trip error below 25 px
                c = np.where(fbr & (fb < 25))[0]
                if len(c):
                    i = c[np.argmax(fb[c])]
                    fb_ex = dict(t=t, id=int(ids[i]), p0=pts[i].tolist(), p1=q[i].tolist(), back=b[i].tolist(),
                                 err_px=float(fb[i]))
            k = np.where(keep)[0]
            if len(k) >= 15:
                F, inl = cv2.findFundamentalMat(pts[k], q[k], cv2.FM_RANSAC, F_PX, F_CONF)
                if F is not None and F.shape == (3, 3):
                    out_i = k[inl.ravel() == 0]
                    keep[out_i] = False
                    st["epipolar_rejected"] = int(len(out_i))
                    if epi_ex is None and len(out_i):
                        x0 = np.c_[pts[out_i], np.ones(len(out_i))]
                        line = x0 @ F.T  # epipolar lines in the current frame
                        d = np.abs((line * np.c_[q[out_i], np.ones(len(out_i))]).sum(1)) / np.hypot(*line[:, :2].T)
                        j = int(np.argmax(np.where(d < 60, d, -1)))
                        epi_ex = dict(t=t, id=int(ids[out_i[j]]), p0=pts[out_i[j]].tolist(), p1=q[out_i[j]].tolist(),
                                      line=line[j].tolist(), dist_px=float(d[j]))
                    F = F.tolist()
                else:
                    F = None
            ids, pts, ages = ids[keep], q[keep], ages[keep] + 1
            # crowding: an older track wins a spacing conflict
            order = np.argsort(-ages, kind="stable")
            taken = np.zeros(len(ids), bool)
            keep2 = np.zeros(len(ids), bool)
            for i in order:
                if not taken[i]:
                    keep2[i] = True
                    taken |= np.linalg.norm(pts - pts[i], axis=1) < SPACING
            st["crowded_out"] = int((~keep2).sum())
            ids, pts, ages = ids[keep2], pts[keep2], ages[keep2]
        ids, pts, ages, next_id, st["replenished"] = replenish(img, ids, pts, ages, next_id)
        frames.append(dict(frame=int(FRAMES[t]), ids=ids.tolist(), pts=np.round(pts, 2).tolist(), ages=ages.tolist(),
                           stats=st, F=F))

    # the highlighted track: alive in the most frames, with a clear move into a mid-sequence frame
    life = {}
    for fr in frames:
        for i in fr["ids"]:
            life[i] = life.get(i, 0) + 1
    t_hi = 6
    a, b = frames[t_hi - 1], frames[t_hi]
    pa, pb = dict(zip(a["ids"], a["pts"])), dict(zip(b["ids"], b["pts"]))
    cand = [i for i in pa if i in pb and 12 < np.hypot(*np.subtract(pb[i], pa[i])) < 40]
    hi = max(cand, key=lambda i: (life[i], np.hypot(*np.subtract(pb[i], pa[i]))))
    pyr_prev = cv2.buildOpticalFlowPyramid(imgs[t_hi - 1], (WIN, WIN), LEVELS, withDerivatives=False)[1]
    pyr_cur = cv2.buildOpticalFlowPyramid(imgs[t_hi], (WIN, WIN), LEVELS, withDerivatives=False)[1]
    p0 = np.float32(pa[hi])
    c2f = []  # estimate after refining down to level l: LK on levels l..3 only, scaled back to full resolution
    for lvl in range(LEVELS, -1, -1):
        q, ok = lk(pyr_prev[lvl], pyr_cur[lvl], p0[None] / 2 ** lvl, level=LEVELS - lvl)
        c2f.append(dict(level=lvl, est=(q[0] * 2 ** lvl).tolist(), ok=bool(ok[0])))
    for lvl, im in enumerate(pyr_prev):
        cv2.imwrite(str(o / f"pyr_{lvl}.png"), im)
    json.dump(dict(source="pack_collab60d/frames/agent0", every_nth_recorded_frame=20, size=[w, h], frames=frames,
                   fb_example=fb_ex, epipolar_example=epi_ex,
                   highlight=dict(t=t_hi, id=int(hi), life=life[hi], p0=pa[hi], p1=pb[hi], coarse_to_fine=c2f),
                   params=dict(win=WIN, max_level=LEVELS, fb_px=FB_PX, f_ransac_px=F_PX, target=TARGET,
                               spacing_px=SPACING)), open(o / "tracks.json", "w"))
    for fr in frames:
        print(fr["frame"], len(fr["ids"]), fr["stats"])
    print("fb", fb_ex, "\nepi", epi_ex, "\nhi", hi, life[hi], c2f)
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
