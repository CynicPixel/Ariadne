"""Report figures and devpost stills from recorded runs (host python: numpy + Pillow). From the repo root:

    python3 docs/Latex/figures/src/make_stills.py

Writes docs/Latex/figures/*.png (report), copies them to submission/media/stills/, and writes the derived CSVs the
report's pgfplots figures read (docs/Latex/figures/data/*.csv). Every input is a recorded-run artifact; the sources
are listed in docs/Latex/figures/SOURCES.md.
"""
import colorsys
import csv
import glob
import json
import math
import os
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
FIG = os.path.join(ROOT, "docs/Latex/figures")
DATA = os.path.join(FIG, "data")
DEV = os.path.join(ROOT, "submission/media/stills")
PACK = os.path.join(ROOT, "python/video/data/pack_collab60d")
FINAL = os.path.join(ROOT, "evaluation/runs/final_stack_e2e/20261026T000432Z/run")
PREMERGE = glob.glob(os.path.join(ROOT, "evaluation/runs/premerge_aws/20261024T083919Z-cpux86-73e3fa/gate/report/*/figures"))[0]
SPLAT = os.path.join(ROOT, "build/video_assets/splat_626abe_render_20261024")
WORLDGEN = os.path.join(ROOT, "evaluation/runs/ph6/worldgen_gsplat_20261026T211816Z")


def font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/TTF/DejaVuSans.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def tag(img, text, size=18):
    d = ImageDraw.Draw(img)
    f = font(size)
    x0, y0, x1, y1 = d.textbbox((0, 0), text, font=f)
    d.rectangle([6, 6, 18 + x1 - x0, 16 + y1 - y0], fill=(0, 0, 0))
    d.text((12, 8), text, fill=(255, 255, 255), font=f)
    return img


def save(img, name):
    p = os.path.join(FIG, name)
    img.save(p, optimize=True)
    shutil.copy(p, os.path.join(DEV, name))
    print(f"{name}: {img.width}x{img.height}")


def hstack(imgs, height, gap=8, bg=(255, 255, 255)):
    imgs = [i.convert("RGB").resize((round(i.width * height / i.height), height), Image.LANCZOS) for i in imgs]
    out = Image.new("RGB", (sum(i.width for i in imgs) + gap * (len(imgs) - 1), height), bg)
    x = 0
    for i in imgs:
        out.paste(i, (x, 0))
        x += i.width + gap
    return out


def id_colour(i):
    r, g, b = colorsys.hsv_to_rgb((i * 0.618033988749895) % 1.0, 0.85, 1.0)
    return int(255 * r), int(255 * g), int(255 * b)


def tracking():
    """One recorded agent-0 frame with each track's trail over the previous sampled frames (tracks.json: OpenCV 5
    calcOpticalFlowPyrLK replayed offline on every 20th recorded frame, win 21, 4 levels, FB 0.5 px, F-RANSAC 1 px)."""
    d = json.load(open(os.path.join(ROOT, "python/video/data/tracks/tracks.json")))
    frames = d["frames"]
    last = frames[-1]
    hist = {}
    for f in frames[-6:]:
        for i, p in zip(f["ids"], f["pts"]):
            hist.setdefault(i, []).append(tuple(p))
    img = Image.open(os.path.join(PACK, f"frames/agent0/agent0_f{last['frame']}.png")).convert("RGB")
    dr = ImageDraw.Draw(img)
    for i in last["ids"]:
        pts = hist[i]
        c = id_colour(i)
        if len(pts) > 1:
            dr.line(pts, fill=c, width=2)
        x, y = pts[-1]
        dr.ellipse([x - 3, y - 3, x + 3, y + 3], fill=c)
    s = last["stats"]
    tag(img, f"recorded frame {last['frame']} · {len(last['ids'])} tracks · LK lost {s['lk_lost']} · FB rejected "
             f"{s['fb_rejected']} · epipolar rejected {s['epipolar_rejected']} · replenished {s['replenished']}", 17)
    save(img, "fig_tracking.png")


def slabs():
    """The operator's occupancy slab (Phase 4 TSDF/ESDF map, label image: 0 unknown, 1 free, 2 occupied) at three
    times of the final-stack recorded run."""
    lut = {0: (128, 128, 128), 1: (255, 255, 255), 2: (20, 20, 20)}
    times = (20000, 55000, 95000)
    arrs = [np.array(Image.open(os.path.join(FINAL, f"snapshots/slab_{t}.png"))) for t in times]
    ys, xs = np.nonzero(arrs[-1])                      # crop all panels to the last panel's mapped extent
    y0, y1, x0, x1 = max(ys.min() - 8, 0), ys.max() + 8, max(xs.min() - 8, 0), xs.max() + 8
    panels = []
    for t, a in zip(times, arrs):
        a = a[y0:y1, x0:x1]
        rgb = np.zeros(a.shape + (3,), np.uint8)
        for k, c in lut.items():
            rgb[a == k] = c
        panels.append(tag(Image.fromarray(rgb).resize((a.shape[1] * 3, a.shape[0] * 3), Image.NEAREST), f"t = {t // 1000} s"))
    save(hstack(panels, 520), "fig_slabs.png")


def decision():
    """One recorded decision: the OpenCV-rendered map the orchestrator sees via observe_image, plus both agents'
    rectified keyframes at that instant."""
    ts = sorted({int(os.path.basename(p).split("ms_")[0][1:]) for p in glob.glob(os.path.join(FINAL, "decisions/t*ms_map.png"))})
    t = min(ts, key=lambda x: abs(x - 50000))
    m = Image.open(os.path.join(FINAL, f"decisions/t{t}ms_map.png")).convert("RGB")
    a = np.array(m).astype(int)
    ys, xs = np.nonzero(np.abs(a - a[0, 0]).sum(axis=2) > 30)   # crop away the uniform unknown border
    m = m.crop((max(xs.min() - 20, 0), max(ys.min() - 20, 0), xs.max() + 20, ys.max() + 20))
    k0 = Image.open(os.path.join(FINAL, f"decisions/t{t}ms_a0_kf.png"))
    k1 = Image.open(os.path.join(FINAL, f"decisions/t{t}ms_a1_kf.png"))
    right = Image.new("RGB", (k0.width, k0.height * 2 + 8), (255, 255, 255))
    right.paste(tag(k0.convert("RGB"), "agent 0 keyframe"), (0, 0))
    right.paste(tag(k1.convert("RGB"), "agent 1 keyframe"), (0, k0.height + 8))
    save(hstack([tag(m.convert("RGB"), f"decision map, t = {t / 1000:g} s"), right], 700), "fig_decision.png")
    return t


def splat():
    a = tag(Image.open(os.path.join(SPLAT, "stills/still_0.png")).convert("RGB"), "novel view · 2.59 M Gaussians", 30)
    b = tag(Image.open(os.path.join(SPLAT, "align/merged.png")).convert("RGB"), "3 agents' submaps in one frame", 30)
    save(hstack([a, b], 540), "fig_splat.png")
    for n in ("still_0", "still_1", "still_2", "still_3"):
        shutil.copy(os.path.join(SPLAT, f"stills/{n}.png"), os.path.join(DEV, f"splat_626abe_{n}.png"))
    if os.path.isdir(WORLDGEN):
        ks = ("005", "017", "029")   # top row: held-out ground-truth views; bottom row: the splat's renders
        top = hstack([tag(Image.open(os.path.join(WORLDGEN, f"target_{k}.png")).convert("RGB"), "held-out view", 11) for k in ks], 180, 4)
        bot = hstack([tag(Image.open(os.path.join(WORLDGEN, f"heldout_{k}.png")).convert("RGB"), "splat render", 11) for k in ks], 180, 4)
        out = Image.new("RGB", (top.width, 364), (255, 255, 255))
        out.paste(top, (0, 0)); out.paste(bot, (0, 184))
        save(out.resize((out.width * 2, out.height * 2), Image.LANCZOS), "fig_worldgen_heldout.png")


def copies():
    for src, name in ((os.path.join(PREMERGE, "trajectories.png"), "fig_premerge_trajectories.png"),
                      (os.path.join(PREMERGE, "ate_over_time.png"), "fig_premerge_ate.png"),
                      (os.path.join(PREMERGE, "map_vs_gt.png"), "fig_premerge_map_vs_gt.png"),
                      (os.path.join(ROOT, "evaluation/runs/2.5/outlier_ramp/ramp.png"), "fig_outlier_ramp.png"),
                      (os.path.join(ROOT, "evaluation/runs/nbv_4.3_gate/coverage_time.png"), "fig_nbv_coverage_time.png"),
                      (os.path.join(ROOT, "evaluation/runs/ph3/bytes_3.6_20261024/figures/fig_bytes.png"), "fig_bytes_3_6.png"),
                      (os.path.join(PACK, "frames/agent0/agent0_f600.png"), "frame_agent0_f600.png"),
                      (os.path.join(PACK, "frames/agent1/agent1_f600.png"), "frame_agent1_f600.png")):
        shutil.copy(src, os.path.join(FIG, name))
        shutil.copy(src, os.path.join(DEV, name))
        print(f"{name}: copied")


def derived_csvs():
    # (1) Known-entry gate separation over the logged D178 sweep: d^2 of every geometry-verified inter-agent
    #     candidate, true = pose error <= 10 deg (the D178 pre-registered definition).
    base = os.path.join(ROOT, "evaluation/runs/collab_d178/logged_4ec3f6")
    rows = []
    for d in sorted(glob.glob(base + "/*/")):
        if not os.path.exists(d + "entry_gate.csv"):
            continue
        eg = {(r["agent_a"], r["kf_a"], r["agent_b"], r["kf_b"]): float(r["last_d2"]) for r in csv.DictReader(open(d + "entry_gate.csv"))}
        for r in csv.DictReader(open(d + "edges.csv")):
            k = (r["agent_a"], r["kf_a"], r["agent_b"], r["kf_b"])
            if r["reason"] == "verified" and k in eg:
                rows.append((eg[k], 1 if float(r["pose_err_deg"]) <= 10 else 0))
    rng = np.random.default_rng(0)   # vertical jitter for the strip plot only
    with open(os.path.join(DATA, "chi2_separation.csv"), "w") as f:
        f.write("d2 jitter true\n")
        for d2, t in rows:
            f.write(f"{max(d2, 1e-3):.6g} {rng.uniform(-0.3, 0.3) + (1 if t else 0):.3f} {t}\n")
    print(f"chi2_separation.csv: {sum(t for _, t in rows)} true, {sum(1 - t for _, t in rows)} false")
    # (2) Gate-off recorded run (collab60d): each agent's 2-D position error vs ground truth over time.
    errs = {}
    for k in range(3):
        est = list(csv.DictReader(open(os.path.join(PACK, f"paths/agent{k}_est.csv"))))
        gt = list(csv.DictReader(open(os.path.join(PACK, f"paths/agent{k}_gt.csv"))))
        errs[k] = [(float(e["t_s"]), math.hypot(float(e["x_m"]) - float(g["x_m"]), float(e["y_m"]) - float(g["y_m"])))
                   for e, g in zip(est, gt)]
    with open(os.path.join(DATA, "collab60d_err.csv"), "w") as f:
        f.write("t a0 a1 a2\n")
        for i in range(len(errs[0])):
            f.write(f"{errs[0][i][0]:.2f} " + " ".join(f"{max(errs[k][i][1], 1e-3):.4f}" for k in range(3)) + "\n")
    print("collab60d_err.csv: max", [round(max(e for _, e in errs[k]), 3) for k in range(3)])


if __name__ == "__main__":
    for p in (FIG, DATA, DEV):
        os.makedirs(p, exist_ok=True)
    tracking(); slabs(); t = decision(); splat(); copies(); derived_csvs()
    print("decision time used:", t)
