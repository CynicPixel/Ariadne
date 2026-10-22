"""The lamp's photometric model and its fit (5.1, D133). numpy only.

Model (SMOKE §4u measured the renderer's law; D131):

    I(x) = A * RID(theta) * [cos(inc)] / (c' + r^2)

  I        linear pixel value in [0, 1] (the sRGB render, decoded), for the calibration plane's albedo
  r        distance from the lamp centre to the surface point x
  theta    angle between the lamp axis and (x - lamp)
  cos(inc) Lambertian incidence at x; the bracket is the "cosine lobe decision" of plan 5.1, decided by held-out fit
  c'       a near-field term the renderer adds (0.5 m^2 for SDF attenuation (0, 0, 1)); 0 for a physical lamp
  RID      relative radiant intensity, 1 on the axis, stored as a table over theta (the renderer's lobe is a flat top
           plus roll-off, not cos^mu)

Fit: for a fixed c', y = I (c' + r^2) / [cos(inc)] should depend on theta alone; its per-theta median gives A * RID.
A wrong c' makes poses at different r disagree at the same theta, so c' minimises the across-pose spread of log y
(golden-section search). Linear in everything but c', so there is no initialisation to get wrong.

Frames: the render's camera link G (x forward, y left, z up), camera at the origin. Results are reported in the
camera's optical frame (x right, y down, z forward), which is what Stage 3 uses.
"""
import math

import numpy as np

RENDER_F = 640.0                 # rig render: hfov 90 deg at 1280 px (SMOKE §4n: exact pinhole, pixel-centre)
RENDER_C = (639.5, 359.5)
PLANE_HALF_THICK = 0.01          # the calibration plane is a 0.02 m box; its lit face is 0.01 m nearer


def srgb_decode(v8):
    c = np.asarray(v8, np.float64) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def g_to_optical(v):
    """Camera link G (x fwd, y left, z up) -> optical (x right, y down, z fwd)."""
    v = np.asarray(v, np.float64)
    return np.stack([-v[..., 1], -v[..., 2], v[..., 0]], -1)


def plane_rotation(pitch, yaw):
    """gz pose R = Rz(yaw) Ry(pitch) (roll 0)."""
    cp, sp, cy, sy = math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1.0]])
    ry = np.array([[cp, 0, sp], [0, 1.0, 0], [-sp, 0, cp]])
    return rz @ ry


def pixel_geometry(pose, lamp_g, shape, stride=4):
    """Per sampled pixel: (row, col), r, theta, cos(inc) for a plane pose dict(d, pitch, yaw) in G."""
    h, w = shape
    rows, cols = np.mgrid[0:h:stride, 0:w:stride]
    ray = np.stack([np.ones(rows.shape), -(cols - RENDER_C[0]) / RENDER_F, -(rows - RENDER_C[1]) / RENDER_F], -1)
    R = plane_rotation(pose["pitch"], pose["yaw"])
    n = R @ np.array([-1.0, 0.0, 0.0])                         # lit face normal, towards the camera
    p0 = np.array([pose["d"], 0.0, 0.0]) + R @ np.array([-PLANE_HALF_THICK, 0.0, 0.0])
    t = (n @ p0) / (ray @ n)
    x = ray * t[..., None]
    lv = x - np.asarray(lamp_g)
    r = np.linalg.norm(lv, axis=-1)
    theta = np.degrees(np.arccos(np.clip(lv[..., 0] / r, -1.0, 1.0)))   # lamp axis = G +x
    cos_inc = -(lv @ n) / r
    return rows, cols, r, theta, cos_inc


def samples(frames, poses, lamp_g, stride=4, min_level=2, max_level=250, min_cos=0.2):
    """Stack usable samples over all poses: pose index, r, theta, cos(inc), linear I."""
    out = []
    for k, (img8, pose) in enumerate(zip(frames, poses)):
        rows, cols, r, th, ci = pixel_geometry(pose, lamp_g, img8.shape[:2], stride)
        v8 = img8[rows, cols]
        ok = (v8 >= min_level) & (v8 <= max_level) & (ci > min_cos) & (r > 0)
        out.append(np.stack([np.full(ok.sum(), k), r[ok], th[ok], ci[ok], srgb_decode(v8[ok])], -1))
    return np.concatenate(out)


def _table(theta, y, bin_deg):
    edges = np.arange(0.0, theta.max() + bin_deg, bin_deg)
    idx = np.digitize(theta, edges) - 1
    centres, vals = [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() >= 20:
            centres.append(0.5 * (edges[b] + edges[b + 1]))
            vals.append(float(np.median(y[m])))
    return np.array(centres), np.array(vals)


def _spread(s, c_prime, use_cos, bin_deg):
    """Across-pose disagreement of log y at the same theta: the c' objective."""
    y = s[:, 4] * (c_prime + s[:, 1] ** 2) / (s[:, 3] if use_cos else 1.0)
    th, tab = _table(s[:, 2], y, bin_deg)
    pred = np.interp(s[:, 2], th, tab)
    ok = pred > 0.05 * tab.max()                 # the lobe's tail is all quantisation; it would dominate log errors
    return float(np.sqrt(np.mean(np.log(y[ok] / pred[ok]) ** 2)))


def fit(s, use_cos=True, bin_deg=1.0, c_range=(0.0, 5.0), iters=60):
    lo, hi = c_range
    g = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = hi - g * (hi - lo), lo + g * (hi - lo)
    fa, fb = _spread(s, a, use_cos, bin_deg), _spread(s, b, use_cos, bin_deg)
    for _ in range(iters):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - g * (hi - lo)
            fa = _spread(s, a, use_cos, bin_deg)
        else:
            lo, a, fa = a, b, fb
            b = lo + g * (hi - lo)
            fb = _spread(s, b, use_cos, bin_deg)
    c_prime = 0.5 * (lo + hi)
    y = s[:, 4] * (c_prime + s[:, 1] ** 2) / (s[:, 3] if use_cos else 1.0)
    th, tab = _table(s[:, 2], y, bin_deg)
    a_gain = float(np.median(y[s[:, 2] < 5.0]))
    return {"c_prime_m2": c_prime, "gain_A": a_gain, "theta_deg": th, "rid": tab / a_gain,
            "rms_log": _spread(s, c_prime, use_cos, bin_deg)}


def predict(model, s, use_cos):
    rid = np.interp(s[:, 2], model["theta_deg"], model["rid"], right=0.0)
    return model["gain_A"] * rid * (s[:, 3] if use_cos else 1.0) / (model["c_prime_m2"] + s[:, 1] ** 2)


def heldout_rms(train, test, use_cos, bin_deg=1.0):
    m = fit(train, use_cos, bin_deg)
    p = predict(m, test, use_cos)
    ok = p > 0.05 * m["gain_A"] / (m["c_prime_m2"] + np.median(test[:, 1]) ** 2)
    return float(np.sqrt(np.mean(np.log(test[ok, 4] / p[ok]) ** 2)))
