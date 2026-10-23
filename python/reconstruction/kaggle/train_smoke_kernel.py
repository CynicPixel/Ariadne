"""Phase 6 trainer smoke on a Kaggle GPU (RECON_6, D190-D196). Needs the ariadne_recon package next to it (kaggle.sh
train-smoke embeds it). Self-generated input, so the answer is known exactly:

  scene   a corridor 1.2 m wide x 2.0 m high, walls/floor/ceiling with an analytic albedo texture in [0.15, 0.85]
  cameras 60 keyframes walking 6 m down the corridor with a gentle yaw sway, plus one held-out frame between each pair
  input   per keyframe: exact depth (ray cast), sigma = 0.002 Z^2, the albedo as the 1-channel training image (the
          proxy's own units, D195)

Measures, per keyframe, what the live path will log (seeded, Gaussians, iterations, ms), and every 10 keyframes the
held-out PSNR of the render of the submap containing the neighbouring keyframe against the true albedo. Then the
re-anchoring check on the GPU: after a pose update the anchor-frame render of every submap is bitwise unchanged.
Output: /kaggle/working/remote.json.
"""
import json
import subprocess
import sys
import time

GSPLAT = "1.5.3"


def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr)[-2000:]


def install_gsplat(torch):
    tv = torch.__version__.split("+")[0]
    pt = "pt" + "".join(tv.split(".")[:2])
    cu = "cu" + torch.version.cuda.replace(".", "")
    idx = f"https://docs.gsplat.studio/whl/{pt}{cu}"
    if sh(f"{sys.executable} -m pip install -q gsplat=={GSPLAT} --index-url {idx} --no-deps")[0] == 0:
        return f"prebuilt wheel ({idx})"
    rc, log = sh(f"{sys.executable} -m pip install -q gsplat=={GSPLAT} --no-deps")
    if rc != 0:
        raise RuntimeError("gsplat install failed:\n" + log)
    return "pypi source (JIT CUDA build)"


def corridor(np, w, h, T_O_C):
    """Ray-cast depth and albedo for a camera (optical frame: x right, y down, z forward) in the corridor."""
    f = 0.5 * w
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    d_C = np.stack([(u - (w - 1) / 2) / f, (v - (h - 1) / 2) / f, np.ones_like(u, float)], -1)
    R, t = T_O_C[:3, :3], T_O_C[:3, 3]
    d = d_C @ R.T
    best = np.full((h, w), np.inf)
    alb = np.zeros((h, w))
    for axis, off in ((0, -0.6), (0, 0.6), (1, -1.0), (1, 1.0)):
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (off - t[axis]) / d[..., axis]
        s[~np.isfinite(s) | (s <= 0)] = np.inf
        P = t + s[..., None] * d
        a, b = (P[..., 1], P[..., 2]) if axis == 0 else (P[..., 0], P[..., 2])
        tex = 0.5 + 0.2 * np.sign(np.sin(6 * a) * np.sin(6 * b)) + 0.15 * np.sin(17 * a + 3 * b)
        hit = s < best
        best[hit], alb[hit] = s[hit], tex[hit]
    Z = best * d_C[..., 2]   # depth along the optical axis
    K = np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1.0]])
    return Z.astype(np.float32), alb.astype(np.float32), K


def pose(np, s):
    yaw = 0.25 * np.sin(s * 1.3)
    c, si = np.cos(yaw), np.sin(yaw)
    T = np.eye(4)
    T[:3, :3] = [[c, 0, si], [0, 1, 0], [-si, 0, c]]
    T[:3, 3] = [0.15 * np.sin(s), 0.05 * np.sin(2 * s), s]
    return T


def main():
    import numpy as np
    import torch
    env = {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda,
           "gpu": torch.cuda.get_device_name(0)}
    t0 = time.time()
    env["gsplat_install"] = install_gsplat(torch)
    sh(f"{sys.executable} -m pip install -q jaxtyping rich")
    import gsplat
    env["gsplat"] = gsplat.__version__
    env["install_s"] = round(time.time() - t0, 1)
    from ariadne_recon.live.trainer import LiveTrainer, TrainParams
    from ariadne_recon.live.export import chunk_bytes

    out = {"env": env, "runs": []}
    for (w, h, iters) in ((320, 180, 15), (640, 360, 10)):
        tr = LiveTrainer(channels=1, p=TrainParams(iters=iters, max_keyframes=30))
        n_kf, rows, psnr = 60, [], []
        for k in range(n_kf):
            s = 0.1 * k
            Z, A, K = corridor(np, w, h, pose(np, s))
            rows.append(tr.add_keyframe(0, k, pose(np, s), 1, K, A, Z, 0.002 * Z ** 2))
            if k % 10 == 9:
                Th = pose(np, s + 0.05)
                Zh, Ah, _ = corridor(np, w, h, Th)
                sm = tr.manager.active(0)
                st = tr.state[sm.sid]
                V = np.linalg.inv(np.linalg.inv(sm.T_O_anchor) @ Th)
                f = lambda a: torch.as_tensor(a, dtype=torch.float32, device="cuda")
                P = st.params
                with torch.no_grad():
                    img, _, _ = gsplat.rasterization(P["means"], torch.nn.functional.normalize(P["quats"], dim=-1),
                                                     torch.exp(P["scales"]), torch.sigmoid(P["opacities"]),
                                                     P["colors"], f(V)[None], f(K)[None], w, h, sh_degree=None,
                                                     packed=False)
                mse = float(((img[0, ..., 0].clamp(0, 1) - f(Ah)) ** 2).mean())
                psnr.append({"kf": k, "psnr_db": round(10 * np.log10(1.0 / max(mse, 1e-12)), 2),
                             "submap": sm.sid, "gaussians": len(st)})
        # Re-anchoring on the GPU: the anchor-frame parameters (hence any anchor-frame render) are untouched.
        snap = {sid: {k: v.detach().clone() for k, v in st.params.items()} for sid, st in tr.state.items()}
        moved = tr.manager.reanchor(0, {sm.anchor_kf: pose(np, 9.0) for sm in tr.manager.submaps}, 2)
        unchanged = all(torch.equal(snap[sid][k], st.params[k]) for sid, st in tr.state.items() for k in st.params)
        sizes = {sid: len(chunk_bytes(tr.params_numpy(sid))) for sid in tr.state}
        ms = [r["ms_total"] for r in rows[1:]]
        out["runs"].append({"w": w, "h": h, "iters": iters, "keyframes": n_kf, "submaps": len(tr.manager.submaps),
                            "ms_per_kf_median": round(float(np.median(ms)), 2),
                            "ms_per_kf_p90": round(float(np.percentile(ms, 90)), 2),
                            "ms_iter_median": rows[-1]["ms_iter_median"], "gaussians_final": rows[-1]["gaussians"],
                            "seeded_median": int(np.median([r["seeded"] for r in rows])),
                            "stable_final": rows[-1]["stable"], "pruned_total": int(sum(r["pruned"] for r in rows)),
                            "exposure_affine_ever": any(r["exposure_affine"] for r in rows),
                            "heldout_psnr": psnr, "reanchor_moved": moved, "reanchor_params_unchanged": unchanged,
                            "chunk_bytes": sizes, "per_kf": rows})
        print(json.dumps({k: v for k, v in out["runs"][-1].items() if k != "per_kf"}), flush=True)
        del tr
        torch.cuda.empty_cache()
    json.dump(out, open("/kaggle/working/remote.json", "w"), indent=2)


if __name__ == "__main__":
    main()
