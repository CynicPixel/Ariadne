"""Phase 6 T4 microbenchmark (RECON_6 §1, D190): runs as a Kaggle GPU kernel. Self-contained.

Why: the live trainer's iterations per keyframe must be derived from the measured ms/iteration on the target GPU,
and no T4 number for gsplat is published (RECON_6 Appendix A.4). This kernel measures, on whatever GPU Kaggle gives:

  train   forward + backward + Adam step of gsplat.rasterization (render_mode "RGB+ED": colour and expected depth,
          the live loss), 0.8 L1 + 0.2 D-SSIM + 0.1 depth L1 (RECON_6 §2), for N in {50k, 100k, 250k, 500k} Gaussians at
          320x180 and 640x360, packed False/True, SH degree 0 (the .splat export is SH0)
  render  the forward pass alone at the same points (no_grad)
  sh3     one train point at SH degree 3, for the cost of view-dependent colour
  append  the live seeding step: grow the parameter tensors by 10k Gaussians and carry the Adam state (zero-padded
          exp_avg / exp_avg_sq), then keep stepping -- our own implementation of the pattern of gsplat's private
          strategy.ops._update_param_with_optimizer, checked here before anything is built on it

Output: /kaggle/working/remote.json (environment, per-point median/p90 ms, peak memory, append check).
The synthetic scene fills the view (a 4 m x 2.4 m x 1-6 m slab in front of the camera, footprint-sized scales);
its content does not change the per-Gaussian cost model, only how many Gaussians each pixel touches. [ASSUMPTION]
that this is representative of a 640x360 indoor keyframe; the live run logs its own ms/iteration to confirm.
"""
import json
import os
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
    rc, log = sh(f"{sys.executable} -m pip install -q gsplat=={GSPLAT} --index-url {idx} --no-deps")
    if rc == 0:
        return f"prebuilt wheel ({idx})"
    rc2, log2 = sh(f"{sys.executable} -m pip install -q gsplat=={GSPLAT} --no-deps")
    if rc2 != 0:
        raise RuntimeError("gsplat install failed:\n" + log + "\n" + log2)
    return "pypi source (JIT CUDA build)"


def scene(torch, n, dev, g):
    """N Gaussians filling the view of a camera at the origin looking down +z."""
    x = (torch.rand(n, generator=g) - 0.5) * 4.0
    y = (torch.rand(n, generator=g) - 0.5) * 2.4
    z = 1.0 + torch.rand(n, generator=g) * 5.0
    means = torch.stack([x, y, z], 1)
    # Footprint-sized: a disc covering ~2 px at 640 px focal-ish scale, log-parametrised like gsplat's trainer.
    scales = torch.log((z / 400.0)[:, None].expand(n, 3).clone() * (0.5 + torch.rand(n, 3, generator=g)))
    quats = torch.nn.functional.normalize(torch.randn(n, 4, generator=g), dim=1)
    opac = torch.logit(torch.full((n,), 0.8))
    sh0 = torch.rand(n, 1, 3, generator=g)
    return {k: v.to(dev).requires_grad_(True) for k, v in
            dict(means=means, scales=scales, quats=quats, opacities=opac, sh0=sh0).items()}


def ssim(torch, a, b):
    """SSIM of [1, H, W, 3] images with the 11x11, sigma 1.5 Gaussian window of the 3DGS reference loss (Kerbl et al.
    2023: L = 0.8 L1 + 0.2 (1 - SSIM)); plain torch so the timing includes it as the live loss will."""
    F = torch.nn.functional
    x, y = a.permute(0, 3, 1, 2), b.permute(0, 3, 1, 2)
    g = torch.exp(-((torch.arange(11, device=a.device) - 5.0) ** 2) / (2 * 1.5 ** 2))
    w = (g[:, None] * g[None, :]) / g.sum() ** 2
    w = w.expand(3, 1, 11, 11).contiguous()
    blur = lambda t: F.conv2d(t, w, padding=5, groups=3)
    mx, my = blur(x), blur(y)
    sxx, syy, sxy = blur(x * x) - mx * mx, blur(y * y) - my * my, blur(x * y) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))).mean()


def append_gaussians(torch, params, opt, new):
    """Grow every parameter by the rows of `new` and carry the Adam state: existing rows keep their moments, new rows
    start at zero (our implementation of the pattern of gsplat's private _update_param_with_optimizer)."""
    for name, p in list(params.items()):
        grown = torch.nn.Parameter(torch.cat([p.detach(), new[name].to(p.device)], 0))
        for group in opt.param_groups:
            if group["params"][0] is p:
                st = opt.state.pop(p, None)
                group["params"][0] = grown
                if st is not None:
                    for k in ("exp_avg", "exp_avg_sq"):
                        pad = torch.zeros((new[name].shape[0],) + st[k].shape[1:], device=p.device, dtype=st[k].dtype)
                        st[k] = torch.cat([st[k], pad], 0)
                    opt.state[grown] = st
        params[name] = grown


def main():
    import numpy as np
    import torch
    env = {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda}
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device")
    env["gpu"] = torch.cuda.get_device_name(0)
    t0 = time.time()
    try:   # an image with gsplat compiled ahead of time (recon-gpu) must not pip-install over it
        import gsplat
        from gsplat import csrc   # noqa: F401
        env["gsplat_install"] = f"preinstalled {gsplat.__version__} (compiled csrc: {csrc.__file__})"
    except ImportError:
        env["gsplat_install"] = install_gsplat(torch)
        sh(f"{sys.executable} -m pip install -q jaxtyping rich")
    import gsplat
    from gsplat import rasterization
    env["gsplat"] = gsplat.__version__
    env["install_s"] = round(time.time() - t0, 1)
    dev = torch.device("cuda")
    out = {"env": env, "points": []}

    def cam(w, h):
        f = 0.5 * w   # 90 deg hfov, the rig's (rig_stereo_v1)
        K = torch.tensor([[[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1]]], device=dev)
        return torch.eye(4, device=dev)[None], K

    def render(p, w, h, packed, sh_deg=0, extra=None):
        vm, K = cam(w, h)
        colors = p["sh0"] if extra is None else torch.cat([p["sh0"], extra], 1)
        return rasterization(p["means"], p["quats"], torch.exp(p["scales"]), torch.sigmoid(p["opacities"]), colors,
                             vm, K, w, h, sh_degree=sh_deg, packed=packed, render_mode="RGB+ED",
                             near_plane=0.01, far_plane=100.0)

    def timed(fn, iters=40, warm=8):
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            a = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            ts.append(1e3 * (time.perf_counter() - a))
        return float(np.median(ts)), float(np.percentile(ts, 90))

    g = torch.Generator().manual_seed(0)
    for (w, h) in ((320, 180), (640, 360)):
        tgt_rgb = torch.rand(1, h, w, 3, device=dev)
        tgt_d = 1.0 + 5.0 * torch.rand(1, h, w, 1, device=dev)
        for n in (50_000, 100_000, 250_000, 500_000):
            for packed in (False, True):
                p = scene(torch, n, dev, g)
                opt = torch.optim.Adam([{"params": [v], "lr": 1e-3} for v in p.values()])
                torch.cuda.reset_peak_memory_stats()

                def step():   # the RECON_6 §2 live loss: 0.8 L1 + 0.2 D-SSIM + depth L1
                    img, _, _ = render(p, w, h, packed)
                    rgb = img[..., :3]
                    loss = (0.8 * (rgb - tgt_rgb).abs().mean() + 0.2 * (1.0 - ssim(torch, rgb, tgt_rgb))
                            + 0.1 * (img[..., 3:] - tgt_d).abs().mean())
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()

                tr = timed(step)

                def fwd():
                    with torch.no_grad():
                        render(p, w, h, packed)

                fw = timed(fwd)
                out["points"].append({"w": w, "h": h, "n": n, "packed": packed, "sh": 0,
                                      "train_ms_med": round(tr[0], 3), "train_ms_p90": round(tr[1], 3),
                                      "render_ms_med": round(fw[0], 3), "render_ms_p90": round(fw[1], 3),
                                      "peak_mem_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1)})
                print(json.dumps(out["points"][-1]), flush=True)
                del p, opt
                torch.cuda.empty_cache()

    # SH degree 3 at one point: 250k, 640x360, unpacked.
    w, h, n = 640, 360, 250_000
    p = scene(torch, n, dev, g)
    shn = (0.01 * torch.randn(n, 15, 3, generator=g)).to(dev).requires_grad_(True)
    opt = torch.optim.Adam([{"params": [v], "lr": 1e-3} for v in list(p.values()) + [shn]])
    tgt_rgb = torch.rand(1, h, w, 3, device=dev)

    def step3():
        img, _, _ = render(p, w, h, False, sh_deg=3, extra=shn)
        loss = (img[..., :3] - tgt_rgb).abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    tr = timed(step3)
    out["sh3"] = {"w": w, "h": h, "n": n, "train_ms_med": round(tr[0], 3), "train_ms_p90": round(tr[1], 3)}
    del p, opt, shn
    torch.cuda.empty_cache()

    # Append: 100k, 10 steps, +10k, 10 more steps; the old rows' Adam moments must survive unchanged.
    w, h = 640, 360
    p = {k: torch.nn.Parameter(v.detach()) for k, v in scene(torch, 100_000, dev, g).items()}
    opt = torch.optim.Adam([{"params": [v], "lr": 1e-3} for v in p.values()])
    tgt_rgb = torch.rand(1, h, w, 3, device=dev)

    def step_a():
        img, _, _ = render(p, w, h, False)
        loss = (img[..., :3] - tgt_rgb).abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    for _ in range(10):
        step_a()
    before = opt.state[p["means"]]["exp_avg"].clone()
    new = {k: v.detach() for k, v in scene(torch, 10_000, dev, g).items()}
    torch.cuda.synchronize()
    a = time.perf_counter()
    append_gaussians(torch, p, opt, new)
    torch.cuda.synchronize()
    append_ms = 1e3 * (time.perf_counter() - a)
    after = opt.state[p["means"]]["exp_avg"]
    kept = bool(torch.equal(after[:100_000], before)) and bool((after[100_000:] == 0).all())
    for _ in range(10):
        step_a()
    out["append"] = {"n_before": 100_000, "n_added": 10_000, "n_after": int(p["means"].shape[0]),
                     "append_ms": round(append_ms, 3), "adam_state_kept": kept,
                     "steps_after_ok": bool(torch.isfinite(p["means"]).all())}
    json.dump(out, open(os.path.join(os.environ.get("ARIADNE_WORK", "/kaggle/working"), "remote.json"), "w"), indent=2)
    print(json.dumps({"sh3": out["sh3"], "append": out["append"], "env": env}, indent=2))


if __name__ == "__main__":
    main()
