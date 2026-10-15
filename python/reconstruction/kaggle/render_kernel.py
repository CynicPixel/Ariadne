"""Remote render step (D44): runs as a Kaggle GPU kernel. Self-contained.

Input : /kaggle/input/<dataset>/render_package.npz (+ .json), written by
        ariadne_recon.render_package (initialized Gaussians, held-out cameras, targets).
Output: /kaggle/working/renders.npz   float32 [C, H, W, 3] in [0, 1], one per view
        /kaggle/working/remote.json   environment (torch / CUDA / gsplat / GPU, how gsplat
                                      was installed) + per-view LPIPS + timing
Modes : SMOKE=1 renders 1000 random Gaussians only (the [SMOKE-TEST] of the remote path).

gsplat 1.4.0 is pinned. Install order: the project's prebuilt wheel index for the
running torch/CUDA (no compile), else PyPI (JIT CUDA build on first use, minutes).
The call mirrors gsplat v1.4.0 examples/simple_trainer.py: rasterization(means,
quats, exp(scales), sigmoid(opacities), cat(sh0, shN), viewmats, Ks, W, H,
sh_degree=3, packed=False, rasterize_mode="classic", near 0.01, far 1e10).
"""
import glob
import json
import os
import subprocess
import sys
import time

GSPLAT = "1.4.0"
LPIPS = "0.1.4"


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
        return f"prebuilt wheel ({idx})", log
    rc2, log2 = sh(f"{sys.executable} -m pip install -q gsplat=={GSPLAT} --no-deps")
    if rc2 != 0:
        raise RuntimeError("gsplat install failed:\n" + log + "\n" + log2)
    return "pypi source (JIT CUDA build)", log + "\n" + log2


def main():
    import numpy as np
    import torch

    env = {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda,
           "cuda_available": torch.cuda.is_available()}
    if not torch.cuda.is_available():
        _, smi = sh("nvidia-smi")
        raise RuntimeError(f"no CUDA device (torch {torch.__version__}, built for CUDA {torch.version.cuda}); "
                           f"nvidia-smi:\n{smi}")
    env["gpu"] = torch.cuda.get_device_name(0)
    t0 = time.time()
    env["gsplat_install"], _ = install_gsplat(torch)
    for dep in ("jaxtyping", "rich"):
        sh(f"{sys.executable} -m pip install -q {dep}")
    import gsplat
    from gsplat import rasterization
    env["gsplat"] = gsplat.__version__
    env["install_s"] = round(time.time() - t0, 1)
    dev = torch.device("cuda")
    out = {"env": env}

    if os.environ.get("SMOKE") == "1":
        g = torch.Generator().manual_seed(0)
        n, w, h = 1000, 320, 240
        means = (torch.randn(n, 3, generator=g) * 0.5 + torch.tensor([0, 0, 3.0])).to(dev)
        quats = torch.rand(n, 4, generator=g).to(dev)
        scales = torch.full((n, 3), 0.03, device=dev)
        opac = torch.full((n,), 0.8, device=dev)
        colors = torch.rand(n, 3, generator=g).to(dev)
        K = torch.tensor([[[300.0, 0, w / 2], [0, 300.0, h / 2], [0, 0, 1]]], device=dev)
        t1 = time.time()
        img, alpha, _ = rasterization(means, quats, scales, opac, colors, torch.eye(4, device=dev)[None], K, w, h)
        torch.cuda.synchronize()
        out["smoke"] = {"shape": list(img.shape), "render_s": round(time.time() - t1, 3),
                        "alpha_mean": float(alpha.mean()), "img_mean": float(img.mean())}
        json.dump(out, open("/kaggle/working/remote.json", "w"), indent=2)
        print(json.dumps(out, indent=2))
        return

    # The dataset mount depth varies (/kaggle/input/<slug>/ or deeper): search, and on failure show the tree.
    pkgs = glob.glob("/kaggle/input/**/render_package.npz", recursive=True)
    if len(pkgs) != 1:
        _, tree = sh("find /kaggle/input -maxdepth 5 | head -50")
        raise RuntimeError(f"expected exactly one render_package.npz under /kaggle/input, found {pkgs}; tree:\n{tree}")
    meta = json.load(open(pkgs[0].replace(".npz", ".json")))
    p = np.load(pkgs[0])
    t = lambda k: torch.from_numpy(p[k]).to(dev)
    sh_deg = int(meta["sh_degree"])
    n = p["means"].shape[0]
    colors = torch.cat([t("sh0"), torch.zeros((n, (sh_deg + 1) ** 2 - 1, 3), device=dev)], 1)
    w, h = int(meta["width"]), int(meta["height"])
    renders = []
    t1 = time.time()
    with torch.no_grad():
        for c in range(p["viewmats"].shape[0]):
            img, _, _ = rasterization(t("means"), t("quats"), torch.exp(t("scales")), torch.sigmoid(t("opacities")),
                                      colors, t("viewmats")[c:c + 1], t("Ks")[c:c + 1], w, h, sh_degree=sh_deg,
                                      packed=False, rasterize_mode="classic", near_plane=0.01, far_plane=1e10)
            renders.append(img[0].clamp(0, 1).cpu().numpy())
    torch.cuda.synchronize()
    out["render_s"] = round(time.time() - t1, 2)
    out["package_sha256"] = meta["sha256"]
    r = np.stack(renders).astype(np.float32)
    np.savez_compressed("/kaggle/working/renders.npz", renders=r, names=p["names"])

    rc, log = sh(f"{sys.executable} -m pip install -q lpips=={LPIPS}")
    if rc == 0:
        import lpips
        net = lpips.LPIPS(net="alex", verbose=False).to(dev)
        tgt = torch.from_numpy(p["targets"].astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(dev) * 2 - 1
        ren = torch.from_numpy(r).permute(0, 3, 1, 2).to(dev) * 2 - 1
        with torch.no_grad():
            out["lpips_alex"] = [float(net(ren[i:i + 1], tgt[i:i + 1])) for i in range(len(r))]
        out["lpips_version"] = LPIPS
    else:
        out["lpips_error"] = log
    json.dump(out, open("/kaggle/working/remote.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "lpips_alex"}, indent=2))


if __name__ == "__main__":
    main()
