"""Phase 6 GPU training on real spools (Kaggle T4, or an AWS g4dn via infra/eval/recon_gpu_aws.sh; RECON_6,
D190/D198/D199/D230). Needs the ariadne_recon package next
to it (kaggle.sh train embeds it) and the private dataset kaggle.sh train uploads: data.tar holding
  spools/<name>/spool.yaml ...   one spool per agent (ariadne_sim_vo --spool, recon_v1.yaml)
  scene/scene.json, mission_*.json   the worldgen scene (briefing placement in the mission world W, D230)
  dataset.json                   provenance: the recording's run id, commit, what built the spools

Runs the live service (ariadne_recon.live.service, --backend gpu, --training-image rgb: the lit demo, D199) over all
spools once per budget in BUDGETS: "batch" = Gaussian-SLAM's 100 iterations per keyframe, "live" = 15 (D198's two
rows), with --diagnostics (training-view PSNR before/after each keyframe's iterations; NOT a held-out metric - the
Phase 6 held-out protocol is not designed yet). Output in /kaggle/working:
  remote.json             versions, gsplat install mode, GPU, per budget: wall, keyframes, submaps, Gaussians,
                          ms per keyframe (median/p95), training-view PSNR init/after (median), freeze check
  splat_<budget>.tar.gz   the service's output dir (table.json, chunks/, service_log.jsonl, manifest.json)
"""
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

GSPLAT = "1.5.3"
# A budget is iterations per keyframe, or [iterations, max keyframes] (a smoke budget: the first N keyframes only).
BUDGETS = json.loads(os.environ.get("ARIADNE_BUDGETS", '{"batch": 100, "live": 15}'))
# Kaggle by default; the in-region AWS GPU run (infra/eval/recon_gpu_aws.sh) sets these to its own directories.
WORK = Path(os.environ.get("ARIADNE_WORK", "/kaggle/working"))
INPUT = Path(os.environ.get("ARIADNE_INPUT", "/kaggle/input"))
PKG = Path(os.environ.get("ARIADNE_PKG", "/kaggle/working/pkg"))


def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr)[-4000:]


def install_gsplat(torch):   # as train_smoke_kernel.py: the prebuilt wheel if one exists, else PyPI (JIT CUDA build)
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


def data_root() -> Path:
    """The dataset's data.tar extracted (or, if Kaggle unpacked it already, the directory holding spools/)."""
    for tar in INPUT.rglob("data.tar"):
        dst = WORK / "data"
        with tarfile.open(tar) as t:
            t.extractall(dst)
        return dst
    for d in INPUT.rglob("spools"):
        if d.is_dir():
            return d.parent
    raise FileNotFoundError(f"no data.tar or spools/ under {INPUT}")


def pct(v, q):
    v = sorted(x for x in v if x is not None)
    return v[min(len(v) - 1, int(q * (len(v) - 1) + 0.5))] if v else None


def main():
    import torch
    out = {"gsplat": GSPLAT, "torch": torch.__version__, "cuda": torch.version.cuda,
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    try:   # an image with gsplat compiled ahead of time (recon-gpu) must not pip-install over it
        import gsplat
        from gsplat import csrc   # noqa: F401  (the compiled module: no JIT build at run time)
        out["install"] = f"preinstalled {gsplat.__version__} (compiled csrc: {csrc.__file__})"
    except ImportError:
        out["install"] = install_gsplat(torch)
        sh(f"{sys.executable} -m pip install -q jaxtyping rich")
    root = data_root()
    out["dataset"] = json.loads((root / "dataset.json").read_text()) if (root / "dataset.json").exists() else None
    spools = sorted(str(p.parent) for p in (root / "spools").glob("*/spool.yaml"))
    out["spools"] = [Path(s).name for s in spools]
    out["budgets"] = {}
    for name, budget in BUDGETS.items():
        iters, max_kf = (budget, 0) if isinstance(budget, int) else budget
        dst = WORK / f"splat_{name}"
        t0 = time.time()
        rc, log = sh(f"cd {PKG} && {sys.executable} -m ariadne_recon.live.service --spool {' '.join(spools)} "
                     f"--backend gpu --training-image rgb --iters {iters} --diagnostics --scene {root / 'scene'} "
                     f"--placement briefing --max-keyframes {max_kf} --out {dst}")
        row = {"iters_per_kf": iters, "max_keyframes": max_kf, "rc": rc, "wall_s": round(time.time() - t0, 1),
               "log_tail": log[-1500:]}
        recs = [json.loads(l) for l in open(dst / "service_log.jsonl")] if (dst / "service_log.jsonl").exists() else []
        if recs:
            ms = [r["ms_total"] for r in recs]
            row.update({
                "keyframes": len(recs), "submaps": len({(r["agent"], r["sid"]) for r in recs}),
                "gaussians_last_per_submap_max": max(r["gaussians"] for r in recs),
                "ms_per_kf_median": pct(ms, 0.5), "ms_per_kf_p95": pct(ms, 0.95),
                "ms_iter_median": pct([r["ms_iter_median"] for r in recs], 0.5),
                "psnr_train_init_median": pct([r["psnr_train_init"] for r in recs], 0.5),
                "psnr_train_after_median": pct([r["psnr_train_after"] for r in recs], 0.5),
                "frozen_bitwise_all": all(r["frozen_bitwise"] in (True, None) for r in recs),
                "per_agent_keyframes": {str(a): sum(r["agent"] == a for r in recs) for a in sorted({r["agent"] for r in recs})}})
        out["budgets"][name] = row
        if dst.exists():
            with tarfile.open(WORK / f"splat_{name}.tar.gz", "w:gz") as t:
                t.add(dst, arcname=f"splat_{name}")
        json.dump(out, open(WORK / "remote.json", "w"), indent=2)   # partial results survive a later failure
        if rc != 0:   # a failed budget (e.g. the smoke) stops the run: later budgets would fail the same way
            break
    json.dump(out, open(WORK / "remote.json", "w"), indent=2)
    return 1 if any(r["rc"] != 0 for r in out["budgets"].values()) else 0   # a failed budget fails the run


if __name__ == "__main__":
    raise SystemExit(main())
