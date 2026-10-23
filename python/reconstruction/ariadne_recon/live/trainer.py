"""The incremental trainer (RECON_6 §2; D190-D193, D195): one keyframe at a time, only the active submap.

Per keyframe of agent k:
  1. assign it to k's active submap (SubmapManager); a new submap starts with no Gaussians
  2. render the submap at the keyframe (no grad) -> alpha; seed new Gaussians where alpha < 0.5 (seeding.py), append
     them to the parameters and to Adam's state (old rows keep their moments: checked on a T4 by bench_kernel.py)
  3. `iters` optimisation steps: >= 40 % on the new keyframe, the rest on uniformly drawn keyframes of the same
     submap (Gaussian-SLAM); loss 0.8 L1 + 0.2 D-SSIM (Kerbl 2023) on the training image + sigma-weighted depth L1
     (§19.3 L_photo + L_depth), over pixels where both are defined
  4. lifecycle (arch §18.4, D191): a Gaussian visible in the keyframe whose camera-frame depth agrees with the
     keyframe's stereo depth within 3 sigma at its centre pixel gains one count; at `stable_count` (15) it is STABLE
     and its gradients are zeroed (frozen); opacity < 0.005 is pruned (gsplat DefaultStrategy's threshold)
  5. exposure (arch §18.7, D192): a per-keyframe affine on the rendered image, identity, trainable only once the
     submap's own step counter has passed 5000

Iterations per keyframe: `iters` fixed (batch/gate path), or `budget_ms` (live): iterations = budget / the measured
median ms per iteration of this submap so far (derived from measurement, D190; at least `min_iters`).

The training image is the albedo proxy (1 channel, D195) unless the run is the RGB ablation (3 channels).
GPU only (gsplat rasterizes in CUDA); the CPU parts live in submaps.py / seeding.py / export.py.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .seeding import SeedParams, seed
from .submaps import SubmapManager

SH_C0 = 0.28209479177387814


@dataclass(frozen=True)
class TrainParams:
    iters: int = 0                # > 0: fixed iterations per keyframe (batch path)
    budget_ms: float = 0.0        # > 0: live, iterations from the measured ms/iteration
    min_iters: int = 3
    new_kf_fraction: float = 0.4  # Gaussian-SLAM: >= 40 % of the iterations on the new keyframe
    lambda_dssim: float = 0.2     # Kerbl 2023
    lambda_depth: float = 0.1     # [ASSUMPTION] depth weight relative to the photometric term; logged, not tuned live
    stable_count: int = 15        # D191: RTG delta_eta 100 frames at 30 Hz -> 15 keyframes at 4.4 KF/s
    prune_opacity: float = 0.005  # gsplat DefaultStrategy
    init_opacity: float = 0.99    # RTG-SLAM's opaque discs
    exposure_after_steps: int = 5000   # arch §18.7 / D192
    max_keyframes: int = 30       # RECON_6: arch 20-40
    lr_means: float = 1.6e-4      # gsplat simple_trainer defaults (means lr scaled by scene scale 1)
    lr_scales: float = 5e-3
    lr_quats: float = 1e-3
    lr_opacities: float = 5e-2
    lr_colors: float = 2.5e-3
    seed: int = 0


def ssim(torch, a, b):
    """SSIM of [1, H, W, C] images, 11x11 sigma 1.5 Gaussian window (the 3DGS reference loss)."""
    F = torch.nn.functional
    c = a.shape[-1]
    x, y = a.permute(0, 3, 1, 2), b.permute(0, 3, 1, 2)
    g = torch.exp(-((torch.arange(11, device=a.device) - 5.0) ** 2) / (2 * 1.5 ** 2))
    w = ((g[:, None] * g[None, :]) / g.sum() ** 2).expand(c, 1, 11, 11).contiguous()
    blur = lambda t: F.conv2d(t, w, padding=5, groups=c)
    mx, my = blur(x), blur(y)
    sxx, syy, sxy = blur(x * x) - mx * mx, blur(y * y) - my * my, blur(x * y) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))


class _SubmapState:
    """The GPU tensors of one submap (anchor frame) and its optimiser."""

    def __init__(self, torch, dev, channels, p: TrainParams):
        self.t, self.dev, self.p = torch, dev, p
        z = lambda *s: torch.zeros(*s, device=dev)
        self.params = {"means": torch.nn.Parameter(z(0, 3)), "scales": torch.nn.Parameter(z(0, 3)),
                       "quats": torch.nn.Parameter(z(0, 4)), "opacities": torch.nn.Parameter(z(0)),
                       "colors": torch.nn.Parameter(z(0, channels))}
        lr = {"means": p.lr_means, "scales": p.lr_scales, "quats": p.lr_quats, "opacities": p.lr_opacities,
              "colors": p.lr_colors}
        self.opt = torch.optim.Adam([{"params": [v], "lr": lr[k], "name": k} for k, v in self.params.items()],
                                    eps=1e-15)
        self.count = z(0)                      # lifecycle confidence (keyframe updates)
        self.stable = torch.zeros(0, dtype=torch.bool, device=dev)
        self.steps = 0
        self.ms_iter: list[float] = []
        self.exposure: dict[int, object] = {}  # kf -> (C+1) x C affine, identity, trainable after the delay
        self.exp_opt = None

    def __len__(self):
        return int(self.params["means"].shape[0])

    def _rebuild(self, new: dict | None, keep=None):
        """Grow (new rows) or shrink (keep mask) every parameter and carry Adam's moments row for row."""
        torch = self.t
        for g in self.opt.param_groups:
            old = g["params"][0]
            name = g["name"]
            data = old.detach()
            st = self.opt.state.pop(old, None)
            if keep is not None:
                data = data[keep]
            if new is not None:
                data = torch.cat([data, new[name]], 0)
            p = torch.nn.Parameter(data)
            g["params"][0] = p
            if st is not None:
                for k in ("exp_avg", "exp_avg_sq"):
                    m = st[k] if keep is None else st[k][keep]
                    if new is not None:
                        m = torch.cat([m, torch.zeros_like(new[name])], 0)
                    st[k] = m
                self.opt.state[p] = st
            self.params[name] = p

    def append(self, means, scales, colors, init_opacity):
        torch = self.t
        n = len(means)
        if n == 0:
            return
        f = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=self.dev)
        q = torch.zeros(n, 4, device=self.dev)
        q[:, 0] = 1.0
        new = {"means": f(means), "scales": torch.log(f(scales))[:, None].repeat(1, 3), "quats": q,
               "opacities": torch.full((n,), float(np.log(init_opacity / (1 - init_opacity))), device=self.dev),
               "colors": f(colors)}
        self._rebuild(new)
        self.count = torch.cat([self.count, torch.zeros(n, device=self.dev)])
        self.stable = torch.cat([self.stable, torch.zeros(n, dtype=torch.bool, device=self.dev)])

    def prune(self, threshold):
        torch = self.t
        keep = torch.sigmoid(self.params["opacities"].detach()) >= threshold
        if bool(keep.all()):
            return 0
        n = int((~keep).sum())
        self._rebuild(None, keep)
        self.count, self.stable = self.count[keep], self.stable[keep]
        return n


class LiveTrainer:
    def __init__(self, channels: int, p: TrainParams, seed_params: SeedParams = SeedParams(), device: str = "cuda"):
        import torch
        from gsplat import rasterization
        self.torch, self.raster = torch, rasterization
        self.dev = torch.device(device)
        self.p, self.sp, self.channels = p, seed_params, channels
        self.manager = SubmapManager(p.max_keyframes)
        self.state: dict[int, _SubmapState] = {}
        self.frames: dict[int, dict] = {}      # kf -> tensors of its training data (active submaps only)
        self.rng = np.random.default_rng(p.seed)
        self.log: list[dict] = []

    # --- rendering -------------------------------------------------------------------------------------------------
    def _render(self, st: _SubmapState, kf: int, grad: bool):
        torch = self.torch
        fr = self.frames[kf]
        P = st.params
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            img, alpha, meta = self.raster(P["means"], torch.nn.functional.normalize(P["quats"], dim=-1),
                                           torch.exp(P["scales"]), torch.sigmoid(P["opacities"]), P["colors"],
                                           fr["viewmat"][None], fr["K"][None], fr["w"], fr["h"], sh_degree=None,
                                           packed=False, render_mode="RGB+ED", near_plane=0.01, far_plane=100.0)
        return img[0], alpha[0, ..., 0], meta

    # --- one keyframe ----------------------------------------------------------------------------------------------
    def add_keyframe(self, agent: int, kf: int, T_O_C: np.ndarray, pose_version: int, K: np.ndarray,
                     image: np.ndarray, Z: np.ndarray, sigma: np.ndarray) -> dict:
        torch = self.torch
        t0 = time.perf_counter()
        sm, opened = self.manager.add_keyframe(agent, kf, T_O_C, pose_version)
        if opened:
            self.state[sm.sid] = _SubmapState(torch, self.dev, self.channels, self.p)
            for other in [s for s in self.manager.submaps if s.agent == agent and s.closed and s.sid in self.state]:
                self._close(other.sid)
        st = self.state[sm.sid]
        h, w = Z.shape
        T_A_C = sm.T_anchor_C[kf]
        img = np.asarray(image, np.float32).reshape(h, w, -1)
        f = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=self.dev)
        valid = np.isfinite(img).all(axis=2) & np.isfinite(Z) & np.isfinite(sigma) & (sigma <= self.sp.max_sigma_m)
        self.frames[kf] = {"viewmat": f(np.linalg.inv(T_A_C)), "K": f(K), "w": w, "h": h, "sid": sm.sid,
                           "img": f(np.nan_to_num(img)), "Z": f(np.nan_to_num(Z)), "S": f(np.nan_to_num(sigma, nan=1.0)),
                           "mask": torch.as_tensor(valid, device=self.dev)}
        alpha = None
        if len(st):
            _, a, _ = self._render(st, kf, grad=False)
            alpha = a.detach().cpu().numpy()
        existing = st.params["means"].detach().cpu().numpy() if len(st) else None
        s = seed(Z, sigma, img, K, T_A_C, alpha, existing, self.sp, self.rng)
        st.append(s.means, s.scales, s.colors, self.p.init_opacity)
        n_iter = self._iterations(st)
        losses = [self._step(st, kf if i < max(1, round(self.p.new_kf_fraction * n_iter)) else
                             int(self.rng.choice(sm.keyframes))) for i in range(n_iter)]
        newly_stable = self._lifecycle(st, kf)
        pruned = st.prune(self.p.prune_opacity)
        torch.cuda.synchronize()
        rec = {"agent": agent, "kf": kf, "sid": sm.sid, "opened": opened, "seeded": len(s.means), "gaussians": len(st),
               "iters": n_iter, "loss_first": losses[0] if losses else None, "loss_last": losses[-1] if losses else None,
               "stable": int(st.stable.sum()), "newly_stable": newly_stable, "pruned": pruned, "steps": st.steps,
               "exposure_affine": st.steps >= self.p.exposure_after_steps,
               "ms_iter_median": float(np.median(st.ms_iter)) if st.ms_iter else None,
               "ms_total": 1e3 * (time.perf_counter() - t0)}
        self.log.append(rec)
        return rec

    def _iterations(self, st: _SubmapState) -> int:
        if self.p.iters > 0:
            return self.p.iters
        if self.p.budget_ms > 0:
            if not st.ms_iter:
                return self.p.min_iters
            return max(self.p.min_iters, int(self.p.budget_ms / float(np.median(st.ms_iter[-50:]))))
        raise ValueError("TrainParams: set iters (batch) or budget_ms (live)")

    def _exposure(self, st: _SubmapState, kf: int):
        torch = self.torch
        if kf not in st.exposure:
            c = self.channels
            A = torch.zeros(c + 1, c, device=self.dev)
            A[:c, :c] = torch.eye(c, device=self.dev)
            st.exposure[kf] = torch.nn.Parameter(A)
            if st.exp_opt is None:
                st.exp_opt = torch.optim.Adam([st.exposure[kf]], lr=1e-3)
            else:
                st.exp_opt.add_param_group({"params": [st.exposure[kf]], "lr": 1e-3})
        return st.exposure[kf]

    def _step(self, st: _SubmapState, kf: int) -> float:
        torch = self.torch
        t0 = time.perf_counter()
        fr = self.frames[kf]
        img, _, _ = self._render(st, kf, grad=True)
        rgb, depth = img[..., : self.channels], img[..., self.channels:]
        use_exp = st.steps >= self.p.exposure_after_steps
        if use_exp:   # D192: identity until the submap's own step counter passes the delay
            A = self._exposure(st, kf)
            rgb = rgb @ A[: self.channels] + A[self.channels]
        m = fr["mask"][..., None].float()
        n = m.sum().clamp(min=1.0)
        l1 = ((rgb - fr["img"]).abs() * m).sum() / (n * self.channels)
        dssim = ((1.0 - ssim(torch, rgb[None], fr["img"][None])) * m.permute(2, 0, 1)[None]).sum() / (n * self.channels)
        ld = ((depth[..., 0] - fr["Z"]).abs() / fr["S"].clamp(min=1e-3) * fr["mask"]).sum() / n
        loss = (1 - self.p.lambda_dssim) * l1 + self.p.lambda_dssim * dssim + self.p.lambda_depth * ld
        st.opt.zero_grad(set_to_none=True)
        if st.exp_opt is not None:
            st.exp_opt.zero_grad(set_to_none=True)
        loss.backward()
        if bool(st.stable.any()):   # STABLE Gaussians are frozen (arch §18.4)
            for v in st.params.values():
                if v.grad is not None:
                    v.grad[st.stable] = 0
        st.opt.step()
        if use_exp:
            st.exp_opt.step()
        st.steps += 1
        torch.cuda.synchronize()
        st.ms_iter.append(1e3 * (time.perf_counter() - t0))
        return float(loss.detach())

    def _lifecycle(self, st: _SubmapState, kf: int) -> int:
        """Count consistent observations: visible in the keyframe and the Gaussian's camera-frame depth within 3 sigma
        of the stereo depth at its centre pixel (D191)."""
        torch = self.torch
        if not len(st):
            return 0
        fr = self.frames[kf]
        with torch.no_grad():
            V = fr["viewmat"]
            X = st.params["means"] @ V[:3, :3].T + V[:3, 3]
            z = X[:, 2]
            K = fr["K"]
            u = torch.round(K[0, 0] * X[:, 0] / z.clamp(min=1e-6) + K[0, 2]).long()
            v = torch.round(K[1, 1] * X[:, 1] / z.clamp(min=1e-6) + K[1, 2]).long()
            inside = (z > 0.01) & (u >= 0) & (u < fr["w"]) & (v >= 0) & (v < fr["h"])
            ui, vi = u.clamp(0, fr["w"] - 1), v.clamp(0, fr["h"] - 1)
            ok = inside & fr["mask"][vi, ui] & ((z - fr["Z"][vi, ui]).abs() <= 3 * fr["S"][vi, ui])
            st.count += ok.float()
            before = int(st.stable.sum())
            st.stable |= st.count >= self.p.stable_count
            return int(st.stable.sum()) - before

    def _close(self, sid: int):
        """A closed submap is frozen: drop its frames' training tensors (bounded memory) but keep its Gaussians."""
        for kf in [k for k, fr in self.frames.items() if fr["sid"] == sid]:
            del self.frames[kf]
        st = self.state[sid]
        st.opt.state.clear()

    # --- export ----------------------------------------------------------------------------------------------------
    def params_numpy(self, sid: int) -> dict:
        st = self.state[sid]
        P = {k: v.detach().cpu().numpy() for k, v in st.params.items()}
        c = np.clip(P["colors"], 0, 1)
        rgb = np.repeat(c, 3, axis=1) if c.shape[1] == 1 else c[:, :3]
        return {"means": P["means"], "scales": P["scales"], "quats": P["quats"], "opacities": P["opacities"],
                "sh0": ((rgb - 0.5) / SH_C0)[:, None, :].astype(np.float32)}
