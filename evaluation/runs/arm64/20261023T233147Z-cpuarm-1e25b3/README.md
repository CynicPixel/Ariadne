# arm64 GravityPgo divergence: root cause (A1 diagnostic, 2026-10-24 IST; not a gate)

**Verdict: the arm64 numerical path is correct. The test feeds different noise to the two architectures.**

`GravityPgo.RecoversTheReferenceGravityErrorAndReducesTilt` (`tests/unit/test_gravity_pgo.cpp`) gets 0.62 deg on
x86 and 2.71 deg on Graviton (run `20261023T193621Z-cpuarm-7f8c96`, at commit b5b9ee4). The cause is in the
fixture, in `Make()`:

    p.g_C = (Rot(s * Vec3(nz(gen), nz(gen), 0)) * gC).normalized();

- The order in which C++ evaluates function arguments is unspecified ([expr.call]), and GCC's order differs by
  target.
  - GCC x86-64 evaluates the two `nz(gen)` right to left: the first draw goes to y.
  - GCC aarch64 evaluates them left to right: the first draw goes to x.
- So every gravity prior gets its two noise components swapped across architectures. That is a different noise
  realisation, not a rounding difference.
  - The odometry chain matches to 12 digits (tilt0 3.78076835862948 vs …6304): its draws are sequenced (Eigen's
    comma initialiser, element assignment).
  - The priors differ, and so do the initial energy (26.397 on arm64 vs 35.112 on x86) and the optimum.

## Evidence

This run: c8g.4xlarge spot (Neoverse-V2), gcc 13.3.0, `-O3 -ffp-contract=off`, OpenCV 5.0.0 (carotene + KleidiCV
HAL), commit a66dac6, diag `infra/eval/gpgo_diag.cpp` with `infra/eval/gpgo_arm64.sh`. The run used the diag
before the ORDER switch and the `noisy` mode were added; the modes it ran are unchanged.

| | x86 (local, gcc 13.3) | arm64 (this run) |
|---|---|---|
| input digest / E0 | `e91cd138…` / 35.1122 | `ced7640a…` / 26.3970 |
| x86 with ORDER=0 (aarch64's order) | `537b4900…` / **26.3970** | — |
| gerr, default / tight tolerances | 0.6238 / 0.6236 | 2.7104 / 2.7104 |
| x86 with ORDER=0, default / tight | **2.7104 / 2.7104**, tilt 3.1702 | 2.7104, tilt 3.1702 |
| seeds 0–99 passing the test's bounds | 17/100 (own order); **14/100** (ORDER=0) | **14/100** |

- **x86 forced to aarch64's draw order reproduces arm64 to the printed digits** (`x86_order.txt`).
  - The remaining digest difference (`537b…` vs `ced7…`) is ULP-level: libm and Eigen NEON. tilt0 agrees to 1e-12.
- **arm64's solve is converged and well-conditioned.**
  - Tight tolerances agree (`tight.txt`).
  - Perturbing every initial pose and prior by 1e-12 leaves gerr at 2.7104 (`perturb.txt`). On x86, even 1e-4
    relative moves it by < 0.01 deg.
- **The HAL is not involved.** The LM driver's CV_64F `cv::Mat` ops (MatExpr scale, add, mul, sum, norm, sqrt) are
  bit-exact or ≤ 7e-16 against scalar loops (`halcheck.txt`).
  - A scalar re-implementation of `LevMarqBase::optimize` gives the same 10 iterations and 2.710381 deg
    (`pure_vec.txt`).
- **Eigen vectorisation is not involved.** `-DEIGEN_DONT_VECTORIZE` gives the same result (`pure_novec.txt`,
  `trace_novec.txt`). The worst LDLT relative residual is 8.8e-10.
- **FMA is not involved.** A fully FMA-contracted x86 build (`-mfma -ffp-contract=fast`) lands on 0.623837 deg.
- **The test is underpowered on both architectures.** Over seeds 0–99 (`seeds.txt`), "gerr < 1.0 deg and tilt
  reduced" holds on 14–17 %; median gerr is about 2 deg, p90 about 4 deg. Default and tight tolerances agree, so these
  are optima, not early stops. Seed 3 passes on x86 by luck.

## The same pattern elsewhere

`sequencing_proposal.txt` is a dry run that reads only. It lists 44 parenthesised initialisers with ≥ 2 RNG draws in
13 test files: pcm, gnc, certificate, joint_robust, degeneracy, loop_backend, window_ba(_levmarq / _inertial),
inertial_window_fixture, lie, gravity_prior, gravity_pgo.
- None are in `cpp/` (production code).
- Braced initialisers `{a(g), b(g)}` are evaluated left to right by rule ([dcl.init.list]), even when they resolve
  to an ordinary constructor. The proposed fix is to brace them. That makes every architecture draw the same noise
  (aarch64's current order).
- This plausibly explains the other cross-arch differences in run 7f8c96: JointRobust shared gravity 3.04/1.13 vs
  2.77/0.90, the Certificate planted-outlier seed, and the GNC near-miss acceptances. [UNVERIFIED] per test until
  the fix runs on both architectures.

## Not explained by this

`PgoLevMarq.NoisyGraphReachesAStationaryPoint` takes 23 iterations on arm64 and 9 on x86. Its fixture is sequenced,
and E0 (919.8791) and the final E (5.7441) match on both.
- With rel-energy and step tolerances of 1e-14, the last LM attempts accept or reject on rounding noise. FMA moves the
  x86 count only 9 → 10 (`gpgo_diag noisy`), so the arm64 count is not fully explained.
- It passes on both. But arm64's final max|grad| is 5.5e-7, only 1.8× under the test's 1e-6 bound. Open; the
  `noisy` mode traces it on the next arm64 run.
