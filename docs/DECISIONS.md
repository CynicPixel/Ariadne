# Local Decisions

Dated decisions taken during implementation. Anything here that changes the
architecture must be mirrored back into `ARIADNE_IMPLEMENTATION_ARCHITECTURE.md` per §49.

---

## 2026-08-31 — D1: Tangent-vector ordering is `[rho; phi]` (translation first)

Arch §11.3 pins the *perturbation convention* (left-multiplicative) but not the
*ordering* of the 6-vector. Chosen: `[translation; rotation]`, matching Sophus.

**Why it matters:** the SE(3) adjoint's block layout depends on this ordering. GTSAM
orders it `[rotation; translation]`, so when GTSAM is used as a test oracle (§11.6) the
ordering must be swapped **at that boundary**, never inside `cpp/core`.
Enforced by `SE3.AdjointMovesPerturbationsBetweenFrames` in `tests/numerical/test_lie.cpp`,
which fails on a wrong ordering while every round-trip test still passes.

## 2026-08-31 — D2: Eigen pinned at 3.4.1, not 5.0.x

Arch §27.1 says "Eigen 3.4". Upstream now also offers 5.0.1. Doc wins: pinned **3.4.1**
(latest 3.4.x). Revisit only if a needed feature is 5.x-only; it would be an arch change.

## 2026-08-31 — D3: MCAP via the standalone C++ SDK, not rosbag2

Arch §5.9 names "rosbag2 MCAP" as the format; arch §3's dependency rule requires replay
to work from files alone, ROS-free. These are compatible because MCAP is a container, not
a ROS API: we read/write it with the standalone Foxglove MCAP C++ SDK, so `cpp/core` never
sees a ROS type and replay needs no ROS install. Bags stay rosbag2-compatible.

## 2026-08-31 — D4: `SE3Log` inverts the left Jacobian numerically

`V^-1` is computed with a 3x3 `Eigen::inverse()` rather than its closed form. `V` is
always invertible for `|phi| < 2*pi`, the cost is negligible, and this removes a class of
transcription bugs from a coefficient formula that is easy to get subtly wrong.
Alternative (closed form) may be added later **with a test against this version**.

## 2026-08-31 — D5: `-ffp-contract=off` project-wide, no fast-math

The Phase 0 gate requires the degradation operator `D` to be bit-identical across
machines. Compilers may fuse `a*b+c` into an FMA, changing the last bit depending on
target ISA — differently on amd64 vs Graviton. Contraction is therefore disabled
globally rather than only for `D`, so the property cannot be lost by a file moving
between targets.

## 2026-08-31 — D6 `[OPEN — needs maintainer decision]`: `cv::XFeat` does not exist

CLAUDE.md §1 locks XFeat as the primary feature extractor. `docs/ALGORITHMS.md` records
this as low-risk because XFeat is "a native OpenCV 5 class", and at line 63 documents a
concrete signature:
`cv::XFeat::create(modelPath, maxKeypoints, scoreThreshold, inputSize, backendId, targetId)`.

**That class does not exist in OpenCV 5.0.0 or contrib** (evidence:
`SMOKE_TEST_RESULTS.md` §2 — exhaustive case-insensitive search of both trees). The
catalogue's claim is a factual error, and it is load-bearing in six places there
(§1.2 shortlist, the OpenCV-5 class table, the recommended per-agent front-end, and the
EC3R-SLAM "strongest cross-domain signal" argument).

**The locked decision itself may still survive.** What is false is "OpenCV ships a class
for it", not "XFeat is usable". Verified upstream facts (2026-08-31):
- `github.com/verlab/accelerated_features`, Apache-2.0 (license confirmed in catalogue L36).
- Ships **PyTorch** weights only: `weights/xfeat.pt` (6.2 MB), `xfeat-lighterglue.pt`.
  **No ONNX export is published** — we would own that step.
- The model is small (~10 conv blocks; `modules/model.py` is 4.5 KB).

**Relevant fact that applies to every option:** OpenCV 5 ships ALIKED/DISK/LightGlue as
classes but **not** their models — all three take a `modelPath` to an ONNX file. So a
learned feature path requires sourcing/exporting ONNX weights no matter which is chosen;
that cost is not unique to XFeat, and §7.4's DNN execution contract governs all of them.

Options, none yet chosen:
- **(a) Keep XFeat primary, via ONNX + `cv::dnn`.** Export `xfeat.pt` ourselves under
  §7.4 (ORT fallback allowed, log which ran). Preserves the locked decision and the
  catalogue's benchmark case. Risk: the export must survive `cv::dnn`'s ONNX coverage —
  a `[SMOKE-TEST]` gate, not an assumption.
- **(b) Promote ALIKED to primary** via the §49 ritual, demoting XFeat to an ablation.
  Native class, but C8 rates ALIKED ~5 FPS-class on CPU, and the catalogue's speed
  argument for XFeat (16.7 ms; the only learned method above 1 FPS on embedded) is
  exactly what motivated the choice.
- **(c) Defer to the Phase 2 opening**, recorded now, decided when 2.1 starts.

Does not block Phase 0. Blocks Phase 2.1.

## 2026-09-06 — D7: manifests pin the PEELED COMMIT, never the tag-object SHA

`5.0.0` is an annotated tag in both OpenCV repos. `git ls-remote --tags --refs` returns the
**tag object** SHA; the commit is the peeled `refs/tags/5.0.0^{}` row. The two differ, and an
earlier revision of `SMOKE_TEST_RESULTS.md` recorded tag objects as commits — which would have
looked like a provenance mismatch against the image's own recorded SHAs during a reproducibility
audit (§36).

**Rule:** every §28.2 manifest field naming an upstream revision records the commit, obtained
from `git rev-parse HEAD` inside the build (as `opencv-base.Dockerfile` already does) or from
the `^{}` row of `git ls-remote --tags <url> '<tag>*'`. Never from `--refs`.

Correct commits for OpenCV 5.0.0: core `40738fb16ceddb5fb3fea747585f7ce6abb0605b`,
contrib `755e50675d97db9b7d449d8bd6b09888646f6c6e`.

## 2026-09-06 — D8: contrib build set is exactly C7's four modules

`xfeatures2d` was in the first `BUILD_LIST` speculatively and **removed**: nothing in the
architecture uses SURF/BRIEF/FREAK/DAISY, and it fetches `boostdesc`/`vgg` blobs at *configure*
time from a slow third-party host, which stalled the first build entirely. Built contrib set is
`ximgproc`, `xphoto`, `optflow`, `quality` — exactly the modules C7 names.
