# Status

Dated entries. Newest first. Phase tracking checkboxes live in `ARIADNE_PHASED_PLAN.md`.

---

## 2026-09-06 — Phase 0: OpenCV verified, determinism substrate landed

### 0.2 smoke tests — now `[BUILD-VERIFIED]` on amd64

`ariadne/opencv-base:5.0.0` built and verified. `infra/scripts/smoke_test.cpp` compiles,
links and runs inside it; **the compile-and-link succeeding is the evidence**. Full results
in `SMOKE_TEST_RESULTS.md` §4b. Highlights:

- **HashTSDF constructs** → §20.1 satisfied, the from-scratch TSDF fallback is not needed.
- `ALIKED`/`DISK`/`LightGlueMatcher`/`ANNIndex` present; `ALIKED::create` resolves in the
  installed `.so`. **XFeat still absent** (issue I1 below).
- `DisparityWLSFilter` and the `quality` module link → C7 contrib path works.
- `build_info_hash = 9127192543017384188` for the §28.2 manifest.
- Headers install to `include/opencv5/`, **not** `opencv4/`.

Two build corrections came out of this (D7, D8): `xfeatures2d` removed from `BUILD_LIST`
(it stalled the first build on configure-time blob downloads and nothing in the
architecture uses it), and manifests must pin the **peeled commit**, not the tag-object
SHA that `git ls-remote --refs` returns for an annotated tag.

### 0.4/0.7 — deterministic primitives, the substrate C6 rests on

- `core/random/philox.hpp` — Philox4x32-10 counter-based RNG. **Passes all three
  authoritative known-answer vectors** from the Random123 reference implementation, so our
  stream matches the published algorithm rather than merely being self-consistent.
  Stateless by construction: value = f(key, counter), so thread count and iteration order
  cannot change a single bit (both asserted under real concurrency).
- `core/math/det_math.hpp` — `DetExp`/`DetLog` built from correctly-rounded operations
  only. libm's exp/log are *not* specified to be correctly rounded and legitimately differ
  between glibc versions and between x86-64 and AArch64; calling them inside `D` would
  break cross-machine bit-identity silently, one ulp at a time. libm is used as the
  **accuracy oracle in tests**, never as the runtime path.

### Determinism gate — measured, not asserted

`infra/scripts/check_determinism.sh` (repeatable; belongs in CI):

```
PASS g++ -O0/-O2/-O3 · clang++ -O0/-O2/-O3   →  6/6 bit-identical
det_math_digest = 8409701359450404215
philox_digest   = 215287054852263674
```

**The control proves the test has teeth:** `-ffast-math` *changes* the det_math digest
(`8409…` → `12633…`), which is exactly the failure D5's `-ffp-contract=off` guards against.
The Philox digest is unchanged even under `-ffast-math`, as an integer-only algorithm should be.

Precise caveat: on this host the *default* (no `-ffp-contract=off`) also matched, because
the default `-march` has no FMA. **On Arm, FMA is baseline and the default would contract**,
so the flag is load-bearing for the arm64 arm, not decorative.

**Tests: 48/48 green.**

### Not started

0.4 serialization (FlatBuffers + `ByteRecord`), 0.5 logging/manifest/seed plumbing,
0.6 CI + Terraform, 0.7 MCAP replay and the `D` operator itself (its primitives now exist),
0.3 COOL day-1.

---

## 2026-08-31 — Phase 0 opened


### Done and verified

- **0.1 (partial)** Monorepo tree per arch §3 (106 directories); CMake superbuild with
  hermetic `FetchContent` pins; `.gitignore` / `.dockerignore`; this doc plus
  `DECISIONS.md` and `SMOKE_TEST_RESULTS.md`.
- **0.4 (partial)** `cpp/core` — Eigen-only, no OpenCV, per the §3 dependency rule:
  - `core/math/lie.hpp` — SO(3)/SE(3) exp/log/adjoint, left-multiplicative convention,
    `[rho; phi]` ordering (D1), stable small-angle and near-pi branches.
  - `core/time/timestamp.hpp` — integer-nanosecond `Timestamp`/`Duration`, §10.6
    time-offset contract carried explicitly as a stated simplification.
  - `core/uncertainty/covariance.hpp` — SE(3) covariance transport, composition and
    inverse rules derived for the left convention.
  - **22/22 tests green** (`ctest`). Covariance composition and inverse are checked
    against 200k-sample Monte Carlo, not against a re-derivation; a control test
    confirms the naive no-adjoint rule is rejected by the same check.
- **0.2 (partial)** Smoke tests at `[SOURCE-VERIFIED]` level — see `SMOKE_TEST_RESULTS.md`.
  C3's module-naming discrepancy is **resolved**; HashTSDF **confirmed present**, so the
  from-scratch TSDF fallback is not needed.

### In flight

- `ariadne/opencv-base:5.0.0` amd64 image building locally (`-j4`). Promotes the
  smoke-test rows from `[SOURCE-VERIFIED]` to `[BUILD-VERIFIED]`.

### Not started

0.4 serialization (FlatBuffers + `ByteRecord`), 0.5 logging/manifest/seeds,
0.6 CI + Terraform, 0.7 MCAP replay + degradation operator `D`, 0.3 COOL day-1.

---

## Open issues

### I1 `[BLOCKING PHASE 2.1]` — XFeat is not in OpenCV 5

CLAUDE.md §1 locks XFeat as the primary feature extractor; it has no implementation in
OpenCV 5.0.0 or contrib. Evidence in `SMOKE_TEST_RESULTS.md` §2, decision record D6.
Needs the §49 update ritual. **Does not block Phase 0.**

### I2 `[SCHEDULE]` — the phase durations do not sum to the calendar

Phase durations total **62 days** (`4+6+10+8+7+7+5+6+9`) plus Phase 8's stated buffer,
against **57 days** to the Oct 26 platform deadline and **55** to the Oct 24 internal
target. The plan's own mitigations are gates-not-dates and the `∥A/∥B/∥C` parallel
workstreams; the compressible slack is Phase 2 (10 d) and Phase 8 (9 d), and §10.7
`attitude_only` is the named fallback for a Phase 2 overrun. Flagged now so the decision
is taken deliberately at the Phase 1 exit gate rather than discovered in October.

### I3 `[OPEN]` — arm64 has no local host

Cross-**toolchain** determinism is now proven (gcc + clang, 3 optimization levels each);
cross-**architecture** is not, and is the half that actually matters for Graviton. The
digests above are committed so the arm64 check is a one-line comparison when a box exists.

Phase 0.2/0.6 require an arm64 arm. This box is amd64-only; QEMU timings would be
meaningless and the COOL benchmark must run on a real `m8g` regardless. Deferred to a
Graviton box or GitHub Arm runner. Recorded as `[PENDING]`, not as passing.

### I4 `[COSMETIC]` — doc filename drift

Arch §3's tree names `docs/ARCHITECTURE.md`; the real file is
`docs/ARIADNE_IMPLEMENTATION_ARCHITECTURE.md`. Source of truth not renamed.
