#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Phase 0 determinism gate: the deterministic primitives must produce
# bit-identical output across toolchains, optimization levels and (in CI)
# architectures.
#
# Run locally:   infra/scripts/check_determinism.sh
# Run in CI:     same, on both an amd64 and an arm64 runner; the digests
#                printed must match the RECORDED values below on every host.
#
# Why a digest and not a value comparison: a digest over raw IEEE bits cannot
# be fooled by a comparison that tolerates a one-ulp drift, and one number is
# cheap to compare across machines and to paste into a run manifest.
# ---------------------------------------------------------------------------
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# Recorded 2026-09-08 on x86-64, gcc 13.3.0 and clang 18.1.3.
# These are OUTPUTS of the implementation, not a specification of it. A
# deliberate change to det_math.hpp or philox.hpp means re-recording them AND
# re-deriving every golden degradation digest that depends on them.
#
# Re-recorded 2026-09-08 after two fixes: DetExp's overflow/underflow scaling
# and PhiloxStream's counter partition. The previous constants were NOT wrong
# for the code that produced them -- but note that the first two changed only
# because the sweeps above were extended. With the original sweeps, both bugs
# were byte-identical to correct code. Nothing published depended on them.
EXPECTED_DET_MATH="11806531094783593156"
EXPECTED_PHILOX="14182439601419266695"
EXPECTED_DEGRADE="8535765251477677907"

cat > "$WORK/digest.cpp" <<'CPP'
#include "core/math/det_math.hpp"
#include "core/random/philox.hpp"
#include "perception/degradation/degradation.hpp"
#include <cstdio>
using namespace ariadne::core;
using namespace ariadne::perception;
int main() {
    uint64_t d = 1469598103934665603ull;
    auto absorb = [&d](double v) {
        uint64_t b = DoubleToBits(v);
        for (int i = 0; i < 8; ++i) { d ^= (b >> (i*8)) & 0xFFull; d *= 1099511628211ull; }
    };
    for (int i = -2000; i <= 2000; ++i) absorb(DetExp(i * 0.25));
    for (int i = 1; i <= 4000; ++i)     absorb(DetLog(i * 0.125));
    // BOUNDARY COVERAGE. The sweep above spans only x in [-500, 500], which is
    // why it stayed byte-identical through a real DetExp bug: the range
    // reduction produced k=1024 (resp. -1075) only for |x| near the overflow
    // and underflow limits, and Exp2Int saturated there. A digest that never
    // evaluates the boundary cannot protect it.
    for (int i = 0; i <= 400; ++i) absorb(DetExp(708.0 + i * 0.00445));
    for (int i = 0; i <= 400; ++i) absorb(DetExp(-745.1332191019411 + i * 0.0028));
    std::printf("det_math_digest=%llu\n", (unsigned long long)d);

    uint64_t r = 1469598103934665603ull;
    auto absorb64 = [&r](uint64_t b) {
        for (int i = 0; i < 8; ++i) { r ^= (b >> (i*8)) & 0xFFull; r *= 1099511628211ull; }
    };
    for (uint64_t site = 0; site < 20000; ++site) {
        absorb64(PhiloxStream(20260906ull, 17ull, site).NextBits64());
    }
    // HIGH SITE BITS AND MULTI-BLOCK DRAWS. As above, the sweep over sites
    // 0..20000 drawing 2 words each left block=0 and site>>32 == 0, so it was
    // blind to the counter layout entirely -- it stayed byte-identical through
    // the repartition that fixed a real stream collision. These two loops
    // exercise the site high half and the block index, which is where the
    // (site, block) aliasing lived.
    for (uint64_t hi = 0; hi < 64; ++hi) {
        absorb64(PhiloxStream(20260906ull, 17ull,
                              (hi << 32) | 0x1234ull).NextBits64());
    }
    // The ENDS of the declared domain, per D18: a digest must reach the limits
    // of the contract it guards, not just a comfortable interior.
    absorb64(PhiloxStream(20260906ull, 17ull, PhiloxStream::kMaxSite).NextBits64());
    absorb64(PhiloxStream(20260906ull, 17ull, PhiloxStream::kMaxSite - 1).NextBits64());
    {
        // Draw to the last legal block (kMaxBlock = 65535, 4 words each).
        PhiloxStream deep(20260906ull, 17ull, 99ull);
        for (uint64_t i = 0; i < 2 * (PhiloxStream::kMaxBlock + 1); ++i) {
            absorb64(deep.NextBits64());
        }
    }
    std::printf("philox_digest=%llu\n", (unsigned long long)r);

    // The Phase 0 exit criterion names D specifically: bit-identical for a
    // fixed (seed, frame_id) across machines. This exercises the whole chain,
    // including the Poisson and Gaussian samplers and the dust field.
    ImageU8 img(64, 64, 3);
    for (int y = 0; y < 64; ++y)
      for (int x = 0; x < 64; ++x)
        for (int c = 0; c < 3; ++c)
          img.At(x, y, c) = (unsigned char)((x*3 + y*5 + c*7) & 0xFF);
    ImageF32 depth(64, 64);
    for (int y = 0; y < 64; ++y)
      for (int x = 0; x < 64; ++x) depth.At(x, y) = 0.5f + 0.1f * (float)x;
    DegradationParams p;
    p.radiance_scale = 0.08; p.photon_scale = 8000.0;
    p.read_noise_electrons = 3.0; p.vignette_strength = 0.35;
    p.prnu_sigma = 0.02; p.analog_gain = 2.0;
    p.enable_dust = true; p.dust_beta = 0.25;
    p.dust_airlight = 0.55; p.dust_field_strength = 0.6;
    ImageU8 out = Degrade(img, depth, p, 20260906ull, 7ull);
    uint64_t g = 1469598103934665603ull;
    for (unsigned char v : out.data) { g ^= v; g *= 1099511628211ull; }
    std::printf("degrade_digest=%llu\n", (unsigned long long)g);
    return 0;
}
CPP

status=0
ran=0

for CXX in g++ clang++; do
  command -v "$CXX" >/dev/null 2>&1 || { echo "skip: $CXX not installed"; continue; }
  for OPT in -O0 -O2 -O3; do
    # -ffp-contract=off mirrors the project-wide setting in CMakeLists.txt.
    # It matters most on Arm, where FMA is baseline and the compiler default
    # WOULD contract a*b+c into an FMA and change the last bit.
    "$CXX" -std=c++17 "$OPT" -ffp-contract=off -I "$REPO_ROOT/cpp" \
        "$WORK/digest.cpp" -o "$WORK/digest"
    out="$("$WORK/digest")"
    det="$(echo "$out" | sed -n 's/^det_math_digest=//p')"
    phi="$(echo "$out" | sed -n 's/^philox_digest=//p')"
    deg="$(echo "$out" | sed -n 's/^degrade_digest=//p')"
    ran=$((ran + 1))

    # The digest line is printed on BOTH paths, in one fixed machine-readable
    # form. The cross-architecture job diffs these lines, so it must be able to
    # compare two FAILING runs -- previously the failure path printed no '='
    # at all, the extracting grep matched nothing and the comparison job died
    # on an empty match instead of reporting the mismatch.
    verdict=PASS
    if [ "$det" != "$EXPECTED_DET_MATH" ] || [ "$phi" != "$EXPECTED_PHILOX" ] \
       || [ "$deg" != "$EXPECTED_DEGRADE" ]; then
      verdict=FAIL
      status=1
    fi
    printf '  %-4s  %-9s %-4s  det=%s philox=%s degrade=%s\n' \
           "$verdict" "$CXX" "$OPT" "$det" "$phi" "$deg"
    if [ "$verdict" = FAIL ]; then
      [ "$det" = "$EXPECTED_DET_MATH" ] || printf '        det_math got %s want %s\n' "$det" "$EXPECTED_DET_MATH"
      [ "$phi" = "$EXPECTED_PHILOX" ]   || printf '        philox   got %s want %s\n' "$phi" "$EXPECTED_PHILOX"
      [ "$deg" = "$EXPECTED_DEGRADE" ]  || printf '        degrade  got %s want %s\n' "$deg" "$EXPECTED_DEGRADE"
    fi
  done
done

if [ "$ran" -eq 0 ]; then
  echo "ERROR: no compiler available; nothing was verified." >&2
  exit 2
fi

echo
echo "arch=$(uname -m)  configurations=$ran"
if [ "$status" -eq 0 ]; then
  echo "DETERMINISM=PASS"
else
  echo "DETERMINISM=FAIL — bit-identity lost. Do NOT ship results until resolved;"
  echo "every named degradation config that already produced published results is suspect."
fi
exit "$status"
