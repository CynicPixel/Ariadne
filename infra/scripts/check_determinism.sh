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
# Added 2026-10-01 (D41): the read-time lens operator; same on g++/clang++ -O0..-O3.
EXPECTED_LENS="1253886791886396038"

cat > "$WORK/digest.cpp" <<'CPP'
// The digest computations live in exactly one place (core/determinism/
// digests.hpp, perception/degradation/degradation_digest.hpp), shared with the
// tests and the run-manifest writer, so this script cannot drift from them.
#include "core/determinism/digests.hpp"
#include "perception/degradation/degradation_digest.hpp"
#include "perception/degradation/lens_digest.hpp"
#include <cstdio>
int main() {
    std::printf("det_math_digest=%llu\n", (unsigned long long)ariadne::core::DetMathDigest());
    std::printf("philox_digest=%llu\n", (unsigned long long)ariadne::core::PhiloxDigest());
    std::printf("degrade_digest=%llu\n", (unsigned long long)ariadne::perception::DegradeDigest());
    std::printf("lens_digest=%llu\n", (unsigned long long)ariadne::perception::LensDigest());
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
    len="$(echo "$out" | sed -n 's/^lens_digest=//p')"
    ran=$((ran + 1))

    # The digest line is printed on BOTH paths, in one fixed machine-readable
    # form. The cross-architecture job diffs these lines, so it must be able to
    # compare two FAILING runs -- previously the failure path printed no '='
    # at all, the extracting grep matched nothing and the comparison job died
    # on an empty match instead of reporting the mismatch.
    verdict=PASS
    if [ "$det" != "$EXPECTED_DET_MATH" ] || [ "$phi" != "$EXPECTED_PHILOX" ] \
       || [ "$deg" != "$EXPECTED_DEGRADE" ] || [ "$len" != "$EXPECTED_LENS" ]; then
      verdict=FAIL
      status=1
    fi
    printf '  %-4s  %-9s %-4s  det=%s philox=%s degrade=%s lens=%s\n' \
           "$verdict" "$CXX" "$OPT" "$det" "$phi" "$deg" "$len"
    if [ "$verdict" = FAIL ]; then
      [ "$det" = "$EXPECTED_DET_MATH" ] || printf '        det_math got %s want %s\n' "$det" "$EXPECTED_DET_MATH"
      [ "$phi" = "$EXPECTED_PHILOX" ]   || printf '        philox   got %s want %s\n' "$phi" "$EXPECTED_PHILOX"
      [ "$deg" = "$EXPECTED_DEGRADE" ]  || printf '        degrade  got %s want %s\n' "$deg" "$EXPECTED_DEGRADE"
      [ "$len" = "$EXPECTED_LENS" ]     || printf '        lens     got %s want %s\n' "$len" "$EXPECTED_LENS"
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
