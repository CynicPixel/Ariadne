#!/usr/bin/env bash
# Payload for: infra/scripts/spinup.sh run --profile cpu -- "$(cat infra/scripts/tests/determinism_graviton.sh)"
# Runs the determinism gate on an Amazon Linux 2023 Graviton instance with that
# distro's own compilers (gcc 11 / clang 15 at time of writing), so it tests
# cross-architecture AND cross-compiler-version bit-identity. validation.md §4e.
rc=0
dnf install -y -q gcc-c++ clang >/dev/null 2>&1 || { echo "compiler install failed"; exit 3; }
{ uname -a; lscpu | grep -E 'Model name|Architecture|^Flags'; g++ --version | head -1; clang++ --version | head -1; } > "$ARIADNE_RESULTS/host.txt"
cat "$ARIADNE_RESULTS/host.txt"
./infra/scripts/check_determinism.sh > "$ARIADNE_RESULTS/determinism.txt" 2>&1 || rc=$?
cat "$ARIADNE_RESULTS/determinism.txt"
exit $rc
