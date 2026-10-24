#!/bin/bash
# Build and run wire_dump.cpp on whatever host this runs on (AWS cpuarm via spinup.sh, or locally in ubuntu:24.04):
# configure the repo WITHOUT OpenCV, build test_protocol (it pulls the pinned Eigen 3.4.1, FlatBuffers v25.12.19,
# zstd v1.5.7 and generates wire_generated.h), then compile wire_dump.cpp with test_serialization.cpp's own compile
# flags (compile_commands.json) and link the same static zstd. Output: $OUT (default $ARIADNE_RESULTS or .)/wire_dump_<arch>.txt
set -euo pipefail
OUT="${OUT:-${ARIADNE_RESULTS:-.}}"
command -v cmake >/dev/null && command -v g++ >/dev/null && command -v ninja >/dev/null || \
  { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq g++ cmake ninja-build git python3 >/dev/null; }
B=/tmp/wdbuild
cmake -S . -B $B -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON > $OUT/wire_dump_configure.log 2>&1
cmake --build $B --target test_protocol --parallel "${JOBS:-$(nproc)}" > $OUT/wire_dump_build.log 2>&1
CMD=$(python3 - "$B" <<'PY'
import json, sys, shlex
for e in json.load(open(sys.argv[1] + "/compile_commands.json")):
    if e["file"].endswith("tests/protocol/test_serialization.cpp"):
        a = shlex.split(e["command"]) if "command" in e else e["arguments"]
        out = []
        skip = False
        for x in a:
            if skip: skip = False; continue
            if x in ("-o", "-c"): skip = x == "-o"; continue
            if x.endswith("test_serialization.cpp"): continue
            out.append(x)
        print(" ".join(shlex.quote(x) for x in out)); break
PY
)
ZSTD=$(find $B -name libzstd.a | head -1)
eval "$CMD infra/eval/wire_dump/wire_dump.cpp $ZSTD -o $B/wire_dump"
{ echo "# $(uname -m) $(g++ --version | head -1)"; $B/wire_dump; } > $OUT/wire_dump_$(uname -m).txt
cat $OUT/wire_dump_$(uname -m).txt
