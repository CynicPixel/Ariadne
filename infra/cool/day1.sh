#!/bin/bash
# COOL day-1 check (plan 0.3; register 0.3, INV2). Runs as root ON THE COOL AMI (Ubuntu 24.04, Graviton4), launched by
#   infra/scripts/spinup.sh run --profile cool --type c8g.4xlarge --max-minutes 90 -- 'bash infra/cool/day1.sh'
# Every step writes to $ARIADNE_RESULTS immediately, so a later failure never loses an earlier step's evidence.
#   1. inspect /opt/cool: layout, sonames, CMake config, COOL's own build information and module list
#   2. build STOCK OpenCV 5.0.0 + contrib natively on the same instance: pinned peeled commits, the exact BUILD_LIST and
#      flags of infra/docker/opencv-base.Dockerfile, into /opt/stock
#   3. loader-switch probe: one binary built against stock, run with stock and with /opt/cool/cpp_sdk/lib first;
#      /proc/self/maps shows which libraries actually loaded (the gate for "same binary, loader switch only")
#   4. per-op output checksums + first timings at 1 thread and at all cores
set -uo pipefail
R="${ARIADNE_RESULTS:?}"
step() { echo "=== $(date -u +%H:%M:%S) $*" | tee -a "$R/steps.txt"; }

step "1 inspect"
uname -a > "$R/uname.txt"; lscpu > "$R/lscpu.txt"; cat /etc/os-release > "$R/os-release.txt"; nproc > "$R/nproc.txt"
ls -la /opt/cool > "$R/opt_cool.txt" 2>&1
ls -la /opt/cool/cpp_sdk/lib > "$R/cool_lib_ls.txt" 2>&1
for f in /opt/cool/cpp_sdk/lib/libopencv_*.so*; do
  [ -L "$f" ] && continue
  echo "$(basename "$f") $(readelf -d "$f" 2>/dev/null | sed -n 's/.*SONAME.*\[\(.*\)\]/\1/p')"
done > "$R/cool_sonames.txt" 2>&1
find /opt/cool -maxdepth 6 -name "OpenCVConfig*.cmake" > "$R/cool_cmake_configs.txt" 2>&1
ls /opt/cool/cpp_sdk/include > "$R/cool_include_ls.txt" 2>&1
( . /opt/cool/venvs/python_3.12/bin/activate && python3 -c \
  "import cv2;print('version',cv2.__version__);print('file',cv2.__file__);print(cv2.getBuildInformation())" ) \
  > "$R/cool_buildinfo.txt" 2>&1

step "2 stock build"
export DEBIAN_FRONTEND=noninteractive
apt-get update -q > "$R/apt.txt" 2>&1
apt-get install -y -q build-essential cmake ninja-build git pkg-config zlib1g-dev libpng-dev libjpeg-dev \
  libtiff-dev >> "$R/apt.txt" 2>&1
gcc --version > "$R/gcc.txt" 2>&1; cmake --version >> "$R/gcc.txt" 2>&1
mkdir -p /src && cd /src
git clone -q --depth 1 --branch 5.0.0 https://github.com/opencv/opencv.git opencv
git clone -q --depth 1 --branch 5.0.0 https://github.com/opencv/opencv_contrib.git opencv_contrib
{ echo "opencv $(git -C opencv rev-parse HEAD) (pinned 40738fb16ceddb5fb3fea747585f7ce6abb0605b)"
  echo "contrib $(git -C opencv_contrib rev-parse HEAD) (pinned 755e50675d97db9b7d449d8bd6b09888646f6c6e)"; } \
  > "$R/stock_commits.txt"
t0=$(date +%s)
cmake -S opencv -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/stock \
  -DOPENCV_EXTRA_MODULES_PATH=/src/opencv_contrib/modules \
  -DBUILD_LIST=core,imgproc,imgcodecs,videoio,features,flann,dnn,video,geometry,calib,stereo,ptcloud,photo,objdetect,ximgproc,xphoto,optflow,quality \
  -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_DOCS=OFF \
  -DBUILD_opencv_python3=OFF -DBUILD_opencv_apps=OFF -DWITH_GTK=OFF -DWITH_QT=OFF -DHIGHGUI_ENABLE_PLUGINS=OFF \
  -DWITH_CUDA=OFF -DWITH_IPP=OFF -DOPENCV_GENERATE_PKGCONFIG=ON -DBUILD_SHARED_LIBS=ON > "$R/stock_cmake.txt" 2>&1 \
  && cmake --build build --parallel "$(nproc)" > "$R/stock_build.txt" 2>&1 \
  && cmake --install build > "$R/stock_install.txt" 2>&1 \
  || { step "stock build FAILED"; tail -40 "$R/stock_build.txt"; exit 1; }
echo "stock_build_seconds $(( $(date +%s) - t0 ))" >> "$R/steps.txt"
for f in /opt/stock/lib/libopencv_*.so*; do
  [ -L "$f" ] && continue
  echo "$(basename "$f") $(readelf -d "$f" | sed -n 's/.*SONAME.*\[\(.*\)\]/\1/p')"
done > "$R/stock_sonames.txt"

step "3 loader-switch probe"
cd /opt/ariadne/src
g++ -O2 -std=c++17 infra/cool/probe.cpp -I/opt/stock/include/opencv5 -L/opt/stock/lib \
  -lopencv_video -lopencv_imgproc -lopencv_core -o /tmp/probe > "$R/probe_build.txt" 2>&1 \
  || { step "probe build FAILED"; cat "$R/probe_build.txt"; exit 1; }
ldd /tmp/probe > "$R/probe_ldd_stock.txt" 2>&1
LD_LIBRARY_PATH=/opt/cool/cpp_sdk/lib:/opt/stock/lib ldd /tmp/probe > "$R/probe_ldd_cool.txt" 2>&1

step "4 equivalence + first timings"
for th in 1 "$(nproc)"; do
  LD_LIBRARY_PATH=/opt/stock/lib /tmp/probe "$th" > "$R/probe_stock_t$th.txt" 2>&1
  LD_LIBRARY_PATH=/opt/cool/cpp_sdk/lib:/opt/stock/lib /tmp/probe "$th" > "$R/probe_cool_t$th.txt" 2>&1
done
step "done"
