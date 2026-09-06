# Third-party pins. Fetched in-tree rather than apt-installed so the build is
# hermetic and needs no sudo: the same versions resolve on the dev box, in CI,
# and on Graviton. Every version here is mirrored into the run manifest (§28.2).
include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

# Eigen 3.4.1 — latest 3.4.x, which is what arch §27.1's "Eigen 3.4" pins.
# (Eigen 5.0.1 exists upstream; deliberately NOT taken — see docs/DECISIONS.md.)
# SOURCE_SUBDIR names a directory with no CMakeLists.txt on purpose: that makes
# FetchContent populate the sources WITHOUT running Eigen's own build system,
# which we do not need (Eigen is header-only) and do not want polluting our cache.
FetchContent_Declare(eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        3.4.1
  GIT_SHALLOW    TRUE
  SOURCE_SUBDIR  do-not-configure-eigen)
FetchContent_MakeAvailable(eigen)
add_library(ariadne_eigen INTERFACE)
target_include_directories(ariadne_eigen SYSTEM INTERFACE ${eigen_SOURCE_DIR})

# FlatBuffers is the locked serializer (CLAUDE.md §1): schema-versioned,
# zero-copy read, and deterministic size accounting at the single serialization
# boundary §4.4 requires. Building it here also builds `flatc`, which generates
# headers from the .fbs schemas at build time, so no generated code is committed
# and the schema is the single source of truth.
set(FLATBUFFERS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_INSTALL OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_BUILD_FLATLIB ON CACHE BOOL "" FORCE)
set(FLATBUFFERS_BUILD_FLATC ON CACHE BOOL "" FORCE)
set(FLATBUFFERS_BUILD_FLATHASH OFF CACHE BOOL "" FORCE)
FetchContent_Declare(flatbuffers
  GIT_REPOSITORY https://github.com/google/flatbuffers.git
  GIT_TAG        v25.12.19
  GIT_SHALLOW    TRUE)
FetchContent_MakeAvailable(flatbuffers)

# zstd: applied to message classes 4-5 (MATCH_RESPONSE, POSE_EDGE) per §15.7.
# Descriptors are high-entropy, so the achieved ratio is MEASURED and reported,
# never assumed. Its CMake lives in build/cmake, not at the repo root.
set(ZSTD_BUILD_PROGRAMS OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(ZSTD_BUILD_STATIC ON CACHE BOOL "" FORCE)
set(ZSTD_LEGACY_SUPPORT OFF CACHE BOOL "" FORCE)
FetchContent_Declare(zstd
  GIT_REPOSITORY https://github.com/facebook/zstd.git
  GIT_TAG        v1.5.7
  GIT_SHALLOW    TRUE
  SOURCE_SUBDIR  build/cmake)
FetchContent_MakeAvailable(zstd)

if(ARIADNE_BUILD_TESTS)
  FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.18.0
    GIT_SHALLOW    TRUE)
  FetchContent_MakeAvailable(googletest)
endif()
