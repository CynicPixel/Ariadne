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

if(ARIADNE_BUILD_TESTS)
  FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.18.0
    GIT_SHALLOW    TRUE)
  FetchContent_MakeAvailable(googletest)
endif()
