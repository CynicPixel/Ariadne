cmake_minimum_required(VERSION 3.22)
project(ariadne VERSION 0.1.0 LANGUAGES CXX)

# C++17 is the floor OpenCV 5 imposes (arch §27.4). Pinned, not "or newer",
# so a compiler defaulting to C++20 cannot silently change overload resolution
# between the dev box and CI.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Determinism (arch §5.6 / Phase 0 gate: D must be bit-identical ACROSS MACHINES).
# -ffp-contract=off forbids the compiler from fusing a*b+c into an FMA, which
# would change the last bit of a result depending on whether the target has FMA
# — i.e. differently on amd64 vs Graviton. We never enable -ffast-math anywhere.
add_compile_options(-ffp-contract=off)

option(ARIADNE_BUILD_TESTS "Build the test suite" ON)

include(cmake/AriadneDependencies.cmake)

add_subdirectory(cpp/core)

if(ARIADNE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
