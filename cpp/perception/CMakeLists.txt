# ariadne_perception: image-domain code. Header-only for now. Depends on core
# and, deliberately, NOT on OpenCV -- the degradation operator is the
# determinism substrate and must be testable without it (arch §3).
add_library(ariadne_perception INTERFACE)
target_include_directories(ariadne_perception INTERFACE ${CMAKE_SOURCE_DIR}/cpp)
target_link_libraries(ariadne_perception INTERFACE ariadne::core)
add_library(ariadne::perception ALIAS ariadne_perception)
