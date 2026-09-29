# ariadne_runtime: replay, metrics and process-level glue. Sits above the domain
# libraries per the arch §3 dependency direction, and still carries no ROS,
# simulator or AWS types.
#
# A compiled library rather than header-only because MCAP requires exactly one
# translation unit to define MCAP_IMPLEMENTATION.
add_library(ariadne_runtime STATIC
  replay/mcap_impl.cpp)
target_include_directories(ariadne_runtime PUBLIC ${CMAKE_SOURCE_DIR}/cpp)
target_link_libraries(ariadne_runtime PUBLIC ariadne::core ariadne::perception ariadne_mcap)
add_library(ariadne::runtime ALIAS ariadne_runtime)

# ariadne_runtime_opencv: the runtime pieces that must see OpenCV. A separate
# target so ariadne_runtime (replay, metrics, the manifest) keeps building on a
# host without OpenCV; it exists only when OpenCV was found.
if(OpenCV_FOUND)
  add_library(ariadne_runtime_opencv STATIC
    manifest/opencv_identity.cpp)
  target_link_libraries(ariadne_runtime_opencv PUBLIC ariadne::runtime opencv_core)
  add_library(ariadne::runtime_opencv ALIAS ariadne_runtime_opencv)

  add_executable(ariadne_identity tools/ariadne_identity.cpp)
  target_link_libraries(ariadne_identity PRIVATE ariadne::runtime_opencv)
  # COMPONENT so the image installs only our binaries, not the install rules
  # that FetchContent dependencies (GoogleTest, yaml-cpp) register.
  install(TARGETS ariadne_identity RUNTIME DESTINATION bin COMPONENT ariadne)
endif()
