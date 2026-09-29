include(GoogleTest)

add_executable(test_numerical
  numerical/test_lie.cpp
  numerical/test_covariance.cpp
  numerical/test_time.cpp
  numerical/test_philox.cpp
  numerical/test_det_math.cpp
  numerical/test_distributions.cpp
  numerical/test_degradation.cpp
  numerical/test_pinhole.cpp
  numerical/test_ba.cpp
  numerical/test_pose_graph.cpp)
target_link_libraries(test_numerical PRIVATE ariadne::core ariadne::perception ariadne::slam
  GTest::gtest_main)
gtest_discover_tests(test_numerical)

add_executable(test_protocol
  protocol/test_serialization.cpp)
target_link_libraries(test_protocol PRIVATE ariadne::core GTest::gtest_main)
gtest_discover_tests(test_protocol)

# The Phase 0 exit gate: generate -> serialize -> replay -> score, twice.
add_executable(test_phase0_gate
  end_to_end/test_phase0_gate.cpp
  end_to_end/test_mcap_replay.cpp)
target_link_libraries(test_phase0_gate PRIVATE ariadne::runtime GTest::gtest_main)
gtest_discover_tests(test_phase0_gate)

add_executable(test_unit
  unit/test_observability.cpp
  unit/test_config.cpp
  unit/test_metrics.cpp
  unit/test_perception_config.cpp
  unit/test_euroc.cpp)
target_link_libraries(test_unit PRIVATE ariadne::runtime ariadne::perception ariadne::slam GTest::gtest_main)
# Tests read the shipped configs from the source tree.
target_compile_definitions(test_unit PRIVATE ARIADNE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
gtest_discover_tests(test_unit)

# Tests that need OpenCV exist only where OpenCV does (the ariadne-core image).
if(OpenCV_FOUND)
  add_executable(test_opencv
    unit/test_opencv_identity.cpp
    unit/test_perception_cv.cpp
    unit/test_stereo_vo.cpp)
  target_link_libraries(test_opencv PRIVATE ariadne::runtime_opencv ariadne::perception_opencv
    ariadne::slam_opencv GTest::gtest_main)
  target_compile_definitions(test_opencv PRIVATE ARIADNE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
  gtest_discover_tests(test_opencv)
endif()
