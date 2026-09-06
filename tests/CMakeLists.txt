include(GoogleTest)

add_executable(test_numerical
  numerical/test_lie.cpp
  numerical/test_covariance.cpp
  numerical/test_time.cpp
  numerical/test_philox.cpp
  numerical/test_det_math.cpp
  numerical/test_distributions.cpp
  numerical/test_degradation.cpp)
target_link_libraries(test_numerical PRIVATE ariadne::core ariadne::perception GTest::gtest_main)
gtest_discover_tests(test_numerical)

add_executable(test_protocol
  protocol/test_serialization.cpp)
target_link_libraries(test_protocol PRIVATE ariadne::core GTest::gtest_main)
gtest_discover_tests(test_protocol)

# The Phase 0 exit gate: generate -> serialize -> replay -> score, twice.
add_executable(test_phase0_gate
  end_to_end/test_phase0_gate.cpp)
target_link_libraries(test_phase0_gate PRIVATE ariadne::runtime GTest::gtest_main)
gtest_discover_tests(test_phase0_gate)
