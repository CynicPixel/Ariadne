include(GoogleTest)

add_executable(test_numerical
  numerical/test_lie.cpp
  numerical/test_covariance.cpp
  numerical/test_time.cpp
  numerical/test_philox.cpp
  numerical/test_det_math.cpp)
target_link_libraries(test_numerical PRIVATE ariadne::core GTest::gtest_main)
gtest_discover_tests(test_numerical)
