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
