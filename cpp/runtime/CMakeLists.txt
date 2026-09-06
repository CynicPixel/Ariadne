# ariadne_runtime: replay, metrics and process-level glue. Sits above the
# domain libraries per the arch §3 dependency direction, and still carries no
# ROS, simulator or AWS types.
add_library(ariadne_runtime INTERFACE)
target_include_directories(ariadne_runtime INTERFACE ${CMAKE_SOURCE_DIR}/cpp)
target_link_libraries(ariadne_runtime INTERFACE ariadne::core)
add_library(ariadne::runtime ALIAS ariadne_runtime)
