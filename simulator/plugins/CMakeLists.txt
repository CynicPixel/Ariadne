# gz-sim system plugins and sim-side tools for the Ariadne simulator. Built in the
# ariadne-sim image with ROS Jazzy sourced (gz-sim 8 / sdformat 14 from its vendor
# packages; rosbag2 for the canonical-stream writer):
#   cmake -S simulator/plugins -B build/sim-plugins && cmake --build build/sim-plugins
# The trajectory math is included from cpp/core (header-only, dependency-free) so
# the plugin runs exactly the code the core test suite checks against Python.
cmake_minimum_required(VERSION 3.22)
project(ariadne_sim_plugins CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
find_package(gz-sim8 REQUIRED)
find_package(gz-plugin2 REQUIRED COMPONENTS register)
find_package(gz-transport13 REQUIRED)
find_package(gz-msgs10 REQUIRED)
find_package(sdformat14 REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(rosbag2_storage REQUIRED)
find_package(ros_gz_bridge REQUIRED)
find_package(rosgraph_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# The plugin and the canonical-stream writer evaluate the same inline trajectory
# code (MoverScript.hh) and must produce bit-identical doubles (D39). With
# -ffp-contract=off the compiler may not fuse a*b+c into an FMA in one inlining
# context and not in another (GCC's default in GNU mode fuses where the target has
# FMA, e.g. aarch64/Graviton), so every operation rounds exactly as written.
set(ARIADNE_SIM_FLAGS -Wall -Wextra -Werror -ffp-contract=off)
set(ARIADNE_SIM_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../../cpp)

add_library(ariadne_kinematic_mover SHARED kinematic_mover/KinematicMover.cc)
target_include_directories(ariadne_kinematic_mover PRIVATE ${ARIADNE_SIM_INCLUDES})
target_compile_options(ariadne_kinematic_mover PRIVATE ${ARIADNE_SIM_FLAGS})
target_link_libraries(ariadne_kinematic_mover PRIVATE
  gz-sim8::gz-sim8 gz-plugin2::register gz-transport13::gz-transport13 gz-msgs10::gz-msgs10)

# ros_cdr.cpp is std-only: compiled in directly so the tool round-trips every
# message through the runtime's own decoder without linking the runtime library.
add_executable(ariadne_canonical_streams tools/canonical_streams.cc ../../cpp/runtime/replay/ros_cdr.cpp)
target_include_directories(ariadne_canonical_streams PRIVATE ${ARIADNE_SIM_INCLUDES})
target_compile_options(ariadne_canonical_streams PRIVATE ${ARIADNE_SIM_FLAGS})
target_link_libraries(ariadne_canonical_streams PRIVATE
  sdformat14::sdformat14 rclcpp::rclcpp rosbag2_cpp::rosbag2_cpp rosbag2_storage::rosbag2_storage)

# The sim recorder (D40): gz-transport -> rosbag2 with no DDS hop, converting with
# ros_gz_bridge's own installed convert_gz_to_ros (same bytes the bridge publishes).
add_executable(ariadne_gz_recorder tools/gz_recorder.cc)
target_compile_options(ariadne_gz_recorder PRIVATE ${ARIADNE_SIM_FLAGS})
target_link_libraries(ariadne_gz_recorder PRIVATE
  ros_gz_bridge::ros_gz_bridge gz-transport13::gz-transport13 gz-msgs10::gz-msgs10
  rclcpp::rclcpp rosbag2_cpp::rosbag2_cpp rosbag2_storage::rosbag2_storage
  ${rosgraph_msgs_TARGETS} ${sensor_msgs_TARGETS} ${geometry_msgs_TARGETS})

install(TARGETS ariadne_kinematic_mover LIBRARY DESTINATION lib)
install(TARGETS ariadne_canonical_streams ariadne_gz_recorder RUNTIME DESTINATION bin)
