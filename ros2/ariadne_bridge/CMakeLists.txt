# ariadne_bridge: RuntimeLink, an rclcpp component loaded next to ros_gz_bridge::RosGzBridge in one container
# (intra-process, no DDS hop: D40's loss was on that hop). Built in the ariadne/sim image:
#   colcon build --base-paths ros2 --build-base build/ros2/build --install-base build/ros2/install
cmake_minimum_required(VERSION 3.22)
project(ariadne_bridge CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosgraph_msgs REQUIRED)
find_package(yaml-cpp REQUIRED)

add_library(runtime_link SHARED src/runtime_link.cpp)
target_compile_options(runtime_link PRIVATE -Wall -Wextra -Werror)
target_include_directories(runtime_link PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../cpp)
target_link_libraries(runtime_link rclcpp::rclcpp rclcpp_components::component ${sensor_msgs_TARGETS}
  ${geometry_msgs_TARGETS} ${std_msgs_TARGETS} ${rosgraph_msgs_TARGETS} yaml-cpp)
rclcpp_components_register_nodes(runtime_link "ariadne_bridge::RuntimeLink")

install(TARGETS runtime_link LIBRARY DESTINATION lib)
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
ament_package()
