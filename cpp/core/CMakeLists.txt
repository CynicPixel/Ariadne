# ariadne_core: the foundation layer. It depends on Eigen and NOTHING else --
# no OpenCV, no ROS, no AWS SDK (arch §3 dependency rule). That is what makes
# replay testable from files alone and keeps CPU unit tests GPU-free.
find_package(Threads REQUIRED)

add_library(ariadne_core INTERFACE)
target_include_directories(ariadne_core INTERFACE ${CMAKE_SOURCE_DIR}/cpp)
target_link_libraries(ariadne_core INTERFACE ariadne_eigen Threads::Threads)
add_library(ariadne::core ALIAS ariadne_core)
