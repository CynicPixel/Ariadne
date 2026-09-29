# ariadne_slam: the from-scratch estimator (arch §10-§14). Depends on core
# only (§3: slam and perception are peers; runtime glues them). The
# OpenCV-backed pieces (PnP, triangulation) get their own target, created only
# when OpenCV is found, so the estimator math builds and tests everywhere.
add_library(ariadne_slam INTERFACE)
target_include_directories(ariadne_slam INTERFACE ${CMAKE_SOURCE_DIR}/cpp)
target_link_libraries(ariadne_slam INTERFACE ariadne::core)
add_library(ariadne::slam ALIAS ariadne_slam)

# ariadne_slam_opencv: the parts built on OpenCV 5 geometry primitives
# (solvePnPRansac, triangulatePoints). Exists only when OpenCV was found.
if(OpenCV_FOUND)
  add_library(ariadne_slam_opencv STATIC
    odometry/stereo_vo.cpp)
  target_link_libraries(ariadne_slam_opencv PUBLIC ariadne::slam opencv_core opencv_geometry)
  add_library(ariadne::slam_opencv ALIAS ariadne_slam_opencv)
endif()
