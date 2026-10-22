# ariadne_planning: the Phase 4 mission-intelligence libraries (arch §20-§22). Depends on core, plus OpenCV core
# for the 1.8 headers it builds on (occupancy_grid.hpp: the voxel walk and the tri-state constants). The 1.8
# placeholder headers stay header-only and unchanged (the ablation baseline, D45).
if(OpenCV_FOUND)
  add_library(ariadne_planning STATIC
    volumetric/tsdf_map.cpp
    volumetric/esdf.cpp
    volumetric/frontiers3d.cpp
    volumetric/traversability.cpp)
  target_include_directories(ariadne_planning PUBLIC ${CMAKE_SOURCE_DIR}/cpp)
  target_link_libraries(ariadne_planning PUBLIC ariadne::core opencv_core opencv_imgproc)
  add_library(ariadne::planning ALIAS ariadne_planning)
endif()
