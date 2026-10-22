# ariadne_planning: the Phase 4 mission-intelligence libraries (arch §20-§22; 4.5 egress: route/dstar_lite, route/egress). Depends on core, plus OpenCV core
# for the 1.8 headers it builds on (occupancy_grid.hpp: the voxel walk and the tri-state constants). The 1.8
# placeholder headers stay header-only and unchanged (the ablation baseline, D45).
if(OpenCV_FOUND)
  add_library(ariadne_planning STATIC
    volumetric/tsdf_map.cpp
    volumetric/esdf.cpp
    volumetric/frontiers3d.cpp
    volumetric/traversability.cpp
    route/dstar_lite.cpp
    route/egress.cpp
    mission/nbv.cpp
    mission/phase4_planner.cpp)
  target_include_directories(ariadne_planning PUBLIC ${CMAKE_SOURCE_DIR}/cpp)
  target_link_libraries(ariadne_planning PUBLIC ariadne::core opencv_core opencv_imgproc)
  add_library(ariadne::planning ALIAS ariadne_planning)
endif()
