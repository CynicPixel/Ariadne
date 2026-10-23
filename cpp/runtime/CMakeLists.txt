# ariadne_runtime: replay, metrics and process-level glue. Sits above the domain
# libraries per the arch §3 dependency direction, and still carries no ROS,
# simulator or AWS types.
#
# A compiled library rather than header-only because MCAP requires exactly one
# translation unit to define MCAP_IMPLEMENTATION.
add_library(ariadne_runtime STATIC
  replay/mcap_impl.cpp
  replay/ros_cdr.cpp)
target_include_directories(ariadne_runtime PUBLIC ${CMAKE_SOURCE_DIR}/cpp)
target_link_libraries(ariadne_runtime PUBLIC ariadne::core ariadne::perception ariadne::slam ariadne_mcap)
add_library(ariadne::runtime ALIAS ariadne_runtime)

# ariadne_runtime_opencv: the runtime pieces that must see OpenCV. A separate
# target so ariadne_runtime (replay, metrics, the manifest) keeps building on a
# host without OpenCV; it exists only when OpenCV was found.
if(OpenCV_FOUND)
  add_library(ariadne_runtime_opencv STATIC
    manifest/opencv_identity.cpp)
  target_link_libraries(ariadne_runtime_opencv PUBLIC ariadne::runtime opencv_core)
  add_library(ariadne::runtime_opencv ALIAS ariadne_runtime_opencv)

  add_executable(ariadne_identity tools/ariadne_identity.cpp)
  target_link_libraries(ariadne_identity PRIVATE ariadne::runtime_opencv)
  # COMPONENT so the image installs only our binaries, not the install rules
  # that FetchContent dependencies (GoogleTest, yaml-cpp) register.
  install(TARGETS ariadne_identity RUNTIME DESTINATION bin COMPONENT ariadne)

  add_executable(ariadne_euroc_frontend tools/ariadne_euroc_frontend.cpp)
  target_link_libraries(ariadne_euroc_frontend PRIVATE
    ariadne::runtime_opencv ariadne::perception_opencv opencv_imgcodecs)
  install(TARGETS ariadne_euroc_frontend RUNTIME DESTINATION bin COMPONENT ariadne)

  add_executable(ariadne_euroc_vo tools/ariadne_euroc_vo.cpp)
  target_link_libraries(ariadne_euroc_vo PRIVATE
    ariadne::runtime_opencv ariadne::perception_opencv ariadne::slam_opencv opencv_imgcodecs)
  install(TARGETS ariadne_euroc_vo RUNTIME DESTINATION bin COMPONENT ariadne)

  # The same pipeline over a TUM RGB-D sequence: sensor depth through a virtual rig (2.6, D100).
  add_executable(ariadne_tum_vo tools/ariadne_tum_vo.cpp)
  target_link_libraries(ariadne_tum_vo PRIVATE
    ariadne::runtime_opencv ariadne::perception_opencv ariadne::slam_opencv opencv_imgcodecs)
  install(TARGETS ariadne_tum_vo RUNTIME DESTINATION bin COMPONENT ariadne)

  # The same pipeline over a simulator recording (stereo_vo_runner.hpp, D41 lens).
  add_executable(ariadne_sim_vo tools/ariadne_sim_vo.cpp)
  target_link_libraries(ariadne_sim_vo PRIVATE
    ariadne::runtime_opencv ariadne::perception_opencv ariadne::slam_opencv opencv_imgcodecs)
  install(TARGETS ariadne_sim_vo RUNTIME DESTINATION bin COMPONENT ariadne)

  # 3.3 (D102): the comms node's link state along a sim mission's GT trajectories.
  add_executable(ariadne_comms_probe tools/ariadne_comms_probe.cpp)
  target_link_libraries(ariadne_comms_probe PRIVATE
    ariadne::runtime_opencv ariadne::perception_opencv opencv_imgcodecs)
  install(TARGETS ariadne_comms_probe RUNTIME DESTINATION bin COMPONENT ariadne)

  # 2.1 retrieval evaluation dump (design PERCEPTION_2.1 §11.5): EigenPlaces vs the mean-bit placeholder on EuRoC.
  add_executable(ariadne_place_eval tools/ariadne_place_eval.cpp)
  target_link_libraries(ariadne_place_eval PRIVATE ariadne::runtime_opencv ariadne::perception_opencv opencv_imgcodecs)

  # Sim-rig reference frames for the RGB exports (PERCEPTION_2.1 §14): left_rgb as colour PNGs.
  add_executable(ariadne_frame_dump tools/ariadne_frame_dump.cpp)
  target_link_libraries(ariadne_frame_dump PRIVATE ariadne::runtime_opencv opencv_imgcodecs)

  # L5: per-image latency of the learned models at EuRoC and sim sizes (p50/p95, one caller, pinned pool).
  add_executable(ariadne_dnn_timing tools/ariadne_dnn_timing.cpp)
  target_link_libraries(ariadne_dnn_timing PRIVATE ariadne::perception_opencv opencv_imgcodecs)

  # 2.1 hard-pair evaluation: loop-candidate verification (PERCEPTION_2.1 §12.4). Offline, no runner hunk.
  add_executable(ariadne_loop_verify tools/ariadne_loop_verify.cpp)
  target_link_libraries(ariadne_loop_verify PRIVATE ariadne::runtime_opencv ariadne::perception_opencv opencv_imgcodecs)

  # 1.6 placeholder collaboration over a sim recording (design COLLAB_1.6).
  add_executable(ariadne_sim_collab tools/ariadne_sim_collab.cpp)
  target_link_libraries(ariadne_sim_collab PRIVATE
    ariadne::runtime_opencv ariadne::perception_opencv ariadne::slam_opencv opencv_imgcodecs opencv_features)
  install(TARGETS ariadne_sim_collab RUNTIME DESTINATION bin COMPONENT ariadne)

  # 1.9 the §30.2 report gate (design HARNESS_1.9): registry dump + MissingMetrics/UnknownMetrics.
  add_executable(ariadne_report_check tools/ariadne_report_check.cpp)
  target_link_libraries(ariadne_report_check PRIVATE ariadne::runtime_opencv)
  install(TARGETS ariadne_report_check RUNTIME DESTINATION bin COMPONENT ariadne)

  # 1.8 placeholder mission layer over a 1.7 spool (design MISSION_1.8).
  add_executable(ariadne_mission tools/ariadne_mission.cpp)
  target_link_libraries(ariadne_mission PRIVATE ariadne::runtime_opencv opencv_imgcodecs opencv_imgproc)
  install(TARGETS ariadne_mission RUNTIME DESTINATION bin COMPONENT ariadne)

  # The 1.1 acceptance test: rig recovery from a sim recording (design SIM_1.1-1.3 §6).
  add_executable(ariadne_sim_calib tools/ariadne_sim_calib.cpp)
  target_link_libraries(ariadne_sim_calib PRIVATE
    ariadne::runtime_opencv opencv_core opencv_imgproc opencv_imgcodecs opencv_objdetect opencv_calib opencv_geometry
    yaml-cpp::yaml-cpp)
  install(TARGETS ariadne_sim_calib RUNTIME DESTINATION bin COMPONENT ariadne)

  # ChArUco boards for the generated scenes' entrance rooms (worldgen design §3/§6).
  add_executable(ariadne_charuco_assets tools/ariadne_charuco_assets.cpp)
  target_link_libraries(ariadne_charuco_assets PRIVATE opencv_core opencv_objdetect opencv_imgcodecs)
  install(TARGETS ariadne_charuco_assets RUNTIME DESTINATION bin COMPONENT ariadne)
endif()
