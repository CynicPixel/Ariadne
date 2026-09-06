find_package(Threads REQUIRED)

# Generate the wire header from the schema at build time. Generated code is
# never committed, so the .fbs cannot drift from what the code compiles against.
set(ARIADNE_WIRE_FBS ${CMAKE_CURRENT_SOURCE_DIR}/serialization/wire.fbs)
set(ARIADNE_WIRE_GEN_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
set(ARIADNE_WIRE_HEADER ${ARIADNE_WIRE_GEN_DIR}/wire_generated.h)

add_custom_command(
  OUTPUT ${ARIADNE_WIRE_HEADER}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${ARIADNE_WIRE_GEN_DIR}
  COMMAND flatc --cpp --scoped-enums -o ${ARIADNE_WIRE_GEN_DIR} ${ARIADNE_WIRE_FBS}
  DEPENDS ${ARIADNE_WIRE_FBS} flatc
  COMMENT "flatc: generating wire_generated.h from wire.fbs"
  VERBATIM)

add_custom_target(ariadne_wire_gen DEPENDS ${ARIADNE_WIRE_HEADER})

# ariadne_core: the foundation layer. Depends on Eigen, FlatBuffers and zstd.
# NOT on OpenCV, ROS or the AWS SDK (arch §3 dependency rule). That is what
# makes replay testable from files alone and CPU unit tests GPU-free.
add_library(ariadne_core INTERFACE)
target_include_directories(ariadne_core INTERFACE
  ${CMAKE_SOURCE_DIR}/cpp
  ${ARIADNE_WIRE_GEN_DIR})
target_link_libraries(ariadne_core INTERFACE
  ariadne_eigen
  flatbuffers
  libzstd_static
  Threads::Threads)
add_dependencies(ariadne_core ariadne_wire_gen)
add_library(ariadne::core ALIAS ariadne_core)
