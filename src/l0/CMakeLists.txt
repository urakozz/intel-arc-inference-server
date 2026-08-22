add_library(b70_l0 STATIC
  error.cc context.cc memory.cc queue.cc cmdlist.cc fence.cc)
target_include_directories(b70_l0 PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(b70_l0 PUBLIC PkgConfig::ZE)
