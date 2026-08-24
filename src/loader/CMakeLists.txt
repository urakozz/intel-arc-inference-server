add_library(b70_loader STATIC snapshot.cc safetensors.cc quant.cc)
target_include_directories(b70_loader PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(b70_loader PUBLIC b70_l0)
