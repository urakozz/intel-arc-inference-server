cmake_minimum_required(VERSION 3.22)
project(b70_inference_server LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
add_compile_options(-Wall -Wextra -Werror)

find_package(PkgConfig REQUIRED)
pkg_check_modules(ZE REQUIRED IMPORTED_TARGET level-zero)

include(cmake/ocloc.cmake)

# Adds -DB70_KERNEL_DIR="<abs path>" so tests and probes can find .bin files.
function(b70_target_kernel_dir TARGET)
  target_compile_definitions(${TARGET} PRIVATE B70_KERNEL_DIR="${B70_KERNEL_DIR}")
endfunction()

enable_testing()
add_subdirectory(src/kernels)
add_subdirectory(src/l0)
add_subdirectory(tools/probe)
add_subdirectory(tests)
