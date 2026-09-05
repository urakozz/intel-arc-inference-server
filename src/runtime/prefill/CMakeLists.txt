# The prefill path's HOST code - the half that g++ compiles.
#
# `runtime::prefill::Context` itself lives in libb70_prefill.so, built by icpx
# (src/sycl/CMakeLists.txt), because its members are SYCL objects and only an
# icpx link registers the device images. Everything here is plain C++17 that
# calls into that .so across the SYCL-free `context.h` ABI: the module/kernel
# cache (which cannot live on `Context` - it would drag the b70_l0 archive into
# the .so) and the ops built out of `Context::launch`.
#
# It is a SEPARATE archive from `b70_runtime` on purpose: b70_runtime is linked
# by every decode binary and test, and none of those may acquire a dependency
# on libb70_prefill.so. A target that wants these symbols links
# `b70_prefill_host` and calls `b70_link_prefill()`, which is what supplies the
# .so and the two oneAPI rpaths.
# `engine_prefill.cc` defines `runtime::Engine::prefill` (and its two
# accessors) HERE rather than in b70_runtime, for exactly the reason above: the
# member calls into the .so, and b70_runtime must stay clean of it. A target
# that links b70_runtime alone and never calls prefill() links what it always
# linked; a target that calls it links this archive too.
#
# b70_loader is a new PUBLIC dependency: `dequant_to_bf16` takes a
# `loader::DeviceWeight` and the walk reads `loader::LoadedModel`. It is the
# same g++ static archive b70_runtime already links, so nothing about the .so
# split changes.
add_library(b70_prefill_host STATIC
  kernels.cc gdn.cc dequant.cc attn.cc step.cc engine_prefill.cc profile.cc)
target_include_directories(b70_prefill_host PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(b70_prefill_host PUBLIC b70_l0 b70_model b70_loader b70_runtime)
b70_target_kernel_dir(b70_prefill_host)
