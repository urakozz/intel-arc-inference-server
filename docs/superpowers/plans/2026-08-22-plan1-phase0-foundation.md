# Plan 1 of 3 - Foundation + Phase 0 probes + the GEMV

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A native build on the box, Level Zero RAII wrappers, the three
phase-0 probes with their numbers in the docs, and the production int4 GEMV
kernel chosen by measurement - everything spec 1 Sections 4, 5, 9.2 and the
`l0/` and `kernels/` parts of its layout require.

**Architecture:** Host code is C++17 under `g++`, linking only `ze_loader`.
Device code is OpenCL C compiled at build time by `ocloc` for `bmg-g31` into
native binaries that the runtime loads with `zeModuleCreate`. Everything
builds and runs on the box (`box`) via `tools/box.sh`, which rsyncs
the tree and invokes cmake/ctest over ssh. No Docker, no `icpx`.

**Tech Stack:** C++17, CMake ≥ 3.22 (box has 4.2.3), `g++` 15.2, Level Zero
loader 1.32 (`pkg-config level-zero`), `ocloc` 26.27, OpenCL C 3.0 with
`cl_intel_subgroups`, Python 3 for one probe script.

**Spec:** `docs/superpowers/specs/2026-08-22-phase0-decode-core-design.md`
(Sections 2, 4, 5, 6.3, 9.2, 11, 13). Plans 2 (loader + model + oracle) and
3 (remaining kernels + runtime + CLI + golden tests) are written after this
plan's probe results exist, because they decide the canonical layout and `S`.

## Global Constraints

- C++17, `-Wall -Wextra -Werror`, `g++`; host links `ze_loader` only.
- OpenCL C 3.0; every kernel `__attribute__((intel_reqd_sub_group_size(16)))`;
  fp32 accumulation; bf16 carried as `ushort`, widened with `<< 16`.
- Determinism: no floating-point atomics, no work-stealing, no
  data-dependent work-group counts; reductions are fixed-tree or two-stage.
- `ocloc compile -device bmg-g31 -output <name> -output_no_suffix -out_dir <dir>`
  produces `<dir>/<name>.bin` (verified on the box 2026-08-22).
- The box builds with `JOBS=8` (doc 09: 8 jobs is the fastest setting; 16+
  swaps). Builds run on the box only; the Mac never compiles this.
- The checkpoint's GPTQ v1 convention: `qzeros` words are `0x77777777`,
  `g_idx` is the identity; dequant is `w = (q − 8) · scale`, `q` = unsigned nibble.
- Every kernel lands with a section in `docs/12-kernels.md` (what it
  computes, lane/work-group assignment, alternatives rejected, measured
  number). A kernel task is not done without it.
- Measured numbers go into the docs in the task that produced them, with the
  command that produced them.
- Commit after every task; messages `feat:`/`test:`/`docs:`/`build:`.

---

### Task 1: Build skeleton, `ocloc` rule, box sync script

**Files:**
- Modify: `CMakeLists.txt` (replace the hello-world file entirely)
- Create: `cmake/ocloc.cmake`
- Create: `src/kernels/CMakeLists.txt`, `src/kernels/noop.cl`
- Create: `tools/box.sh`
- Create: `tests/CMakeLists.txt`, `tests/check.h`
- Delete: `main.cpp`

**Interfaces:**
- Produces: CMake function `add_ocloc_kernel(<name> SOURCE <file.cl> [DEFINES k=v ...] [DEPENDS <files>])`
  → builds `${B70_KERNEL_DIR}/<name>.bin`; CMake cache var `B70_KERNEL_DIR`
  (`${CMAKE_BINARY_DIR}/kernels`); compile definition
  `B70_KERNEL_DIR="<abs path>"` on every test/probe target via
  `b70_target_kernel_dir(<target>)`; `tools/box.sh {sync|build|test [regex]|run <cmd>}`;
  `tests/check.h` macros `CHECK(cond)`, `CHECK_EQ(a,b)`, `CHECK_NEAR(a,b,tol)`.

- [ ] **Step 1: Replace `CMakeLists.txt`**

```cmake
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
add_subdirectory(tests)
```

- [ ] **Step 2: Create `cmake/ocloc.cmake`**

```cmake
# ocloc AOT rule. One call = one device binary for bmg-g31.
find_program(OCLOC_EXECUTABLE ocloc REQUIRED)
set(B70_KERNEL_DIR "${CMAKE_BINARY_DIR}/kernels" CACHE INTERNAL "directory of compiled device binaries")
set(B70_OCLOC_DEVICE "bmg-g31" CACHE STRING "ocloc -device target (B70 = Battlemage G31)")
file(MAKE_DIRECTORY "${B70_KERNEL_DIR}")

# add_ocloc_kernel(<name> SOURCE <file.cl> [DEFINES A=1 B=2 ...] [DEPENDS <files>])
# Produces ${B70_KERNEL_DIR}/<name>.bin (ZE_MODULE_FORMAT_NATIVE). <name> is the
# variant name the runtime asks for, e.g. gemv_M1_K5120_N5120_S1_L0.
function(add_ocloc_kernel NAME)
  cmake_parse_arguments(K "" "SOURCE" "DEFINES;DEPENDS" ${ARGN})
  if(NOT K_SOURCE)
    message(FATAL_ERROR "add_ocloc_kernel(${NAME}): SOURCE is required")
  endif()
  set(opts "-cl-std=CL3.0")
  foreach(d IN LISTS K_DEFINES)
    string(APPEND opts " -D${d}")
  endforeach()
  set(out "${B70_KERNEL_DIR}/${NAME}.bin")
  add_custom_command(
    OUTPUT "${out}"
    COMMAND "${OCLOC_EXECUTABLE}" compile
            -file "${K_SOURCE}" -device "${B70_OCLOC_DEVICE}"
            -output "${NAME}" -output_no_suffix -out_dir "${B70_KERNEL_DIR}"
            -options "${opts}"
    DEPENDS "${K_SOURCE}" ${K_DEPENDS}
    COMMENT "ocloc ${NAME}"
    VERBATIM)
  add_custom_target("kernel_${NAME}" ALL DEPENDS "${out}")
endfunction()
```

- [ ] **Step 3: Create the first kernel and its CMake file**

`src/kernels/noop.cl`:
```c
// noop: the smallest kernel. Used by the L0 smoke test and by probe_replay,
// where ~700 of these in one command list measure per-kernel fixed cost.
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void noop(__global uint* restrict out) {
  if (get_global_id(0) == 0) out[0] = 42u;
}
```

`src/kernels/CMakeLists.txt`:
```cmake
add_ocloc_kernel(noop SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/noop.cl)
```

- [ ] **Step 4: Create `tests/CMakeLists.txt` and `tests/check.h`**

`tests/CMakeLists.txt` (tests are added by later tasks):
```cmake
# Each test is add_executable + b70_target_kernel_dir + add_test. No framework:
# tests/check.h exits non-zero on the first failed CHECK.
```

`tests/check.h`:
```cpp
#pragma once
#include <cmath>
#include <cstdio>
#include <cstdlib>

#define CHECK(cond)                                                             \
  do {                                                                          \
    if (!(cond)) {                                                              \
      std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

#define CHECK_EQ(a, b)                                                          \
  do {                                                                          \
    auto _a = (a); auto _b = (b);                                               \
    if (!(_a == _b)) {                                                          \
      std::fprintf(stderr, "%s:%d: CHECK_EQ failed: %s != %s\n", __FILE__,      \
                   __LINE__, #a, #b);                                           \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

#define CHECK_NEAR(a, b, tol)                                                   \
  do {                                                                          \
    double _a = (a); double _b = (b); double _t = (tol);                        \
    if (!(std::fabs(_a - _b) <= _t)) {                                          \
      std::fprintf(stderr, "%s:%d: CHECK_NEAR failed: %s=%g %s=%g tol=%g\n",    \
                   __FILE__, __LINE__, #a, _a, #b, _b, _t);                     \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)
```

- [ ] **Step 5: Create `tools/box.sh`** and `chmod +x` it

```bash
#!/usr/bin/env bash
# Sync this tree to the box and build / test there. The Mac never compiles.
#   tools/box.sh sync            rsync the tree
#   tools/box.sh build           sync + cmake configure + build
#   tools/box.sh test [regex]    sync + build + ctest (optionally -R regex)
#   tools/box.sh run <cmd...>    run a shell command in the remote tree
# Env: BOX (ssh target), REMOTE_DIR (relative to $HOME on the box), JOBS.
set -euo pipefail
BOX="${BOX:-user@box}"
REMOTE_DIR="${REMOTE_DIR:-b70-inference-server}"
JOBS="${JOBS:-8}"
cd "$(dirname "$0")/.."

sync_tree() {
  rsync -az --delete \
    --exclude build --exclude .git --exclude 'cmake-build-*' --exclude .idea \
    ./ "$BOX:$REMOTE_DIR/"
}
configure_and_build() {
  ssh "$BOX" "cd '$REMOTE_DIR' && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release > /dev/null && cmake --build build -j$JOBS"
}

case "${1:-}" in
  sync)  sync_tree ;;
  build) sync_tree; configure_and_build ;;
  test)  sync_tree; configure_and_build
         ssh "$BOX" "cd '$REMOTE_DIR' && ctest --test-dir build --output-on-failure ${2:+-R $2}" ;;
  run)   shift; ssh "$BOX" "cd '$REMOTE_DIR' && $*" ;;
  *)     echo "usage: $0 sync|build|test [regex]|run <cmd...>" >&2; exit 2 ;;
esac
```

- [ ] **Step 6: Delete `main.cpp`** (`git rm main.cpp`).

- [ ] **Step 7: Build on the box and verify the kernel binary exists**

Run: `tools/box.sh build && tools/box.sh run ls -la build/kernels/`
Expected: build succeeds; listing shows `noop.bin` (and a `noop.spv` - ignore it).

Run: `tools/box.sh test`
Expected: `No tests were found!!!` from ctest, exit 0 (ctest returns 0 with zero tests only when `--output-on-failure` is used without `--no-tests=error`; if it returns non-zero, that is fine for this task - the assertion is that configure and build succeed).

- [ ] **Step 8: Commit**

```bash
git add CMakeLists.txt cmake/ocloc.cmake src/kernels tools/box.sh tests/CMakeLists.txt tests/check.h
git rm -q main.cpp
git commit -m "build: cmake skeleton, ocloc AOT rule for bmg-g31, box sync script"
```

---

### Task 2: Level Zero wrappers - context, memory, queue, command lists, fence

**Files:**
- Create: `src/l0/error.h`, `src/l0/error.cc`
- Create: `src/l0/context.h`, `src/l0/context.cc`
- Create: `src/l0/memory.h`, `src/l0/memory.cc`
- Create: `src/l0/queue.h`, `src/l0/queue.cc`
- Create: `src/l0/cmdlist.h`, `src/l0/cmdlist.cc`
- Create: `src/l0/fence.h`, `src/l0/fence.cc`
- Create: `src/l0/CMakeLists.txt`
- Modify: `CMakeLists.txt` (add `add_subdirectory(src/l0)` before `add_subdirectory(tests)`)
- Create: `tests/l0/copy_test.cc`
- Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Produces (namespace `l0`):
  - `const char* result_name(ze_result_t)`; `struct Error : std::runtime_error { ze_result_t result; }`;
    macro `ZE_CHECK(call)` - throws `l0::Error`; aborts the process on `ZE_RESULT_ERROR_DEVICE_LOST`.
  - `class Context { explicit Context(uint32_t device_index = 0); ze_context_handle_t handle() const; ze_device_handle_t device() const; const ze_device_properties_t& props() const; const ze_device_compute_properties_t& compute() const; std::string name() const; uint32_t eu_count() const; }`
  - `enum class MemKind { Device, Host, Shared }; class Mem { Mem(Context&, MemKind, size_t bytes, size_t align = 64); void* ptr() const; size_t size() const; MemKind kind() const; template<class T> T* as() const; }`
  - `class Queue { Queue(Context&, uint32_t ordinal = 0); ze_command_queue_handle_t handle() const; void execute(CmdList&, Fence* fence = nullptr); void synchronize(); }`
  - `class CmdList { static CmdList immediate(Context&, uint32_t ordinal = 0); static CmdList regular(Context&, uint32_t ordinal = 0); void copy(void* dst, const void* src, size_t bytes); void fill(void* dst, uint32_t pattern, size_t bytes); void launch(Kernel&, uint32_t gx, uint32_t gy = 1, uint32_t gz = 1); void close(); void reset(); bool is_immediate() const; ze_command_list_handle_t handle() const; }`
    (`launch` is declared here and implemented in Task 3 together with `Kernel`.)
  - `class Fence { explicit Fence(Queue&); ze_fence_handle_t handle() const; void wait(); /* host-sync then reset */ }`
- All classes are move-only; destructors release handles; no class knows about models or kernels.

- [ ] **Step 1: Write the failing test `tests/l0/copy_test.cc`**

```cpp
// Round-trips 1 MiB host -> device -> host through the immediate list.
#include <cstdint>
#include <cstring>
#include <vector>
#include "check.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/memory.h"

int main() {
  l0::Context ctx(0);
  std::printf("device: %s, EUs: %u\n", ctx.name().c_str(), ctx.eu_count());
  CHECK(ctx.eu_count() > 0);

  const size_t n = 1u << 20;
  std::vector<uint8_t> src(n), dst(n, 0);
  for (size_t i = 0; i < n; ++i) src[i] = static_cast<uint8_t>(i * 7 + 3);

  l0::Mem dev(ctx, l0::MemKind::Device, n);
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  imm.copy(dev.ptr(), src.data(), n);
  imm.copy(dst.data(), dev.ptr(), n);
  CHECK_EQ(std::memcmp(src.data(), dst.data(), n), 0);

  l0::Mem shared(ctx, l0::MemKind::Shared, 64);
  imm.fill(shared.ptr(), 0xA5A5A5A5u, 64);
  CHECK_EQ(shared.as<uint32_t>()[0], 0xA5A5A5A5u);
  CHECK_EQ(shared.as<uint32_t>()[15], 0xA5A5A5A5u);
  std::puts("copy_test OK");
  return 0;
}
```

Add to `tests/CMakeLists.txt`:
```cmake
add_executable(copy_test l0/copy_test.cc)
target_include_directories(copy_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(copy_test PRIVATE b70_l0)
add_test(NAME copy_test COMMAND copy_test)
```

- [ ] **Step 2: Run it to verify it fails to build**

Run: `tools/box.sh test copy_test`
Expected: configure fails - `b70_l0` target does not exist / headers missing.

- [ ] **Step 3: Write `src/l0/error.h` and `error.cc`**

`error.h`:
```cpp
#pragma once
#include <level_zero/ze_api.h>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace l0 {
const char* result_name(ze_result_t r);
struct Error : std::runtime_error {
  ze_result_t result;
  Error(const char* call, ze_result_t r)
      : std::runtime_error(std::string(call) + " -> " + result_name(r)), result(r) {}
};
}  // namespace l0

// Every Level Zero call goes through this. DEVICE_LOST means the driver is
// wedged (doc 06 has the history); nothing sensible can follow, so abort.
#define ZE_CHECK(call)                                                        \
  do {                                                                        \
    ze_result_t _r = (call);                                                  \
    if (_r != ZE_RESULT_SUCCESS) {                                            \
      if (_r == ZE_RESULT_ERROR_DEVICE_LOST) {                                \
        std::fprintf(stderr, "FATAL: ZE_RESULT_ERROR_DEVICE_LOST in %s\n", #call); \
        std::abort();                                                         \
      }                                                                       \
      throw ::l0::Error(#call, _r);                                           \
    }                                                                         \
  } while (0)
```

`error.cc`:
```cpp
#include "l0/error.h"

namespace l0 {
const char* result_name(ze_result_t r) {
  switch (r) {
#define C(x) case x: return #x;
    C(ZE_RESULT_SUCCESS) C(ZE_RESULT_NOT_READY) C(ZE_RESULT_ERROR_DEVICE_LOST)
    C(ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY) C(ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY)
    C(ZE_RESULT_ERROR_MODULE_BUILD_FAILURE) C(ZE_RESULT_ERROR_MODULE_LINK_FAILURE)
    C(ZE_RESULT_ERROR_UNINITIALIZED) C(ZE_RESULT_ERROR_UNSUPPORTED_VERSION)
    C(ZE_RESULT_ERROR_UNSUPPORTED_FEATURE) C(ZE_RESULT_ERROR_INVALID_ARGUMENT)
    C(ZE_RESULT_ERROR_INVALID_NULL_HANDLE) C(ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE)
    C(ZE_RESULT_ERROR_INVALID_NULL_POINTER) C(ZE_RESULT_ERROR_INVALID_SIZE)
    C(ZE_RESULT_ERROR_UNSUPPORTED_SIZE) C(ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT)
    C(ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT) C(ZE_RESULT_ERROR_INVALID_ENUMERATION)
    C(ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION) C(ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT)
    C(ZE_RESULT_ERROR_INVALID_NATIVE_BINARY) C(ZE_RESULT_ERROR_INVALID_GLOBAL_NAME)
    C(ZE_RESULT_ERROR_INVALID_KERNEL_NAME) C(ZE_RESULT_ERROR_INVALID_FUNCTION_NAME)
    C(ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION) C(ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION)
    C(ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX) C(ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE)
    C(ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE) C(ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED)
    C(ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE) C(ZE_RESULT_ERROR_OVERLAPPING_REGIONS)
    C(ZE_RESULT_ERROR_UNKNOWN)
#undef C
    default: {
      static thread_local char buf[32];
      std::snprintf(buf, sizeof buf, "ze_result_t(0x%x)", static_cast<unsigned>(r));
      return buf;
    }
  }
}
}  // namespace l0
```

- [ ] **Step 4: Write `src/l0/context.h` and `context.cc`**

`context.h`:
```cpp
#pragma once
#include <level_zero/ze_api.h>
#include <cstdint>
#include <string>

namespace l0 {
// Driver init + one GPU device + one context. device_index counts GPU
// devices of the first driver (ONEAPI_DEVICE_SELECTOR is not consulted).
class Context {
 public:
  explicit Context(uint32_t device_index = 0);
  ~Context();
  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;

  ze_context_handle_t handle() const { return ctx_; }
  ze_driver_handle_t driver() const { return driver_; }
  ze_device_handle_t device() const { return dev_; }
  const ze_device_properties_t& props() const { return props_; }
  const ze_device_compute_properties_t& compute() const { return compute_; }
  std::string name() const { return props_.name; }
  uint32_t eu_count() const {
    return props_.numSlices * props_.numSubslicesPerSlice * props_.numEUsPerSubslice;
  }

 private:
  ze_driver_handle_t driver_ = nullptr;
  ze_device_handle_t dev_ = nullptr;
  ze_context_handle_t ctx_ = nullptr;
  ze_device_properties_t props_{};
  ze_device_compute_properties_t compute_{};
};
}  // namespace l0
```

`context.cc`:
```cpp
#include "l0/context.h"
#include <vector>
#include "l0/error.h"

namespace l0 {
Context::Context(uint32_t device_index) {
  ZE_CHECK(zeInit(ZE_INIT_FLAG_GPU_ONLY));
  uint32_t n_drivers = 0;
  ZE_CHECK(zeDriverGet(&n_drivers, nullptr));
  if (n_drivers == 0) throw std::runtime_error("no Level Zero driver");
  std::vector<ze_driver_handle_t> drivers(n_drivers);
  ZE_CHECK(zeDriverGet(&n_drivers, drivers.data()));
  driver_ = drivers[0];

  uint32_t n_dev = 0;
  ZE_CHECK(zeDeviceGet(driver_, &n_dev, nullptr));
  std::vector<ze_device_handle_t> devs(n_dev);
  ZE_CHECK(zeDeviceGet(driver_, &n_dev, devs.data()));
  std::vector<ze_device_handle_t> gpus;
  for (auto d : devs) {
    ze_device_properties_t p{};
    p.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
    ZE_CHECK(zeDeviceGetProperties(d, &p));
    if (p.type == ZE_DEVICE_TYPE_GPU) gpus.push_back(d);
  }
  if (device_index >= gpus.size())
    throw std::runtime_error("GPU index " + std::to_string(device_index) + " out of range (" +
                             std::to_string(gpus.size()) + " GPUs)");
  dev_ = gpus[device_index];
  props_.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
  ZE_CHECK(zeDeviceGetProperties(dev_, &props_));
  compute_.stype = ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES;
  ZE_CHECK(zeDeviceGetComputeProperties(dev_, &compute_));

  ze_context_desc_t cd{};
  cd.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
  ZE_CHECK(zeContextCreate(driver_, &cd, &ctx_));
}

Context::~Context() {
  if (ctx_) zeContextDestroy(ctx_);
}
}  // namespace l0
```

- [ ] **Step 5: Write `src/l0/memory.h` and `memory.cc`**

`memory.h`:
```cpp
#pragma once
#include <cstddef>
#include "l0/context.h"

namespace l0 {
enum class MemKind { Device, Host, Shared };

// One allocation. Device memory is not host-accessible; Host memory is
// host-resident and device-visible; Shared migrates (used for the control block).
class Mem {
 public:
  Mem(Context& ctx, MemKind kind, size_t bytes, size_t align = 64);
  ~Mem();
  Mem(Mem&& o) noexcept;
  Mem& operator=(Mem&&) = delete;
  Mem(const Mem&) = delete;

  void* ptr() const { return ptr_; }
  size_t size() const { return bytes_; }
  MemKind kind() const { return kind_; }
  template <class T> T* as() const { return static_cast<T*>(ptr_); }

 private:
  Context* ctx_;
  MemKind kind_;
  size_t bytes_;
  void* ptr_ = nullptr;
};
}  // namespace l0
```

`memory.cc`:
```cpp
#include "l0/memory.h"
#include "l0/error.h"

namespace l0 {
Mem::Mem(Context& ctx, MemKind kind, size_t bytes, size_t align)
    : ctx_(&ctx), kind_(kind), bytes_(bytes) {
  ze_device_mem_alloc_desc_t dd{};
  dd.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
  ze_host_mem_alloc_desc_t hd{};
  hd.stype = ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
  // Allocations above the device's maxMemAllocSize need the relaxed-limits
  // extension; the 8 GB bandwidth probe and the 2.5 GB lm_head both do.
  ze_relaxed_allocation_limits_exp_desc_t relaxed{};
  relaxed.stype = ZE_STRUCTURE_TYPE_RELAXED_ALLOCATION_LIMITS_EXP_DESC;
  relaxed.flags = ZE_RELAXED_ALLOCATION_LIMITS_EXP_FLAG_MAX_SIZE;
  if (bytes > ctx.props().maxMemAllocSize) dd.pNext = &relaxed;

  switch (kind) {
    case MemKind::Device:
      ZE_CHECK(zeMemAllocDevice(ctx.handle(), &dd, bytes, align, ctx.device(), &ptr_));
      break;
    case MemKind::Host:
      ZE_CHECK(zeMemAllocHost(ctx.handle(), &hd, bytes, align, &ptr_));
      break;
    case MemKind::Shared:
      ZE_CHECK(zeMemAllocShared(ctx.handle(), &dd, &hd, bytes, align, ctx.device(), &ptr_));
      break;
  }
}

Mem::~Mem() {
  if (ptr_) zeMemFree(ctx_->handle(), ptr_);
}

Mem::Mem(Mem&& o) noexcept : ctx_(o.ctx_), kind_(o.kind_), bytes_(o.bytes_), ptr_(o.ptr_) {
  o.ptr_ = nullptr;
}
}  // namespace l0
```

- [ ] **Step 6: Write `src/l0/queue.h`, `queue.cc`, `fence.h`, `fence.cc`**

`queue.h`:
```cpp
#pragma once
#include "l0/context.h"

namespace l0 {
class CmdList;
class Fence;

// One asynchronous compute queue on the device's compute ordinal.
class Queue {
 public:
  explicit Queue(Context& ctx, uint32_t ordinal = 0);
  ~Queue();
  Queue(const Queue&) = delete;
  Queue& operator=(const Queue&) = delete;

  ze_command_queue_handle_t handle() const { return q_; }
  Context& context() const { return *ctx_; }
  // Submits a closed regular command list. fence may be null.
  void execute(CmdList& list, Fence* fence = nullptr);
  void synchronize();

 private:
  Context* ctx_;
  ze_command_queue_handle_t q_ = nullptr;
};
}  // namespace l0
```

`queue.cc`:
```cpp
#include "l0/queue.h"
#include "l0/cmdlist.h"
#include "l0/error.h"
#include "l0/fence.h"

namespace l0 {
Queue::Queue(Context& ctx, uint32_t ordinal) : ctx_(&ctx) {
  ze_command_queue_desc_t d{};
  d.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
  d.ordinal = ordinal;
  d.index = 0;
  d.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
  d.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
  ZE_CHECK(zeCommandQueueCreate(ctx.handle(), ctx.device(), &d, &q_));
}
Queue::~Queue() {
  if (q_) zeCommandQueueDestroy(q_);
}
void Queue::execute(CmdList& list, Fence* fence) {
  ze_command_list_handle_t h = list.handle();
  ZE_CHECK(zeCommandQueueExecuteCommandLists(q_, 1, &h, fence ? fence->handle() : nullptr));
}
void Queue::synchronize() { ZE_CHECK(zeCommandQueueSynchronize(q_, UINT64_MAX)); }
}  // namespace l0
```

`fence.h`:
```cpp
#pragma once
#include "l0/queue.h"

namespace l0 {
// Host-side completion signal for one queue submission. wait() blocks, then
// resets, so the same fence serves every replay.
class Fence {
 public:
  explicit Fence(Queue& q);
  ~Fence();
  Fence(const Fence&) = delete;
  Fence& operator=(const Fence&) = delete;
  ze_fence_handle_t handle() const { return f_; }
  void wait();

 private:
  ze_fence_handle_t f_ = nullptr;
};
}  // namespace l0
```

`fence.cc`:
```cpp
#include "l0/fence.h"
#include "l0/error.h"

namespace l0 {
Fence::Fence(Queue& q) {
  ze_fence_desc_t d{};
  d.stype = ZE_STRUCTURE_TYPE_FENCE_DESC;
  ZE_CHECK(zeFenceCreate(q.handle(), &d, &f_));
}
Fence::~Fence() {
  if (f_) zeFenceDestroy(f_);
}
void Fence::wait() {
  ZE_CHECK(zeFenceHostSynchronize(f_, UINT64_MAX));
  ZE_CHECK(zeFenceReset(f_));
}
}  // namespace l0
```

- [ ] **Step 7: Write `src/l0/cmdlist.h` and `cmdlist.cc`** (`launch` is declared now, defined in Task 3)

`cmdlist.h`:
```cpp
#pragma once
#include <cstddef>
#include <cstdint>
#include "l0/context.h"

namespace l0 {
class Kernel;

// Two flavours. immediate(): synchronous - every append executes and completes
// before returning; used for uploads and tests. regular(): in-order, recorded
// once, closed, executed many times by Queue::execute - the decode list.
class CmdList {
 public:
  static CmdList immediate(Context& ctx, uint32_t ordinal = 0);
  static CmdList regular(Context& ctx, uint32_t ordinal = 0);
  ~CmdList();
  CmdList(CmdList&& o) noexcept;
  CmdList(const CmdList&) = delete;
  CmdList& operator=(const CmdList&) = delete;

  void copy(void* dst, const void* src, size_t bytes);
  void fill(void* dst, uint32_t pattern, size_t bytes);
  // Appends a launch with gx*gy*gz work-groups; the kernel's group size must
  // already be set (Kernel::group_size).
  void launch(Kernel& k, uint32_t gx, uint32_t gy = 1, uint32_t gz = 1);
  void close();
  void reset();
  bool is_immediate() const { return immediate_; }
  ze_command_list_handle_t handle() const { return l_; }

 private:
  CmdList(ze_command_list_handle_t l, bool immediate) : l_(l), immediate_(immediate) {}
  ze_command_list_handle_t l_ = nullptr;
  bool immediate_;
};
}  // namespace l0
```

`cmdlist.cc` (without `launch`, which Task 3 adds):
```cpp
#include "l0/cmdlist.h"
#include "l0/error.h"

namespace l0 {
CmdList CmdList::immediate(Context& ctx, uint32_t ordinal) {
  ze_command_queue_desc_t qd{};
  qd.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
  qd.ordinal = ordinal;
  qd.mode = ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS;
  qd.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
  ze_command_list_handle_t l = nullptr;
  ZE_CHECK(zeCommandListCreateImmediate(ctx.handle(), ctx.device(), &qd, &l));
  return CmdList(l, true);
}

CmdList CmdList::regular(Context& ctx, uint32_t ordinal) {
  ze_command_list_desc_t d{};
  d.stype = ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC;
  d.commandQueueGroupOrdinal = ordinal;
  d.flags = ZE_COMMAND_LIST_FLAG_IN_ORDER;
  ze_command_list_handle_t l = nullptr;
  ZE_CHECK(zeCommandListCreate(ctx.handle(), ctx.device(), &d, &l));
  return CmdList(l, false);
}

CmdList::~CmdList() {
  if (l_) zeCommandListDestroy(l_);
}
CmdList::CmdList(CmdList&& o) noexcept : l_(o.l_), immediate_(o.immediate_) { o.l_ = nullptr; }

void CmdList::copy(void* dst, const void* src, size_t bytes) {
  ZE_CHECK(zeCommandListAppendMemoryCopy(l_, dst, src, bytes, nullptr, 0, nullptr));
}
void CmdList::fill(void* dst, uint32_t pattern, size_t bytes) {
  ZE_CHECK(zeCommandListAppendMemoryFill(l_, dst, &pattern, sizeof pattern, bytes, nullptr, 0, nullptr));
}
void CmdList::close() { ZE_CHECK(zeCommandListClose(l_)); }
void CmdList::reset() { ZE_CHECK(zeCommandListReset(l_)); }
}  // namespace l0
```

- [ ] **Step 8: `src/l0/CMakeLists.txt` and hook into the root**

```cmake
add_library(b70_l0 STATIC
  error.cc context.cc memory.cc queue.cc cmdlist.cc fence.cc)
target_include_directories(b70_l0 PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(b70_l0 PUBLIC PkgConfig::ZE)
```

Root `CMakeLists.txt`: insert `add_subdirectory(src/l0)` after `add_subdirectory(src/kernels)`.

- [ ] **Step 9: Build and run the test**

Run: `tools/box.sh test copy_test`
Expected: prints the device name (`Intel(R) Arc(TM) Pro B70 Graphics`), `EUs: 256`, `copy_test OK`; ctest `100% tests passed`.

If `zeMemAllocShared` is rejected with `ZE_RESULT_ERROR_UNSUPPORTED_FEATURE`, add `ZE_FLAT_HIERARCHY=FLAT` to the ssh environment in `tools/box.sh` (the reference stack sets it, doc 10) and retry before changing code.

- [ ] **Step 10: Commit**

```bash
git add src/l0 tests/l0/copy_test.cc tests/CMakeLists.txt CMakeLists.txt
git commit -m "feat(l0): context, memory, queue, command lists, fence wrappers + copy test"
```

---

### Task 3: Module and kernel wrappers + first replayed launch

**Files:**
- Create: `src/l0/module.h`, `src/l0/module.cc`, `src/l0/kernel.h`, `src/l0/kernel.cc`
- Create: `src/kernels/kernels.h`
- Modify: `src/l0/cmdlist.cc` (add `launch`), `src/l0/CMakeLists.txt`
- Create: `tests/l0/launch_test.cc`; Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Produces (namespace `l0`):
  - `class Module { Module(Context&, const std::string& bin_path); Kernel kernel(const char* name); ze_module_handle_t handle() const; }` - loads a `ZE_MODULE_FORMAT_NATIVE` file; on `MODULE_BUILD_FAILURE` the exception message includes the driver build log.
  - `class Kernel { template<class T> void arg(uint32_t index, const T& value); void arg_ptr(uint32_t index, const void* p); void group_size(uint32_t x, uint32_t y = 1, uint32_t z = 1); ze_kernel_handle_t handle() const; }`
  - `void CmdList::launch(Kernel&, gx, gy, gz)` → `zeCommandListAppendLaunchKernel` with group counts.
- Produces (namespace `kernels`): `std::string path(const std::string& variant)` → `std::string(B70_KERNEL_DIR) + "/" + variant + ".bin"`.

- [ ] **Step 1: Write the failing test `tests/l0/launch_test.cc`**

```cpp
// Loads noop.bin, records one launch in a regular list, executes it twice
// (replay), and checks the write happened. This is the smallest replay.
#include <cstdint>
#include "check.h"
#include "kernels/kernels.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "l0/queue.h"

int main() {
  l0::Context ctx(0);
  l0::Queue q(ctx);
  l0::Fence fence(q);
  l0::Module mod(ctx, kernels::path("noop"));
  l0::Kernel k = mod.kernel("noop");

  l0::Mem out(ctx, l0::MemKind::Shared, 64);
  out.as<uint32_t>()[0] = 0;
  k.arg_ptr(0, out.ptr());
  k.group_size(16);

  l0::CmdList list = l0::CmdList::regular(ctx);
  list.launch(k, 1);
  list.close();

  for (int replay = 0; replay < 2; ++replay) {
    out.as<uint32_t>()[0] = 0;
    q.execute(list, &fence);
    fence.wait();
    CHECK_EQ(out.as<uint32_t>()[0], 42u);
  }
  std::puts("launch_test OK");
  return 0;
}
```

`tests/CMakeLists.txt` add:
```cmake
add_executable(launch_test l0/launch_test.cc)
target_include_directories(launch_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(launch_test PRIVATE b70_l0)
b70_target_kernel_dir(launch_test)
add_dependencies(launch_test kernel_noop)
add_test(NAME launch_test COMMAND launch_test)
```

- [ ] **Step 2: Run to verify it fails to build**

Run: `tools/box.sh test launch_test` - Expected: missing headers `l0/module.h`, `kernels/kernels.h`.

- [ ] **Step 3: Write `src/kernels/kernels.h`**

```cpp
#pragma once
#include <string>

#ifndef B70_KERNEL_DIR
#error "B70_KERNEL_DIR must be defined (use b70_target_kernel_dir in CMake)"
#endif

namespace kernels {
// Absolute path of a compiled device binary by variant name.
inline std::string path(const std::string& variant) {
  return std::string(B70_KERNEL_DIR) + "/" + variant + ".bin";
}
}  // namespace kernels
```

- [ ] **Step 4: Write `module.h`/`module.cc`, `kernel.h`/`kernel.cc`, and `CmdList::launch`**

`module.h`:
```cpp
#pragma once
#include <string>
#include "l0/context.h"

namespace l0 {
class Kernel;
// A native (ocloc-built) device binary loaded into the context.
class Module {
 public:
  Module(Context& ctx, const std::string& bin_path);
  ~Module();
  Module(const Module&) = delete;
  Module& operator=(const Module&) = delete;
  Kernel kernel(const char* name);
  ze_module_handle_t handle() const { return m_; }

 private:
  ze_module_handle_t m_ = nullptr;
};
}  // namespace l0
```

`module.cc`:
```cpp
#include "l0/module.h"
#include <fstream>
#include <iterator>
#include <vector>
#include "l0/error.h"
#include "l0/kernel.h"

namespace l0 {
Module::Module(Context& ctx, const std::string& bin_path) {
  std::ifstream f(bin_path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open kernel binary: " + bin_path);
  std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

  ze_module_desc_t d{};
  d.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
  d.format = ZE_MODULE_FORMAT_NATIVE;
  d.inputSize = bytes.size();
  d.pInputModule = bytes.data();
  ze_module_build_log_handle_t log = nullptr;
  ze_result_t r = zeModuleCreate(ctx.handle(), ctx.device(), &d, &m_, &log);
  if (r != ZE_RESULT_SUCCESS) {
    std::string text;
    if (log) {
      size_t n = 0;
      zeModuleBuildLogGetString(log, &n, nullptr);
      text.resize(n);
      zeModuleBuildLogGetString(log, &n, text.data());
      zeModuleBuildLogDestroy(log);
    }
    throw std::runtime_error("zeModuleCreate(" + bin_path + ") -> " + result_name(r) + "\n" + text);
  }
  if (log) zeModuleBuildLogDestroy(log);
}
Module::~Module() {
  if (m_) zeModuleDestroy(m_);
}
Kernel Module::kernel(const char* name) { return Kernel(*this, name); }
}  // namespace l0
```

`kernel.h`:
```cpp
#pragma once
#include <cstdint>
#include "l0/error.h"

namespace l0 {
class Module;
// One kernel function with its argument state. Arguments are set by value;
// pointers are passed as the pointer value (sizeof(void*)).
class Kernel {
 public:
  Kernel(Module& m, const char* name);
  ~Kernel();
  Kernel(Kernel&& o) noexcept : k_(o.k_) { o.k_ = nullptr; }
  Kernel(const Kernel&) = delete;
  Kernel& operator=(const Kernel&) = delete;

  template <class T>
  void arg(uint32_t index, const T& value) {
    ZE_CHECK(zeKernelSetArgumentValue(k_, index, sizeof(T), &value));
  }
  void arg_ptr(uint32_t index, const void* p) { arg<const void*>(index, p); }
  void group_size(uint32_t x, uint32_t y = 1, uint32_t z = 1) {
    ZE_CHECK(zeKernelSetGroupSize(k_, x, y, z));
  }
  ze_kernel_handle_t handle() const { return k_; }

 private:
  ze_kernel_handle_t k_ = nullptr;
};
}  // namespace l0
```

`kernel.cc`:
```cpp
#include "l0/kernel.h"
#include "l0/module.h"

namespace l0 {
Kernel::Kernel(Module& m, const char* name) {
  ze_kernel_desc_t d{};
  d.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
  d.pKernelName = name;
  ZE_CHECK(zeKernelCreate(m.handle(), &d, &k_));
}
Kernel::~Kernel() {
  if (k_) zeKernelDestroy(k_);
}
}  // namespace l0
```

Append to `cmdlist.cc` (inside `namespace l0`, after `fill`), and add `#include "l0/kernel.h"`:
```cpp
void CmdList::launch(Kernel& k, uint32_t gx, uint32_t gy, uint32_t gz) {
  ze_group_count_t gc{gx, gy, gz};
  ZE_CHECK(zeCommandListAppendLaunchKernel(l_, k.handle(), &gc, nullptr, 0, nullptr));
}
```

`src/l0/CMakeLists.txt`: add `module.cc kernel.cc` to the source list.

- [ ] **Step 5: Build and run**

Run: `tools/box.sh test launch_test` - Expected: `launch_test OK`, tests passed.

- [ ] **Step 6: Commit**

```bash
git add src/l0 src/kernels/kernels.h tests/l0/launch_test.cc tests/CMakeLists.txt
git commit -m "feat(l0): module + kernel wrappers, regular-list launch, replay smoke test"
```

---

### Task 4: `probe_bw` - Level Zero read bandwidth → doc 01

**Files:**
- Create: `src/kernels/bw_sum.cl`; Modify: `src/kernels/CMakeLists.txt`
- Create: `tools/probe/probe_bw.cc`, `tools/probe/CMakeLists.txt`, `tools/probe/timer.h`
- Modify: `CMakeLists.txt` (add `add_subdirectory(tools/probe)`)
- Modify: `docs/01-hardware.md`

**Interfaces:**
- Produces: `tools/probe/timer.h` - `struct Timer { void start(); double ms() const; }` (steady_clock) used by every probe; CMake convention for probes: `add_executable(probe_x probe_x.cc)`, link `b70_l0`, `b70_target_kernel_dir`, `add_dependencies(probe_x kernel_<name>)`.

- [ ] **Step 1: Kernel `src/kernels/bw_sum.cl`**

```c
// bw_sum: read-only bandwidth probe. Every work-item strides over uint4s and
// folds them with XOR; the result is written only if it equals a constant the
// data almost surely never produces, which keeps the loads live without atomics.
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void bw_sum(__global const uint4* restrict in, ulong n_vec,
                     __global uint* restrict out) {
  uint acc = 0u;
  for (ulong i = get_global_id(0); i < n_vec; i += get_global_size(0)) {
    uint4 v = in[i];
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x9E3779B9u) out[get_group_id(0)] = acc;
}
```

`src/kernels/CMakeLists.txt` add: `add_ocloc_kernel(bw_sum SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/bw_sum.cl)`

- [ ] **Step 2: `tools/probe/timer.h`**

```cpp
#pragma once
#include <chrono>
struct Timer {
  std::chrono::steady_clock::time_point t0;
  void start() { t0 = std::chrono::steady_clock::now(); }
  double ms() const {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  }
};
```

- [ ] **Step 3: `tools/probe/probe_bw.cc`**

```cpp
// probe_bw: device-to-device read bandwidth with a plain L0 launch.
// Method (doc 01): 5 warm-ups, 20 timed iterations, buffers 2 GB and 8 GB.
// Timing wraps execute+fence; at >3 ms per kernel the ~20 us submit cost is <1%.
#include <cstdio>
#include <cstdint>
#include <vector>
#include "kernels/kernels.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "l0/queue.h"
#include "timer.h"

int main() {
  l0::Context ctx(0);
  l0::Queue q(ctx);
  l0::Fence fence(q);
  l0::Module mod(ctx, kernels::path("bw_sum"));
  l0::Kernel k = mod.kernel("bw_sum");
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  l0::Mem out(ctx, l0::MemKind::Device, 1 << 20);

  const uint32_t wg = 256;
  const uint32_t groups = ctx.eu_count() * 8 * 16 / wg * 4;  // 4x oversubscribed threads
  std::printf("| buffer | GB/s (median of 20) | min | max |\n|---|---|---|---|\n");
  for (size_t gb : {2ul, 8ul}) {
    const size_t bytes = gb << 30;
    l0::Mem buf(ctx, l0::MemKind::Device, bytes);
    imm.fill(buf.ptr(), 0x12345678u, bytes);
    const uint64_t n_vec = bytes / 16;
    k.arg_ptr(0, buf.ptr());
    k.arg(1, n_vec);
    k.arg_ptr(2, out.ptr());
    k.group_size(wg);
    l0::CmdList list = l0::CmdList::regular(ctx);
    list.launch(k, groups);
    list.close();
    std::vector<double> gbps;
    for (int it = 0; it < 25; ++it) {
      Timer t; t.start();
      q.execute(list, &fence);
      fence.wait();
      double ms = t.ms();
      if (it >= 5) gbps.push_back(double(bytes) / (ms * 1e6));
    }
    std::sort(gbps.begin(), gbps.end());
    std::printf("| %zu GB | %.0f | %.0f | %.0f |\n", gb, gbps[gbps.size() / 2], gbps.front(), gbps.back());
  }
  return 0;
}
```
(add `#include <algorithm>` for `std::sort`.)

- [ ] **Step 4: `tools/probe/CMakeLists.txt` and root hook**

```cmake
add_executable(probe_bw probe_bw.cc)
target_include_directories(probe_bw PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(probe_bw PRIVATE b70_l0)
b70_target_kernel_dir(probe_bw)
add_dependencies(probe_bw kernel_bw_sum)
```
Root `CMakeLists.txt`: add `add_subdirectory(tools/probe)` after `src/l0`.

- [ ] **Step 5: Run on the box**

Run: `tools/box.sh build && tools/box.sh run ./build/tools/probe/probe_bw`
Expected: two rows, medians within 5% of 600 GB/s. If the 8 GB allocation fails despite the relaxed-limits descriptor, report the error text and drop to 4 GB - note it in the doc row.

- [ ] **Step 6: Record in `docs/01-hardware.md`**

Under "## Memory bandwidth - measured", add after the existing table:

```markdown
Re-measured with a plain Level Zero launch (`tools/probe/probe_bw`,
`bw_sum.cl`, 2026-08-__): <paste the two rows>. Agrees with the torch probe
within <x>%. Use 600 GB/s.
```
Fill the blanks with the numbers printed.

- [ ] **Step 7: Commit**

```bash
git add src/kernels/bw_sum.cl src/kernels/CMakeLists.txt tools/probe CMakeLists.txt docs/01-hardware.md
git commit -m "feat(probe): L0 read-bandwidth probe + measured number in doc 01"
```

---

### Task 5: `probe_replay` - per-kernel cost inside a replayed list → doc 07 #5, doc 05, doc 04

**Files:**
- Create: `src/kernels/ctrl_read.cl`; Modify: `src/kernels/CMakeLists.txt`
- Create: `tools/probe/probe_replay.cc`; Modify: `tools/probe/CMakeLists.txt`
- Modify: `docs/07-open-questions.md` (#5), `docs/05-perf-model.md` (measure item 5), `docs/04-architecture.md` (fusion decision)

- [ ] **Step 1: Kernel `src/kernels/ctrl_read.cl`**

```c
// ctrl_read: the shape of every decode kernel's first instruction - read one
// dword of the shared-memory control block. probe_replay times this against
// noop to price a control-block read per kernel.
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void ctrl_read(__global const uint* restrict ctrl, __global uint* restrict out) {
  if (get_global_id(0) == 0) out[0] = ctrl[0] + 1u;
}
```
CMake: `add_ocloc_kernel(ctrl_read SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/ctrl_read.cl)`

- [ ] **Step 2: `tools/probe/probe_replay.cc`**

```cpp
// probe_replay: fixed cost per kernel inside a replayed regular command list.
// One in-order list with N launches, closed once, executed 1000x with a fence.
// Reports us per replay and us per kernel for N in {1, 250, 700}, for noop
// and for ctrl_read (which reads a shared-memory dword, like every decode
// kernel will). Decision rule is in spec 1 Section 4.1.
#include <algorithm>
#include <cstdio>
#include <vector>
#include "kernels/kernels.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "l0/queue.h"
#include "timer.h"

static double median_us(l0::Queue& q, l0::Fence& f, l0::CmdList& list, int reps) {
  std::vector<double> us;
  for (int i = 0; i < reps + 20; ++i) {
    Timer t; t.start();
    q.execute(list, &f);
    f.wait();
    if (i >= 20) us.push_back(t.ms() * 1e3);
  }
  std::sort(us.begin(), us.end());
  return us[us.size() / 2];
}

int main() {
  l0::Context ctx(0);
  l0::Queue q(ctx);
  l0::Fence fence(q);
  l0::Mem out(ctx, l0::MemKind::Device, 4096);
  l0::Mem ctrl(ctx, l0::MemKind::Shared, 64);
  ctrl.as<uint32_t>()[0] = 7;

  struct Case { const char* bin; const char* fn; bool has_ctrl; };
  const Case cases[] = {{"noop", "noop", false}, {"ctrl_read", "ctrl_read", true}};
  std::printf("| kernel | N | us/replay | us/kernel |\n|---|---|---|---|\n");
  {
    l0::CmdList empty = l0::CmdList::regular(ctx);
    empty.close();
    std::printf("| (empty list) | 0 | %.1f | - |\n", median_us(q, fence, empty, 1000));
  }
  for (const Case& c : cases) {
    l0::Module mod(ctx, kernels::path(c.bin));
    l0::Kernel k = mod.kernel(c.fn);
    if (c.has_ctrl) { k.arg_ptr(0, ctrl.ptr()); k.arg_ptr(1, out.ptr()); }
    else { k.arg_ptr(0, out.ptr()); }
    k.group_size(16);
    for (int n : {1, 250, 700}) {
      l0::CmdList list = l0::CmdList::regular(ctx);
      for (int i = 0; i < n; ++i) list.launch(k, 1);
      list.close();
      double us = median_us(q, fence, list, 1000);
      std::printf("| %s | %d | %.1f | %.2f |\n", c.fn, n, us, us / n);
    }
  }
  return 0;
}
```

`tools/probe/CMakeLists.txt` add:
```cmake
add_executable(probe_replay probe_replay.cc)
target_include_directories(probe_replay PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(probe_replay PRIVATE b70_l0)
b70_target_kernel_dir(probe_replay)
add_dependencies(probe_replay kernel_noop kernel_ctrl_read)
```

- [ ] **Step 3: Run on the box**

Run: `tools/box.sh build && tools/box.sh run ./build/tools/probe/probe_replay`
Expected: a table with 7 rows. Sanity: `us/kernel` at N=700 should be between 0.5 and 10; the empty-list row is the submit+fence floor.

- [ ] **Step 4: Record and decide**

- `docs/07-open-questions.md` #5: replace the body with the table and one sentence: "Measured 2026-08-__. Per-kernel cost is X µs → fusion is / is not on the phase-1 critical path (spec 1 §4.1 rule: ≥ 3 µs yes, < 1 µs no)."
- `docs/05-perf-model.md`, "What to measure" item 5: mark ✅ with the N=700 number and the empty-list floor.
- `docs/04-architecture.md`, "Kernel count is a first-class design input": replace "~250" with "~650 (spec 1 §9.1)" and add the measured µs/kernel and the resulting decision in one sentence.

- [ ] **Step 5: Commit**

```bash
git add src/kernels/ctrl_read.cl src/kernels/CMakeLists.txt tools/probe docs/04-architecture.md docs/05-perf-model.md docs/07-open-questions.md
git commit -m "feat(probe): per-kernel replay cost probe; numbers in docs 04/05/07"
```

---

### Task 6: bf16 / int4 helpers and the CPU GEMV reference

Pure host code; no GPU. This is the shared meaning of the bits for the
kernel tests (Tasks 7-9) and, in plan 2, the loader.

**Files:**
- Create: `src/common/bf16.h`, `src/common/int4.h`, `src/common/CMakeLists.txt`
- Create: `tests/kernels/gemv_ref.h`, `tests/kernels/int4_test.cc`
- Modify: `CMakeLists.txt` (add `add_subdirectory(src/common)`), `tests/CMakeLists.txt`

**Interfaces (namespace `common`):**
- `inline float bf16_to_f32(uint16_t)`; `inline uint16_t f32_to_bf16(float)` (round-to-nearest-even); `inline uint16_t f32_to_f16(float)`; `inline float f16_to_f32(uint16_t)`.
- `struct Int4Gptq { uint32_t K, N; std::vector<uint32_t> qweight; /* [K/8][N] */ std::vector<uint16_t> scales; /* f16 [K/64][N] */ }` -
  `static Int4Gptq random(uint32_t K, uint32_t N, uint32_t seed)`;
  `float at(uint32_t k, uint32_t n) const` → `(nibble − 8) · scale`; `size_t bytes() const` (qweight + scales).
- `tests/kernels/gemv_ref.h`: `void gemv_ref(const Int4Gptq& w, const std::vector<uint16_t>& x_bf16 /*[M][K]*/, uint32_t M, std::vector<float>& out /*[M][N]*/)` - double accumulation.

- [ ] **Step 1: Failing test `tests/kernels/int4_test.cc`**

```cpp
#include <cstdint>
#include "check.h"
#include "common/bf16.h"
#include "common/int4.h"

int main() {
  // bf16 round trip on representable values, and RNE on a tie.
  CHECK_EQ(common::bf16_to_f32(common::f32_to_bf16(1.5f)), 1.5f);
  CHECK_EQ(common::f32_to_bf16(1.0f), uint16_t(0x3F80));
  CHECK_EQ(common::f16_to_f32(uint16_t(0x3C00)), 1.0f);
  CHECK_EQ(common::f16_to_f32(common::f32_to_f16(0.015625f)), 0.015625f);

  // GPTQ nibble semantics: word 0 holds k = 0..7 for column n, low nibble first.
  common::Int4Gptq w;
  w.K = 64; w.N = 16;
  w.qweight.assign(w.K / 8 * w.N, 0u);
  w.scales.assign(w.K / 64 * w.N, common::f32_to_f16(0.5f));
  w.qweight[0 * w.N + 3] = 0x0000000Fu;   // k=0, n=3 -> nibble 15 -> (15-8)*0.5 = 3.5
  w.qweight[1 * w.N + 3] = 0x00000080u;   // k=9 (row 1, nibble 1 -> k=8+1), n=3 -> nibble 8 -> 0
  CHECK_NEAR(w.at(0, 3), 3.5, 1e-6);
  CHECK_NEAR(w.at(1, 3), (0 - 8) * 0.5, 1e-6);
  CHECK_NEAR(w.at(9, 3), 0.0, 1e-6);
  CHECK_EQ(w.bytes(), size_t(w.K / 8 * w.N * 4 + w.K / 64 * w.N * 2));

  common::Int4Gptq r = common::Int4Gptq::random(128, 64, 1);
  CHECK_EQ(r.qweight.size(), size_t(128 / 8 * 64));
  CHECK(r.at(5, 7) >= -8.0f * 0.08f && r.at(5, 7) <= 7.0f * 0.08f);   // scales are in [0.02, 0.08]
  std::puts("int4_test OK");
  return 0;
}
```

`tests/CMakeLists.txt` add:
```cmake
add_executable(int4_test kernels/int4_test.cc)
target_include_directories(int4_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
add_test(NAME int4_test COMMAND int4_test)
```

- [ ] **Step 2: Run to verify it fails** - `tools/box.sh test int4_test` → missing headers.

- [ ] **Step 3: `src/common/bf16.h`**

```cpp
#pragma once
#include <cstdint>
#include <cstring>

namespace common {
inline float bf16_to_f32(uint16_t h) {
  uint32_t u = uint32_t(h) << 16; float f; std::memcpy(&f, &u, 4); return f;
}
// Round-to-nearest-even; NaN inputs are not expected and not handled.
inline uint16_t f32_to_bf16(float f) {
  uint32_t u; std::memcpy(&u, &f, 4);
  uint32_t rounding = ((u >> 16) & 1u) + 0x7FFFu;
  return uint16_t((u + rounding) >> 16);
}
// IEEE half <-> float, normal range only (scales are ~1e-3..1; never subnormal).
inline float f16_to_f32(uint16_t h) {
  uint32_t sign = (h >> 15) & 1u, exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu;
  uint32_t u;
  if (exp == 0) u = sign << 31;                                   // zero / subnormal -> 0
  else if (exp == 31) u = (sign << 31) | 0x7F800000u | (man << 13);
  else u = (sign << 31) | ((exp + 112u) << 23) | (man << 13);
  float f; std::memcpy(&f, &u, 4); return f;
}
inline uint16_t f32_to_f16(float f) {
  uint32_t u; std::memcpy(&u, &f, 4);
  uint32_t sign = (u >> 16) & 0x8000u;
  int32_t exp = int32_t((u >> 23) & 0xFFu) - 127 + 15;
  uint32_t man = u & 0x7FFFFFu;
  if (exp <= 0) return uint16_t(sign);                            // flush to zero
  if (exp >= 31) return uint16_t(sign | 0x7C00u);
  uint32_t half = sign | (uint32_t(exp) << 10) | (man >> 13);
  uint32_t rem = man & 0x1FFFu;                                   // RNE on the 13 dropped bits
  if (rem > 0x1000u || (rem == 0x1000u && (half & 1u))) half += 1u;
  return uint16_t(half);
}
}  // namespace common
```

- [ ] **Step 4: `src/common/int4.h`**

```cpp
#pragma once
#include <cstdint>
#include <random>
#include <vector>
#include "common/bf16.h"

namespace common {
// GPTQ layout 0 (spec 1 §6.3): qweight[K/8][N] u32, nibble i of word (r, n)
// is the weight at k = r*8 + i, column n; scales f16 [K/64][N]; symmetric,
// zero point 8 (stored as 7 in the checkpoint's v1 qzeros, which we drop).
struct Int4Gptq {
  uint32_t K = 0, N = 0;
  std::vector<uint32_t> qweight;
  std::vector<uint16_t> scales;

  static constexpr uint32_t kGroup = 64;

  float at(uint32_t k, uint32_t n) const {
    uint32_t word = qweight[(k / 8) * N + n];
    int q = int((word >> (4 * (k % 8))) & 0xFu);
    return float(q - 8) * f16_to_f32(scales[(k / kGroup) * N + n]);
  }
  size_t bytes() const { return qweight.size() * 4 + scales.size() * 2; }

  // Uniform random nibbles, scales in [0.02, 0.08]: typical magnitude of a
  // 27B checkpoint's group scales, so outputs are O(1) for unit inputs.
  static Int4Gptq random(uint32_t K, uint32_t N, uint32_t seed) {
    Int4Gptq w; w.K = K; w.N = N;
    std::mt19937 rng(seed);
    w.qweight.resize(size_t(K / 8) * N);
    for (auto& v : w.qweight) v = rng();
    w.scales.resize(size_t(K / kGroup) * N);
    std::uniform_real_distribution<float> sd(0.02f, 0.08f);
    for (auto& s : w.scales) s = f32_to_f16(sd(rng));
    return w;
  }
};
}  // namespace common
```

`src/common/CMakeLists.txt`: `add_library(b70_common INTERFACE)` + `target_include_directories(b70_common INTERFACE ${CMAKE_SOURCE_DIR}/src)`. Root: `add_subdirectory(src/common)`.

- [ ] **Step 5: `tests/kernels/gemv_ref.h`**

```cpp
#pragma once
#include <cstdint>
#include <vector>
#include "common/bf16.h"
#include "common/int4.h"

// Reference y[m][n] = sum_k x[m][k] * W[k][n] with double accumulation.
inline void gemv_ref(const common::Int4Gptq& w, const std::vector<uint16_t>& x_bf16,
                     uint32_t M, std::vector<float>& out) {
  out.assign(size_t(M) * w.N, 0.f);
  std::vector<float> xf(size_t(M) * w.K);
  for (size_t i = 0; i < xf.size(); ++i) xf[i] = common::bf16_to_f32(x_bf16[i]);
  for (uint32_t n = 0; n < w.N; ++n) {
    for (uint32_t m = 0; m < M; ++m) {
      double acc = 0.0;
      for (uint32_t k = 0; k < w.K; ++k) acc += double(xf[size_t(m) * w.K + k]) * double(w.at(k, n));
      out[size_t(m) * w.N + n] = float(acc);
    }
  }
}

inline std::vector<uint16_t> random_bf16(size_t n, uint32_t seed, float lo = -1.f, float hi = 1.f) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> d(lo, hi);
  std::vector<uint16_t> v(n);
  for (auto& e : v) e = common::f32_to_bf16(d(rng));
  return v;
}
```
(`#include <random>` at top.)

- [ ] **Step 6: Run** - `tools/box.sh test int4_test` → `int4_test OK`.

- [ ] **Step 7: Commit**

```bash
git add src/common tests/kernels/int4_test.cc tests/kernels/gemv_ref.h tests/CMakeLists.txt CMakeLists.txt
git commit -m "feat(common): bf16/f16 conversions, GPTQ int4 layout-0 accessor, CPU GEMV reference"
```

---

### Task 7: `gemv.cl` - the int4 GEMV, both layouts, split-K, `M` loop

**Files:**
- Create: `src/kernels/gemv.cl`
- Modify: `src/kernels/CMakeLists.txt` (variant matrix), `src/kernels/kernels.h` (variant names)
- Modify: `src/common/int4.h` (add `tiled()` - layout 1 repack)
- Create: `tests/kernels/gemv_harness.h`, `tests/kernels/gemv_test.cc`; Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Kernel `gemv(const uint* w, const half* scales, const ushort* x, float* out)`; defines `M K N S LAYOUT`; work-group 64 (`reqd_work_group_size`), grid `(N/64, S)`; writes `out[(s*M + m)*N + n]` fp32.
- `kernels::gemv_variant(M,K,N,S,L)` → `"gemv_M1_K5120_N5120_S1_L0"`.
- `common::Int4Gptq::tiled()` → `std::vector<uint32_t>` in layout 1: per `(n_tile, g)` 136 u32 = 128 u32 nibbles `[k_octet(8)][n(16)]` + 8 u32 holding 16 f16 scales; tile index `n_tile * G + g`.
- `tests/kernels/gemv_harness.h`: `struct GemvCase { uint32_t M, K, N, S, L; }`, `struct GemvResult { std::vector<float> out; double us_per_launch; size_t weight_bytes; }`, `GemvResult run_gemv(l0::Context&, l0::Queue&, l0::Fence&, const common::Int4Gptq&, const std::vector<uint16_t>& x, GemvCase, int timed_launches)`, `double max_abs_err(const std::vector<float>& got, const std::vector<float>& ref)`, `double tol_for(const std::vector<float>& ref)` = `1e-4 * max|ref| + 1e-5`.

- [ ] **Step 1: Failing test `tests/kernels/gemv_test.cc`**

```cpp
// gemv correctness vs the CPU reference, a few representative variants.
// probe_gemv (Task 9) runs the full matrix; this keeps ctest fast.
#include <cstdio>
#include "check.h"
#include "common/int4.h"
#include "gemv_harness.h"
#include "gemv_ref.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/queue.h"

static void run_case(l0::Context& ctx, l0::Queue& q, l0::Fence& f, GemvCase c) {
  common::Int4Gptq w = common::Int4Gptq::random(c.K, c.N, 1234 + c.N);
  std::vector<uint16_t> x = random_bf16(size_t(c.M) * c.K, 99);
  std::vector<float> ref;
  gemv_ref(w, x, c.M, ref);
  GemvResult r = run_gemv(ctx, q, f, w, x, c, /*timed_launches=*/0);
  double err = max_abs_err(r.out, ref), tol = tol_for(ref);
  std::printf("gemv M=%u K=%u N=%u S=%u L=%u  max_abs_err=%.3g tol=%.3g\n", c.M, c.K, c.N, c.S, c.L, err, tol);
  CHECK(err <= tol);
}

int main() {
  l0::Context ctx(0);
  l0::Queue q(ctx);
  l0::Fence f(q);
  run_case(ctx, q, f, {1, 5120, 5120, 1, 0});
  run_case(ctx, q, f, {1, 5120, 5120, 4, 1});
  run_case(ctx, q, f, {1, 17408, 5120, 8, 0});
  run_case(ctx, q, f, {1, 5120, 34816, 2, 1});
  run_case(ctx, q, f, {2, 5120, 5120, 1, 0});
  std::puts("gemv_test OK");
  return 0;
}
```

`tests/CMakeLists.txt` add:
```cmake
add_executable(gemv_test kernels/gemv_test.cc)
target_include_directories(gemv_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/kernels)
target_link_libraries(gemv_test PRIVATE b70_l0)
b70_target_kernel_dir(gemv_test)
add_dependencies(gemv_test kernel_gemv_M1_K5120_N5120_S1_L0 kernel_gemv_M1_K5120_N5120_S4_L1
  kernel_gemv_M1_K17408_N5120_S8_L0 kernel_gemv_M1_K5120_N34816_S2_L1 kernel_gemv_M2_K5120_N5120_S1_L0)
add_test(NAME gemv_test COMMAND gemv_test)
```

- [ ] **Step 2: Run to verify it fails** - `tools/box.sh test gemv_test` → missing `gemv_harness.h` / variant targets.

- [ ] **Step 3: Write `src/kernels/gemv.cl`**

```c
// gemv.cl - int4 (group 64, symmetric) weights x bf16 activations, M in [1,8].
// Spec 1 §9.2. Weight-stationary: a subgroup owns 16 consecutive n and streams
// its K-range once; the M activation rows are re-read per k-group from L1/L2
// (16 lanes hit the same line: a broadcast load, no shuffles). Split-K: grid
// dim 1 is the slice s; slices write separate fp32 partials and the consumer
// (prep) sums them, so no atomics and bitwise-identical replays.
//
// Compile-time: M (tokens), K, N, S (split-K slices, S | K/64), LAYOUT (0 GPTQ
// native: w[K/8][N] u32 + scales[K/64][N] f16; 1 tiled: per (n_tile, g) 512 B
// nibbles [k_octet][n] then 32 B of f16 scales, contiguous along K).
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

#ifndef M
#define M 1
#endif
#ifndef LAYOUT
#define LAYOUT 0
#endif
#define SG 16
#define SG_PER_WG 4
#define WG_N (SG * SG_PER_WG)
#define GROUP 64
#define G (K / GROUP)
#define G_PER_S (G / S)
#define TILE_U32 136   /* 128 u32 of nibbles + 8 u32 of scales = 544 B */

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }

// 8 nibbles of one u32 word times 8 consecutive activations.
inline float dot8(uint word, ushort8 xv) {
  float a = 0.f;
  a += (float)((int)((word      ) & 0xFu) - 8) * bf16f(xv.s0);
  a += (float)((int)((word >>  4) & 0xFu) - 8) * bf16f(xv.s1);
  a += (float)((int)((word >>  8) & 0xFu) - 8) * bf16f(xv.s2);
  a += (float)((int)((word >> 12) & 0xFu) - 8) * bf16f(xv.s3);
  a += (float)((int)((word >> 16) & 0xFu) - 8) * bf16f(xv.s4);
  a += (float)((int)((word >> 20) & 0xFu) - 8) * bf16f(xv.s5);
  a += (float)((int)((word >> 24) & 0xFu) - 8) * bf16f(xv.s6);
  a += (float)((int)((word >> 28) & 0xFu) - 8) * bf16f(xv.s7);
  return a;
}

__attribute__((intel_reqd_sub_group_size(SG)))
__attribute__((reqd_work_group_size(WG_N, 1, 1)))
__kernel void gemv(__global const uint* restrict w,
                   __global const half* restrict scales,
                   __global const ushort* restrict x,
                   __global float* restrict out) {
  const uint lane = get_sub_group_local_id();
  const uint n_tile = get_group_id(0) * SG_PER_WG + get_sub_group_id();
  const uint n = n_tile * SG + lane;
  const uint s = get_group_id(1);
  const uint g0 = s * G_PER_S;
  const uint g1 = g0 + G_PER_S;

  float acc[M];
  for (int m = 0; m < M; ++m) acc[m] = 0.f;

  for (uint g = g0; g < g1; ++g) {
    uint wv[8];
    float scale;
#if LAYOUT == 0
    __global const uint* wp = w + (size_t)(g * 8) * N + n;
    for (int j = 0; j < 8; ++j) wv[j] = wp[(size_t)j * N];
    scale = (float)scales[(size_t)g * N + n];
#else
    __global const uint* tile = w + ((size_t)n_tile * G + g) * TILE_U32;
    uint8 blk = intel_sub_group_block_read8(tile);
    wv[0] = blk.s0; wv[1] = blk.s1; wv[2] = blk.s2; wv[3] = blk.s3;
    wv[4] = blk.s4; wv[5] = blk.s5; wv[6] = blk.s6; wv[7] = blk.s7;
    ushort sh = intel_sub_group_block_read_us((__global const ushort*)(tile + 128));
    scale = (float)as_half(sh);
#endif
    float gacc[M];
    for (int m = 0; m < M; ++m) gacc[m] = 0.f;
    for (int j = 0; j < 8; ++j) {
      for (int m = 0; m < M; ++m) {
        ushort8 xv = vload8(0, x + (size_t)m * K + g * GROUP + j * 8);
        gacc[m] += dot8(wv[j], xv);
      }
    }
    for (int m = 0; m < M; ++m) acc[m] += gacc[m] * scale;
  }

  for (int m = 0; m < M; ++m) out[((size_t)s * M + m) * N + n] = acc[m];
}
```

- [ ] **Step 4: Variant matrix in `src/kernels/CMakeLists.txt` and names in `kernels.h`**

Append to `src/kernels/CMakeLists.txt`:
```cmake
set(GEMV_CL ${CMAKE_CURRENT_SOURCE_DIR}/gemv.cl)
function(add_gemv_variant M K N S L)
  add_ocloc_kernel(gemv_M${M}_K${K}_N${N}_S${S}_L${L} SOURCE ${GEMV_CL}
                   DEFINES M=${M} K=${K} N=${N} S=${S} LAYOUT=${L})
endfunction()
# Probe matrix (spec 1 §4.2): the five int4 shapes the 27B runs after load-time
# fusion, both layouts, S in {1,2,4,8,16}. All S divide K/64 (80 and 272).
foreach(L 0 1)
  foreach(S 1 2 4 8 16)
    add_gemv_variant(1 5120 5120  ${S} ${L})
    add_gemv_variant(1 5120 14336 ${S} ${L})
    add_gemv_variant(1 5120 16384 ${S} ${L})
    add_gemv_variant(1 5120 34816 ${S} ${L})
    add_gemv_variant(1 17408 5120 ${S} ${L})
  endforeach()
endforeach()
add_gemv_variant(2 5120 5120 1 0)   # spec 1 §9: the M loop must compile at M=2
```

Append to `src/kernels/kernels.h` inside `namespace kernels`:
```cpp
inline std::string gemv_variant(unsigned M, unsigned K, unsigned N, unsigned S, unsigned L) {
  return "gemv_M" + std::to_string(M) + "_K" + std::to_string(K) + "_N" + std::to_string(N) +
         "_S" + std::to_string(S) + "_L" + std::to_string(L);
}
```

- [ ] **Step 5: Add `tiled()` to `src/common/int4.h`** (inside `struct Int4Gptq`)

```cpp
  // Layout 1 (spec 1 §6.3): per (n_tile of 16, k_group of 64) one 544-byte
  // tile: 128 u32 nibbles indexed [k_octet j][lane l] = word (g*8+j, n_tile*16+l),
  // then 16 f16 scales packed two per u32. Tiles ordered g-inner, n_tile-outer,
  // so one subgroup streams contiguous memory along K.
  static constexpr uint32_t kTileU32 = 136;
  std::vector<uint32_t> tiled() const {
    const uint32_t G = K / kGroup, NT = N / 16;
    std::vector<uint32_t> t(size_t(NT) * G * kTileU32);
    for (uint32_t nt = 0; nt < NT; ++nt)
      for (uint32_t g = 0; g < G; ++g) {
        uint32_t* tile = &t[(size_t(nt) * G + g) * kTileU32];
        for (uint32_t j = 0; j < 8; ++j)
          for (uint32_t l = 0; l < 16; ++l)
            tile[j * 16 + l] = qweight[size_t(g * 8 + j) * N + nt * 16 + l];
        for (uint32_t l = 0; l < 16; l += 2)
          tile[128 + l / 2] = uint32_t(scales[size_t(g) * N + nt * 16 + l]) |
                              (uint32_t(scales[size_t(g) * N + nt * 16 + l + 1]) << 16);
      }
    return t;
  }
```

- [ ] **Step 6: Write `tests/kernels/gemv_harness.h`**

```cpp
#pragma once
// Shared by gemv_test and probe_gemv: upload, launch, time, read back.
// Timing: `timed_launches` launches are recorded in ONE regular list cycling
// through NB identical copies of the weights at different addresses, so that
// consecutive launches miss the 24 MB L2 (doc 01) and the number is DRAM
// bandwidth, not cache bandwidth. us_per_launch = list time / launches.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include "common/bf16.h"
#include "common/int4.h"
#include "kernels/kernels.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "l0/queue.h"
#include <chrono>

struct GemvCase { uint32_t M, K, N, S, L; };
struct GemvResult { std::vector<float> out; double us_per_launch = 0; size_t weight_bytes = 0; };

inline double max_abs_err(const std::vector<float>& got, const std::vector<float>& ref) {
  double e = 0;
  for (size_t i = 0; i < ref.size(); ++i) e = std::max(e, double(std::fabs(got[i] - ref[i])));
  return e;
}
inline double tol_for(const std::vector<float>& ref) {
  double mx = 0;
  for (float v : ref) mx = std::max(mx, double(std::fabs(v)));
  return 1e-4 * mx + 1e-5;
}

inline GemvResult run_gemv(l0::Context& ctx, l0::Queue& q, l0::Fence& fence,
                           const common::Int4Gptq& w, const std::vector<uint16_t>& x,
                           GemvCase c, int timed_launches) {
  GemvResult r;
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  // Weight copies: enough that one cycle through them exceeds 3x L2 (72 MB).
  std::vector<uint32_t> tiled;
  const uint32_t* wsrc = w.qweight.data();
  size_t wbytes = w.qweight.size() * 4;
  if (c.L == 1) { tiled = w.tiled(); wsrc = tiled.data(); wbytes = tiled.size() * 4; }
  r.weight_bytes = w.bytes();
  const int NB = timed_launches ? std::max(2, int((72u << 20) / wbytes) + 1) : 1;

  std::vector<l0::Mem> wbufs, sbufs;
  for (int i = 0; i < NB; ++i) {
    wbufs.emplace_back(ctx, l0::MemKind::Device, wbytes);
    imm.copy(wbufs.back().ptr(), wsrc, wbytes);
    sbufs.emplace_back(ctx, l0::MemKind::Device, w.scales.size() * 2);
    imm.copy(sbufs.back().ptr(), w.scales.data(), w.scales.size() * 2);
  }
  l0::Mem xbuf(ctx, l0::MemKind::Device, x.size() * 2);
  imm.copy(xbuf.ptr(), x.data(), x.size() * 2);
  const size_t out_n = size_t(c.S) * c.M * c.N;
  l0::Mem obuf(ctx, l0::MemKind::Device, out_n * 4);

  l0::Module mod(ctx, kernels::path(kernels::gemv_variant(c.M, c.K, c.N, c.S, c.L)));
  l0::Kernel k = mod.kernel("gemv");
  k.group_size(64);
  auto bind = [&](int i) {
    k.arg_ptr(0, wbufs[i].ptr());
    k.arg_ptr(1, sbufs[i].ptr());
    k.arg_ptr(2, xbuf.ptr());
    k.arg_ptr(3, obuf.ptr());
  };

  // Correctness launch.
  {
    l0::CmdList list = l0::CmdList::regular(ctx);
    bind(0);
    list.launch(k, c.N / 64, c.S);
    list.close();
    q.execute(list, &fence);
    fence.wait();
  }
  if (timed_launches > 0) {
    l0::CmdList list = l0::CmdList::regular(ctx);
    for (int i = 0; i < timed_launches; ++i) { bind(i % NB); list.launch(k, c.N / 64, c.S); }
    list.close();
    std::vector<double> us;
    for (int rep = 0; rep < 8; ++rep) {
      auto t0 = std::chrono::steady_clock::now();
      q.execute(list, &fence);
      fence.wait();
      double total = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
      if (rep >= 3) us.push_back(total / timed_launches);
    }
    std::sort(us.begin(), us.end());
    r.us_per_launch = us[us.size() / 2];
  }
  std::vector<float> partials(out_n);
  imm.copy(partials.data(), obuf.ptr(), out_n * 4);
  r.out.assign(size_t(c.M) * c.N, 0.f);
  for (uint32_t s = 0; s < c.S; ++s)
    for (size_t i = 0; i < size_t(c.M) * c.N; ++i) r.out[i] += partials[size_t(s) * c.M * c.N + i];
  return r;
}
```

Note: `l0::Mem` is move-constructible, so `std::vector<l0::Mem>::emplace_back` works; reserve `NB` first to avoid moves during growth (`wbufs.reserve(NB); sbufs.reserve(NB);`).

- [ ] **Step 7: Build and run**

Run: `tools/box.sh test gemv_test`
Expected: five lines with `max_abs_err` ≤ `tol`, `gemv_test OK`. Build time: ~50 `ocloc` compiles (1-3 s each) on the first build only.

If layout 1 fails while layout 0 passes, the suspects are, in order: the `intel_sub_group_block_read8` element order (component `i` of lane `l` must be `tile[i*16 + l]` - write a 1-tile variant test printing lane values), then the scale packing order in `tiled()`.

- [ ] **Step 8: Commit**

```bash
git add src/kernels/gemv.cl src/kernels/CMakeLists.txt src/kernels/kernels.h src/common/int4.h tests/kernels/gemv_harness.h tests/kernels/gemv_test.cc tests/CMakeLists.txt
git commit -m "feat(kernels): int4 g64 GEMV with split-K, M loop, GPTQ-native and tiled layouts + test"
```

---

### Task 8: `gemv_bf16.cl` - the bf16 GEMV (`lm_head`, `a‖b`)

**Files:**
- Create: `src/kernels/gemv_bf16.cl`; Modify: `src/kernels/CMakeLists.txt`, `src/kernels/kernels.h`
- Modify: `src/common/bf16.h` (add `Bf16Tiled`)
- Modify: `tests/kernels/gemv_harness.h` (add `run_gemv_bf16`), `tests/kernels/gemv_ref.h` (add `gemv_bf16_ref`)
- Create: `tests/kernels/gemv_bf16_test.cc`; Modify: `tests/CMakeLists.txt`

**Interfaces:**
- Kernel `gemv_bf16(const ushort* w, const ushort* x, float* out)`; defines `M K N`; grid `(N/64)`; `out[m*N + n]` fp32.
- `kernels::gemv_bf16_variant(M,K,N)` → `"gemv_bf16_M1_K5120_N248320"`.
- `common::Bf16Tiled { uint32_t K, N; std::vector<uint16_t> data; static Bf16Tiled from_rowmajor(const uint16_t* w /*[N][K]*/, uint32_t K, uint32_t N); size_t bytes() const; }` - canonical bf16 layout: tiles `[n_tile][k_octet][8 k][16 n]`, element `(k, n)` at `((n/16 * K/8 + k/8) * 8 + k%8) * 16 + n%16`.
- `gemv_bf16_ref(const uint16_t* w_rowmajor, K, N, const std::vector<uint16_t>& x, M, std::vector<float>& out)`.
- `run_gemv_bf16(ctx, q, fence, const Bf16Tiled&, x, M, timed_launches)` → `GemvResult`.

- [ ] **Step 1: Failing test `tests/kernels/gemv_bf16_test.cc`**

```cpp
#include <cstdio>
#include "check.h"
#include "common/bf16.h"
#include "gemv_harness.h"
#include "gemv_ref.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/queue.h"

static void run_case(l0::Context& ctx, l0::Queue& q, l0::Fence& f, uint32_t M, uint32_t K, uint32_t N) {
  std::vector<uint16_t> w_rm = random_bf16(size_t(N) * K, 7 + N, -0.05f, 0.05f);   // [N][K]
  std::vector<uint16_t> x = random_bf16(size_t(M) * K, 11);
  std::vector<float> ref;
  gemv_bf16_ref(w_rm.data(), K, N, x, M, ref);
  common::Bf16Tiled w = common::Bf16Tiled::from_rowmajor(w_rm.data(), K, N);
  GemvResult r = run_gemv_bf16(ctx, q, f, w, x, M, 0);
  double err = max_abs_err(r.out, ref), tol = tol_for(ref);
  std::printf("gemv_bf16 M=%u K=%u N=%u  max_abs_err=%.3g tol=%.3g\n", M, K, N, err, tol);
  CHECK(err <= tol);
}

int main() {
  l0::Context ctx(0);
  l0::Queue q(ctx);
  l0::Fence f(q);
  run_case(ctx, q, f, 1, 5120, 128);
  run_case(ctx, q, f, 1, 5120, 248320);
  std::puts("gemv_bf16_test OK");
  return 0;
}
```

`tests/CMakeLists.txt` add:
```cmake
add_executable(gemv_bf16_test kernels/gemv_bf16_test.cc)
target_include_directories(gemv_bf16_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/kernels)
target_link_libraries(gemv_bf16_test PRIVATE b70_l0)
b70_target_kernel_dir(gemv_bf16_test)
add_dependencies(gemv_bf16_test kernel_gemv_bf16_M1_K5120_N128 kernel_gemv_bf16_M1_K5120_N248320)
add_test(NAME gemv_bf16_test COMMAND gemv_bf16_test)
```

- [ ] **Step 2: Run to verify it fails** - `tools/box.sh test gemv_bf16_test`.

- [ ] **Step 3: `src/kernels/gemv_bf16.cl`**

```c
// gemv_bf16.cl - bf16 weights x bf16 activations, M in [1,8]. Used for lm_head
// (N = 248320, 2.5 GB read per token) and the padded a||b projection (N = 128).
// Same lane-per-n structure as gemv.cl. Weights are in the canonical bf16 tile
// layout [n_tile][k_octet][8 k][16 n], so one intel_sub_group_block_read_us8
// per 8 k streams 256 contiguous bytes per subgroup. No split-K: lm_head has
// 15520 subgroups and fills the device on N alone.
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
#ifndef M
#define M 1
#endif
#define SG 16
#define SG_PER_WG 4
#define WG_N (SG * SG_PER_WG)
#define K8 (K / 8)

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }

__attribute__((intel_reqd_sub_group_size(SG)))
__attribute__((reqd_work_group_size(WG_N, 1, 1)))
__kernel void gemv_bf16(__global const ushort* restrict w,
                        __global const ushort* restrict x,
                        __global float* restrict out) {
  const uint lane = get_sub_group_local_id();
  const uint n_tile = get_group_id(0) * SG_PER_WG + get_sub_group_id();
  const uint n = n_tile * SG + lane;
  __global const ushort* tile = w + (size_t)n_tile * K8 * 128;

  float acc[M];
  for (int m = 0; m < M; ++m) acc[m] = 0.f;

  for (uint k8 = 0; k8 < K8; ++k8, tile += 128) {
    ushort8 wv = intel_sub_group_block_read_us8(tile);
    for (int m = 0; m < M; ++m) {
      ushort8 xv = vload8(0, x + (size_t)m * K + k8 * 8);
      float a = acc[m];
      a += bf16f(wv.s0) * bf16f(xv.s0); a += bf16f(wv.s1) * bf16f(xv.s1);
      a += bf16f(wv.s2) * bf16f(xv.s2); a += bf16f(wv.s3) * bf16f(xv.s3);
      a += bf16f(wv.s4) * bf16f(xv.s4); a += bf16f(wv.s5) * bf16f(xv.s5);
      a += bf16f(wv.s6) * bf16f(xv.s6); a += bf16f(wv.s7) * bf16f(xv.s7);
      acc[m] = a;
    }
  }
  for (int m = 0; m < M; ++m) out[(size_t)m * N + n] = acc[m];
}
```

`src/kernels/CMakeLists.txt` append:
```cmake
set(GEMV_BF16_CL ${CMAKE_CURRENT_SOURCE_DIR}/gemv_bf16.cl)
function(add_gemv_bf16_variant M K N)
  add_ocloc_kernel(gemv_bf16_M${M}_K${K}_N${N} SOURCE ${GEMV_BF16_CL} DEFINES M=${M} K=${K} N=${N})
endfunction()
add_gemv_bf16_variant(1 5120 248320)   # lm_head
add_gemv_bf16_variant(1 5120 128)      # in_proj_a || in_proj_b, zero-padded 96 -> 128
```

`src/kernels/kernels.h` append:
```cpp
inline std::string gemv_bf16_variant(unsigned M, unsigned K, unsigned N) {
  return "gemv_bf16_M" + std::to_string(M) + "_K" + std::to_string(K) + "_N" + std::to_string(N);
}
```

- [ ] **Step 4: `Bf16Tiled` in `src/common/bf16.h`** (append inside `namespace common`; add `#include <vector>`)

```cpp
// Canonical bf16 weight layout (spec 1 §9.2): tiles [n_tile][k_octet][8 k][16 n].
// A subgroup reads one 256 B tile per 8 k with a single block read.
struct Bf16Tiled {
  uint32_t K = 0, N = 0;
  std::vector<uint16_t> data;
  static Bf16Tiled from_rowmajor(const uint16_t* w, uint32_t K, uint32_t N) {
    Bf16Tiled t; t.K = K; t.N = N;
    t.data.resize(size_t(N) * K);
    const uint32_t K8 = K / 8;
    for (uint32_t n = 0; n < N; ++n)
      for (uint32_t k = 0; k < K; ++k)
        t.data[((size_t(n / 16) * K8 + k / 8) * 8 + k % 8) * 16 + n % 16] = w[size_t(n) * K + k];
    return t;
  }
  size_t bytes() const { return data.size() * 2; }
};
```

- [ ] **Step 5: Reference and harness additions**

`tests/kernels/gemv_ref.h` append:
```cpp
inline void gemv_bf16_ref(const uint16_t* w_rowmajor, uint32_t K, uint32_t N,
                          const std::vector<uint16_t>& x_bf16, uint32_t M, std::vector<float>& out) {
  out.assign(size_t(M) * N, 0.f);
  for (uint32_t n = 0; n < N; ++n)
    for (uint32_t m = 0; m < M; ++m) {
      double acc = 0.0;
      for (uint32_t k = 0; k < K; ++k)
        acc += double(common::bf16_to_f32(x_bf16[size_t(m) * K + k])) *
               double(common::bf16_to_f32(w_rowmajor[size_t(n) * K + k]));
      out[size_t(m) * N + n] = float(acc);
    }
}
```

`tests/kernels/gemv_harness.h` append:
```cpp
inline GemvResult run_gemv_bf16(l0::Context& ctx, l0::Queue& q, l0::Fence& fence,
                                const common::Bf16Tiled& w, const std::vector<uint16_t>& x,
                                uint32_t M, int timed_launches) {
  GemvResult r;
  r.weight_bytes = w.bytes();
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  const int NB = timed_launches ? std::max(2, int((72u << 20) / w.bytes()) + 1) : 1;
  std::vector<l0::Mem> wbufs; wbufs.reserve(NB);
  for (int i = 0; i < NB; ++i) {
    wbufs.emplace_back(ctx, l0::MemKind::Device, w.bytes());
    imm.copy(wbufs.back().ptr(), w.data.data(), w.bytes());
  }
  l0::Mem xbuf(ctx, l0::MemKind::Device, x.size() * 2);
  imm.copy(xbuf.ptr(), x.data(), x.size() * 2);
  l0::Mem obuf(ctx, l0::MemKind::Device, size_t(M) * w.N * 4);
  l0::Module mod(ctx, kernels::path(kernels::gemv_bf16_variant(M, w.K, w.N)));
  l0::Kernel k = mod.kernel("gemv_bf16");
  k.group_size(64);
  auto bind = [&](int i) { k.arg_ptr(0, wbufs[i].ptr()); k.arg_ptr(1, xbuf.ptr()); k.arg_ptr(2, obuf.ptr()); };
  {
    l0::CmdList list = l0::CmdList::regular(ctx);
    bind(0); list.launch(k, w.N / 64); list.close();
    q.execute(list, &fence); fence.wait();
  }
  if (timed_launches > 0) {
    l0::CmdList list = l0::CmdList::regular(ctx);
    for (int i = 0; i < timed_launches; ++i) { bind(i % NB); list.launch(k, w.N / 64); }
    list.close();
    std::vector<double> us;
    for (int rep = 0; rep < 8; ++rep) {
      auto t0 = std::chrono::steady_clock::now();
      q.execute(list, &fence); fence.wait();
      double total = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
      if (rep >= 3) us.push_back(total / timed_launches);
    }
    std::sort(us.begin(), us.end());
    r.us_per_launch = us[us.size() / 2];
  }
  r.out.resize(size_t(M) * w.N);
  imm.copy(r.out.data(), obuf.ptr(), r.out.size() * 4);
  return r;
}
```

- [ ] **Step 6: Build and run** - `tools/box.sh test gemv_bf16_test` → both cases pass. (The N=248320 CPU reference takes a few seconds; that is expected.)

- [ ] **Step 7: Commit**

```bash
git add src/kernels/gemv_bf16.cl src/kernels/CMakeLists.txt src/kernels/kernels.h src/common/bf16.h tests/kernels tests/CMakeLists.txt
git commit -m "feat(kernels): bf16 GEMV in canonical tile layout for lm_head and a||b + test"
```

---

### Task 9: `probe_gemv` - the measurement that picks the layout and `S`

**Files:**
- Create: `tools/probe/probe_gemv.cc`; Modify: `tools/probe/CMakeLists.txt`
- Modify: `docs/05-perf-model.md` (item 6), `docs/08-decode-vs-prefill.md` ("The decode kernel must split K"), `docs/02-formats.md` (canonical layout decision), spec §6.3 (record the winner)

- [ ] **Step 1: `tools/probe/probe_gemv.cc`**

```cpp
// probe_gemv: the production GEMV at every production shape, both layouts,
// S in {1,2,4,8,16}; GB/s of weight bytes (int4 nibbles + f16 scales), and a
// correctness check against the CPU reference at every configuration.
// Decision rules (spec 1 §4.2): layout = higher GB/s summed over the five
// int4 shapes; S per shape = smallest S within 3% of that shape's best.
#include <cstdio>
#include <map>
#include <vector>
#include "common/int4.h"
#include "gemv_harness.h"
#include "gemv_ref.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/queue.h"

int main() {
  l0::Context ctx(0);
  l0::Queue q(ctx);
  l0::Fence f(q);
  const double peak = 600.0;  // GB/s, doc 01
  struct Shape { uint32_t K, N; const char* name; };
  const Shape shapes[] = {{5120, 5120, "out/o_proj"}, {5120, 14336, "q|k|v"}, {5120, 16384, "qkv|z"},
                          {5120, 34816, "gate|up"}, {17408, 5120, "down"}};
  const uint32_t Ss[] = {1, 2, 4, 8, 16};
  double layout_sum[2] = {0, 0};
  std::map<std::pair<uint32_t, uint32_t>, std::map<uint32_t, double>> best;  // (L, shape idx) -> S -> GB/s

  std::printf("| shape | K×N | L | S | µs | GB/s | %% of 600 | max abs err | tol |\n|---|---|---|---|---|---|---|---|---|\n");
  for (uint32_t si = 0; si < 5; ++si) {
    const Shape& sh = shapes[si];
    common::Int4Gptq w = common::Int4Gptq::random(sh.K, sh.N, 42 + si);
    std::vector<uint16_t> x = random_bf16(sh.K, 5);
    std::vector<float> ref;
    gemv_ref(w, x, 1, ref);
    const double tol = tol_for(ref);
    for (uint32_t L = 0; L < 2; ++L)
      for (uint32_t S : Ss) {
        GemvResult r = run_gemv(ctx, q, f, w, x, {1, sh.K, sh.N, S, L}, 40);
        double err = max_abs_err(r.out, ref);
        double gbps = double(r.weight_bytes) / (r.us_per_launch * 1e3);
        best[{L, si}][S] = gbps;
        std::printf("| %s | %u×%u | %u | %u | %.1f | %.0f | %.0f%% | %.2g | %.2g |%s\n", sh.name, sh.K, sh.N, L, S,
                    r.us_per_launch, gbps, 100.0 * gbps / peak, err, tol, err <= tol ? "" : "  **WRONG**");
      }
  }
  // lm_head, bf16
  {
    std::vector<uint16_t> w_rm = random_bf16(size_t(248320) * 5120, 3, -0.05f, 0.05f);
    common::Bf16Tiled w = common::Bf16Tiled::from_rowmajor(w_rm.data(), 5120, 248320);
    std::vector<uint16_t> x = random_bf16(5120, 5);
    GemvResult r = run_gemv_bf16(ctx, q, f, w, x, 1, 10);
    double gbps = double(r.weight_bytes) / (r.us_per_launch * 1e3);
    std::printf("| lm_head bf16 | 5120×248320 | B | 1 | %.1f | %.0f | %.0f%% | - | - |\n", r.us_per_launch, gbps, 100.0 * gbps / peak);
  }
  std::puts("\nDecision:");
  for (uint32_t L = 0; L < 2; ++L) {
    for (uint32_t si = 0; si < 5; ++si) {
      double b = 0; for (auto& [S, g] : best[{L, si}]) b = std::max(b, g);
      layout_sum[L] += b;
    }
    std::printf("- layout %u: sum of best GB/s over shapes = %.0f\n", L, layout_sum[L]);
  }
  const uint32_t Lwin = layout_sum[1] > layout_sum[0] ? 1 : 0;
  std::printf("- canonical layout: %u\n", Lwin);
  for (uint32_t si = 0; si < 5; ++si) {
    double b = 0; for (auto& [S, g] : best[{Lwin, si}]) b = std::max(b, g);
    uint32_t pick = 16;
    for (uint32_t S : Ss) if (best[{Lwin, si}][S] >= 0.97 * b) { pick = S; break; }
    std::printf("- %s (%u×%u): S = %u\n", shapes[si].name, shapes[si].K, shapes[si].N, pick);
  }
  return 0;
}
```
(`#include <algorithm>` for `std::max`.)

`tools/probe/CMakeLists.txt` add:
```cmake
add_executable(probe_gemv probe_gemv.cc)
target_include_directories(probe_gemv PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/kernels ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(probe_gemv PRIVATE b70_l0)
b70_target_kernel_dir(probe_gemv)
# depends on every gemv variant; the kernels target list is in src/kernels
get_property(ALL_KERNELS DIRECTORY ${CMAKE_SOURCE_DIR}/src/kernels PROPERTY BUILDSYSTEM_TARGETS)
add_dependencies(probe_gemv ${ALL_KERNELS})
```

- [ ] **Step 2: Run on the box**

Run: `tools/box.sh build && tools/box.sh run ./build/tools/probe/probe_gemv | tee /tmp/probe_gemv.md`
Expected: 50 int4 rows + 1 bf16 row, **no `WRONG`**, and the decision block. Sanity: at least one configuration per shape ≥ 70% of peak; `lm_head` ≥ 80%. If every configuration is < 50%, suspect the timing (check `NB`, the L2 cycling) before the kernel.

- [ ] **Step 3: Record the numbers and the decision**

- `docs/05-perf-model.md` "What to measure" item 6: ✅ + the best row per shape and the `lm_head` row, plus the command.
- `docs/08-decode-vs-prefill.md`, under "Design rules that follow" rule 2: add one sentence with the measured N=5120 numbers at S=1 vs the chosen S - this is the split-K claim confirmed or refuted on this silicon.
- `docs/02-formats.md`, "Design principle: format plurality is a load-time problem": add "Canonical int4 layout chosen 2026-08-__ by `probe_gemv`: layout <L> (<reason in one line from the numbers>)."
- Spec §6.3: add "**Chosen: layout <L>**; `S` per shape: …" after the two layout bullets.
- `docs/12-kernels.md` (new): sections `## gemv` and `## gemv_bf16` following this template - each must be complete before the task is done:

```markdown
## gemv - int4 g64 × bf16, M ∈ [1,8]
**Computes:** out[s][m][n] = Σ_{k ∈ slice s} x[m][k] · (q[k][n] − 8) · scale[k/64][n]
**Work assignment:** why lane-per-n, why 16-wide subgroups, why 64-n work-groups, why split-K over grid dim 1 and not atomics.
**Layouts:** what layout 0 and 1 look like in memory and why the tiled one exists; which won and by how much.
**Rejected:** shuffle-broadcast of activations (OpenVINO's pattern) vs same-line vector loads - say which was measured, or that it was not.
**Measured:** the table from probe_gemv (best S per shape), date, command.
```

- [ ] **Step 4: Commit**

```bash
git add tools/probe/probe_gemv.cc tools/probe/CMakeLists.txt docs/02-formats.md docs/05-perf-model.md docs/08-decode-vs-prefill.md docs/12-kernels.md docs/superpowers/specs/2026-08-22-phase0-decode-core-design.md
git commit -m "feat(probe): GEMV bandwidth matrix; canonical layout and split-K chosen by measurement; docs/12-kernels.md"
```

---

### Task 10: `checkpoint_bytes.py` and phase-0 close-out

**Files:**
- Create: `tools/probe/checkpoint_bytes.py`
- Modify: `docs/03-models.md` (one line pointing at the script), `docs/07-open-questions.md` (#1: "reproduce with …"), `README.md` (phase 0 row: tick what is measured)

- [ ] **Step 1: `tools/probe/checkpoint_bytes.py`**

```python
#!/usr/bin/env python3
"""Byte accounting for a HF snapshot: the W behind every t/s ceiling (doc 03, 05).

Reads model.safetensors.index.json as the manifest and the safetensors headers
(no tensor data), groups bytes by role, and prints W = bytes read per decode
token: int4 qweight + f16 scales + bf16 lm_head + small bf16 tensors. qzeros and
g_idx are dropped at load, the embedding is gathered, the vision tower is
skipped, the MTP head is phase 2.

usage: checkpoint_bytes.py <snapshot_dir>
"""
import collections, json, os, struct, sys

def main(d):
    d = d.rstrip("/") + "/"
    ix = json.load(open(d + "model.safetensors.index.json"))["weight_map"]
    hdr = {}
    for f in sorted(set(ix.values())):
        with open(d + f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(n))
        h.pop("__metadata__", None)
        for k, v in h.items():
            hdr.setdefault(k, {})[f] = (v["dtype"], v["shape"], v["data_offsets"][1] - v["data_offsets"][0])
    missing = [k for k in ix if k not in hdr]
    if missing:
        sys.exit(f"{len(missing)} tensors in index but not in files, e.g. {missing[:3]}")
    groups = collections.OrderedDict((g, [0, 0]) for g in
        ["lm_head", "embed_tokens", "lm.qweight", "lm.scales", "lm.qzeros", "lm.g_idx", "lm.other", "mtp", "visual", "other"])
    for k in ix:
        dt, sh, nb = hdr[k][ix[k]]
        if k.startswith("lm_head"): g = "lm_head"
        elif "embed_tokens" in k: g = "embed_tokens"
        elif k.startswith("mtp."): g = "mtp"
        elif k.startswith("model.visual"): g = "visual"
        elif k.startswith("model.language_model"):
            suf = k.rsplit(".", 1)[-1]
            g = {"qweight": "lm.qweight", "scales": "lm.scales", "qzeros": "lm.qzeros", "g_idx": "lm.g_idx"}.get(suf, "lm.other")
        else: g = "other"
        groups[g][0] += 1; groups[g][1] += nb
    print("| group | tensors | GB |\n|---|---|---|")
    for g, (n, b) in groups.items():
        print(f"| {g} | {n} | {b/1e9:.3f} |")
    W = sum(groups[g][1] for g in ["lm_head", "lm.qweight", "lm.scales", "lm.other"])
    lh = groups["lm_head"][1]
    print(f"\nW = {W/1e9:.3f} GB/token  -> ceiling @600 GB/s = {600e9/W:.1f} t/s")
    print(f"W with lm_head int4 = {(W - lh + lh*4.125/16)/1e9:.3f} GB -> {600e9/(W - lh + lh*4.125/16):.1f} t/s")
    print(f"W with lm_head int8 = {(W - lh/2)/1e9:.3f} GB -> {600e9/(W - lh/2):.1f} t/s")
    print(f"MTP head (phase 2, per draft step) = {groups['mtp'][1]/1e9:.3f} GB")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
```

- [ ] **Step 2: Run on the box and compare with doc 03**

Run: `tools/box.sh run "python3 tools/probe/checkpoint_bytes.py \$(ls -d ~/.cache/huggingface/hub/models--Vishva007--Qwen3.8-27B-W4A16-AutoRound-GPTQ/snapshots/*/ | head -1)"`
Expected: `W = 15.519 GB/token -> ceiling 38.7 t/s`, `lm_head 2.543`, `lm.qweight 12.163`, `lm.scales 0.760`, matching doc 03's table exactly.

- [ ] **Step 3: Docs**

- `docs/03-models.md`, "Byte accounting - `W` is measured": add "Reproduce: `python3 tools/probe/checkpoint_bytes.py <snapshot>`."
- `docs/07-open-questions.md` #1: same one-liner.
- `README.md` phase-ladder row 0: replace the text with the four measured items ticked (`W` ✅, baseline ✅, replay floor ✅ <µs/kernel>, GEMV ✅ <best GB/s at N=5120>) and "bandwidth re-measured ✅ <GB/s>".

- [ ] **Step 4: Commit and tag the end of phase 0**

```bash
git add tools/probe/checkpoint_bytes.py docs/03-models.md docs/07-open-questions.md README.md
git commit -m "feat(probe): checkpoint byte accounting script; phase 0 complete"
git tag phase0-done
```

---

## Plan self-review (done at authoring, 2026-08-22)

**Spec coverage.** §4.1 → Task 5; §4.2 → Tasks 7-9; §4.3 → Task 4; §4.4 →
Task 10; §5 build + `l0/` + `kernels/` + `tests/check.h` → Tasks 1-3; §6.3
both layouts → Task 7 (`tiled()`); §9.2 `gemv` and `gemv_bf16` → Tasks 7-8
(`prep` is plan 3, as the spec says); §11 `gemv_*` tests → Tasks 7-8; §3
item 1 (phase-0 numbers) → Tasks 4, 5, 9; §3 item 4 (`docs/12-kernels.md`)
→ Task 9. Not in this plan by design: loader, model, runtime, CLI, oracle,
`prep`, `gdn_step`, attention, argmax (plans 2-3).

**Type consistency.** `l0::Mem(Context&, MemKind, size_t, size_t)` and
`as<T>()` are used identically in Tasks 2-9; `CmdList::launch(Kernel&, gx,
gy, gz)` declared in Task 2, defined in Task 3, used from Task 3 on;
`kernels::path` / `gemv_variant` / `gemv_bf16_variant` names match their
CMake target names (`kernel_<variant>`); `GemvCase{M,K,N,S,L}` field order is
the same in Tasks 7 and 9; `Int4Gptq::tiled()` returns `std::vector<uint32_t>`
with `kTileU32 = 136`, matching `TILE_U32 136` in `gemv.cl`.

**Known risk the executor should watch.** `std::vector<l0::Mem>` requires the
move constructor declared in Task 2; `reserve()` before `emplace_back`. The
`ze_relaxed_allocation_limits_exp_desc_t` chain in `Mem` is only attached
when the request exceeds `maxMemAllocSize`; if the driver rejects the 8 GB
probe buffer anyway, fall back to 4 GB and say so in doc 01.
