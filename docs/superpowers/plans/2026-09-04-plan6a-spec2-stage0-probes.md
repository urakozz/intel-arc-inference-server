# Spec 2 Stage 0 - Prefill probes P1-P5 and the composed ceiling

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure every term of a prefill step at the production shapes before
one line of prefill is designed into the engine - the SYCL/L0 interop (P1),
`sycl-tla`'s bf16 GEMM (P2), our int4→bf16 dequant scratch (P3), FMHA at
head_dim 256 (P4), Intel's CuTe chunked-GDN kernel (P5) - then **compose them
into a ceiling** at C = 4096 and C = 2048, quote it against vLLM's 1973 t/s with
both labels attached, and put the 90 %-margin bar to the operator as a ruling
request. Every probe carries a **pre-registered prediction committed before the
measurement**.

**Architecture:** T0 builds the one thing everything else needs - an *optional*
`B70_PREFILL` component compiled and linked by `icpx`, `sycl-tla` pinned by sha
and consumed **header-only** (never `add_subdirectory`), and
`runtime::prefill::Context`: an L0 immediate command list plus a SYCL in-order
queue over the **same** `ze_context`/`ze_device`, so the engine's
`zeMemAllocDevice` pointers are valid USM on both sides. P1 proves it. P2-P5 are
four independent instruments on top; each is one probe binary, one markdown
record, one commit, and each has a defined *negative* outcome that is a complete
result rather than a failure. T6 sums them and files the bar.

**Tech Stack:** C++17 for the decode tree (unchanged); **C++20 + SYCL 2020**
(`icpx -fsycl -fsycl-targets=spir64_gen -Xs "-device bmg-g31"`) for the prefill
component; `sycl-tla` (Intel's CUTLASS-SYCL fork, CUTLASS 4.2.1) pinned at
`2db1b7c94cadc52decf0013bd8b6e244fcb37dd4` (v0.9.2+9); OpenCL C 3.0 + ocloc AOT
for `pf_dequant_tile`; pure Level Zero runtime; ctest on the box
(box, `tools/box.sh`, JOBS 44 - the Mac never compiles).

**Spec:** `docs/superpowers/specs/2026-09-04-spec2-prefill-design.md` - §4 is
this plan's scope; §3 is the design every probe tests; §8 is copied verbatim
into Global Constraints below.
**Interface contract (binding):**
`.superpowers/sdd/2026-09-04-plan6-spec2-prefill/interfaces.md`. Six deviations
are requested in the self-review; **none is applied unilaterally** - the
controller rules. Until it does, an implementer writes the interfaces.md name
first and records the exact compiler error.

---

## Global Constraints

### From spec §8, verbatim

- Box workflow `tools/box.sh`; JOBS 44; the **system** oneAPI toolchain
  (`/opt/intel/oneapi/compiler/2026.1/bin/icpx` via full path or
  `setvars.sh`, `docs/10-the-box.md:42`) - the container's SYCL is the
  operator's and stays read-only; `sycl-tla` built once as a static
  dependency, version pinned in the build.
- Probes run on card 1 (`ZE_AFFINITY_MASK=1`) while a server holds card 0;
  record rows only on a provably idle box (zero DRM holders).
- Every number labelled measured vs derived vs estimated/external, with
  grade and conditions; no two values for one quantity without a
  reconciliation sentence.
- Nothing pushed; tags local; work on the branch the operator names.
- Design must not preclude prefix caching: chunk boundaries at multiples
  of 1024 align with the queued block-snapshot scheme
  (`docs/04-architecture.md`, "Follow-on"); no work on it here.

### Added by this plan

- **Every probe is iterate-grade and says so.** Card 1 under
  `ZE_AFFINITY_MASK=1` is not an idle box: a vLLM container may hold card 0 and
  the two cards share PCIe and host memory bandwidth. Nothing Stage 0 produces
  is a record row, nothing goes into `docs/BENCHMARKS.md`, and every table this
  plan writes carries the line
  `grade: iterate (card 1, ZE_AFFINITY_MASK=1, card 0 may be held)`.
- **Pre-registration is a separate commit.** Each of T0/P2/P3/P4/P5 writes its
  prediction into its own `docs/probe-*.md` and **commits it before running the
  probe**. A report whose prediction lands in the same commit as its result is
  rejected at review.
- **Warm-up discipline, `tests/kernels/gemv_harness.h:46-57` verbatim:** replay
  the closed list 8 times, drop the first 3, take the median of the last 5,
  divide by the launches recorded in it. Plus the ramp control
  (`tools/probe/probe_gemv.cc:113-127`): a discarded warm-up configuration of
  the same shape immediately before the first recorded row, printed beside it,
  because the device ramps down while the host computes a CPU reference.
- **Buffers are seeded with xorshift pseudo-random data, never a fill.** The
  B70 losslessly compresses device-local memory, so a repeated word reads back
  above the 608 GB/s theoretical peak (`docs/01-hardware.md`, "Memory
  bandwidth"). The helper to copy is `seed_incompressible` in
  `tools/probe/probe_bw.cc`.
- **Every SYCL probe run exports the four IGC/runtime variables Intel's own
  docs require for these kernels** (`sycl-tla/media/docs/cpp/build/building_with_sycl_support.md:66-71`),
  and each report states that it did:
  ```bash
  export SYCL_PROGRAM_COMPILE_OPTIONS="-ze-opt-large-register-file"
  export IGC_VISAOptions="-perfmodel"
  export IGC_VectorAliasBBThreshold=10000
  export IGC_ExtraOCLOptions="-cl-intel-256-GRF-per-thread"
  ```
  A measurement taken without them is a different measurement and is not
  comparable to one taken with them.
- **Never touch docker, never kill a process.** Card 0's container is the
  operator's; `ZE_AFFINITY_MASK=1` is the whole of this plan's interaction with
  it. No installs anywhere. `~/PycharmProjects/sycl-tla`,
  `~/PycharmProjects/vllm-xpu-kernels` and `~/PycharmProjects/oneDNN` are
  **read-only reference checkouts** - nothing in this plan writes to them.
- **Build and test only through `tools/box.sh`** (`build`, `test`, `run`,
  `pull`). The Mac configures nothing and compiles nothing.
- **`-Wall -Wextra -Werror`** on every new target, the SYCL sub-build included;
  third-party include roots are added `SYSTEM` so their warnings are not ours.
  `-cl-denorms-are-zero` stays forbidden (`cmake/ocloc.cmake:41-43`);
  `-cl-fp32-correctly-rounded-divide-sqrt` stays the ocloc default.
- **Decode is untouched by Stage 0.** No file under `src/kernels/*.cl`,
  `src/runtime/{buffers,capture,engine}.{h,cc}`, `src/loader/`, `src/model/` is
  modified by any task here. The launch/module invariants (774 / 19) are not in
  play. T0's acceptance includes a full-suite run with the component **off** and
  again with it **on**.
- **Every number labelled**, with the four kinds this project already uses
  (`docs/12-kernels.md:47-60`): per-kernel measured / estimated / derived /
  external. No two values for one quantity without a reconciliation sentence in
  the same document.
- **Commit messages** follow the house form (`type(scope): summary`) and end
  with:
  `Claude-Session: `

### Dependency note

**T0 is the only ordering constraint.** P2, P3, P4 and P5 are mutually
independent - different probe binaries, different record files, no shared source
file - and **may be run in parallel by four implementers** once T0 has landed.
P3 needs no SYCL at all (its kernel is OpenCL C, its harness the existing L0
wrapper set), so it can start even if T0 stalls on the pin; the other three
cannot. T6 consumes all five.

```
T0 ──┬── P2 ─┐
     ├── P3 ─┤        (P3 has no SYCL dependency; it may start before T0 lands)
     ├── P4 ─┼── T6
     └── P5 ─┘
```

One cross-probe coupling, resolved so it does **not** become an ordering
dependency: `sycl-tla`'s `LayoutB` tag does not by itself say whether the B
operand must be `[K][N]` row-major or `[N][K]` in memory (P2 step 3 settles it).
Rather than make P3 wait, **P3 builds and measures both orientations** - they
have different write patterns and therefore different GB/s, which is worth
knowing regardless - and T6 picks the one P2 needs.

---

## Task T0 - the SYCL build foundation, `runtime::prefill::Context`, and P1 (interop smoke)

**Files:**
- Create: `cmake/prefill.cmake` - detection, the icpx sub-build, the consumer helper
- Modify: `CMakeLists.txt` - one `include(cmake/prefill.cmake)` after `include(cmake/ocloc.cmake)`
- Create: `src/sycl/CMakeLists.txt` - the **standalone** sub-project whose CXX compiler is icpx
- Create: `src/sycl/tla_pin.h.in` - the generated header that stamps the measured `sycl-tla` sha into every probe
- Create: `src/sycl/context.cc` - the SYCL/L0 interop implementation
- Create: `src/sycl/tla_smoke.cc` - compile-only proof the pin parses under our flags
- Create: `src/runtime/prefill/context.h` - SYCL-free; included by g++ TUs
- Create: `src/runtime/prefill/context_sycl.h` - the typed `sycl::queue&` accessor; icpx TUs only
- Create: `tools/probe/probe_interop.cc` - **P1**
- Create: `tests/prefill/context_test.cc` - the failing test that drives T0
- Modify: `tests/CMakeLists.txt` - register `context_test`, guarded
- Modify: `tools/probe/CMakeLists.txt` - a comment naming where the SYCL probes are built (no target added)
- Create: `docs/probe-interop-2026-09-04.md` - P1's pre-registration and result

**Interfaces:**
- **Consumes:** `l0::Context` (`src/l0/context.h:23`) - `handle()`, `device()`,
  `driver()`; `l0::Kernel::handle()` (`src/l0/kernel.h:25`, inline); `l0::Mem`
  (`src/l0/memory.h:10`); `kernels::path()` (`src/kernels/kernels.h:9`); the
  compiled `noop`, `ctrl_read` and `bw_sum` binaries
  (`src/kernels/CMakeLists.txt:1-3`).
- **Produces:**
  ```cpp
  // src/runtime/prefill/context.h                        (SYCL-free)
  namespace runtime::prefill {
  struct KernelArg { const void* ptr; size_t size; };     // zeKernelSetArgumentValue form
  class Context {
   public:
    Context(ze_context_handle_t ze_ctx, ze_device_handle_t ze_dev);
    explicit Context(l0::Context& c);                     // inline, delegates
    ~Context();
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    void wait();                                          // BOTH queues drained
    void launch(ze_kernel_handle_t k, uint32_t gx, uint32_t gy, uint32_t gz,
                std::initializer_list<KernelArg> args);
    void launch(l0::Kernel& k, uint32_t gx, uint32_t gy, uint32_t gz,
                std::initializer_list<KernelArg> args);    // inline, forwards handle()
    void* sycl_queue_raw()  const;                         // see context_sycl.h
    void* sycl_context_raw() const;                        // sycl::context*, for interop queries
    ze_context_handle_t      ze_context() const;
    ze_device_handle_t       ze_device()  const;
    ze_command_list_handle_t l0_list()    const;
   private:
    struct Impl; Impl* p_;
  };
  }
  // src/runtime/prefill/context_sycl.h                   (icpx TUs only)
  namespace runtime::prefill {
  inline sycl::queue&   sycl(Context& cx)      { return *static_cast<sycl::queue*>(cx.sycl_queue_raw()); }
  inline sycl::context& sycl_ctx(Context& cx)  { return *static_cast<sycl::context*>(cx.sycl_context_raw()); }
  }
  ```
  and the CMake contract every later target uses:
  ```cmake
  B70_PREFILL_ENABLED          # ON/OFF, set by cmake/prefill.cmake
  b70_link_prefill(<target>)   # links libb70_prefill.so, sets BUILD_RPATH, orders the sub-build
  b70_sycl_probe(<name> ...)   # inside src/sycl/CMakeLists.txt: one icpx-built probe executable
  ```

- [ ] **Step 1 - pre-register P1, and commit it alone.** Create
  `docs/probe-interop-2026-09-04.md` with predictions and nothing measured:
  ```markdown
  # P1 - SYCL↔Level Zero interop smoke (spec 2 §4)

  grade: iterate (card 1, ZE_AFFINITY_MASK=1, card 0 may be held)

  ## Pre-registered predictions (written 2026-09-04, BEFORE any measurement)

  | quantity | prediction | basis |
  |---|---|---|
  | USM round-trip | a `zeMemAllocDevice` buffer written by a SYCL kernel and read by an OpenCL C kernel on the L0 immediate list holds the SYCL kernel's values, **bit-exact** | the L0 backend extension: a pointer from `zeMemAllocDevice` on the SAME `ze_context` is a valid interop USM pointer (`sycl_ext_oneapi_backend_level_zero.md`, "Construct a SYCL object from a Level-Zero handle") |
  | `sycl::get_pointer_type` on that pointer | **unknown** - recorded, not asserted | the extension does not promise the runtime tracks a foreign allocation; the probe reports what this driver says |
  | L0 immediate-list per-launch overhead (`noop`) | ≤ 10 µs at N = 1, ≤ 2 µs amortised at N = 1024 | `probe_replay`, docs/07 #5: 9.9 µs for a 1-launch regular-list replay, 0.52 µs/kernel at N = 700 |
  | SYCL in-order queue per-launch overhead (empty `single_task`) | ≤ 10 µs at N = 1, ≤ 2 µs at N = 1024 | same class of work; **no published figure for this queue on this box - this is the first** |
  | cross-queue handoff, both directions | **≤ 20 µs** | spec §4's P1 row |

  ## The budget this has to fit inside (derived, and it is tight)

  A prefill chunk alternates our OpenCL C kernels and `sycl-tla` GEMMs about
  **6 times per layer** (norm → GEMM → activation → GEMM → attention/GDN →
  GEMM), so **≈ 384 handoffs per chunk over 64 layers**. Spec §3.6 requires the
  interop cost under **1 %** of chunk time. Against a C = 2048 chunk of ~1.0 s
  (half of vLLM's 2.076 s for pp4096, `explorer-2 §4`), 1 % is 10 ms - i.e.
  **26 µs per handoff**. The 20 µs prediction is therefore the design's headroom
  and not a comfortable margin. **If the measurement lands above 26 µs the
  finding is that the first cut must batch our OpenCL C kernels between GEMMs
  (fewer, larger handoffs), and P1 records that arithmetic explicitly rather
  than only a pass/fail.**
  ```
  Commit: `docs(probe): pre-register P1 interop predictions`
- [ ] **Step 2 - verify the pin on the box and record it.**
  ```bash
  tools/box.sh run "git -C ~/PycharmProjects/sycl-tla rev-parse HEAD; \
    git -C ~/PycharmProjects/sycl-tla describe --tags --always; \
    git -C ~/PycharmProjects/sycl-tla remote -v; \
    git -C ~/PycharmProjects/sycl-tla status --short; \
    sed -n '37,39p' ~/PycharmProjects/sycl-tla/include/cutlass/version.h"
  ```
  **The pin this plan sets: `2db1b7c94cadc52decf0013bd8b6e244fcb37dd4`** -
  `intel/sycl-tla`, `v0.9.2-9-g2db1b7c9`, CUTLASS 4.2.1, which is what the box's
  checkout is at. It is chosen over the revision `vllm-xpu-kernels` pins
  (`87f6850680a580654b9ea2c80dbc01aeb36ad231`,
  `~/PycharmProjects/vllm-xpu-kernels/CMakeLists.txt:323-325`) for two reasons
  recorded in `cmake/prefill.cmake`: it is already on the box, so nothing
  clones; and it carries the current FMHA tree
  (`applications/flash_attention_v2/{collective,kernel}`, non-legacy) whose
  generic `FMHAConfigGenWithTileShape` is P4's cheapest route to head_dim 256.
  **P5's step 2 owns the risk that Intel's CuTe GDN kernel does not compile
  against this revision, and names `87f68506…` as its documented retry.**
- [ ] **Step 3 - write the failing test.** `tests/prefill/context_test.cc`:
  ```cpp
  // The prefill execution context: one ze_context, two queues (spec 2 §3.6).
  // Needs a B70 and the `noop` binary and nothing else -- no checkpoint, so no
  // ctest label. It is also the ABI check for the two-compiler build: this
  // translation unit is g++-compiled and calls into an icpx-linked .so across a
  // std::initializer_list and a std::runtime_error.
  #include <cstdio>
  #include "check.h"
  #include "kernels/kernels.h"
  #include "l0/context.h"
  #include "l0/module.h"
  #include "runtime/prefill/context.h"

  int main() {
    l0::Context ctx(0);
    runtime::prefill::Context cx(ctx);
    CHECK(cx.ze_context() == ctx.handle());
    CHECK(cx.ze_device() == ctx.device());
    CHECK(cx.l0_list() != nullptr);
    CHECK(cx.sycl_queue_raw() != nullptr);
    CHECK(cx.sycl_context_raw() != nullptr);
    cx.wait();                                   // both queues, nothing queued
    l0::Module mod(ctx, kernels::path("noop"));
    l0::Kernel k = mod.kernel("noop");
    cx.launch(k, 1, 1, 1, {});                   // reqd_work_group_size supplies the group
    cx.wait();
    std::printf("context_test OK\n");
    return 0;
  }
  ```
  Register it in `tests/CMakeLists.txt`, right after the `l0` group:
  ```cmake
  # The prefill execution context (spec 2 §3.6). Built only when the optional
  # SYCL component is on (cmake/prefill.cmake); with it off this block is
  # skipped entirely and the suite is exactly what it was.
  if(B70_PREFILL_ENABLED)
    add_executable(context_test prefill/context_test.cc)
    target_include_directories(context_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
    target_link_libraries(context_test PRIVATE b70_l0)
    b70_link_prefill(context_test)
    b70_target_kernel_dir(context_test)
    add_dependencies(context_test kernel_noop)
    add_test(NAME context_test COMMAND context_test)
  endif()
  ```
- [ ] **Step 4 - run it and watch it fail.** `tools/box.sh test context_test`
  → CMake does not create the target at all, because `B70_PREFILL_ENABLED` does
  not exist yet. Record the exact ctest output (`No tests were found!!!`); that
  is the failing state.
- [ ] **Step 5 - write `cmake/prefill.cmake`.** Exact content:
  ```cmake
  # The OPTIONAL SYCL prefill component (spec 2 §3.2, §3.6).
  #
  # It adds NOTHING to the decode build: no compile-flag change, no new
  # dependency on any existing target, no change to the 774-launch list. Its one
  # artifact is libb70_prefill.so, built by a SEPARATE CMake sub-build whose CXX
  # compiler is icpx (src/sycl/CMakeLists.txt). Two reasons it must be that shape:
  #
  #   * `-fsycl` device code is embedded in the object file as a fat binary and
  #     is only turned into a REGISTERED device image by the clang driver's LINK
  #     step (sycl-post-link + clang-offload-wrapper). A g++ link of icpx objects
  #     drops the device images silently -- the binary builds and then cannot
  #     find its kernels at run time. So the SYCL objects must be LINKED by
  #     icpx: a shared library, linked by icpx, whose static initialisers
  #     register its images when a g++-linked binary loads it.
  #   * CMake supports exactly one CXX compiler per project, so the icpx half
  #     is a sub-project driven by ExternalProject_Add.
  #
  # The .so links NO project static library -- only libze_loader and the SYCL
  # runtime -- so nothing in the decode build has to become -fPIC. The SYCL
  # PROBES are executables and may link the g++-built b70_l0 archive directly,
  # which needs no PIC either.
  set(B70_ICPX "/opt/intel/oneapi/compiler/2026.1/bin/icpx"
      CACHE FILEPATH "oneAPI C++ compiler for the prefill component (docs/10-the-box.md:42)")
  set(B70_PREFILL "AUTO" CACHE STRING "SYCL prefill component: AUTO, ON or OFF")
  set_property(CACHE B70_PREFILL PROPERTY STRINGS AUTO ON OFF)
  set(B70_SYCL_TLA_SRC_DIR "$ENV{HOME}/PycharmProjects/sycl-tla"
      CACHE PATH "sycl-tla checkout (read-only); consumed header-only")
  # v0.9.2-9-g2db1b7c9, CUTLASS 4.2.1. Chosen over the revision
  # vllm-xpu-kernels pins (87f6850680a580654b9ea2c80dbc01aeb36ad231,
  # vllm-xpu-kernels/CMakeLists.txt:323-325) because it is what the box already
  # has -- so nothing clones -- and because it carries the current FMHA tree
  # whose generic FMHAConfigGenWithTileShape is P4's route to head_dim 256.
  set(B70_SYCL_TLA_REVISION "2db1b7c94cadc52decf0013bd8b6e244fcb37dd4"
      CACHE STRING "sycl-tla pin; a checkout at another sha is a WARNING, recorded in the probe report")

  set(B70_PREFILL_ENABLED OFF)
  if(NOT B70_PREFILL STREQUAL "OFF")
    if(EXISTS "${B70_ICPX}")
      execute_process(COMMAND "${B70_ICPX}" --version
                      RESULT_VARIABLE _icpx_rc
                      OUTPUT_VARIABLE _icpx_out ERROR_VARIABLE _icpx_out)
    else()
      set(_icpx_rc 1)
      set(_icpx_out "no such file: ${B70_ICPX}")
    endif()
    if(_icpx_rc EQUAL 0)
      set(B70_PREFILL_ENABLED ON)
      string(REGEX REPLACE "\n.*" "" _icpx_line "${_icpx_out}")
      message(STATUS "b70: prefill component ON  -- ${_icpx_line}")
    elseif(B70_PREFILL STREQUAL "ON")
      message(FATAL_ERROR "B70_PREFILL=ON but icpx is unusable:\n${_icpx_out}")
    else()
      message(STATUS "b70: prefill component OFF -- icpx unusable (${_icpx_out})")
    endif()
  endif()

  if(B70_PREFILL_ENABLED)
    include(ExternalProject)
    get_filename_component(_icpx_bin "${B70_ICPX}" DIRECTORY)
    get_filename_component(_oneapi_root "${_icpx_bin}" DIRECTORY)
    set(B70_ONEAPI_LIB "${_oneapi_root}/lib" CACHE INTERNAL "SYCL runtime lib dir")
    set(B70_PREFILL_INSTALL "${CMAKE_BINARY_DIR}/sycl-install")
    set(B70_PREFILL_LIB "${B70_PREFILL_INSTALL}/lib/libb70_prefill.so")
    ExternalProject_Add(b70_prefill_ext
      SOURCE_DIR  "${CMAKE_SOURCE_DIR}/src/sycl"
      BINARY_DIR  "${CMAKE_BINARY_DIR}/sycl-build"
      INSTALL_DIR "${B70_PREFILL_INSTALL}"
      CMAKE_ARGS
        -DCMAKE_CXX_COMPILER=${B70_ICPX}
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DB70_ROOT=${CMAKE_SOURCE_DIR}
        -DB70_KERNEL_DIR=${B70_KERNEL_DIR}
        -DB70_OCLOC_DEVICE=${B70_OCLOC_DEVICE}
        -DB70_ONEAPI_LIB=${B70_ONEAPI_LIB}
        -DB70_L0_LIB=$<TARGET_FILE:b70_l0>
        -DB70_SYCL_TLA_SRC_DIR=${B70_SYCL_TLA_SRC_DIR}
        -DB70_SYCL_TLA_REVISION=${B70_SYCL_TLA_REVISION}
      BUILD_ALWAYS 1
      BUILD_BYPRODUCTS "${B70_PREFILL_LIB}"
      DEPENDS b70_l0)
    add_library(b70_prefill INTERFACE)
    target_link_libraries(b70_prefill INTERFACE "${B70_PREFILL_LIB}")
    target_include_directories(b70_prefill INTERFACE "${CMAKE_SOURCE_DIR}/src")
  endif()

  # Link a g++-built target against the prefill component. Also orders the
  # sub-build ahead of it and records the two run-time library directories:
  # ours, and the oneAPI SYCL runtime -- which is NOT on a non-interactive ssh
  # LD_LIBRARY_PATH (docs/10-the-box.md:42), so without the rpath every probe
  # and test dies at exec with "libsycl.so.8: cannot open shared object file".
  function(b70_link_prefill TARGET)
    if(NOT B70_PREFILL_ENABLED)
      message(FATAL_ERROR "b70_link_prefill(${TARGET}) but the component is off")
    endif()
    target_link_libraries(${TARGET} PRIVATE b70_prefill)
    add_dependencies(${TARGET} b70_prefill_ext)
    set_property(TARGET ${TARGET} APPEND PROPERTY
                 BUILD_RPATH "${B70_PREFILL_INSTALL}/lib" "${B70_ONEAPI_LIB}")
  endfunction()
  ```
  and one line in `CMakeLists.txt`, immediately after `include(cmake/ocloc.cmake)`:
  ```cmake
  include(cmake/prefill.cmake)
  ```
- [ ] **Step 6 - write `src/sycl/CMakeLists.txt`.** Exact content:
  ```cmake
  cmake_minimum_required(VERSION 3.22)
  # The icpx half of the build. Configured ONLY by cmake/prefill.cmake through
  # ExternalProject_Add, never by the top-level project: it sets
  # CMAKE_CXX_COMPILER to icpx, which one CMake project cannot do twice.
  #
  # C++20, not the decode tree's C++17: Intel's CuTe chunked-GDN kernel (P5) is
  # C++20 (vllm-xpu-kernels/cmake/utils.cmake:580, CMakeLists.txt:498). sycl-tla
  # itself asks only for C++17 (sycl-tla/CMakeLists.txt:233-237) and compiles at
  # 20. Nothing the decode build compiles is in this project; the one header
  # that crosses the boundary, src/runtime/prefill/context.h, is plain C++17 and
  # SYCL-free.
  #
  # 3.22 is sycl-tla's own floor when SYCL is on (sycl-tla/CMakeLists.txt:29-30).
  project(b70_prefill_sycl LANGUAGES CXX)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
  add_compile_options(-Wall -Wextra -Werror)

  foreach(v B70_ROOT B70_KERNEL_DIR B70_OCLOC_DEVICE B70_ONEAPI_LIB B70_L0_LIB
            B70_SYCL_TLA_SRC_DIR B70_SYCL_TLA_REVISION)
    if(NOT DEFINED ${v})
      message(FATAL_ERROR "${v} must be passed in by cmake/prefill.cmake")
    endif()
  endforeach()

  find_package(PkgConfig REQUIRED)
  pkg_check_modules(ZE REQUIRED IMPORTED_TARGET level-zero)

  # AOT for the B70. These are sycl-tla's own flags for this part, read out of
  # sycl-tla/cmake/FindDPCPP.cmake and spelled with the -Xs shorthand:
  #   compile: -fsycl -fno-sycl-instrument-device-code -fsycl-targets=spir64_gen   (:41,:73,:111)
  #   link:    + -Xsycl-target-backend=spir64_gen "-device bmg-g31"                (:112)
  #            + -Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier,
  #              +SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate  (:114-122)
  # `-Xs` is the driver's shorthand for -Xsycl-target-backend. The two SPIR-V
  # extensions beyond split_barrier are what the 2D block loads and the DPAS
  # matrix-multiply-accumulate lower through; without them the GEMM and FMHA
  # kernels fail to translate. bmg-g31 is the B70 (cmake/ocloc.cmake:4).
  set(B70_SYCL_COMPILE -fsycl -fno-sycl-instrument-device-code -fsycl-targets=spir64_gen)
  set(B70_SYCL_LINK ${B70_SYCL_COMPILE}
      -Xs "-device ${B70_OCLOC_DEVICE}"
      -Xspirv-translator
      -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate)

  # --- sycl-tla, pinned, HEADER-ONLY ----------------------------------------
  # We deliberately do NOT add_subdirectory() it. Its README says the library
  # "is header-only ... and does not need to be built to be used by other
  # projects" (sycl-tla/README.md:179-181), and its top-level CMakeLists has a
  # long list of effects an external consumer does not want: it FORCES
  # CMAKE_INSTALL_PREFIX to `install` (:246-248) and CMAKE_BUILD_TYPE to Release
  # (:376-380), sets CMAKE_POSITION_INDEPENDENT_CODE ON (:382), forces
  # CMAKE_CXX_STANDARD 17 (:235-237), requires a Python3 interpreter (:255),
  # FetchContents google/benchmark unless CUTLASS_ENABLE_BENCHMARKS is turned
  # off explicitly -- it defaults ON unconditionally (:293, :1320-1323) -- and,
  # through tools/util, ExternalProjects oneMKL when MKLROOT is unset
  # (cmake/onemkl.cmake:38,54-77). We need include paths and two -D's.
  #
  # NOTE: `applications/` (the FMHA tree) and `benchmarks/` are on NO exported
  # target and are NOT installed -- sycl-tla adds them ad hoc per example
  # (examples/06_bmg_flash_attention/CMakeLists.txt:31). They are include roots
  # here for exactly that reason.
  if(NOT EXISTS "${B70_SYCL_TLA_SRC_DIR}/include/cute/tensor.hpp")
    message(FATAL_ERROR
      "sycl-tla not found at ${B70_SYCL_TLA_SRC_DIR}. This build never clones -- run:\n"
      "  git clone https://github.com/intel/sycl-tla.git ${B70_SYCL_TLA_SRC_DIR}\n"
      "  git -C ${B70_SYCL_TLA_SRC_DIR} checkout ${B70_SYCL_TLA_REVISION}")
  endif()
  execute_process(COMMAND git -C "${B70_SYCL_TLA_SRC_DIR}" rev-parse HEAD
                  OUTPUT_VARIABLE B70_TLA_SHA OUTPUT_STRIP_TRAILING_WHITESPACE
                  RESULT_VARIABLE _tla_rc ERROR_QUIET)
  if(NOT _tla_rc EQUAL 0)
    set(B70_TLA_SHA "unknown-not-a-git-checkout")
  endif()
  if(NOT B70_TLA_SHA STREQUAL B70_SYCL_TLA_REVISION)
    message(WARNING "b70 prefill: sycl-tla HEAD is ${B70_TLA_SHA}, pin is "
                    "${B70_SYCL_TLA_REVISION} -- every probe prints the sha it was "
                    "built against; record the deviation in the probe report")
  endif()
  # Stamped into a generated header so no probe can report a number without
  # naming the revision that produced it.
  configure_file(tla_pin.h.in "${CMAKE_CURRENT_BINARY_DIR}/tla_pin.h" @ONLY)

  add_library(sycl_tla INTERFACE)
  target_include_directories(sycl_tla SYSTEM INTERFACE
    ${B70_SYCL_TLA_SRC_DIR}/include
    ${B70_SYCL_TLA_SRC_DIR}/tools/util/include
    ${B70_SYCL_TLA_SRC_DIR}/applications
    ${B70_SYCL_TLA_SRC_DIR}/benchmarks
    ${B70_SYCL_TLA_SRC_DIR}/examples/common)
  # The two definitions sycl-tla's own build adds for an Intel SYCL target
  # (sycl-tla/CMakeLists.txt:214 and :187). A bare -D gives the macro the value
  # 1, which is what the FMHA tree's `#if (SYCL_INTEL_TARGET == 35)` Xe3p branch
  # tests against -- so 1 correctly selects the Xe2/BMG branch.
  #
  # We do NOT define CUTLASS_VERSIONS_GENERATED: it guards the include of the
  # GENERATED header cutlass/version_extended.h (sycl-tla/CMakeLists.txt:884-887,
  # :398), which only sycl-tla's own CMake produces and this build does not run.
  # vllm-xpu-kernels does define it (CMakeLists.txt:389) along with
  # CUTLASS_ENABLE_HEADERS_ONLY (:386, a CMake option, meaningless as a macro);
  # if a build error ever names version_extended.h, that difference is the first
  # thing to check.
  target_compile_definitions(sycl_tla INTERFACE CUTLASS_ENABLE_SYCL SYCL_INTEL_TARGET)
  # sycl-tla does NOT add this for SYCL builds -- :244 puts it in the CUDA-only
  # flag set -- and its own docs tell you to pass it by hand
  # (media/docs/cpp/build/building_with_sycl_support.md:41). Without it a
  # template error in the mainloop prints thousands of lines.
  target_compile_options(sycl_tla INTERFACE -ftemplate-backtrace-limit=0)

  # --- the component ---------------------------------------------------------
  add_library(b70_prefill SHARED context.cc)
  target_include_directories(b70_prefill PRIVATE ${B70_ROOT}/src)
  target_compile_options(b70_prefill PRIVATE ${B70_SYCL_COMPILE})
  target_link_options(b70_prefill PRIVATE ${B70_SYCL_LINK} -Wl,-rpath,${B70_ONEAPI_LIB})
  target_link_libraries(b70_prefill PRIVATE PkgConfig::ZE)
  install(TARGETS b70_prefill LIBRARY DESTINATION lib)

  # Compile-only proof that the pin resolves and its headers parse under our
  # flags. It defines one symbol nobody calls; it exists so that a broken pin
  # fails here in seconds rather than inside P2's mainloop instantiation.
  add_library(b70_tla_smoke STATIC tla_smoke.cc)
  target_include_directories(b70_tla_smoke PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
  target_compile_options(b70_tla_smoke PRIVATE ${B70_SYCL_COMPILE})
  target_link_libraries(b70_tla_smoke PRIVATE sycl_tla)

  # --- the SYCL probes -------------------------------------------------------
  # Their sources live in tools/probe/ with every other probe; only their BUILD
  # is here, because they contain SYCL kernels and so must be compiled AND
  # linked by icpx (cmake/prefill.cmake states why). Each probe gets the test
  # helpers' include dirs too, so gemv_harness.h / attn_ref.h / gdn_ref.h are
  # reachable -- the probes reuse those references rather than restating them.
  function(b70_sycl_probe NAME)
    add_executable(${NAME} ${B70_ROOT}/tools/probe/${NAME}.cc)
    target_include_directories(${NAME} PRIVATE
      ${B70_ROOT}/src ${B70_ROOT}/tests ${B70_ROOT}/tests/kernels
      ${B70_ROOT}/tools/probe ${CMAKE_CURRENT_BINARY_DIR})
    target_compile_definitions(${NAME} PRIVATE B70_KERNEL_DIR="${B70_KERNEL_DIR}")
    target_compile_options(${NAME} PRIVATE ${B70_SYCL_COMPILE})
    target_link_options(${NAME} PRIVATE ${B70_SYCL_LINK} -Wl,-rpath,${B70_ONEAPI_LIB})
    target_link_libraries(${NAME} PRIVATE b70_prefill ${B70_L0_LIB} PkgConfig::ZE ${ARGN})
    install(TARGETS ${NAME} RUNTIME DESTINATION bin)
  endfunction()
  b70_sycl_probe(probe_interop)
  ```
- [ ] **Step 7 - write `src/sycl/tla_pin.h.in` and `src/sycl/tla_smoke.cc`.**
  ```c
  // tla_pin.h.in -- generated by src/sycl/CMakeLists.txt. Every probe prints
  // these two strings, so no measurement can be reported without naming the
  // sycl-tla revision that produced it.
  #pragma once
  #define B70_SYCL_TLA_SHA "@B70_TLA_SHA@"
  #define B70_SYCL_TLA_PIN "@B70_SYCL_TLA_REVISION@"
  ```
  ```cpp
  // tla_smoke.cc -- compile-only: does the pinned sycl-tla parse under our
  // flags? Nothing calls this. See src/sycl/CMakeLists.txt.
  #include <cute/tensor.hpp>
  #include <cutlass/cutlass.h>
  #include <cutlass/gemm/device/gemm_universal_adapter.h>
  #include "tla_pin.h"
  namespace {
  int tla_smoke_rank() { return int(cute::rank(cute::Shape<cute::_1, cute::_1>{})); }
  }  // namespace
  const char* b70_tla_sha() { return B70_SYCL_TLA_SHA; }
  int b70_tla_smoke() { return tla_smoke_rank(); }
  ```
- [ ] **Step 8 - write the two headers** exactly as quoted in **Interfaces**
  above. `context.h`'s header comment must state three things, because each is a
  design decision a reader will otherwise undo:
  1. *this header is SYCL-free because g++ translation units include it*;
  2. *it includes `l0/context.h` and `l0/kernel.h` and that costs **no** link
     dependency - the only members it touches (`Context::handle()`,
     `Context::device()`, `Kernel::handle()`) are inline, which is why
     `libb70_prefill.so` links no project archive*;
  3. *`sycl_queue_raw()`/`sycl_context_raw()` have exactly two legitimate
     callers, the two inline accessors in `context_sycl.h`.*
- [ ] **Step 9 - discover the exact interop struct spelling on this SYCL
  version, and record it.**
  ```bash
  tools/box.sh run "grep -n 'struct\\|backend_input_t\\|ownership\\|make_context\\|make_device\\|make_queue' \
    /opt/intel/oneapi/compiler/2026.1/include/sycl/ext/oneapi/backend/level_zero.hpp | head -60"
  ```
  Paste, verbatim, into `docs/probe-interop-2026-09-04.md`: the definitions of
  `backend_input_t<backend::ext_oneapi_level_zero, context>` and `<…, device>`,
  their member names **in declaration order**, and the `ownership` enum's
  spelling. **Write `src/sycl/context.cc`'s aggregate initialisers from that
  output**, not from the draft below.
- [ ] **Step 10 - implement `src/sycl/context.cc`.** The shape, to be spelled
  with the field names Step 9 recorded:
  ```cpp
  #include "runtime/prefill/context.h"
  #include <sycl/sycl.hpp>
  #include <sycl/ext/oneapi/backend/level_zero.hpp>
  #include "runtime/prefill/context_sycl.h"
  #include <cstdio>
  #include <stdexcept>
  #include <string>
  #include <vector>

  namespace runtime::prefill {
  namespace {
  constexpr sycl::backend kL0 = sycl::backend::ext_oneapi_level_zero;
  void zc(ze_result_t r, const char* what) {
    if (r == ZE_RESULT_SUCCESS) return;
    char b[24];
    std::snprintf(b, sizeof b, "0x%X", unsigned(r));
    throw std::runtime_error(std::string("prefill::Context: ") + what + " failed: " + b);
  }
  }  // namespace

  struct Context::Impl {
    ze_context_handle_t ctx;
    ze_device_handle_t dev;
    ze_command_list_handle_t list;
    sycl::device sdev;
    sycl::context sctx;
    sycl::queue sq;
  };

  Context::Context(ze_context_handle_t ze_ctx, ze_device_handle_t ze_dev) {
    // The L0 side: one IMMEDIATE command list on ordinal 0 -- the same shape
    // l0::CmdList::immediate builds (src/l0/cmdlist.cc:7-15) but ASYNCHRONOUS,
    // because prefill's ordering is by wait()/events and a synchronous append
    // would serialise the host on every kernel.
    ze_command_queue_desc_t qd{};
    qd.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
    qd.ordinal = 0;
    qd.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
    qd.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
    ze_command_list_handle_t list = nullptr;
    zc(zeCommandListCreateImmediate(ze_ctx, ze_dev, &qd, &list),
       "zeCommandListCreateImmediate");

    // The SYCL side: device and context by INTEROP over the same handles, so
    // the engine's zeMemAllocDevice pointers are valid USM in SYCL kernels
    // (that is the whole requirement; P1 proves it). The QUEUE is a plain
    // in-order sycl::queue ON that context, NOT an interop queue: the
    // extension's queue interop wants a ze_command_queue we would then own and
    // drain twice, and it is the CONTEXT that makes the pointers shared.
    // ownership::keep everywhere -- l0::Context still owns the handles and
    // still destroys them.
    sycl::device sdev = sycl::make_device<kL0>(ze_dev);
    sycl::context sctx = sycl::make_context<kL0>(
        sycl::backend_input_t<kL0, sycl::context>{
            std::vector<sycl::device>{sdev}, ze_ctx,
            sycl::ext::oneapi::level_zero::ownership::keep});
    p_ = new Impl{ze_ctx, ze_dev, list, sdev, sctx,
                  sycl::queue(sctx, sdev, sycl::property::queue::in_order{})};
  }

  Context::~Context() {
    if (p_) { zeCommandListDestroy(p_->list); delete p_; }
  }

  void Context::wait() {
    p_->sq.wait_and_throw();
    zc(zeCommandListHostSynchronize(p_->list, UINT64_MAX), "zeCommandListHostSynchronize");
  }

  void Context::launch(ze_kernel_handle_t k, uint32_t gx, uint32_t gy, uint32_t gz,
                       std::initializer_list<KernelArg> args) {
    uint32_t i = 0;
    for (const KernelArg& a : args)
      zc(zeKernelSetArgumentValue(k, i++, a.size, a.ptr), "zeKernelSetArgumentValue");
    // The group size comes from the kernel's own reqd_work_group_size. Every
    // prefill OpenCL C kernel declares one, as every decode kernel does, so
    // this is one fewer argument a caller can get wrong; a kernel without one
    // is rejected by name rather than launched at a guessed width. This is how
    // interfaces.md's group-size-free `launch` signature is honoured.
    ze_kernel_properties_t kp{};
    kp.stype = ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES;
    zc(zeKernelGetProperties(k, &kp), "zeKernelGetProperties");
    if (kp.requiredGroupSizeX == 0)
      throw std::runtime_error("prefill::Context::launch: kernel has no reqd_work_group_size");
    zc(zeKernelSetGroupSize(k, kp.requiredGroupSizeX, kp.requiredGroupSizeY,
                            kp.requiredGroupSizeZ), "zeKernelSetGroupSize");
    ze_group_count_t g{gx, gy, gz};
    zc(zeCommandListAppendLaunchKernel(p_->list, k, &g, nullptr, 0, nullptr),
       "zeCommandListAppendLaunchKernel");
  }

  void* Context::sycl_queue_raw()   const { return &p_->sq; }
  void* Context::sycl_context_raw() const { return &p_->sctx; }
  ze_context_handle_t Context::ze_context() const { return p_->ctx; }
  ze_device_handle_t  Context::ze_device()  const { return p_->dev; }
  ze_command_list_handle_t Context::l0_list() const { return p_->list; }
  }  // namespace runtime::prefill
  ```
- [ ] **Step 11 - build and run the test.** `tools/box.sh test context_test` →
  green. Two failure modes and their fixes, so neither is debugged by guessing:
  - `recompile with -fPIC` - something pulled `b70_l0` into the **.so**. The
    .so's only link inputs are `PkgConfig::ZE` and the SYCL runtime. Fix the
    CMake; do **not** add `-fPIC` to the decode build.
  - `libsycl.so.*: cannot open shared object file` at exec - the `BUILD_RPATH`
    in `b70_link_prefill` is missing or the target did not go through it.
  Record the `.so` size and the first successful build's wall time.
- [ ] **Step 12 - write P1, `tools/probe/probe_interop.cc`.** Five sections,
  each printing one markdown table; every timing an 8-replay/drop-3 median after
  a discarded warm-up; the first line printed is
  `sycl-tla sha <B70_SYCL_TLA_SHA> (pin <B70_SYCL_TLA_PIN>)` and the second the
  four IGC variables as read back from `getenv`, so the record cannot omit them.
  1. **USM round-trip.** 64 MiB from `zeMemAllocDevice` on `cx.ze_context()` /
     `cx.ze_device()` (through `l0::Mem`, the engine's own allocator). Print
     `sycl::get_pointer_type(p, sycl_ctx(cx))` and
     `sycl::get_pointer_device(p, sycl_ctx(cx))` - **record what they say**; the
     extension permits `unknown` for a foreign allocation. Then a SYCL
     `parallel_for` writes `p[i] = xorshift(i)` as `uint32_t`; `cx.wait()`; the
     OpenCL C `bw_sum` kernel (`src/kernels/bw_sum.cl`, already compiled) reads
     it on the L0 immediate list into a second buffer; `cx.wait()`; the host
     copies both back and asserts the sum equals the host-computed xorshift sum
     **exactly**. Then the reverse: `ctrl_read` writes, a SYCL kernel reads.
  2. **L0 immediate-list per-launch overhead.** `noop`, N ∈ {1, 64, 1024}
     appended back to back then one `wait()`; µs/launch at each N - the same
     shape `probe_replay` uses, so the rows are comparable to docs/07 #5.
  3. **SYCL in-order queue per-launch overhead.** an empty `single_task`, the
     same N sweep, the same reporting.
  4. **Cross-queue handoff**, 256 alternations each, two orderings:
     (a) SYCL kernel → `cx.wait()` → L0 `noop` → `cx.wait()` - the "first cut:
     a queue wait between the two" spec §3.6 explicitly allows;
     (b) an L0 event: the `noop` launch signals a `ze_event_handle_t` from a
     host-visible pool (`l0::EventPool`, `src/l0/event.h:29`) and the SYCL side
     waits on it through `sycl::ext::oneapi::level_zero::make_event` **if that
     constructs on this version** - if it does not, record the exact error and
     report only (a). µs per handoff for each ordering that ran.
  5. **The budget check.** Print `handoffs_per_chunk = 6 × 64 = 384`, the
     measured µs/handoff, their product in ms, and that product as a percentage
     of a 1.0 s C = 2048 chunk and a 2.2 s C = 4096 chunk - with the sentence
     from Step 1's pre-registration about what a miss means.
- [ ] **Step 13 - run P1 on card 1 and record.**
  ```bash
  tools/box.sh run "export SYCL_PROGRAM_COMPILE_OPTIONS=-ze-opt-large-register-file; \
    export IGC_VISAOptions=-perfmodel IGC_VectorAliasBBThreshold=10000; \
    export IGC_ExtraOCLOptions=-cl-intel-256-GRF-per-thread; \
    ZE_AFFINITY_MASK=1 ./build/sycl-install/bin/probe_interop"
  tools/box.sh run "ls -l /proc/*/fd 2>/dev/null | grep -c renderD"   # card-0 holder count
  ```
  Append the output to `docs/probe-interop-2026-09-04.md` under `## Measured`,
  with the box conditions (driver version from `l0::Context::name()`/props, the
  DRM holder count above, `ZE_AFFINITY_MASK`) and a
  **prediction-vs-measurement** table, one row per pre-registered quantity,
  marked hit or miss.
- [ ] **Step 14 - prove decode is unaffected.** Two runs, both pasted into the
  task report:
  ```bash
  tools/box.sh run "cmake -S . -B build-off -DCMAKE_BUILD_TYPE=Release -DB70_PREFILL=OFF > /dev/null && cmake --build build-off -j44 && ctest --test-dir build-off -LE checkpoint --output-on-failure"
  tools/box.sh test ''            # the default configure: B70_PREFILL=AUTO -> ON
  ```
  Required: the OFF tree configures, builds and passes with **exactly the
  pre-existing test set**; the AUTO tree passes that set **plus `context_test`**.
  Record both counts.
- [ ] **Step 15 - commit.**
  `feat(prefill): SYCL build foundation + L0 interop context - P1 measured`

---

## Task P2 - `tools/probe/probe_prefill_gemm`: `sycl-tla`'s stock bf16 GEMM at the production shapes

**Files:**
- Create: `tools/probe/probe_prefill_gemm.cc`
- Modify: `src/sycl/CMakeLists.txt` - one line, `b70_sycl_probe(probe_prefill_gemm sycl_tla)`
- Create: `docs/probe-prefill-gemm-2026-09-04.md`

**Interfaces:**
- **Consumes:** `runtime::prefill::Context`, `runtime::prefill::sycl()` (T0);
  `common::bf16.h`'s `f32_to_bf16`/`bf16_to_f32`; the shape table
  (`src/model/qwen35.cc:38-64`); and, as the configuration to copy,
  **`examples/00_bmg_gemm/00_bmg_gemm_with_sycl_queue.cpp`** - *not* the plain
  `00_bmg_gemm.cpp`, because only the queue variant takes our queue:
  `gemm_op.initialize(arguments, nullptr, &q)` and `gemm_op.run(&q)` at
  `:272,275`, where `cudaStream_t` is `typedef sycl::queue*`
  (`include/cutlass/gpu_generics.h:355`). The plain example uses
  `compat::get_default_queue()` (`include/cute/util/compat/device.hpp:888` -
  the namespace is `compat`, **not** `syclcompat`).
- **Produces:** the measured TFLOP/s matrix that fixes the GEMM term of the
  composed ceiling; the **first measured XMX number in this project** (the
  figure T6 puts into `docs/01-hardware.md`); the recorded determinism answer
  spec §6.4 makes a hard gate; and **the B-operand memory layout the GEMM
  actually requires**, which is what `pf_dequant_tile` must write.

**Shapes** (`{K, N}`, all six as `model::Qwen35` binds them) × **M ∈ {512,
1024, 2048, 4096}**:

| name | K | N | 2·K·N (GFLOP per M-row) | note |
|---|---|---|---|---|
| QkvZ | 5120 | 16384 | 0.16777 | GDN, ×48 layers |
| OutProj / OProj | 6144 | 5120 | 0.06291 | one shape, two `LinearId`s |
| GateUp | 5120 | 34816 | 0.35652 | ×64 layers - the big one |
| Down | 17408 | 5120 | 0.17826 | ×64 layers |
| Qkv | 5120 | 14336 | 0.14680 | FA, ×16 layers |
| LmHead | 5120 | 248320 | 2.54320 | **probe cell only** - production runs it at M = 1 (spec §3.5) |

- [ ] **Step 1 - pre-register, and commit alone.** Create
  `docs/probe-prefill-gemm-2026-09-04.md`:
  ```markdown
  # P2 - sycl-tla stock bf16 GEMM at the prefill shapes (spec 2 §4)

  grade: iterate (card 1, ZE_AFFINITY_MASK=1, card 0 may be held)

  ## Pre-registered predictions (written 2026-09-04, BEFORE any measurement)

  1. **≥ 90 TFLOP/s at M = 4096 on gate‖up** (spec §4's P2 row; vLLM's implied
     98.2 TFLOP/s sustained is the yardstick, `explorer-2 §4`).
  2. TFLOP/s rises monotonically in M over {512, 1024, 2048, 4096} at every
     shape: weight bytes amortise over more rows, and the work-group tile is
     `Shape<_256,_256,_32>` (`00_bmg_gemm.cpp:364`), so M = 512 is only two
     tiles deep and M < 256 would not fill one.
  3. `out/o_proj` (6144×5120) is the worst cell at every M - the smallest N,
     and the shape decode also finds hardest (`docs/12`, 533 GB/s).
  4. Two runs of the same cell are **bitwise identical**. Source basis, not
     hope: the configuration omits the `TileScheduler_` template argument, so
     it is the data-parallel `PersistentScheduler`, and
     `include/cutlass/gemm/kernel/xe_gemm.hpp:82-87` static_asserts
     *"Intel Xe does not support specializing the tile scheduler"* - Stream-K
     routes to a **different** kernel (`xe_gemm_cooperative.hpp`) which this
     probe does not instantiate. There is no atomic accumulation into C.
  5. `Gemm::get_workspace_size(arguments) == 0`. It has to be: the queue
     variant treats a non-zero workspace as a hard error
     (`00_bmg_gemm_with_sycl_queue.cpp:263-265`).

  ## What prediction 1 implies for the whole spec - pre-registered, DERIVED

  Per-token GEMM FLOPs, from `model::Qwen35`'s table:
  GDN layer = QkvZ + OutProj + GateUp + Down = 0.76546 GFLOP/row, ×48;
  FA layer = Qkv + OProj + GateUp + Down = 0.74449 GFLOP/row, ×16.
  **Σ = 48.654 GFLOP/row.** (Reconciles with `explorer-2 §1`'s 48.97
  GFLOP/token, which additionally counts the conv, the recurrence and the a‖b
  linear - the two are different quantities, not two values for one.)

  At C = 4096 that is **199.3 TFLOP of GEMM**. At exactly 90 TFLOP/s the GEMM
  term alone is **2.215 s** - **6.7 % ABOVE vLLM's 2.076 s for the entire
  pp4096** - before dequant, attention, GDN, norms or interop. So:

  > **If P2 measures 90 TFLOP/s, the composed ceiling lands under vLLM and spec
  > §2's second bullet fires.** To reach 2.076 s device-side the GEMM term must
  > run at **≥ 96 TFLOP/s** *and* everything else must be free. This is written
  > down before the measurement so T6's conclusion cannot be a hindsight
  > rationalisation of whatever number comes out.
  ```
  Commit: `docs(probe): pre-register P2 sycl-tla GEMM predictions`
- [ ] **Step 2 - record the configuration, from source, before writing code.**
  ```bash
  tools/box.sh run "grep -n 'ElementInput\\|ElementAccumulator\\|ElementOutput\\|Layout[ABCD]\\|GmemTiledCopy\\|TileShape\\|TiledMma\\|PipelineStages\\|DispatchPolicy\\|CollectiveMma\\|CollectiveEpilogue\\|GemmUniversal\\|GemmUniversalAdapter\\|make_cute_packed_stride\\|get_workspace_size\\|can_implement\\|initialize(\\|\\.run(' \
    ~/PycharmProjects/sycl-tla/examples/00_bmg_gemm/00_bmg_gemm_with_sycl_queue.cpp \
    ~/PycharmProjects/sycl-tla/examples/00_bmg_gemm/00_bmg_gemm.cpp"
  ```
  Paste the alias chain into `docs/probe-prefill-gemm-2026-09-04.md` under
  `## The configuration measured`, **with the file:line of every alias.** What
  the tree holds at the pin, so the implementer knows what to expect and reports
  any difference as a finding:
  | alias | value | line |
  |---|---|---|
  | `ElementInputA` / `ElementInputB` | `bfloat16_t` | `:346-347` |
  | `ElementAccumulator` / `ElementOutput` | `float` | `:344,348` |
  | `LayoutA/B/C/D` | `cutlass::layout::RowMajor` (all four) | `:350-353` |
  | `GmemTiledCopyA` / `B` | `void` → the mainloop picks `XE_LOAD_2D` / `XE_LOAD_2D_VNNI` itself | `:360-361`, comment `:355-359` |
  | `TileShape` | `Shape<_256, _256, _32>` | `:364` |
  | `TiledMma` | `TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, bfloat16_t>>, Layout<TileShape>, Layout<Shape<_8,_4,_1>, Stride<_4,_1,_0>>>::TiledMMA` - 32 subgroups, each owning 32×64×32 | `:377`, doc `:370-376` |
  | `PipelineStages` | `2` | `:380` |
  | `GEMMDispatchPolicy` | **`MainloopXeL1Staged<2>`** | `:382` |
  | `TileScheduler_` | **omitted** → `void` → data-parallel | `:425-429` |

  > **One correction to our own record, to be written into the P2 report:**
  > `explorer-3 §2` and `docs/01-hardware.md:81`-area text describe the BMG GEMM
  > mainloop as `MainloopIntelXeXMX16`. At this pin the example uses
  > **`MainloopXeL1Staged<2>`**, and `00_bmg_gemm.cpp:381` says in so many words
  > that `MainloopIntelXeXMX16` is *"for older version of copy/mma atom"*. Both
  > names are real; the newer one is what is measured. State which was measured
  > and that the older name is the superseded policy - do not silently swap the
  > name in the explorer report.
- [ ] **Step 3 - settle the B-operand memory layout, and it is a measurement,
  not a reading.** `LayoutB = RowMajor` with
  `stride_B = make_cute_packed_stride(StrideB{}, make_shape(N, K, L))`
  (`00_bmg_gemm.cpp:234`) is stated over an `(N, K, L)` shape, so the tag alone
  does not tell you whether B must be `[K][N]` row-major (interfaces.md's
  assumption) or `[N][K]`. Settle it with a **16×16×16 GEMM** whose A is the
  identity and whose B holds `B[i][j] = i*16 + j` under one interpretation:
  - build the tiny buffer once, run the GEMM, and compare `C` against **both**
    host references (`B` read as `[K][N]` and as `[N][K]`);
  - exactly one must match bit-for-bit (an identity A makes the GEMM a copy, so
    there is no rounding question);
  - print `stride_B` as CuTe prints it, beside the verdict.
  **Record the answer as `## The B layout the GEMM requires` in the P2 report,
  in one sentence, and flag it to T6.** If it is `[N][K]`, interfaces.md's
  "Dequant scratch is bf16 row-major `[K][N]`, `ldb = N`" needs changing and
  P3's `TRANSPOSED=1` variant is the production one - P3 measures both
  orientations precisely so this discovery costs no rework.
- [ ] **Step 4 - write the probe.** `tools/probe/probe_prefill_gemm.cc`:
  - prints `sycl-tla sha <B70_SYCL_TLA_SHA> (pin <B70_SYCL_TLA_PIN>)` and the
    four IGC variables first, like every probe in this plan;
  - a `Shape { uint32_t K, N; const char* name; }` table with the six rows
    above, names spelled with U+2016 `‖` (as `probe_gemv.cc:217` does, because
    the table is pasted into markdown);
  - inputs allocated through `l0::Mem` on `cx.ze_context()` - the engine's own
    allocator, which is exactly what P1 proved SYCL can read. `A` bf16 `[M][K]`
    row-major (interfaces.md); `B` bf16 in the orientation Step 3 established;
    both filled with the xorshift seeder from `probe_bw.cc`, the raw `uint16_t`
    words masked so the bf16 exponent lands in `[-0.5, 0.5)` - nothing inf or
    NaN, and the K-length accumulation stays well-scaled;
  - output `C` fp32 `[M][N]`, `ldc = N`;
  - the call sequence copied from the queue example: `Gemm::Arguments{kGemm,
    problem_size, {A, stride_A, B, stride_B}, {{1.f, 0.f}, C, stride_C, C,
    stride_D}, hw_info}` with `hw_info.sm_count =
    KernelHardwareInfo::query_device_multiprocessor_count(0)`; assert
    `get_workspace_size == 0`; `can_implement` checked and its status printed
    per cell; `initialize(args, nullptr, &q)` then `run(&q)` with
    `q = runtime::prefill::sycl(cx)`;
  - **per-shape discarded warm-up** before that shape's first recorded M,
    printed as a ramp-control row (`probe_gemv.cc:113-127`);
  - timing: the 8-replay/drop-3 median over a fixed enqueue count, `q.wait()`
    closing each replay - `time_list`'s shape spelled for a SYCL queue;
  - **correctness on a sampled subset**: 4096 output cells picked by a fixed
    xorshift stream (same seed every run, so the sample is reproducible), each
    recomputed on the host in `double` from the same bf16 inputs. Bar:
    `|got − ref| ≤ 5e-3 · max|ref_sampled| + 1e-4`. **Why that bar and not the
    house 1e-4 one** (`gemv_harness.h:38-42`): `sycl-tla` accumulates in fp32 in
    a tile order we do not control, and fp32 accumulation error over K = 17408
    is ≈ √K·2⁻²⁴ ≈ 8e-6 relative, so 5e-3 carries ~600× headroom. **This is a
    structural check** - wrong layout, wrong dtype, wrong stride, a dropped
    k-tile - **not a numerics bar**, and the probe's header says so;
  - **determinism**: every cell run twice into two separate output buffers and
    `memcmp`ed; `identical` / `**DIFFER**` per cell;
  - output: one row-per-cell table
    `| shape | K×N | M | can_implement | ms | TFLOP/s | % of 90 | max abs err | tol | bitwise |`
    and a second, transposed **TFLOP/s matrix** (rows = shapes, columns = M) -
    the one T6 pastes into `docs/12`.
- [ ] **Step 5 - build, and expect the first build to be the hard part.** Two
  things to know before debugging blind: FMHA and GEMM template instantiations
  are compile-memory hogs - sycl-tla routes them into a throttled Ninja job pool
  for that reason (`sycl-tla/SYCL.cmake:73-79`,
  `sycl-tla/CMakeLists.txt:125-134`) - so build this probe **serially**
  (`cmake --build build/sycl-build -j1 --target probe_prefill_gemm`) if the box
  starts swapping; and the include roots are `SYSTEM` (T0 step 6), so a warning
  inside CUTLASS is not an error while a warning in the probe is. Record the
  first error verbatim in the report and fix the probe, never the flags.
- [ ] **Step 6 - run and record.**
  ```bash
  tools/box.sh run "export SYCL_PROGRAM_COMPILE_OPTIONS=-ze-opt-large-register-file; \
    export IGC_VISAOptions=-perfmodel IGC_VectorAliasBBThreshold=10000; \
    export IGC_ExtraOCLOptions=-cl-intel-256-GRF-per-thread; \
    ZE_AFFINITY_MASK=1 ./build/sycl-install/bin/probe_prefill_gemm"
  ```
  Append to the report: both tables, the box conditions, the ramp control, the B
  layout verdict, and a **prediction-vs-measurement** section with one row per
  pre-registered claim marked hit or miss. If prediction 1 misses, the report
  says by how much **and re-runs the "What prediction 1 implies" arithmetic with
  the measured number** - that paragraph is the deliverable, not a footnote.
- [ ] **Step 7 - the XMX line for `docs/01-hardware.md`.** In the report,
  compute and label:
  `measured XMX bf16 throughput = <best TFLOP/s cell> (measured, iterate grade,
  ZE_AFFINITY_MASK=1, sycl-tla <sha>, <shape> at M = <M>)`, with the
  reconciliation sentence for the derived 183.5 TFLOPS (`explorer-2 §2`: vendor
  367 TOPS INT8 ÷ 2, via `docs/01-hardware.md:106-109`'s "~2× bf16") -
  *"183.5 is a vendor-derived peak; this is a measured achieved rate at
  production shapes; the ratio is <x> %."* T6 step 5 lands it.
- [ ] **Step 8 - commit.**
  `feat(probe): P2 - sycl-tla bf16 GEMM at the six prefill shapes, <X> TFLOP/s at M=4096`

---

## Task P3 - `pf_dequant_tile` and `tools/probe/probe_dequant`

**Files:**
- Create: `src/kernels/prefill/dequant.cl` - the OpenCL C kernel `pf_dequant_tile`
- Create: `src/kernels/prefill/CMakeLists.txt` - the `add_ocloc_kernel` variants
- Modify: `src/kernels/CMakeLists.txt` - one `add_subdirectory(prefill)` at the end
- Create: `tests/prefill/dequant_test.cc` - the failing test
- Create: `tools/probe/probe_dequant.cc` - the probe (plain g++; **no SYCL**)
- Modify: `tests/CMakeLists.txt`, `tools/probe/CMakeLists.txt` - the two targets
- Create: `docs/probe-dequant-2026-09-04.md`

**Interfaces:**
- **Consumes:** the loader's tile geometry, which lives in exactly one place -
  `src/common/repack.h`: `repack_int4_layout1` (`:16-29`) and
  `repack_int4_layout0_cols` (`:67-80`); the dequant contract as
  `src/kernels/gemv.cl` implements it (`dot8`, `:47-76` - both the
  mask/subtract form and the `GEMV_DEQ_SHIFT` xor/shift form); the CPU truth
  `common::Int4Gptq::at` (`src/common/int4.h:19-23`); `common::f32_to_bf16`
  (`src/common/bf16.h`); the cross-language fixture
  `tests/golden/dequant_fixture.safetensors` and its test
  `tests/loader/dequant_fixture_test.cc`.
- **Produces:** `pf_dequant_tile`, the kernel `dequant_to_bf16` will bind, in
  **both** output orientations; and the GB/s that fixes the dequant term of the
  composed ceiling.

**The contract, quoted from the sources so the kernel cannot drift from them:**

*Layout 0* (`repack.h:67-80`, `loader/loader.h:22-23`): `qweight[K/8][N]` u32,
nibble `i` of word `(r, n)` is the weight at `k = r*8 + i`; plus a **separate**
`scales[K/64][N]` f16 allocation - `DeviceWeight::scales` is non-null **exactly
for layout 0**.

*Layout 1* (`repack.h:11-15`, `int4.h:26-31`): per `(n_tile of 16, k_group of
64)` one **136-u32 = 544-byte** tile - `tile[j*16 + l]` is the `qweight` word
`(g*8+j, nt*16+l)` for `j ∈ [0,8)`, `l ∈ [0,16)`; then 16 f16 scales at
`tile[128 + l/2]`, **two per u32, even lane in the low half**. Tiles ordered
**g-inner, n_tile-outer**.

*The dequant* (`gemv.cl:47-59`, the `GEMV_DEQ_SHIFT` form, verbatim):
```c
  // OpenCL C defines right shift of a signed integer as arithmetic sign-fill.
  // For every q in [0,16), signext4(q ^ 8) == q - 8. One word-wide xor plus
  // the shift pair per nibble therefore preserves the exact integer values and
  // the dot-product order while removing mask/subtract work.
  const uint u = word ^ 0x88888888u;
  ... (float)(((int)(u << 28)) >> 28) ...
```
so `w = (q − 8) · scale` with the f16 scale widened to fp32, and the bf16
written is `rne_bf16(float(q − 8) * f16_to_f32(scale))` - **the same expression
`common::Int4Gptq::at` computes**, which is why P3's bar is bit-exactness and
not a tolerance.

- [ ] **Step 1 - pre-register, and commit alone.** Create
  `docs/probe-dequant-2026-09-04.md`:
  ```markdown
  # P3 - pf_dequant_tile: int4 tiles -> the bf16 scratch (spec 2 §4)

  grade: iterate (card 1, ZE_AFFINITY_MASK=1, card 0 may be held)

  ## Pre-registered predictions (written 2026-09-04, BEFORE any measurement)

  1. **>= 500 GB/s** (spec §4's P3 row), where GB/s counts **bytes read plus
     bytes written, both once**. The probe prints that denominator in its own
     output, because a different one changes the number by 2x.
  2. Bit-exact against `common::Int4Gptq::at` -> `common::f32_to_bf16`, at every
     shape, in **both** tile layouts and **both** output orientations. Not a
     tolerance: the device and the host evaluate the same fp32 expression and
     round once.
  3. Layout 1 is at least as fast as layout 0 at every shape: one contiguous
     544 B run per subgroup-step against 8 strided 64 B rows plus a scale row --
     the 10-vs-17-message contrast `probe_gemv_loads.cl` measured for decode.
  4. The **`[N][K]` orientation is faster than `[K][N]`**, because the source
     tiles are N-tiled 16 columns wide and a lane owns one column: writing
     `[N][K]` makes each lane's 64 k-values one contiguous 128-byte run, while
     `[K][N]` makes them 64 scattered 2-byte stores at stride 2N. Predicted gap:
     large, not marginal. **Which orientation the GEMM needs is P2 step 3's
     answer**; this probe measures both so the answer costs no rework.

  ## The overhead this decides, and the two-model reconciliation

  Bytes WRITTEN per chunk over the 256 int4 matrices a chunk dequantises
  (48 GDN layers x {QkvZ, OutProj, GateUp, Down} + 16 FA x {Qkv, OProj, GateUp,
  Down}), at 2 B per bf16 element (DERIVED):

  | matrix | K x N | bytes | x layers | subtotal |
  |---|---|---|---|---|
  | QkvZ | 5120 x 16384 | 167.77 MB | 48 | 8053.1 MB |
  | OutProj | 6144 x 5120 | 62.91 MB | 48 | 3019.9 MB |
  | GateUp | 5120 x 34816 | 356.52 MB | 64 | 22817.0 MB |
  | Down | 17408 x 5120 | 178.26 MB | 64 | 11408.5 MB |
  | Qkv | 5120 x 14336 | 146.80 MB | 16 | 2348.8 MB |
  | OProj | 6144 x 5120 | 62.91 MB | 16 | 1006.6 MB |
  | **written** | | | | **48.65 GB** |
  | read (the int4 linears themselves) | | | | **~13.0 GB** |

  * **Model A, streaming writes:** 61.7 GB at 590 GB/s = **104 ms/chunk**.
    Against a 2.215 s C = 4096 chunk (P2's pre-registration) that is **4.7 %**.
  * **Model B, write-allocate** (the bf16 store pulls the line before writing
    it, so a write costs 2x its bytes): 110.3 GB = **187 ms** -- **8.4 %** at
    C = 4096 and **16.8 %** at C = 2048.

  **Model B is what spec §3.1 quotes** ("derived: 8-16 % of chunk time at
  C = 4096-2048") -- the two match to the digit, which is how the spec's figure
  is identified as a write-allocate model. **This probe is the reconciliation:
  its measured GB/s decides which model is right, and the report then replaces
  both with the measured number.** Spec §11's trigger stands: if the measured
  overhead exceeds 20 % the design revisits a second weight layout for prefill.

  Cross-check on the same arithmetic from the other side: spec §3.1 prices the
  gate||up matrix alone at "~1.2 ms at 590 GB/s". Model A gives
  (356.52 write + 94.70 read) MB / 590 GB/s = **0.765 ms**; Model B gives
  (2 x 356.52 + 94.70) / 590 = **1.37 ms**, and 2 x 356.52 / 590 = **1.21 ms**.
  So the spec's 1.2 ms is Model B with the int4 read omitted. One quantity,
  three arithmetics, one measurement to settle it.
  ```
  Commit: `docs(probe): pre-register P3 dequant predictions and the two overhead models`
- [ ] **Step 2 - write the failing test.** `tests/prefill/dequant_test.cc`
  (interfaces.md's name), registered with **no guard** - it needs a B70 and the
  compiled kernels, not the SYCL component:
  ```cpp
  // pf_dequant_tile: every int4 tile layout the loader produces, in both output
  // orientations, BIT-EXACT against common::Int4Gptq::at rounded once to bf16.
  // The same fp32 expression on both sides, so a differing bit is a bug and
  // never a tolerance question -- the discipline
  // tests/loader/dequant_fixture_test.cc established across languages, applied
  // here across devices.
  #include <cstdio>
  #include <vector>
  #include "check.h"
  #include "common/bf16.h"
  #include "common/int4.h"
  #include "dequant_harness.h"     // tests/prefill/dequant_harness.h: upload, launch, read back
  int main() {
    l0::Context ctx(0);
    l0::Queue q(ctx);
    l0::Fence f(q);
    // The test shape: K a multiple of 64 and N a multiple of 16, so both tile
    // layouts are exercised whole and no partial tile is silently skipped.
    const uint32_t K = 256, N = 64;
    common::Int4Gptq w = common::Int4Gptq::random(K, N, 7);
    std::vector<uint16_t> kn(size_t(K) * N), nk(size_t(K) * N);
    for (uint32_t k = 0; k < K; ++k)
      for (uint32_t n = 0; n < N; ++n) {
        const uint16_t v = common::f32_to_bf16(w.at(k, n));
        kn[size_t(k) * N + n] = v;
        nk[size_t(n) * K + k] = v;
      }
    for (uint32_t L = 0; L < 2; ++L)
      for (uint32_t T = 0; T < 2; ++T) {
        std::vector<uint16_t> got = run_dequant(ctx, q, f, w, K, N, L, T);
        const std::vector<uint16_t>& ref = T ? nk : kn;
        for (size_t i = 0; i < ref.size(); ++i) CHECK_EQ(got[i], ref[i]);
        std::printf("pf_dequant_tile L%u T%u: %zu elements bit-exact\n", L, T, ref.size());
      }
    return 0;
  }
  ```
  `tests/prefill/dequant_harness.h` holds `run_dequant`: it uploads `w.tiled()`
  for layout 1 or `w.qweight` + `w.scales` for layout 0, loads
  `pf_dequant_tile_K<K>_N<N>_L<L>_T<T>`, launches grid `(N/16, K/64)`, reads
  back. It is a header, not a copy of `gemv_harness.h`: this kernel has no
  split-K, no activations and no partials to fold.
- [ ] **Step 3 - run it and watch it fail.** `tools/box.sh test dequant_test` →
  the build fails at `add_dependencies(dequant_test kernel_pf_dequant_tile_…)`:
  the kernels do not exist. Record the error.
- [ ] **Step 4 - write `src/kernels/prefill/dequant.cl`.** One entry point;
  `K`, `N`, `LAYOUT`, `TRANSPOSED` are `-D`s; `reqd_work_group_size` declared,
  because T0's `launch` reads the group size from the kernel and throws on a
  kernel that declares none:
  ```c
  // dequant.cl -- pf_dequant_tile: one int4 linear's weights, in whichever tile
  // layout the loader chose for that shape, widened to bf16 into the reusable
  // prefill scratch (spec 2 §3.1).
  //
  // The dequant contract is src/kernels/gemv.cl's, unchanged: w = (q - 8) *
  // scale, symmetric group-64, the f16 scale widened to fp32, the product
  // rounded ONCE to bf16 -- the same expression common::Int4Gptq::at computes on
  // the host, which is why tests/prefill/dequant_test.cc holds this bit-exact
  // rather than to a bar. The xor/shift nibble extraction is gemv.cl's
  // GEMV_DEQ_SHIFT form and preserves the exact integer values.
  //
  // Compile-time: K, N, LAYOUT (0 GPTQ-native + separate scales, 1 tiled with
  // inline scales), TRANSPOSED (0 -> out is bf16 [K][N] row-major, ldb = N;
  // 1 -> out is bf16 [N][K], ldb = K). Which one the GEMM wants is P2 step 3's
  // measurement; both are built because they have different write patterns and
  // therefore different bandwidth, and that difference is one of P3's results.
  //
  // Grid: (N/16, K/64) work-groups of 16 lanes -- one subgroup per (n_tile,
  // k_group), i.e. exactly one 544-byte layout-1 tile per subgroup, the same
  // unit gemv.cl streams. Lane l owns output column n_tile*16 + l and writes
  // that column's 64 k-rows.
  //
  // This kernel is NOT on the decode path, binds no decode buffer, and is not
  // in src/runtime/capture.cc's walk: the 774/19 invariants are untouched.
  #pragma OPENCL EXTENSION cl_khr_fp16 : enable
  #pragma OPENCL EXTENSION cl_intel_subgroups : enable
  #pragma OPENCL EXTENSION cl_intel_subgroups_short : enable
  #define SG 16
  #define GROUP 64
  #define G (K / GROUP)
  #define TILE_U32 136   /* 128 u32 of nibbles + 8 u32 of scales = 544 B */

  // Round-to-nearest-even fp32 -> bf16. This MUST be the transcription of
  // common::f32_to_bf16 (src/common/bf16.h) and not a paraphrase: the
  // bit-exactness bar is the two functions being the same rounding.
  inline ushort rne_bf16(float f) { /* transcribe common/bf16.h here */ }

  __attribute__((intel_reqd_sub_group_size(SG)))
  __attribute__((reqd_work_group_size(SG, 1, 1)))
  __kernel void pf_dequant_tile(__global const uint* restrict w,
                                __global const half* restrict scales,  /* layout 0 only */
                                __global ushort* restrict out) {
    const uint lane = get_sub_group_local_id();
    const uint n_tile = get_group_id(0);
    const uint g = get_group_id(1);
    const uint n = n_tile * SG + lane;
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
    for (int j = 0; j < 8; ++j) {
      const uint u = wv[j] ^ 0x88888888u;
      for (int i = 0; i < 8; ++i) {
        const int qm8 = ((int)(u << (28 - 4 * i))) >> 28;   /* == q - 8, exactly */
        const uint k = g * GROUP + (uint)j * 8u + (uint)i;
        const ushort v = rne_bf16((float)qm8 * scale);
  #if TRANSPOSED
        out[(size_t)n * K + k] = v;          /* [N][K]: lane-contiguous along k */
  #else
        out[(size_t)k * N + n] = v;          /* [K][N]: subgroup-contiguous along n */
  #endif
      }
    }
  }
  ```
  **Read `common/bf16.h` and transcribe `f32_to_bf16` into `rne_bf16`
  literally** - the tie-breaking is the whole bar.
- [ ] **Step 5 - register the variants.** `src/kernels/prefill/CMakeLists.txt`:
  ```cmake
  # Prefill-only kernels (spec 2). None is bound by src/runtime/capture.cc and
  # none changes the decode list's 774 launches or 19 modules.
  set(PF_DEQUANT_CL ${CMAKE_CURRENT_SOURCE_DIR}/dequant.cl)
  function(add_pf_dequant K N L T)
    add_ocloc_kernel(pf_dequant_tile_K${K}_N${N}_L${L}_T${T} SOURCE ${PF_DEQUANT_CL}
                     DEFINES K=${K} N=${N} LAYOUT=${L} TRANSPOSED=${T})
  endfunction()
  # The unit-test shape, both layouts x both orientations.
  foreach(L 0 1)
    foreach(T 0 1)
      add_pf_dequant(256 64 ${L} ${T})
    endforeach()
  endforeach()
  # The six production shapes, both layouts (so P3 can REPORT the layout
  # contrast rather than assume it) x both orientations (P2 step 3 says which
  # the GEMM needs; P3 measures both).
  set(PF_SHAPES "5120 16384" "6144 5120" "5120 34816" "17408 5120" "5120 14336" "5120 248320")
  foreach(row IN LISTS PF_SHAPES)
    separate_arguments(f UNIX_COMMAND "${row}")
    list(GET f 0 K)
    list(GET f 1 N)
    foreach(L 0 1)
      foreach(T 0 1)
        add_pf_dequant(${K} ${N} ${L} ${T})
      endforeach()
    endforeach()
  endforeach()
  ```
  and `add_subdirectory(prefill)` as the last line of
  `src/kernels/CMakeLists.txt`. That is 4 + 24 = **28 new ocloc compiles**; at
  JOBS 44 they are parallel and cheap, but say so in the task report so the
  build-time change is not a surprise.
- [ ] **Step 6 - run the test.** `tools/box.sh test dequant_test` → green, all
  four (layout × orientation) combinations bit-exact. If a bit differs, the
  fault is `rne_bf16` or the tile index arithmetic; the failure message must
  print `k`, `n`, the source word, the nibble and the scale, exactly as
  `dequant_fixture_test.cc:47-55` does.
- [ ] **Step 7 - write `tools/probe/probe_dequant.cc`.** Plain g++, links
  `b70_l0` only - **no SYCL, which is what makes P3 independent of T0.** Per
  shape × layout × orientation: upload the source to device memory, allocate the
  **one reusable 356,515,840-byte scratch** (gate‖up's size, interfaces.md's
  "Layout conventions"), take the 8-replay/drop-3 median over a list of launches
  cycling `NB = max(2, 72 MB / source_bytes + 1)` source copies past the 24 MB
  L2 (`gemv_harness.h:64-70`'s rule), and report:
  ```
  | shape | K×N | L | orient | µs | read MB | write MB | GB/s (r+w) | GB/s (r+2w) | bit-exact |
  ```
  **Both GB/s columns, side by side** - that pair *is* the Model A / Model B
  reconciliation from Step 1, and printing one alone would recreate the
  two-values problem in a new form. Plus a ramp-control row and a drift control
  that re-measures the first shape at the end (`probe_gemv.cc:176-202`).
  Bit-exactness is re-checked here too, on a sampled 1 % of cells, so a
  correctness regression cannot hide behind a good bandwidth number.
- [ ] **Step 8 - run and record.**
  ```bash
  tools/box.sh run "ZE_AFFINITY_MASK=1 ./build/tools/probe/probe_dequant"
  ```
  Append to `docs/probe-dequant-2026-09-04.md`: the table, ramp and drift
  controls, box conditions, prediction-vs-measurement, **and the per-chunk
  overhead recomputed from the measured GB/s** at C = 4096 and C = 2048 against
  P2's measured GEMM time - or, if P2 has not landed, against the 2.215 s
  pre-registration, labelled as such. State explicitly whether spec §11's 20 %
  trigger fires.
- [ ] **Step 9 - answer the double-buffering question.** Spec §4's P3 row also
  asks "whether scratch reuse across layers needs double-buffering". Answer it
  from the measurement, in one paragraph: with one scratch, the dequant of
  matrix *i+1* cannot overlap the GEMM of matrix *i*; a second scratch costs
  another 356 MB of the 32 GB card (against weights 13.7 GB + KV 1.07 GB +
  activations ≤ 0.5 GB at C = 4096, spec §3.6, plus the 128 MB P5 may add) and
  could hide at most the measured dequant time per matrix. Give that number and
  the verdict. **It is a finding for L2, not work in this plan.**
- [ ] **Step 10 - commit.**
  `feat(prefill): pf_dequant_tile + P3 - <X> GB/s, dequant overhead <Y>% at C=4096`

---

## Task P4 - `tools/probe/probe_prefill_attn`: FMHA at head_dim 256, or the priced fallback

**Files:**
- Create: `tools/probe/probe_prefill_attn.cc`
- Modify: `src/sycl/CMakeLists.txt` - `b70_sycl_probe(probe_prefill_attn sycl_tla)`
- Create: `docs/probe-prefill-attn-2026-09-04.md`

**Interfaces:**
- **Consumes:** `runtime::prefill::Context`, `runtime::prefill::sycl()` (T0);
  the cache layout - `kv_k`/`kv_v` bf16 `[16 layers][max_len][4][256]`, per-FA-layer
  slice `kv_stride_ = max_len·4·256·2 B` (`src/runtime/buffers.h:74`,
  `src/runtime/capture.cc:200,520-521`); the oracle `tests/kernels/attn_ref.h`
  (`attn_ref::decode` `:222`, the block merge in `attn_ref::reduce` `:307`,
  `kScale = 0.0625f` = 1/√256 `:125`); and, as the configuration to
  instantiate, **`benchmarks/flash_attention/fmha_configuration.hpp`**'s generic
  `FMHAConfigGenWithTileShape<…, int WgTileQ, int WgTileK, int WgTileV,
  int SgTileQ, int SgTileK, int HeadDimQK, int HeadDimV>` (`:291-311`), which
  builds the three tile shapes from **arbitrary** head dims:
  ```cpp
  299:  using ShapeQK     = Shape<Int<WgTileQ>, Int<WgTileK>, Int<HeadDimQK>>;
  300:  using ShapePV     = Shape<Int<WgTileQ>, Int<WgTileV>, Int<WgTileK>>;
  301:  using ShapeOutput = Shape<Int<WgTileQ>, Int<HeadDimV>>;
  ```
- **Produces:** the inherit-or-own answer for prefill attention (spec §3.3), and
  either µs at C ∈ {1024, 2048, 4096} over depth 4096, or the exact
  instantiation error plus a one-page price for our own flash kernel.

**Two facts established from source before this task starts, which change what
it has to do:**

1. **There is no head_dim static_assert to fight.** `examples/06_bmg_flash_attention/06_xe_fmha_fwd.cpp`
   has an `#if HEAD_DIM ==` chain over **64 / 96 / 128 / 192 only** (`:96-133`,
   the BMG prefill 128 case at `:115-118`: `ShapeQK = Shape<_256,_32,_32>`,
   `ShapePV = Shape<_256,_32,_32>`, `ShapeOut = Shape<_256,_128>`,
   `SubgroupLayoutQK = Layout<Shape<_16,_1,_1>>`) and **no `#else`** - so
   `HEAD_DIM=256` there yields four "unknown type name" errors, not a
   static_assert. `FMHAConfigGenWithTileShape` sidesteps the chain entirely and
   **needs no change to `applications/` at all.** Head dim is *also* a runtime
   field (`FMHAProblemShape::head_size_qk/head_size_vo`,
   `kernel/xe_fmha_fwd_kernel.hpp:51-58`), but it is the tile shapes that decide
   codegen.
2. **Our cache needs no relayout - only custom strides.** The kernel's tensors
   are `shape_K = (seq_len_kv, head_size_qk, num_heads_kv, batch)` with
   `StrideK = Stride<int, _1, int, int>` and
   `shape_V = (head_size_vo, seq_len_kv, num_heads_kv, batch)` with
   `StrideV = Stride<_1, int, int, int>` (`xe_fmha_fwd_runner.hpp:790-795`,
   `:1483-1486`) - **in both, the head-dim mode has stride 1.** Our per-layer
   slice `[pos][4][256]` gives, for kv-head `h`: base `+ h·256`, stride 1 along
   d, stride 1024 along pos, 256 between heads. That is exactly the shape of
   those strides with `ldk = 1024` instead of a packed `256`.
   `make_cute_packed_stride` is a convenience, not a requirement - the Params
   take the strides. **So `attn_chunk` can read `kv_k`/`kv_v` in place.** P4's
   first correctness job is to prove that claim.

- [ ] **Step 1 - pre-register, and commit alone.** Create
  `docs/probe-prefill-attn-2026-09-04.md`:
  ```markdown
  # P4 -- sycl-tla FMHA forward at head_dim 256 (spec 2 §4)

  grade: iterate (card 1, ZE_AFFINITY_MASK=1, card 0 may be held)
  sycl-tla: pin 2db1b7c94cadc52decf0013bd8b6e244fcb37dd4 (v0.9.2+9, CUTLASS 4.2.1)

  ## The shape, which is the whole question

  head_dim **256**, 24 q-heads / 4 kv-heads (GQA 6:1), causal, bf16 K/V read
  from OUR cache -- `[pos][4][256]` bf16 per layer. Intel's shipped BMG prefill
  example is head_dim **128**. Doubling the head dim doubles `VTiles`, and
  `VTiles` is the multiplier on the O accumulator held in registers across the
  whole KV loop (`collective/xe_fmha_fwd_mainloop.hpp:227-228`,
  `FragA = expand_sg_fragment_t<SingleFragA, 1, VTiles>`, with the GEMM-2 loop
  fully unrolled over it at `:781-793`). **So the expected failure mode is
  register spill, not a wrong answer and not an assert.**

  ## Pre-registered predictions (written 2026-09-04, BEFORE any measurement)

  1. **It instantiates**, via `FMHAConfigGenWithTileShape` with
     `HeadDimQK = HeadDimV = 256`. Confidence: medium-high, now that the generic
     path is known to exist. The three asserts that can fire are known and are
     avoided by construction (see below).
  2. **The recommended first config keeps `VTiles = 4`, the same as hdim 128**:
     `WgTileQ = 128, WgTileK = 32, WgTileV = 64, HeadDimQK = HeadDimV = 256`
     -> `ShapeQK = <_128,_32,_256>`, `ShapePV = <_128,_64,_32>`,
     `ShapeOut = <_128,_256>`, `VTiles = 256/64 = 4`. A naive doubling of the
     128 config (`ShapePV = <_256,_32,_32>`, `ShapeOut = <_256,_256>`) gives
     `VTiles = 8` and twice the accumulator on a 256-GRF budget; it is measured
     as a second cell, not as the first.
  3. Causal FMHA over depth 4096 at C = 4096 costs **< 3.2 % of the chunk**
     (`explorer-2 §5(b)`: attention is 3.2 % of the FLOPs of a 4096-deep prefill
     tail chunk, independent of C, and FMHA shares the GEMM's mainloop). Against
     P2's 2.215 s pre-registration: **< 71 ms for all 16 layers**, i.e.
     **< 4.4 ms per layer per chunk**.
  4. **Our cache is readable in place** with `ldk = ldv_seq = 1024` and no copy.
  5. Correctness within **max rel err 1e-2, mean rel err 1e-3** of
     `tests/kernels/attn_ref.h` on elements with |ref| >= 1e-3 * max|ref|.
     Basis: the mainloop downcasts the softmax probabilities to bf16 before the
     PV DPAS (`xe_fmha_fwd_mainloop.hpp:761-762` asserts the f32x2->bf16x2 pack),
     a 2^-9 = 1.95e-3 half-ulp relative perturbation on the weights, and it uses
     an exp2-based online softmax where our reference uses `exp`. 1e-2 is ~5x
     headroom on the dominant term. **This is a structural bar, not a numerics
     contract**; spec §6's real bars are token-level and belong to L1/L3.

  ## The asserts this configuration must not trip (all quoted, with lines)

  | file:line | assert | how the Step-2 config satisfies it |
  |---|---|---|
  | `collective/xe_fmha_fwd_mainloop.hpp:162-165` | `(VTiles * get<1>(TileShapePV)) % get<2>(TileShapeQK) == 0` -- "Head size ... must be divisible by the QK K-tile (BLK_QK_D)" | 4*64 = 256, BLK_QK_D = 256 -> 0 |
  | `benchmarks/flash_attention/fmha_configuration.hpp:115-116` (and `xe_fmha_fwd_runner.hpp:1549-1550`) | `get<0>(TileShapeOutput) == get<0>(TileShapePV)` -- output and PV tiles must agree in Q | both `_128` |
  | `collective/xe_fmha_fwd_mainloop.hpp:958-959` | `size(SGLayoutPV) == size(SGLayoutQK)` -- "Q*K cannot be parallelized in the head size dimension" | keep the D mode of `SubgroupLayoutQK` at `_1`, as every shipped config does; `get_sg_layout_pv` (`:931-941`) maps `(sgQ,sgK,sgD) -> (sgQ,1,sgK)` |
  | `collective/xe_fmha_fwd_mainloop.hpp:761-762` | `tArP` per-work-item element count must be even (f32x2->bf16x2 pack) | bf16 path; a function of the subgroup layout -- if it fires, halve `SgTileK` |
  | `benchmarks/flash_attention/fmha_configuration.hpp:304-305` | `WgTileQ % SgTileQ == 0`, `WgTileK % SgTileK == 0` | pick `SgTileQ`, `SgTileK` as divisors |
  | `fmha_configuration.hpp:73-75` / `06_xe_fmha_fwd.cpp:1591` | `!(Persistent & Causal)` | **non-persistent scheduler**; ours is causal |

  Not an assert but a silent slow path: `kSumDivVT = (kSumSize % VTiles == 0)`
  (`xe_fmha_fwd_mainloop.hpp:756`) selects a faster row-sum reduction at
  `:795-805`. The report states which branch the chosen config takes.
  ```
  Commit: `docs(probe): pre-register P4 FMHA-at-head_dim-256 config and asserts`
- [ ] **Step 2 - instantiate at head_dim 256, and BUILD IT BEFORE writing any
  timing code. This is the task's fork.** `tools/probe/probe_prefill_attn.cc`
  starts as the generic config plus a launcher:
  ```cpp
  // P4: does sycl-tla's FMHA forward instantiate at head_dim 256, and what does
  // it cost? Everything here reads OUR cache layout (bf16 [pos][4][256] per
  // layer, src/runtime/buffers.h:74) through custom strides -- no copy, no
  // relayout. src/kernels/attn.cl is untouched and nothing here is a shipped
  // code path.
  //
  // The configuration is benchmarks/flash_attention/fmha_configuration.hpp's
  // generic FMHAConfigGenWithTileShape (:291-311), NOT the example's
  // `#if HEAD_DIM ==` chain (06_xe_fmha_fwd.cpp:96-133), which enumerates
  // 64/96/128/192 with no #else and therefore does not compile at 256 at all.
  #include "fmha_configuration.hpp"     // sycl-tla/benchmarks/flash_attention
  #include "tla_pin.h"
  // VTiles = HeadDimV / WgTileV = 256 / 64 = 4 -- the same register multiplier
  // the shipped hdim-128 config runs at. See the probe doc's assert table.
  using FMHA256 = FMHAConfigGenWithTileShape</*Causal=*/true, /*BlockScale=*/false,
      /*WgTileQ=*/128, /*WgTileK=*/32, /*WgTileV=*/64,
      /*SgTileQ=*/16, /*SgTileK=*/32, /*HeadDimQK=*/256, /*HeadDimV=*/256>;
  ```
  **Two things must be solved here and both are concrete:**
  - the exact template parameter list and order of `FMHAConfigGenWithTileShape`
    (the comment above names the shape, not the order) - read it:
    ```bash
    tools/box.sh run "sed -n '280,320p' ~/PycharmProjects/sycl-tla/benchmarks/flash_attention/fmha_configuration.hpp"
    ```
    and record it verbatim in the report;
  - **the launch takes no queue.** FMHA does not go through
    `GemmUniversalAdapter` - `xe_fmha_fwd_runner.hpp:937-938` says so - and its
    launcher (`static void run(Params)`, `:939-972`) calls
    `compat::experimental::launch<cutlass::device_kernel<FMHAKernel>, …>` on the
    **default** queue with `kernel_properties{sub_group_size<cute::intel::sg_size>,
    intelex::grf_size<256>}` (`:957-964`). So the probe calls
    `compat::set_default_queue(runtime::prefill::sycl(cx))`
    (`include/cute/util/compat/device.hpp:899`) once at start-up, and **prints
    that it did** - a measurement on a different queue than the engine's is a
    different measurement. Host argument struct at `:999-1022`,
    `can_implement` at `:1029`, `get_workspace_size` returns 0 unconditionally
    (`kernel/xe_fmha_fwd_kernel.hpp:191`), `to_underlying_arguments` at `:1040`.
  Build it. Record the outcome verbatim: on success the compile wall time and
  whether IGC printed a spill warning (`-Xs "-options -print-reg-usage"` is the
  way to see it, and the probe doc records the register count); on failure the
  full first diagnostic with its `sycl-tla` file:line.
- [ ] **Step 3a - IF IT INSTANTIATES: prove the in-place cache read.** Build a
  synthetic cache exactly as `probe_attn.cc` does
  (`tools/probe/probe_attn.cc:20-33`): random bf16 in a model-like range, and
  **rotate over `--kvsets` independent caches** so back-to-back launches do not
  run out of a 24 MB L2 the engine never has (one FA layer's live KV at depth
  4096 is 16.8 MB; in situ ~45 other kernels read ~1.8 GB between two attention
  launches). Then, at C = 64, depth 4096:
  - allocate `kv_k`/`kv_v` in **our** layout, `[4096+C][4][256]` bf16;
  - hand the kernel `ptr_K = kv_k + h·256`, `StrideK = {1024, 1, 256, 0}` and
    the matching V strides, GQA `num_heads_q = 24`, `num_heads_kv = 4`, batch 1,
    one `cumulative_seqlen` pair `{0, C}` (`collective/fmha_fusion.hpp:40-48`:
    `VariableLength{max_length, cumulative_length}` is a plain exclusive prefix
    sum of length batch+1), `seq_len_kv_cache` covering the 4096 prior
    positions, **paged KV off** (page size = the whole cache is expressible as
    "not paged" and is the simpler instantiation; spec §3.3's page-size remark
    is satisfied trivially);
  - compare against a **second run over a packed `[pos][256]` copy of the same
    values** with `ldk = 256`. The two must be **bitwise identical**: same
    values, same order, only the stride differs. That is the in-place claim,
    proved by construction rather than argued.
- [ ] **Step 3b - IF IT INSTANTIATES: correctness against our reference.** For
  C ∈ {64, 256} over depth 4096:
  - reference: `attn_ref::decode(pos = 0, n_act = C, M = C, max_len, attn_q_fp32,
    kv_k, kv_v, attn_part)` - at `pos = 0` its `p > pos + m` test **is** the
    causal mask over the chunk - then merge blocks in ascending order as
    `attn_ref::reduce` does (`:316-325`) and take `acc[d] / sm`, **stopping
    before the output gate**, because `attn_chunk` in interfaces.md returns
    `out` pre-gate;
  - budget it honestly in the probe's header: at C = 256 and depth 4096
    `attn_part` is 24 × 64 × 256 × 258 × 4 B = **406 MB** host-side and the
    reference is scalar - **minutes, not seconds**;
  - report max and mean relative error, the count of elements outside Step 1's
    bar, and which `kSumDivVT` branch the config took.
- [ ] **Step 3c - IF IT INSTANTIATES: timing.** µs per layer per chunk at
  C ∈ {1024, 2048, 4096} over depth 4096, 8-replay/drop-3 after a discarded
  warm-up, for **both** the `VTiles = 4` config and the naive `VTiles = 8`
  doubling (prediction 2's control). Table:
  `| config | VTiles | C | depth | µs/launch | ms × 16 layers | % of P2's chunk | vs the <3.2% prediction |`
- [ ] **Step 4 - IF IT DOES NOT INSTANTIATE: this is the deliverable, and it is
  complete.** Record in `docs/probe-prefill-attn-2026-09-04.md`:
  1. **The exact error** - the full first diagnostic, the `sycl-tla` file:line
     that produced it, and the `static_assert` text if there is one (Step 1's
     table says which are possible and what to change first: `SgTileK` for the
     `tArP`-even assert, `WgTileV` for the divisibility one).
  2. **One page pricing our own flash kernel**, scaled from the measured
     head_dim-128 configuration to 256, arithmetic written out:
     - **Register budget.** BMG in 256-GRF mode - which is what Intel's own
       kernels ask for (`vllm-xpu-kernels/cmake/utils.cmake:607`,
       `-internal_options -cl-intel-256-GRF-per-thread`; FMHA sets
       `grf_size<256>` at launch, `xe_fmha_fwd_runner.hpp:962`) - gives
       256 × 32 B / 16 lanes = **512 B = 128 fp32 per lane** at SIMD16. One
       query row's online-softmax accumulator at head_dim 256 is 256 fp32 plus
       `(mx, sm)`, so **one lane cannot hold a row**: the accumulator must be
       spread across the subgroup, 16 fp32 per lane over 16 lanes - which is
       exactly what `attn.cl` already does (`attn_ref.h:124`,
       `kLanes = 16, kPerLane = 16`). State the resulting tile - `Q_TILE` query
       rows per work-group × 256 accumulator dims spread over the subgroups -
       and solve for the largest `Q_TILE` that fits 128 fp32/lane with room for
       the K/V staging and the score block.
     - **SLM budget.** 128 KB per work-group (`docs/01-hardware.md:16`). An S×S
       bf16 score block at S = 256 is **exactly 131,072 B** - the entire budget
       for one head (`explorer-2 §5`) - so the score block is tiled and
       streamed, never materialised. State the `(Q_TILE × KV_TILE)` block chosen
       and its SLM cost, plus the K/V staging buffers. Note that Intel's own
       FMHA carries **no SLM at all** (`struct SharedStorage {};`,
       `xe_fmha_fwd_mainloop.hpp:285` per `explorer-3 §4`) and keeps P in
       registers - a design we can copy rather than invent.
     - **What transfers for free.** Task 5's register-packed GQA (`greg6`,
       `tools/probe/CMakeLists.txt:81-83`, measured 1.065 ms/token in situ,
       `docs/12-kernels.md:2289`): one K/V walk serves all six q-heads of a
       kv-head, dividing KV traffic by six at GQA 6:1. `intel_sub_group_2d_block_read_*`
       (verified compiling for BMG, `docs/12` `gemv` §) and
       `intel_sub_group_bf16_bf16_matrix_mad_k16` (verified to lower to a real
       `dpas.8x8`, `explorer-3 §1`) are both reachable from OpenCL C.
     - **The price**, in spec §3.3's form: engineer-days, plus the expected
       µs/layer/chunk from a traffic model -
       `bytes = C·24·256·2 (q) + depth·4·256·2·2 (k+v, read once per q-tile
       group); time ≥ bytes / 590 GB/s` - beside the FLOP bound
       `2·2·C·depth·24·256 / <P2's measured TFLOP/s>`. Say which binds.
  3. **The recommendation**: ours in SYCL on `sycl-tla`'s copy/MMA atoms
     (head_dim is a template parameter there) or in OpenCL C - pick one, say
     why, per spec §3.3's fallback bullet.
- [ ] **Step 5 - record and commit** (either branch).
  `feat(probe): P4 - FMHA at head_dim 256 <instantiates: X µs/layer | does not: priced fallback>`

---

## Task P5 - `tools/probe/probe_gdn_chunk`: Intel's CuTe chunked gated delta rule against our state

**Files:**
- Create: `tools/probe/probe_gdn_chunk.cc`
- Modify: `src/sycl/CMakeLists.txt` - the `probe_gdn_chunk` block (given in Step 2)
- Create: `docs/probe-gdn-chunk-2026-09-04.md`

**Interfaces:**
- **Consumes:** `runtime::prefill::Context`, `runtime::prefill::sycl()` (T0);
  our state layouts - `gdn_state` fp32 `[48 layers][48 v-heads][128 k][128 v]`,
  **k-major rows and v columns** (`src/kernels/gdn_step.cl:20`,
  `src/runtime/buffers.h:73`), `conv_ring` bf16 `[48][16 slots][10240]`
  **slot-major, holding the RAW pre-conv qkv values** (`gdn_step.cl:22` and the
  ring-ownership argument at `:75-95`), the fused `qkv‖z` row `[M][16384]` with
  `q` at flat channel 0, `k` at 2048, `v` at 4096
  (`tests/kernels/gdn_ref.h:93`) and `z` at 10240, and `ab_out` fp32 `[M][128]`
  with `a` at `[0,48)`, `b` at `[48,96)` (`gdn_ref.h:96-97`); the CPU oracle
  `gdn_ref::step` (`tests/kernels/gdn_ref.h:113`).
- **Produces:** the inherit-or-own answer for GDN (spec §3.4), **the numerics
  band spec §6 must be written for**, and µs per layer per chunk.

**What the kernel wants - read out of the source, so the layout comparison is a
fact and not a guess.** Target:
`~/PycharmProjects/vllm-xpu-kernels/csrc/xpu/gdn_attn/xe_2/chunk_gated_delta_rule_kernels_xe2.hpp`
at `a397c58eb7781e6fe0d6b3fb7c25d21b5f658784` on `main`.

| their side | file:line | ours | fit |
|---|---|---|---|
| `gdn::kernel_launcher<T, StateT>(sycl::queue&, T* core_attn_out, const T* q, const T* k, const T* v, T* A, T* w, T* u, const float* b, float* a, const float* A_log, const T* dt_bias, StateT* ssm_state, int ssm_state_stride_0, const int* query_start_loc, const int* cache_indices, const bool* has_initial_state, const int* token_indx, int batch_size, int total_virtual_seqlen, int num_k_heads, int head_k_dim, int num_v_heads, int head_v_dim)` - raw USM pointers, **torch-free except one call** | `:1251-1276`; the one torch call is `vllm::xpu::is_bmg()` at `:1351` (`csrc/utils.h:64-65`) | - | callable from a plain SYCL host program once that predicate is provided |
| five `queue.submit`/`parallel_for` back to back with **no explicit dependencies** - correctness requires an **in-order** queue | `:1297, 1330, 1370, 1398, 1433, 1477` | `Context`'s SYCL queue is `property::queue::in_order` (T0) | ✓ by construction |
| BMG takes the *optimised* inverse kernel; PVC the native one, because "PVC has acc issue of sycl tla" | branch at `:1351`, comment `:1386-1388` | B70 = BMG | ✓ - and it means the BMG path is the tested one |
| `q`, `k` `[total_virtual_seqlen][num_k_heads][head_k_dim]`; `v` `[T'][num_v_heads][head_v_dim]`; out `[total_seqlen][num_v_heads][head_v_dim]`; strides `num_k_heads·head_k_dim` outer, 1 inner | `:1506-1524`, `:1034-1040` | our `qkv‖z` is one fused `[C][16384]` row, q/k/v at flat channel 0 / 2048 / 4096 - and 2048 = 16×128, 6144 = 48×128, so the head/dim structure is **already there**, only the row stride differs (16384 vs 2048/6144) | a **strided gather**, not a relayout; Step 3 copies into their stride and measures the copy |
| `b`, `a` fp32 **`[num_v_heads][T']` - transposed, head-major**; the sigmoid is already folded into `b`; **`a` is mutated in place** | `:1516-1517`, `:1262`, `:82-125` | our `ab_out` fp32 `[C][128]`, `a` at `[0,48)`, `b` at `[48,96)`, token-major | a 48×C transpose plus `sigmoid(b)` per layer per chunk - 48·C fp32 = **786 KB at C = 4096**, negligible |
| `ssm_state` **`[cache_batch][num_v_heads][head_v_dim][head_k_dim]` - V-major then K**; only `stride(0)` is read | `:1519-1520`, tile `:1018-1022`, base `:962-965`, stride `:1539` | ours is `[48][128 k][128 v]` - **K-major then V** | **TRANSPOSED.** A 128×128 fp32 transpose per (layer, v-head): 3.15 MB per layer, **151 MB for all 48** - exactly the "O(state) relayout at chunk boundaries" spec §3.4 permits, ~0.5 ms of traffic per chunk for every layer at 590 GB/s (derived) |
| `ssm_state` dtype may be fp32, fp16 or bf16 (`DISPATCH_STATE_DTYPE`); accumulation is fp32 in registers regardless | `:1599-1616` | ours is fp32 | ✓ - instantiate `StateT = float`, which is spec §3.4's "fp32 throughout" |
| chunk size **64, compile-time `constexpr`**, and structurally baked (4 unrolled 16×16 diagonal blocks `:446`, all ten lower-triangular sub-blocks enumerated `:497-509`); the header only compiles with `-DVLLM_XPU_ENABLE_XE2` | `csrc/xpu/gdn_attn/gdn_attn_utils.h:7-9` | `fla`'s `FLA_CHUNK_SIZE = 64` (`explorer-5 §2`) - the same number the reference vLLM runs | ✓ |
| head dims are runtime ints but **must be multiples of 64** - unchecked loops `head_v_dim / chunk_size` at `:1111, 1183, 1215` and `head_k_dim / chunk_size` at `:864, 1184`; the only shape check is `num_v_heads % num_k_heads == 0` at `:1541` | | 128 / 128; 48 % 16 = 0 | ✓, and the probe asserts both itself since the kernel will not |
| q/k/v are **chunk-padded**: token at `pre_chunks·64 + token_in_seq`; `padding_size = batch·(chunk−1)` | `:1558`; the layout is spelled out in `tests/gdn_attn/test_gated_delta_rule.py:73-77` | batch 1, C a multiple of 64 → 63 pad rows | ✓, cheap |
| conv1d and the q/k l2norm are applied **beforehand**, by separate kernels | `gdn_attn_interface.cpp:465, 490, 517-518`; the q-scale is `rsqrt(head_k_dim)` folded into q only, `xe_2/l2norm_kernel.hpp:35,53` | ours are **inside** `gdn_step`; our q-scale is `1/√128` in fp32 (`gdn_ref.h:98`) - the same value | the batched conv1d is ours either way (spec §3.4 bullet 1); the probe applies conv + l2norm on the host so the measurement is **the delta rule alone**, and says so |
| the caller allocates scratch `A` `[num_v_heads][T'+pad][64]`, `w` `[…][head_k_dim]`, `u` `[…][head_v_dim]` in the io dtype - the torch entry does it with `torch::zeros` | `:1560-1568` | - | at C = 4096, bf16: 48·4159·(64+128+128)·2 B = **128 MB** of extra scratch. **Add it to the C = 4096 memory budget** (spec §3.6) |
| `has_initial_state` may be `nullptr` at the torch layer but the device code **dereferences it unconditionally** | `:1588-1590` vs `:763, :950` | - | the probe always passes a real `bool[1]` |
| `chunk_prepare_kernel` **destroys `a` in place** (rewrites it to the per-chunk cumsum of `softplus(a+dt_bias)·−exp(A_log)`) | `:82-125` | - | the probe re-materialises `a` before **every** replay, or its 8 replays measure 8 different inputs |
| C++**20**; defines `-DVLLM_XPU_ENABLE_XE2` plus the CUTLASS set; link `-flink-huge-device-code` and the SPIR-V ext list | `vllm-xpu-kernels/CMakeLists.txt:260-296, 386-391`; `cmake/utils.cmake:580, 607` | T0's sub-build is C++20 and already sets `CUTLASS_ENABLE_SYCL`/`SYCL_INTEL_TARGET` and the SPIR-V ext list | this task adds `VLLM_XPU_ENABLE_XE2`, `-flink-huge-device-code` and the three include roots |
| `gemm.hpp` (the device GEMM helpers it includes) **has no `#pragma once`** | `csrc/xpu/gdn_attn/xe_2/gemm.hpp` | - | include it exactly once per TU; a second include is a redefinition error, not a warning |

**Pin risk, owned here.** T0 pins `sycl-tla` at `2db1b7c9` (v0.9.2+9, CUTLASS
4.2.1); `vllm-xpu-kernels` pins `87f6850680a580654b9ea2c80dbc01aeb36ad231`
(`CMakeLists.txt:323-325`). The header needs `cute/tensor.hpp`, `XE_DPAS_TT`,
`TiledMMAHelper`, `cutlass/kernel_hardware_info.h`,
`cutlass/platform/platform.h`, `cutlass/tensor_ref.h` and four
`cutlass/util/*` headers. If it does not compile against the pin, **the
documented retry is a second checkout at vLLM's revision**, configured with
`-DB70_SYCL_TLA_SRC_DIR=<that checkout>`; the report says which revision built
it, and T6 records whether the project needs two pins or one.

- [ ] **Step 1 - pre-register, and commit alone.** Create
  `docs/probe-gdn-chunk-2026-09-04.md` containing **the layout table above**
  (source-derived, so it belongs in the pre-registration, not the results) plus:
  ```markdown
  # P5 -- Intel's CuTe chunked gated delta rule vs our state (spec 2 §4)

  grade: iterate (card 1, ZE_AFFINITY_MASK=1, card 0 may be held)
  kernel: vllm-xpu-kernels a397c58eb7781e6fe0d6b3fb7c25d21b5f658784, main
  sycl-tla: pin 2db1b7c94cadc52decf0013bd8b6e244fcb37dd4 (retry: 87f68506... )

  ## Pre-registered predictions (written 2026-09-04, BEFORE any measurement)

  1. **It builds** against our toolchain once `vllm::xpu::is_bmg()` is provided
     and the three scratch buffers are allocated by the caller. Confidence:
     medium-high -- `kernel_launcher` (:1252-1276) is torch-free apart from that
     one predicate.
  2. **The state layout does NOT match and needs a 128x128 transpose per
     (layer, v-head)** -- theirs `[v][k]` (:1018-1022), ours `[k][v]`
     (gdn_step.cl:20). Stated as a prediction because it is read from source and
     must be confirmed by a measurement that reproduces the reference THROUGH
     the relayout, in both directions.
  3. **Max relative difference of `gdn_state` after 4096 positions against
     `gdn_ref::step` run 4096 times: <= 5e-2; mean <= 5e-3.** Basis: the chunked
     WY form is algebraically equal but differently rounded, the state is fp32
     on both sides, and 64 chunk-boundary carries at C = 4096 compound. There is
     **no prior measurement of this quantity anywhere in this project** -- it is
     the band spec §6 must be written for and the single most valuable thing P5
     produces.
  4. FLOPs are negligible -- the recurrence is 5.5 MFLOP/token against 772
     MFLOP for the layer's GEMMs (spec §3.4) -- so **µs per layer per chunk
     scales linearly in C and stays under 2 % of the chunk** at every
     C in {1024, 2048, 4096}.
  5. The relayout is not the bottleneck: 151 MB of state transposed per chunk is
     ~0.5 ms at 590 GB/s against a 2.2 s chunk (derived, 0.02 %).
  ```
  Commit: `docs(probe): pre-register P5 CuTe-GDN layout map and numerics band`
- [ ] **Step 2 - make it build, in isolation, before any correctness work. This
  is the task's fork.** Add to `src/sycl/CMakeLists.txt`:
  ```cmake
  # P5: Intel's CuTe chunked gated delta rule (vllm-xpu-kernels, a397c58).
  # Consumed HEADER-ONLY from the operator's read-only checkout; nothing in that
  # tree is written. VLLM_XPU_ENABLE_XE2 is what makes chunk_size_xe2 = 64 exist
  # at all (gdn_attn_utils.h:7-9). -flink-huge-device-code is what Intel's own
  # build uses for these kernels (vllm-xpu-kernels/CMakeLists.txt:288); the
  # SPIR-V extension list is already in B70_SYCL_LINK. The three include roots
  # mirror add_xe2_kernel_library (vllm-xpu-kernels/cmake/utils.cmake:569-577).
  set(B70_VLLM_XPU_KERNELS_DIR "$ENV{HOME}/PycharmProjects/vllm-xpu-kernels"
      CACHE PATH "read-only checkout supplying the CuTe chunked-GDN kernel")
  b70_sycl_probe(probe_gdn_chunk sycl_tla)
  target_include_directories(probe_gdn_chunk SYSTEM PRIVATE
    ${B70_VLLM_XPU_KERNELS_DIR}
    ${B70_VLLM_XPU_KERNELS_DIR}/csrc/xpu/gdn_attn
    ${B70_VLLM_XPU_KERNELS_DIR}/csrc/xpu/gdn_attn/xe_2)
  target_compile_definitions(probe_gdn_chunk PRIVATE VLLM_XPU_ENABLE_XE2)
  target_link_options(probe_gdn_chunk PRIVATE -flink-huge-device-code)
  ```
  and write `tools/probe/probe_gdn_chunk.cc`'s **first** version as nothing but
  a shim that satisfies the header's host-side dependency and calls
  `gdn::kernel_launcher` once with zeroed buffers:
  ```cpp
  // P5: Intel's CuTe chunked gated delta rule against OUR gdn_state /
  // conv_ring. src/kernels/gdn_step.cl is untouched and nothing here is a
  // shipped code path.
  //
  // The header includes <torch/all.h> and "csrc/utils.h" for exactly two
  // things: the torch-level entry chunk_gated_delta_rule_impl_xe2 (:1506, which
  // we never call) and vllm::xpu::is_bmg() at :1351. The plan is to provide the
  // predicate and NOT pull in torch. If the header cannot be compiled without
  // torch, THAT is this task's blocker and Step 5 is the deliverable.
  //
  // gemm.hpp, which it includes, has no #pragma once -- include this header
  // exactly once in exactly one translation unit.
  namespace vllm::xpu { inline bool is_bmg() { return true; } }   // B70 = bmg-g31
  #include "chunk_gated_delta_rule_kernels_xe2.hpp"
  #include "tla_pin.h"
  ```
  **Build it.** Three possible outcomes, all recorded verbatim:
  (a) it compiles - record the wall time, binary size and any IGC spill warning;
  (b) it fails only on `torch/all.h` or `csrc/utils.h` - record the diagnostic,
  then try the minimal shim (a stub `csrc/utils.h` on our own include path
  ahead of theirs, providing only `vllm::xpu::is_bmg()` and
  `vllmGetQueue()`) and record whether that clears it;
  (c) it fails inside `cute`/`cutlass` - record the diagnostic, then **retry
  against vLLM's own `sycl-tla` pin** (`-DB70_SYCL_TLA_SRC_DIR=<checkout at
  87f6850680a580654b9ea2c80dbc01aeb36ad231>`) and record which revision built
  it. Do not modify either read-only checkout.
- [ ] **Step 3 - IF IT BUILDS: wire our layouts to theirs and prove the
  relayout.** In the probe:
  - allocate **our-shaped** inputs: `qkvz` fp32 `[C][16384]` (the GEMV's
    partials shape, `gdn_ref.h:93`), `ab` fp32 `[C][128]`, `conv_ring` bf16
    `[16][10240]`, `gdn_state` fp32 `[48][128][128]`, all xorshift-seeded in a
    model-like range;
  - **on the host**, apply exactly the prologue `gdn_step` applies, by calling
    into `gdn_ref.h`'s own code paths so there is no second implementation of
    it: conv1d + SiLU, the q/k l2norm, the `1/√128` q-scale, and
    `beta = sigmoid(b)`. Leave `a` **raw** - their `chunk_prepare_kernel` does
    `softplus(a + dt_bias)` and the `−exp(A_log)` cumsum itself (`:82-125`), so
    pre-applying it would double it;
  - build their inputs: q/k `[C+63][16][128]`, v `[C+63][48][128]`, b/a fp32
    `[48][C+63]` transposed, state `[1][48][128][128]` **transposed from ours**,
    plus the `A`/`w`/`u` scratch, zeroed;
  - assert, in the probe, the two shape invariants the kernel does not check:
    `head_k_dim % 64 == 0`, `head_v_dim % 64 == 0`;
  - call
    `gdn::kernel_launcher<sycl::ext::oneapi::bfloat16, float>(runtime::prefill::sycl(cx), …)`
    with `batch_size = 1`, `query_start_loc = {0, C}`, `cache_indices = {0}`,
    `has_initial_state = {true}`, `token_indx = nullptr`,
    `ssm_state_stride_0 = 48·128·128`;
  - transpose the state back and compare. **Also run the transpose pair alone
    (ours → theirs → ours) and assert it is bit-exact** - an fp32 transpose must
    be, and it separates a relayout bug from a numerics finding.
- [ ] **Step 4 - IF IT BUILDS: correctness, and the numerics band.** The oracle
  is `gdn_ref::step` (`tests/kernels/gdn_ref.h:113`) called **4096 times, one
  position per call, `M = 1`**, carrying `conv_ring` and `state` forward - that
  is the bit-exact model of what decode does, and it is the self-consistency
  oracle spec §3.7 names for S2. Then:
  - run the chunked kernel over the same 4096 positions **in 64 chunks of 64**,
    carrying state across, so every chunk boundary is crossed (a probe-scale
    mirror of spec §6.2's `C = 1024` multi-chunk gate);
  - report, after 4096 positions: **max and mean relative difference of
    `gdn_state`** over all 48 × 128 × 128 fp32 elements, and the same for the
    per-position output `y`;
  - report the band **after 64, 256, 1024 and 4096 positions**, so the *growth*
    is visible - a band that grows linearly in chunk count is a different
    finding from one that plateaus, and spec §6.3 needs to know which;
  - the reference is scalar and does ≈ 2.4 M fp ops per position over 48 heads:
    **budget minutes for 4096 positions** and say so in the probe's header.
- [ ] **Step 5 - IF IT BUILDS: timing.** µs per layer per chunk at
  C ∈ {1024, 2048, 4096}, 8-replay/drop-3 after a discarded warm-up, with `a`
  **re-materialised before every replay** (it is destroyed in place, `:82-125`).
  Table: `| C | µs/layer | ms × 48 layers | % of P2's chunk | relayout µs | scratch MB |`.
  State in the report that this **excludes** the conv1d and the l2norm - they
  are separate kernels on their side and ours to write either way (spec §3.4
  bullet 1) - and that the five sub-kernels are enqueued with no explicit
  dependencies, so the number is only valid on an in-order queue.
- [ ] **Step 6 - IF IT DOES NOT BUILD, OR THE LAYOUT CANNOT BE ADAPTED IN
  O(state): this is the deliverable, and it is complete.** Record:
  1. **The exact blocker** - the full first diagnostic with file:line, which of
     Step 2's three branches it fell into, and what the retry against vLLM's pin
     did. Or, if it built but the layout cannot be met in O(state), the specific
     requirement and why.
  2. **The "ours" design pricing - spec §3.4's bullets, each with a number:**
     - **Batched conv1d** (depthwise 4-tap causal) over the chunk, seeded from
       the ring's last 3 positions and writing the chunk's last 3 back - the
       explicit seed/writeback that replaces `gdn_step.cl:176-178`'s
       `M + 3 ≤ RING` argument rather than deepening the ring. Traffic:
       `C·10240·2 B` read and written per layer per chunk = 168 MB at C = 4096,
       ×48 layers = 8.06 GB → **13.7 ms at 590 GB/s (derived)**.
     - **The intra-chunk block**: cumulative gate sums, `A = (K·Kᵀ ⊙ β)`,
       `solve_tril`, the `W, U` recompute. GEMM-shaped at 64×128×128 per head -
       price the GEMM-shaped part through **P2's measured TFLOP/s** and the
       triangular solve at the vector-fp32 rate (22.94 TFLOPS, `explorer-2 §2`,
       external), and give the split. Note that Intel's kernel spends a whole
       dedicated kernel on the inverse and has a BMG-specific optimised version
       of it (`:392`, taken because of `is_bmg()`), which is the honest signal
       for how much of the work this term is.
     - **The chunk-to-chunk state scan**: `C/64` sequential steps × 48 heads ×
       128×128 fp32 - the one term that cannot be parallelised over positions.
       At C = 4096 that is 64 steps; give the per-step traffic (48·128·128·4 B =
       3.15 MB read + written) and the resulting floor.
     - **The correctness argument**: fp32 throughout, intra-chunk 64 matching
       the `fla` reference (`explorer-5 §2`), and the same
       `gdn_ref::step`-over-4096-positions band Step 4 would have measured -
       **which S2 must then measure for its own kernel before any gate runs.**
  3. **Engineer-days**, and spec §3.4's sentence stated plainly: this kernel is
     a **correctness** problem, not a throughput one.
- [ ] **Step 7 - record and commit** (either branch).
  `feat(probe): P5 - Intel CuTe chunked GDN <builds: band <X>, <Y> µs/layer | blocked: <Z>, ours priced>`

---

## Task T6 - composition, the bar as a ruling request, and the record corrections

**Files:**
- Modify: `docs/12-kernels.md` - a new `## Prefill - the Stage 0 probe matrix` section
- Modify: `docs/01-hardware.md` - the **measured** XMX rate from P2
- Modify: `docs/06-prior-art.md` - the `gemmstone` record
- Modify: `docs/07-open-questions.md` - the items P1-P5 answer
- Create: `docs/superpowers/specs/2026-09-04-spec2-stage0-ruling-request.md`
- Modify: `.superpowers/sdd/2026-09-04-plan6-spec2-prefill/interfaces.md` - **only** the
  deviations the controller has ruled on, and only after it has ruled

**Interfaces:**
- **Consumes:** all five `docs/probe-*.md` records.
- **Produces:** the composed ceiling at C = 4096 and C = 2048, the bar proposal,
  and the corrections spec §9 requires.

- [ ] **Step 1 - the probe matrix in `docs/12-kernels.md`.** A new top-level
  section after `## \`gdn_step\`` (so prefill is one block, not scattered),
  containing each probe's headline table with its grade line and a link to the
  full record, plus a `### What each probe decided` paragraph per probe in the
  form the rest of the document uses ("what it computes, how the work is
  assigned and **why**, what was rejected, and the numbers that decided it").
  It also carries the **`sycl-tla` sha every number was measured against** -
  once, at the top of the section, and again in any row measured against a
  different revision (P5's retry branch).
- [ ] **Step 2 - the composed ceiling, with every term labelled.** One table
  per chunk width C ∈ {4096, 2048}:

  | term | how it is priced | grade |
  |---|---|---|
  | GEMM | P2's measured ms at M = C, summed over the per-layer shape lists: GDN (QkvZ + OutProj + GateUp + Down) × 48 + FA (Qkv + OProj + GateUp + Down) × 16 | measured (P2), summed |
  | dequant | P3's measured ms per matrix in the orientation P2 needs, × the 256 matrices a chunk dequantises (48 × 4 + 16 × 4) | measured (P3), summed |
  | attention | P4's measured µs/layer × 16 - **or**, on P4's fallback branch, its priced estimate, labelled **estimated** and carried with its basis | measured (P4) or estimated |
  | GDN | P5's measured µs/layer × 48 + the state relayout - **or** its priced estimate, same rule | measured (P5) or estimated |
  | norms / SiLU / gated-head / attn_prep / embed | the model below | **derived** |
  | interop | P1's measured µs/handoff × 384 | measured (P1), scaled |
  | `lm_head` | decode's existing S = 1 route, **once per chunk, not × M** (spec §3.5): 4378.6 µs bf16, or the int4 row - `docs/12` `gemv_bf16` §; **state which checkpoint** | measured (decode), unchanged |

  **The `× M` model for the small kernels, stated once and used everywhere.**
  Spec §4 says "priced from their decode rows × M", and a literal ×M is wrong
  for these because their decode rows are launch-bound, not bandwidth-bound -
  `prep_res_fold` moves 20 KB in 2.012 µs, i.e. 10 GB/s
  (`docs/15-step-anatomy.md:530`). Use:

  > `t_M(kernel) = max( t_1(kernel), bytes_per_token(kernel) × M / 590 GB/s )`
  > - **derived, and a LOWER bound**, because it assumes the widened kernel
  > reaches full bandwidth, which S4 has not written yet. `t_1` from
  > `docs/15-step-anatomy.md:161-168` and `:530`: `prep_res_fold` **2.012**,
  > `prep_norm_finish` **1.739**, `prep_silu_mul` **9.7**,
  > `prep_gated_head` **1.6**, `attn_prep` **3.2** µs/launch, and
  > `embed_gather` **3.65 µs/token** (`docs/12-kernels.md:1276`).
  > Launch counts per token: 129 / 129 / 64 / 48 / 16 / 1.
  > `bytes_per_token` from each kernel's "Traffic per token" table in `docs/12`.

  Report `t_1 × M` in one adjacent column as the **upper** bound, with the
  sentence that the truth is between them and that L1 measures it. That is a
  bounded range, not two values for one quantity, and the table says so.
  Sanity anchor to include: at M = 4096 the literal `t_1 × M` for
  `prep_res_fold` alone is 2.012 µs × 4096 × 129 = **1063 ms**, while the
  bandwidth model gives 20 KB × 4096 × 129 / 590 GB/s = **18 ms** - a 59× spread,
  which is precisely why the model matters and why quoting only one of them
  would decide the verdict by accident.
- [ ] **Step 3 - the verdict paragraph, which is the point of the whole stage.**
  Immediately under the ceiling tables:
  - the composed ceiling in seconds and in t/s (`4096 / seconds`), for both C;
  - it beside **vLLM's 1973 t/s** with **both labels attached**, as spec §2
    requires: *device-side (loader-excluded, first-token-inclusive)* vs
    *HTTP-inclusive* - llama-benchy's `est_ppt = ttfr − latency`, which covers
    HTTP upload, server JSON parse, tokenisation, scheduling, prefill and the
    first streamed byte (`explorer-5 §1`) - plus the chunk width, the
    checkpoint, and **vLLM's own chunking**: `max_num_batched_tokens` defaults
    to 2048 on a 32 GB card and the V1 scheduler enforces it even for a solo
    request, so vLLM's pp4096 is **at least two sequential ~2048-token steps**
    (`explorer-5 §2`). The comparison names its own asymmetry.
  - **and, if the ceiling lands under vLLM, this paragraph says so** - spec §2's
    second absolute. P2's pre-registered arithmetic (90 TFLOP/s ⇒ 2.215 s for
    the GEMM term alone ⇒ 6.7 % above vLLM's whole 2.076 s) makes that the
    *expected* outcome, so the paragraph states, term by term, **which terms are
    Intel's code running at Intel's rate and which are ours** - the "skill or
    silicon" answer spec §7 asks for.
- [ ] **Step 4 - the bar, as a ruling request.** Create
  `docs/superpowers/specs/2026-09-04-spec2-stage0-ruling-request.md`:
  - the composed ceiling, one number per C, with its term-by-term derivation;
  - the **90 % default** spec §2 recommends, applied: `bar = 0.90 × ceiling`, in
    t/s, for both C;
  - the two absolutes restated with the measured numbers in them;
  - **the questions the operator is being asked**, numbered and answerable:
    (1) is 90 % the margin, or another; (2) which C is the gate's chunk width;
    (3) does the gate quote device-side `pp` only, or is spec 3's
    HTTP-inclusive number required before the row is published; (4) if the
    ceiling is under 1973, does the ladder still run (spec §5) or does the
    re-assessment memo come first; (5) the interface deviations listed in this
    plan's self-review;
  - **no work starts on Stage 1 until this is ruled** (spec §2's stopping rule).
- [ ] **Step 5 - `docs/01-hardware.md` gains the measured XMX rate.** Insert
  after the `## What the matrix engine can actually do` dispatch-policy table,
  before `Battlemage is **pre-Xe3p**` (`:96`):
  ```markdown
  ### What it actually delivers - measured 2026-09-04

  `tools/probe/probe_prefill_gemm` (P2, spec 2 §4), `sycl-tla` at <sha>, the
  stock BMG bf16 GEMM configuration (`Shape<_256,_256,_32>`, 32 subgroups,
  `MMA_Atom<XE_DPAS_TT<8, float, bfloat16_t>>`, `MainloopXeL1Staged<2>`, no
  SLM), at the six production prefill shapes × M ∈ {512, 1024, 2048, 4096}:

  **<X> TFLOP/s** at the best cell (<shape>, M = <M>) - **measured, iterate
  grade** (card 1 under `ZE_AFFINITY_MASK=1`; card 0 may have been held).
  Full matrix: [probe-prefill-gemm-2026-09-04.md](probe-prefill-gemm-2026-09-04.md).

  This **replaces the derived 183.5 TFLOPS** that spec 2's planning documents
  used (vendor 367 TOPS INT8 ÷ 2, via the "~2× bf16" line above). The two are
  not the same quantity and the derived figure is not withdrawn: 183.5 is a
  vendor-derived *peak*, <X> is a *measured achieved rate at production shapes*,
  and the ratio is <X/183.5> %. Anything that needs a ceiling uses the derived
  number and says so; anything that needs a rate uses this one.
  ```
  That paragraph is the no-two-values reconciliation and is not optional. Also
  correct the mainloop name in the same edit if the surrounding text names
  `MainloopIntelXeXMX16` as the BMG policy: at the measured pin the example uses
  `MainloopXeL1Staged<2>` and `00_bmg_gemm.cpp:381` calls the older name the one
  "for older version of copy/mma atom". Both names are real; say which was
  measured, and do not silently rewrite `explorer-3 §2`.
- [ ] **Step 6 - the `gemmstone` record.** First establish what is actually
  written, because the claim spec §1 corrects may already be gone:
  ```bash
  grep -n -i "closed\|proprietary\|not open\|black box\|opaque" docs/06-prior-art.md
  ```
  As of this plan's authoring that grep returns **nothing**, and
  `docs/06-prior-art.md:59` already reads *"Intel's own JIT GEMM generator
  (`gemmstone`)"*. So the edit is **additive, not a correction**, and T6 says so
  rather than pretending to fix a claim that is not there. Add to the
  `### \`oneDNN\` - read this one properly` block (`:65-82`):
  ```markdown
  **And it is where vLLM's XPU int4 prefill runs too, not only OpenVINO's.**
  `XPUwNa16LinearKernel.apply_weights`
  (`vllm/model_executor/kernels/linear/mixed_precision/xpu.py:105-122`) →
  `torch.ops._xpu_C.int4_gemm_w4a16` →
  `vllm-xpu-kernels/csrc/xpu/onednn/onednn_matmul.cpp:257-276` →
  `dnnl_matmul_w4a16_int4` (`int4_gemm_w4a16.h:12-161`), with the primitive
  cache keyed on `m` (`onednn_ext.h:895-942`) - the same entry point at M = 1 and
  at M = 2048, a different generated microkernel. **It is Apache-2.0 and
  readable**, at `~/PycharmProjects/oneDNN/src/gpu/intel/gemm/jit/`, with a
  cost-model kernel selector (`selector/kernel_evaluator.cpp:93-491`,
  `evaluateSCore :129-191`, `evaluateECore :260-386`) over a tuned catalog.
  Source: explorer-5 §3, spec 2 §1's "one correction to our own record".
  ```
  If the grep **does** return such a claim (a document changed under this plan),
  correct that sentence in place instead and record both the before and after
  text in the task report.
- [ ] **Step 7 - close the open questions the probes answered.**
  `docs/07-open-questions.md`:
  - **#5** (per-kernel fixed cost, `:120-134`) gains P1's immediate-list and
    SYCL-queue rows **beside** the captured-list rows already there, labelled as
    a *different execution model* and therefore a companion measurement, not a
    contradiction of the 0.52 µs/kernel figure.
  - **#9** (native int8×int4 DPAS, `:259-286`) gains one sentence: spec 2 §10
    puts it out of scope, and `explorer-3 §1` records that the int4 named
    builtins **compile but do not lower** on this driver - the instruction is
    real and reachable only through inline vISA. The question is now answered as
    "reachable, not via OpenCL C builtins, not a work item".
  - a **new item** for the one thing Stage 0 discovered and did not resolve: the
    box's driver/IGC stack (`libze_intel_gpu.so.1.15.39122`, IGC 2.38.x, ocloc
    26.27, `docs/10-the-box.md:40`) is **newer than `sycl-tla`'s validated CI
    matrix for Xe2** (Compute Runtime 26.01, IGC 2.27,
    `sycl-tla/README.md:118-122`). Record it as a known-unvalidated combination
    with the probe results as the only evidence either way.
- [ ] **Step 8 - full suite, both configurations, then commit.**
  ```bash
  tools/box.sh run "cmake -S . -B build-off -DCMAKE_BUILD_TYPE=Release -DB70_PREFILL=OFF > /dev/null && cmake --build build-off -j44 && ctest --test-dir build-off -LE checkpoint --output-on-failure"
  tools/box.sh test ''
  ```
  Commit:
  `docs(prefill): Stage 0 composed ceiling <X> t/s device-side vs vLLM 1973 - bar ruling requested`

---

## Plan self-review (2026-09-04, at authoring)

### Spec coverage - §4's rows to tasks

| spec §4 row | what it demands | task | steps |
|---|---|---|---|
| **P1 - interop smoke** | SYCL queue from our L0 handles; USM pointer round-trip; event ordering; per-launch overhead | **T0** | 1, 9-13 |
| **P2 - `sycl-tla` GEMM** | TFLOP/s per shape and M; determinism (no split-K atomics); prediction ≥ 90 TFLOP/s at M = 4096 | **P2** | 1-8 (determinism from source at 2, measured at 4) |
| **P3 - dequant scratch** | `dequant_tile` GB/s and ms per matrix; prediction ≥ 500 GB/s; the 8-16 % claim; the double-buffering question | **P3** | 1-10 (the 8-16 % claim at 1 and 8; double-buffering at 9) |
| **P4 - FMHA at head_dim 256** | does it instantiate; µs at C ∈ {1024, 2048, 4096} over depth 4096 | **P4** | 1-5, with the fallback fully specified at 4 |
| **P5 - Intel's CuTe GDN kernel** | builds; layout fit vs `gdn_state`/`conv_ring`; correctness vs the fp32 recurrent reference; state max-rel-diff after 4096 positions; µs per layer per chunk | **P5** | 1-7, with the blocker branch fully specified at 6 |
| **Stage 0 output** - docs/12 matrix, composed ceiling, 90 %-margin bar proposal, the operator's ruling | | **T6** | 1-4 |
| **Stage 0 output** - first **measured** XMX for `docs/01-hardware.md` | | **P2** step 7 → **T6** step 5 | |
| **§9 records** - `docs/06-prior-art.md`'s oneDNN/`gemmstone` record | | **T6** | 6 |
| **§8 constraints** | all five bullets | Global Constraints, verbatim | |
| **§3.7 parallelism** - "S1-S3 each own their probe; Stage 0 is itself parallel" | | Dependency note | |
| **§3.6** - "Stage 0 measures the interop cost and requires it under 1 % of chunk time" | | **T0** step 1's budget arithmetic + step 12 §5 | |
| **§11 risks** - dequant overhead > 20 % revisits a second layout; FMHA may not instantiate; CuTe GDN may not fit; toolchain coupling | | P3 step 8; P4 step 4; P5 step 6; T0 step 2 + P5's pin-risk paragraph | |

Rows deliberately **not** covered here, with the reason: §5's ladder
(L1/L2/L3), §6's correctness bars, §7's gate, and the
`runtime::prefill::{gemm,gdn,attn}.h` implementations are Stage 1 and belong to
plans 6b-6e. T0 builds the `Context` those headers need and nothing more of
them. `Engine::prefill`, the `PersistentBuffers`/`DecodeScratch`/`PrefillScratch`
split and `b70-decode --pp` are stream S4's and appear nowhere in this plan.

### Placeholder scan

No `TBD`, no `TODO`, no "add error handling", no "similar to task N". Every
CMake block, kernel, test and header in this plan is quoted in full. The
angle-bracket tokens that remain (`<X>`, `<sha>`, `<shape>`, `<M>`, `<Y>`,
`<Z>`) are all **measurement outputs** appearing in commit messages and doc
templates that the measuring step fills - the convention plan 5 used and its own
self-review declared (`2026-08-26-plan5-spec1.7-mbu-push.md:283-286`).

Two places name a transcription rather than quoting it, deliberately, because
quoting would create a second copy of a numerics contract:
`pf_dequant_tile`'s `rne_bf16` (P3 step 4 - *"read `common/bf16.h` and
transcribe `f32_to_bf16` literally; the tie-breaking is the whole bar"*), and
P5's host prologue (Step 3 - *"call into `gdn_ref.h`'s own code paths so there
is no second implementation"*).

Eight things this plan cannot know without running something. Each is written as
*discover X by running Y, record it in Z*, with Y a concrete command:
1. the exact `backend_input_t` field spelling for this SYCL version - T0 step 9;
2. `sycl-tla`'s alias chain at the pin, and whether the mainloop policy is
   `MainloopXeL1Staged` as read here - P2 step 2;
3. **whether the GEMM's B operand must be `[K][N]` or `[N][K]` in memory** -
   P2 step 3, settled by a 16×16×16 identity-A GEMM against both host
   interpretations (and P3 measures both orientations so the answer costs no
   rework);
4. the exact template parameter order of `FMHAConfigGenWithTileShape` - P4
   step 2;
5. whether FMHA instantiates at head_dim 256, and which assert fires if not -
   P4 step 2, with the five possible asserts and their fixes tabulated in
   step 1;
6. whether the CuTe GDN header compiles without torch, and against which
   `sycl-tla` revision - P5 step 2's three branches;
7. whether `sycl::ext::oneapi::level_zero::make_event` constructs on this
   version (P1's event-ordered handoff, ordering (b)) - T0 step 12 §4, which
   falls back to the queue-wait ordering the spec already permits;
8. whether `docs/06-prior-art.md` still contains the claim spec §1 corrects -
   T6 step 6, grep given, both branches specified.

### Type consistency against interfaces.md

| interfaces.md | this plan | status |
|---|---|---|
| `namespace runtime::prefill`; host code under `src/runtime/prefill/`, SYCL under `src/sycl/`, OpenCL C prefill kernels under `src/kernels/prefill/` with a `pf_` prefix | same, exactly | ✓ |
| `struct KernelArg { const void* ptr; size_t size; }` | identical | ✓ |
| activations bf16 row-major `[M][K]`, `lda = K`; GEMM output fp32 `[M][N]`, `ldc = N` | P2 step 4 | ✓ |
| dequant scratch **bf16 row-major `[K][N]`, `ldb = N`**, one buffer of 356,515,840 B | P3 steps 4 and 7 build and size exactly that buffer - **but also the `[N][K]` orientation**, pending P2 step 3 | **deviation 6 (conditional)** |
| `M` always runtime, grids `ceil(M/tile)`, no per-M binaries | P2 uses `sycl-tla`'s runtime M; `pf_dequant_tile` has no M at all | ✓ |
| persistent buffers keep decode's layouts: `kv_k`/`kv_v` `[pos][4][256]` bf16, `gdn_state` `[layer][48][128][128]` fp32, `conv_ring` per `buffers.h` | P4's cache facts, P5's layout table - both read in place, P5 with a documented O(state) transpose | ✓ |
| probe names `probe_prefill_gemm` (P2, P3), `probe_prefill_attn` (P4), `probe_gdn_chunk` (P5), `probe_interop` (P1) | **P3 is `tools/probe/probe_dequant`, a separate binary** | **deviation 4** |
| tests `tests/prefill/dequant_test.cc` | identical (P3 step 2), plus a local `dequant_harness.h` | ✓ |
| `Context(l0::Device& dev, l0::Ctx& ctx)` | `Context(l0::Context&)` | **deviation 1** |
| `sycl::queue& sycl();` | `void* sycl_queue_raw()` + free `sycl(Context&)`, plus `sycl_context_raw()`/`sycl_ctx()` | **deviation 2** |
| `void launch(l0::Kernel& k, …)` | plus a `ze_kernel_handle_t` overload | **deviation 3** |
| `loader::Linear` in `dequant_to_bf16`'s signature | the type is `loader::DeviceWeight` | **deviation 5** |
| vLLM bar **1973 t/s pp4096**, HTTP-inclusive; today **121 s** for 4096 ids; per-token forward **48.97 GFLOP**; derived XMX peak **183.5 TFLOPS**; decode gate rows **32.22 / 29.33** at `2a7df0b` | quoted with those exact values and labels in T0 §budget, P2 step 1, T6 steps 3 and 5 | ✓ |

### Interface changes requested (NOT applied - the controller rules)

1. **`Context(l0::Device& dev, l0::Ctx& ctx)` → `Context(l0::Context& ctx)`.**
   `l0::Device` and `l0::Ctx` do not exist. `src/l0/context.h:23` defines a
   single class `l0::Context` owning the driver, device and context handles
   together, exposing `driver()`, `device()`, `handle()`. The two-argument form
   cannot be written.
2. **`sycl::queue& sycl();` → `void* sycl_queue_raw() const;` plus
   `inline sycl::queue& runtime::prefill::sycl(Context&)` in a second header,
   `context_sycl.h`** (and the same for `sycl::context`, which P1 needs for
   `sycl::get_pointer_type`). A member returning `sycl::queue&` forces every
   including translation unit to have SYCL headers; the engine, the tests and
   every g++ target that constructs a `Context` do not and cannot. The split
   keeps one class with one definition in every TU. If the controller prefers
   the member, the alternative is that **`context.h` becomes icpx-only and the
   whole engine link moves to icpx** - a much larger ripple, and a decision for
   plan 6e, not for Stage 0.
3. **`launch` gains a `ze_kernel_handle_t` overload**, with the `l0::Kernel&`
   form kept and inline-forwarding to it. This is what lets
   `libb70_prefill.so` link no project archive, which is what avoids making the
   decode build `-fPIC` (`cmake/prefill.cmake` states the chain). Additive: the
   interfaces.md signature still compiles and still works.
   *Also note, not a change:* `launch` sets the group size from the kernel's own
   `reqd_work_group_size` via `zeKernelGetProperties` and throws by name on a
   kernel that declares none - the interfaces.md signature has no group-size
   argument, and this is how it is honoured rather than guessed.
4. **P3's probe is `tools/probe/probe_dequant`, a separate binary**, not a mode
   of `probe_prefill_gemm` as interfaces.md's "Probes" list implies
   (`probe_prefill_gemm (P2, P3)`). `probe_dequant` needs **no SYCL at all** -
   its kernel is OpenCL C and its harness the existing L0 wrapper set - so
   folding it into the SYCL binary would make P3 depend on T0 and on the
   `sycl-tla` pin for nothing, and would break the parallelism spec §4
   explicitly wants ("Stage 0 is itself parallel").
5. **`dequant_to_bf16(Context&, const loader::Linear& lin, uint16_t* scratch)`
   → `const loader::DeviceWeight& lin`.** interfaces.md's own comment says
   "find it in `src/loader/`"; the struct is `loader::DeviceWeight`
   (`src/loader/loader.h:19-26`), carrying `mem` (the canonical bytes),
   `scales` (a `unique_ptr<l0::Mem>`, non-null **exactly for layout 0**),
   `shape` (`model::GemvShape{K, N, S, layout}`) and `kind`. Two notes for
   plan 6b: it carries **no raw `qweight` pointer** - the pointer is
   `mem.ptr()` - and its `S` field is decode's split-K, meaningless on the
   prefill path. No host wrapper is written in this plan (P3 produces the
   kernel), so this is a heads-up, not a blocker here.
6. **Conditional, pending P2 step 3: the dequant scratch may have to be bf16
   `[N][K]` (`ldb = K`), not `[K][N]` (`ldb = N`).** `sycl-tla`'s example sets
   `LayoutB = cutlass::layout::RowMajor` over an `(N, K, L)` shape
   (`00_bmg_gemm.cpp:351,234`), which does not by itself say which memory order
   the B operand needs. P2 step 3 settles it with an identity-A GEMM; P3 builds
   and measures both orientations so the answer costs no rework. **If it is
   `[N][K]`, interfaces.md's "Layout conventions" bullet needs changing** - and
   the change is in our favour: the int4 tiles are N-tiled 16 columns wide with
   one column per lane, so `[N][K]` makes each lane's 64 k-values one contiguous
   128-byte run instead of 64 scattered 2-byte stores at stride 2N. The buffer's
   **size is unchanged** (356,515,840 B) either way.

### Known tensions, ruled here

- **The plan's own pre-registered arithmetic predicts the spec misses its
  headline comparison.** P2's ≥ 90 TFLOP/s prediction, if met exactly, puts the
  GEMM term alone 6.7 % above vLLM's entire pp4096 time. That is written into
  P2's pre-registration deliberately, before any measurement, so T6's verdict
  cannot be a hindsight rationalisation - and because spec §2's second absolute
  ("if the composed ceiling itself lands under vLLM, the memo says so") is the
  outcome the arithmetic points at. Stage 0's job is to establish it with
  evidence, not to avoid it.
- **A C++20 sub-build beside a C++17 main build, and two compilers over one
  `libstdc++`.** The decode tree stays C++17 (`CMakeLists.txt:4`); the icpx
  sub-project is C++20 because Intel's CuTe GDN kernel is
  (`vllm-xpu-kernels/cmake/utils.cmake:580`), while `sycl-tla` itself asks only
  for C++17 (`sycl-tla/CMakeLists.txt:233-237`) and compiles at 20. They never
  share a translation unit: the only header crossing the boundary is
  `src/runtime/prefill/context.h`, which is plain C++17 and SYCL-free. The ABI
  risk is real, and T0 step 3's `context_test` - a g++ binary calling into an
  icpx-linked `.so` across a `std::initializer_list` and a `std::runtime_error`
  - exists precisely to catch it, which is why it is the *first* thing written
  and not an afterthought.
- **Two `sycl-tla` pins are in play and this plan sets one.**
  `2db1b7c9` (the box's checkout, v0.9.2+9) is the pin; `87f68506` is what
  `vllm-xpu-kernels` builds against and is P5's documented retry. If P5 needs
  the second one, the project has two third-party revisions in one build tree
  and T6 records that as a finding for plan 6d rather than this plan silently
  carrying both.
- **`ZE_AFFINITY_MASK=1` and `l0::Context(0)` look contradictory and are not.**
  Under a mask the driver renumbers, so `--device 0` / `l0::Context(0)` inside a
  masked process **is** card 1 - `src/l0/context.h:14-22` says exactly this.
  Every probe here passes 0 and is run under the mask; the masking is the
  environment's job and no probe reads the variable.
- **The box is outside `sycl-tla`'s validated matrix.** Its CI validates Xe2 on
  Compute Runtime 26.01 / IGC 2.27 (`sycl-tla/README.md:118-122`); the box runs
  `libze_intel_gpu.so.1.15.39122` with IGC 2.38.x and ocloc 26.27
  (`docs/10-the-box.md:40`). Newer, not older, so the expected failure mode is a
  codegen surprise rather than a missing feature - and it is the reason every
  probe records the driver version beside its numbers. T6 step 7 files it as an
  open question.




