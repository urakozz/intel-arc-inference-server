# Checking on the Mac

The engine is written on a laptop and proved on the box (doc 10). When the box is
busy or unreachable, code still gets written - by agents as much as by hand - and
it used to be checked by a page of hand-typed `clang++` lines. `tools/mac_check.sh`
is that page as one command. It never touches the box: no ssh, no `BOX`, no
`tools/box.env`.

```sh
tools/mac_check.sh                         # sections 1-4, base main
tools/mac_check.sh --quick                 # 2 and 4 only on what changed against main
tools/mac_check.sh --base spec12b-int8-kv --kernels  # another base; also the Mac OpenCL runs
tools/mac_check.sh --allow-kernel-changes  # a branch that means to move a cmdline
```

It prints a section per check, then a table, and exits non-zero if any section
FAILs. Measured on the Intel MacBook Pro (i9-9980HK, 16 threads) at `36346cc`
plus this kit, `--kernels`, 80 s in all with the host build warm:

| section | result on main | time |
|---|---|---|
| host | 24 pass, 0 fail, 109 disabled (105 need a device, 4 box-only data) | 35 s warm; the first build is ~2.5 min (the Rust tokenizer crate) |
| l0 | 162 pass, 0 fail, 10 skip (SYCL / cutlass / oneDNN) | 31 s |
| cmdlines | 280 variants: +0, -0, ~0 against main | ~11 s with kernels |
| opencl | 268 distinct command lines pass, 0 fail | 3 s |
| kernels | gemv_i8w (bit-exact) and argmax agree with their host references on the UHD 630 | |

## Requirements

Xcode's clang and CMake (3.22 or later; the box uses 4.2). The Level Zero
**headers** - not the runtime, which does not exist for macOS: a checkout of
`https://github.com/oneapi-src/level-zero`, with `L0_INCLUDE` pointing at its
`include/` (default `~/PycharmProjects/level-zero/include`). `cargo` is
optional: with it the tokenizer library builds and `template_agnes_test` runs
against the Agnes snapshot if `~/.cache/huggingface` has it.

## What each section does

**1 host - the host tests, under ctest.** `cmake --preset mac-host`
(`CMakePresets.json`) configures the real tree with `B70_HOST_ONLY=ON`, then
builds and runs `ctest`. `cmake/host_only.cmake` replaces the three things the
full build takes from the box - the `level-zero` pkg-config package (an
INTERFACE target carrying only the headers), `ocloc` (a stub
`add_ocloc_kernel` that makes the `kernel_<name>` targets exist and produces
nothing) and `icpx` (the SYCL component is off) - and then **derives** which
tests can run instead of listing them. Every `add_test` is recorded, and at the
end of `tests/` each test is DISABLED if its executable links `b70_l0`,
`b70_runtime`, `b70_prefill_host` or `b70_prefill`; depends on a kernel binary or
on `b70-decode`/`b70-serve`; carries the `checkpoint` label; is not an executable
of the tree (the `b70_cli_reject` shell blocks, `tools/probe/*.sh`); names an
absolute path that does not exist here; or reads the Qwen3.8 `tokenizer.json`
through `tok::default_tokenizer_json()` (the one hand-kept list, because that
dependency is in the test's code, not its command line). The reasons are in
`build/mac-host/host_only_skipped.txt`. `all` is narrowed to the enabled tests'
executables, so the device libraries are never built.

Deriving rather than listing is the point: `tests/CMakeLists.txt` is edited by
every branch in flight, and a second list of "the host tests" would be wrong the
first time someone added one. A new host test runs on the Mac without anyone
touching this configuration; a new device test is disabled for the reasons above
without anyone touching it either. One case is worth knowing: the loader's host
tests link `b70_loader`, which links `b70_l0`. They are run, because static
linking pulls in no Level Zero object for the pure parts they use - and a test
that does call into the device fails to **link** here, which is the right signal.
`b70_loader` is also why the headers are needed at all.

A plain `cmake --preset mac-host && cmake --build --preset mac-host && ctest
--preset mac-host` is the same section without the script. The host build is the
real `-Wall -Wextra -Werror` build with Apple clang, not g++, so a warning only
one of the two compilers emits can still separate them.

**2 l0 - every C++ source against the Level Zero headers.** Every tracked or new
`.cc` outside `third_party/` (src, tests, tools/probe) is compiled with
`-fsyntax-only -std=c++17 -Wall -Wextra -Werror` and the include paths the targets
use (`tools/mac/l0_syntax.sh`). This is the check the device code gets: it parses
and type-checks against `ze_api.h`; it does not link and does not run. SYCL
sources are skipped, by place (`src/sycl`, and every `tools/probe/<name>.cc`
that `src/sycl/CMakeLists.txt` declares with `b70_sycl_probe`), by a direct
`#include <sycl/...>`-style line, or by the compiler's own "file not found" for
one that reaches SYCL through a project header. Their headers exist only in the
oneAPI install.

**3 cmdlines - existing kernel binaries untouched.** `tools/kernel_cmdlines` runs
on this tree and on the base's `src/kernels` (extracted with `git archive`), and
the two lists are compared by variant name: **added** (new names - expected on a
kernel branch), **removed**, and **changed** (same name, different defines or
options - a different binary under an old name). Removed or changed lines fail
the section unless `--allow-kernel-changes`. This is the Mac half of the rule
that a change must not move a binary it does not mean to (spec 14 G0); the box's
half compares the binaries' checksums.

**4 opencl - every variant's command line through clang.** Each distinct
(source, defines, options) line of the list - 268 for 280 variants - is compiled
with `clang -x cl -cl-std=CL3.0 -target spir64 -fsyntax-only
-cl-fp32-correctly-rounded-divide-sqrt -D...` (`tools/mac/cl_syntax.sh`), the
`cmake/ocloc.cmake` command line with clang in place of ocloc. `spir64` is the
target that has the CL 3.0 optional features ocloc has for `bmg-g31`. Intel-only
options (`-cl-intel-256-GRF-per-thread`) are dropped. Apple's clang declares none
of the Intel functions, so `tools/mac/opencl/intel_shim.h` is force-included:
the `cl_intel_subgroups` block reads and shuffles and their `_short` / `_char`
forms as their extension specs define them, and the 2D block I/O, DPAS and split
barrier functions **as this repo uses them** - only those names, because ocloc
exposes part of each family and the pattern would admit a name ocloc rejects (one
already did: `pf_gemm.cl` records it). A kernel that needs a new 2D block shape
fails here as "undeclared"; add it to the shim once ocloc on the box has built
it, and say so in the commit. Each variant's `#error` guards are exercised by its
defines, which is most of what a wrong variant line gets wrong.

**5 kernels (`--kernels`) - indicative runs on the Mac's GPU.**
`tools/mac/clrun` is a small library over Apple's OpenCL 1.2 framework: build a
`.cl` from this repo with given defines on the Mac's GPU, run it on host buffers.
The subgroup operations are emulated by the same shim in its `B70_CL_EMULATE`
mode - a "subgroup" is 16 consecutive work-items of a 1-D group, and a block read
is the plain load it is specified to equal - so only kernels whose cross-lane
traffic is block reads can run; reductions, shuffles, DPAS and 2D block I/O are
left undeclared there and fail the build by name rather than run on a wrong
stand-in. Three drivers ship: `gemv_i8w_run [M K N]` (the int8 `lm_head`, against
its formula at a K-scaled tolerance; it is bit-exact on the UHD 630),
`argmax_run [M VOCAB VOCAB_USED]` (both stages and the control-block bookkeeping,
exact, with a tie at the top of a row and a larger logit in the masked tail),
`pf_moe_run` (spec 15d: the prefill MoE block's sort at Ornith's shape on random,
skewed and adversarial routes, the bf16 / int8 gathers, the expert dequant with an
empty expert skipped and the combine, all exact against `tests/kernels/pf_moe_ref.h`;
its grouped GEMMs are DPAS and stay on the box) and `k2_run` (spec 18b: K2-Horizon's
portable kernels - the grouped norm, SiLU x up, both attention-prep builds, the sigmoid
routers with their padded lanes, the MoE gate||up / down and MoVA's value experts - at
K2's real shapes against `tests/kernels/k2_ref.h`; it also takes two host sources) and
`moe_run` (spec 15e: the decode MoE block `moe.cl` at Ornith's shape against
`tests/kernels/moe_ref.h` at M = 1, and at M = 4 - the MTP verify lists' rows - every
row bitwise the M = 1 binary's on that row, in order and reversed; `moe_down` at DN_KS 1,
the Mac's 256-lane work-group cap, as `k2_run` does). A new
driver is a `tools/mac/clrun/<kernel>_run.cc` with a `main()` over `clrun::Device`,
`Program` and `Buffer`, and its name in section 5's loop in `tools/mac_check.sh`, which
builds the drivers into `build/mac-check/clrun/` with `src/` and `tests/` on the include
path (so a driver can use a test's host reference).
`B70_MAC_CL_DEVICE=AMD` picks another device by name; the CPU device rejects
argmax's 256-wide work-group.

**Beside the sections: the pipeline-parallel protocols under ThreadSanitizer.**
`tools/mac/pp_prefill_tsan.sh` (after a host build) rebuilds `pp_prefill_protocol_test` (spec
16c's prefill order on two host threads standing in for the cards) and `pp_protocol_test` (spec
16b's peer hand-off) with `-fsanitize=thread` and runs them: the order hands every per-chunk
buffer between the host and the "devices" through an event or a queue, or the sanitizer says
where it does not. It proves the host protocol race-free, not the cards' events or PCIe.

## What it does not prove

**Anything about the B70.** The section that runs kernels runs them on a different
GPU, through a different compiler, with the subgroup made of work-items instead of
SIMD lanes. Agreement there says the indexing, the guards and the arithmetic order
are right on *some* device; it says nothing about timing, about ocloc's ISA build
and its register budget, about DPAS, 2D block I/O or cross-lane behaviour, and
nothing about rounding where the two compilers contract a multiply-add
differently. The OpenCL syntax check is Apple's clang, a few versions from Intel's
fork inside ocloc; a corner where the two frontends disagree is possible in either
direction. The host build is Apple clang, not the box's g++. And the golden gate,
the checkpoint tests and every device test only run on the box: a green table here
is the precondition for putting a branch on the box, not a substitute for it.
