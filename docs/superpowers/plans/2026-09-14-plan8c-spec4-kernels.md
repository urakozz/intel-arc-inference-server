# Spec 4 / Stage 2 - K2-Horizon kernels Implementation Plan

**Status (2026-09-14): written, NOT dispatched.** Implementation waits for the operator's ruling on the prefill GEMM direction; these plans may be adjusted after it.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every device binary K2's decode list needs, each green against a host reference: new kernels for the grouped norm, SiLU·mul per slot, router top-k, the expert-slot GEMV, the two expert combines and the K2 attention trio; K2-shaped variants of the existing GEMV, bf16 GEMV, residual fold, embed gather and argmax.

**Architecture:** New OpenCL C sources live in `src/kernels/k2/`, and every existing `.cl` is compiled at K2's shapes by define only. Routing follows spec §3.3 option A: a router kernel writes `{id, weight}` pairs, ascending by id, into a small device buffer. A slot kernel reads its id from that buffer on the device and indexes the flat expert buffer at `id × stride`, so launch arguments stay frozen.

Every slot-shaped kernel takes its slot as `slot_base + get_group_id(1)`. The same binary therefore runs one launch per slot (grid y = 1, which spec §3.3 specifies) or one launch covering every slot (grid y = k), which is left as a tuning option for later. Rounding follows the 27B's discipline: torch rounds per op, so these kernels do too. Each reference repeats its kernel's operation order verbatim.

**Tech Stack:** OpenCL C 3.0 via ocloc AOT (`add_ocloc_kernel`, `-cl-fp32-correctly-rounded-divide-sqrt` by default), Level Zero through `src/l0`, C++17 host references, ctest via `tools/box.sh`.

**Spec:** `docs/superpowers/specs/2026-09-14-spec4-k2-horizon-decode-core-design.md` (§3.2 semantics, §3.3 decode list and kernels, §4 kernel tests, §8 risks 2 and 6)

## Global Constraints

- Branch `spec1.7-codex-exp`; never push. Commits end with `Claude-Session: `.
- Build/test only via `tools/box.sh` (JOBS 44); the Mac never compiles. Long box jobs detached (`setsid nohup … > $HOME/<name>.log 2>&1 < /dev/null &`).
- **No existing `.cl`, `.h`, `.cc` or test file is edited.** `src/kernels/CMakeLists.txt` gets one appended `add_subdirectory(k2)`; `tests/CMakeLists.txt` gets appended blocks.
- `-cl-denorms-are-zero` is forbidden (cmake/ocloc.cmake rejects it). **Never `rsqrt`; `1.0f / sqrt(x)`.**
- Every K2 kernel is compiled at M = 1 only (spec 4 decodes one sequence; prefill is spec 5).
- **The torch-semantics defines come from plan 8a's facts doc §6** (`docs/k2-stage0-facts-2026-09-14.md`). Task 1 Step 1 applies them. If §6 is absent (the operator has not run plan 8a Task 5), keep the pre-registered defaults below and note it in the commit message.
- GPU work on device 1: `ZE_AFFINITY_MASK=1`.

---

## File structure

| path | responsibility |
|---|---|
| `src/kernels/k2/CMakeLists.txt` | every K2 variant, plus the torch-semantics defines |
| `src/kernels/k2/k2_kernels.h` | variant names, the ONE host home for each name |
| `src/kernels/k2/k2_norm.cl` | `k2_norm_finish` - grouped RMSNorm stage B |
| `src/kernels/k2/k2_silu.cl` | `k2_silu_mul` - SiLU(gate)·up per slot |
| `src/kernels/k2/k2_route.cl` | `k2_router_topk` - sigmoid, selection bias, top-k, bf16 weights |
| `src/kernels/k2/k2_experts.cl` | `k2_gemv_slot`, `k2_value_combine`, `k2_moe_combine` |
| `src/kernels/k2/k2_attn.cl` | `k2_attn_prep`, `k2_attn_decode`, `k2_attn_reduce` |
| `tests/kernels/k2/k2_dev.h` | the device harness these tests share |
| `tests/kernels/k2/k2_ref.h` | host references: norm, silu, route, experts |
| `tests/kernels/k2/k2_attn_ref.h` | host reference: the attention trio |
| `tests/kernels/k2/k2_shapes_test.cc` | reused sources at K2 shapes: GEMV, bf16 GEMV, embed, argmax |
| `tests/kernels/k2/k2_prep_test.cc`, `k2_route_test.cc`, `k2_experts_test.cc`, `k2_attn_test.cc` | one per new source |

---

### Task 1: K2 variant scaffold, and the existing sources at K2's shapes

**Files:**
- Create: `src/kernels/k2/CMakeLists.txt`, `src/kernels/k2/k2_kernels.h`, `tests/kernels/k2/k2_dev.h`, `tests/kernels/k2/k2_shapes_test.cc`
- Modify: `src/kernels/CMakeLists.txt` (append one line), `tests/CMakeLists.txt` (append)

**Interfaces:**
- Consumes: `add_ocloc_kernel`, `${CTRL_DEFINES}`, `${ATTN_BLOCK}` (set in `src/kernels/CMakeLists.txt` before the new `add_subdirectory`), `kernels::gemv_variant`, `kernels::gemv_bf16_variant`, `kernels::gemv_bf16_tiling`, `kernels::prep_res_fold_variant`, `tests/kernels/gemv_harness.h` (`run_gemv`, `run_gemv_bf16`, `GemvCase`), `tests/kernels/gemv_ref.h` (`gemv_ref`, `gemv_bf16_ref`, `random_bf16`), `common::Int4Gptq::random`, `common::Bf16Tiled::from_rowmajor`.
- Produces: the names below in `namespace kernels::k2` (used by Tasks 2-5 and plan 8e's capture); CMake cache variables `K2_SCORE_SCALE`, `K2_SOFTPLUS_BETA`, `K2_TOPK_SUM_T4`, `K2_TOPK_SUM_T8`, `K2_TOPK_TIE_LOW`; the `k2test::Dev` harness.

- [ ] **Step 1: Apply plan 8a's measured semantics.** Read `docs/k2-stage0-facts-2026-09-14.md` §6. The defaults below are the pre-registered values; change a value only where §6 recorded a candidate at `1.000000` that differs from it:

| cache variable | default | §6 question |
|---|---|---|
| `K2_SCORE_SCALE` | `0.08838834764831844f` (fp32 of 128^-0.5) | Q2 - the bf16 alternative is `0.08837890625f` |
| `K2_TOPK_SUM_T4` | `0` (SEQ) | Q5 k=4 - `1` PAIR |
| `K2_TOPK_SUM_T8` | `0` (SEQ) | Q5 k=8 - `1` PAIR, `2` LANES4 |
| `K2_TOPK_TIE_LOW` | `1` | Q6 - `0` if topk returned the higher index |

- [ ] **Step 2: Write `src/kernels/k2/CMakeLists.txt`**

```cmake
# Spec 4: K2-Horizon's device binaries (plan 8c). New sources are in this directory; the
# existing ones - gemv.cl, gemv_bf16.cl, prep.cl, embed_gather.cl, argmax.cl - are compiled at
# K2's shapes BY DEFINE ONLY and are not edited. Every variant is M = 1.
set(K2_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(B70_KSRC ${CMAKE_SOURCE_DIR}/src/kernels)

# The torch-semantics defines: pre-registered in plan 8a Task 5, measured in
# docs/k2-stage0-facts-2026-09-14.md §6. CACHE INTERNAL (implies FORCE) so an edit here takes
# effect on the next configure, and so tests/CMakeLists.txt sees the same values for the host
# references.
set(K2_SCORE_SCALE "0.08838834764831844f" CACHE INTERNAL "attention score scale (facts §6 Q2)")
set(K2_SOFTPLUS_BETA "0.6931471805599453f" CACHE INTERNAL "softplus beta = ln 2, fp32")
set(K2_TOPK_SUM_T4 0 CACHE INTERNAL "top-4 weight sum order: 0 SEQ, 1 PAIR (facts §6 Q5)")
set(K2_TOPK_SUM_T8 0 CACHE INTERNAL "top-8 weight sum order: 0 SEQ, 1 PAIR, 2 LANES4 (facts §6 Q5)")
set(K2_TOPK_TIE_LOW 1 CACHE INTERNAL "equal selection scores: 1 lower id first (facts §6 Q6)")

# --- existing sources at K2 shapes -------------------------------------------------------
# int4 GEMV, layout 0, S = 1: the fused attention rows, o_proj, the dense MLP, the shared
# expert. No GEMV_BLOCK2D / GEMV_DEQ_SHIFT: those are the 27B's measured cells only.
foreach(shape "2560;10240" "2560;9280" "4096;2560" "2560;12288" "6144;2560" "2560;1536" "768;2560"
              "2560;1024")
  list(GET shape 0 K)
  list(GET shape 1 N)
  add_ocloc_kernel(gemv_M1_K${K}_N${N}_S1_L0 SOURCE ${B70_KSRC}/gemv.cl
                   DEFINES M=1 K=${K} N=${N} S=1 LAYOUT=0)
endforeach()
# (K2560 N1024 is test-only: k2_experts_test's bitwise reference for the value-expert slot.)

# bf16 GEMV: the MoE router (100 rows padded to 128, the {16,16} tiling) and lm_head.
add_ocloc_kernel(gemv_bf16_M1_K2560_N128_C16_S16 SOURCE ${B70_KSRC}/gemv_bf16.cl
                 DEFINES M=1 K=2560 N=128 COLS_PER_WG=16 KSPLIT=16)
add_ocloc_kernel(gemv_bf16_M1_K2560_N250624 SOURCE ${B70_KSRC}/gemv_bf16.cl
                 DEFINES M=1 K=2560 N=250624)

# The norm's stage A is prep.cl's prep_res_fold at K2's grid: 10 chunks of 256, 5 per group.
# SP0 for layer 0's input norm, SP1 after every K2 mixer/MLP (all S = 1).
foreach(SP 0 1)
  add_ocloc_kernel(prep_res_fold_M1_K2560_SP${SP}_G10 SOURCE ${B70_KSRC}/prep.cl
                   DEFINES M=1 K=2560 S_PREV=${SP} FOLD_G=10)
endforeach()

add_ocloc_kernel(k2_embed_gather_M1 SOURCE ${B70_KSRC}/embed_gather.cl
                 DEFINES ${CTRL_DEFINES} M=1 HIDDEN=2560 VOCAB=250624)
# VOCAB_USED = VOCAB: K2's vocabulary is fully used (spec §1), so nothing is masked.
add_ocloc_kernel(k2_argmax_stage1_M1 SOURCE ${B70_KSRC}/argmax.cl
                 DEFINES ${CTRL_DEFINES} M=1 VOCAB=250624 VOCAB_USED=250624)
add_ocloc_kernel(k2_argmax_stage2 SOURCE ${B70_KSRC}/argmax.cl
                 DEFINES ${CTRL_DEFINES} VOCAB=250624 VOCAB_USED=250624)

# --- new sources (Tasks 2-5 append below) ------------------------------------------------
```

Append to `src/kernels/CMakeLists.txt`:

```cmake
# Spec 4: K2-Horizon's binaries. After everything above, so CTRL_DEFINES and ATTN_BLOCK exist.
add_subdirectory(k2)
```

- [ ] **Step 3: Write `src/kernels/k2/k2_kernels.h`**

```cpp
#pragma once
#include <string>

#include "kernels/kernels.h"

// K2-Horizon's variant names - the ONE host home for each (spec 4, plan 8c). The device-side
// homes are src/kernels/k2/CMakeLists.txt; a disagreement names a binary that does not exist,
// and runtime capture throws on it.
namespace kernels::k2 {

inline constexpr unsigned kFoldG = 10;      // prep_res_fold at K2560: 10 chunks, 5 per norm group
inline constexpr unsigned kAttnBlock = 64;  // == ATTN_BLOCK (src/kernels/CMakeLists.txt)

inline std::string gemv(unsigned K, unsigned N) { return kernels::gemv_variant(1, K, N, 1, 0); }
inline std::string gemv_bf16(unsigned K, unsigned N) {
  return kernels::gemv_bf16_variant(1, K, N, kernels::gemv_bf16_tiling(N));
}
inline std::string res_fold(unsigned s_prev) {
  return kernels::prep_res_fold_variant(1, 2560, s_prev, kFoldG);
}
inline std::string norm_finish() { return "k2_norm_finish"; }
inline std::string silu_mul(unsigned inter) { return "k2_silu_mul_I" + std::to_string(inter); }
inline std::string router_topk(unsigned experts, unsigned k) {
  return "k2_router_topk_E" + std::to_string(experts) + "_T" + std::to_string(k);
}
inline std::string gemv_slot(unsigned K, unsigned N) {
  return "k2_gemv_slot_K" + std::to_string(K) + "_N" + std::to_string(N);
}
inline std::string value_combine() { return "k2_value_combine"; }
inline std::string moe_combine() { return "k2_moe_combine"; }
inline std::string attn_prep(bool dense) {
  return std::string("k2_attn_prep_") + (dense ? "dense" : "sparse");
}
inline std::string attn_decode(unsigned maxlen) {
  return "k2_attn_decode_L" + std::to_string(maxlen) + "_B" + std::to_string(kAttnBlock);
}
inline std::string attn_reduce(unsigned maxlen) {
  return "k2_attn_reduce_L" + std::to_string(maxlen) + "_B" + std::to_string(kAttnBlock);
}
inline std::string embed_gather() { return "k2_embed_gather_M1"; }
inline std::string argmax_stage1() { return "k2_argmax_stage1_M1"; }
inline std::string argmax_stage2() { return "k2_argmax_stage2"; }

}  // namespace kernels::k2
```

- [ ] **Step 4: Write the shared harness** - `tests/kernels/k2/k2_dev.h`:

```cpp
#pragma once
// The device harness every K2 kernel test shares: one context, one queue, launches appended to
// ONE in-order regular list and executed once (how runtime capture appends them).
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "kernels/kernels.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/kernel.h"
#include "l0/memory.h"
#include "l0/module.h"
#include "l0/queue.h"

namespace k2test {

struct Launch {
  l0::Kernel* k;
  uint32_t gx, gy;
};

struct Dev {
  l0::Context ctx{0};
  l0::Queue q{ctx};
  l0::Fence fence{q};
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  std::vector<std::unique_ptr<l0::Module>> mods;
  std::vector<std::unique_ptr<l0::Kernel>> kerns;

  l0::Kernel& kernel(const std::string& variant, const char* entry, uint32_t wg) {
    mods.push_back(std::make_unique<l0::Module>(ctx, kernels::path(variant)));
    kerns.push_back(std::make_unique<l0::Kernel>(mods.back()->kernel(entry)));
    kerns.back()->group_size(wg);
    return *kerns.back();
  }
  void run(std::initializer_list<Launch> ls) {
    l0::CmdList list = l0::CmdList::regular(ctx);
    for (const Launch& l : ls) list.launch(*l.k, l.gx, l.gy);
    list.close();
    q.execute(list, &fence);
    fence.wait();
  }
  template <class T> l0::Mem up(const std::vector<T>& v, l0::MemKind kind = l0::MemKind::Device) {
    l0::Mem m(ctx, kind, v.size() * sizeof(T));
    imm.copy(m.ptr(), v.data(), v.size() * sizeof(T));
    return m;
  }
  template <class T> std::vector<T> down(const l0::Mem& m, size_t n) {
    std::vector<T> v(n);
    imm.copy(v.data(), m.ptr(), n * sizeof(T));
    return v;
  }
};

inline std::vector<uint16_t> rand_bf16(size_t n, uint32_t seed, float lo, float hi) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> d(lo, hi);
  std::vector<uint16_t> v(n);
  for (auto& e : v) e = common::f32_to_bf16(d(rng));
  return v;
}
inline std::vector<float> rand_f32(size_t n, uint32_t seed, float sigma) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> d(0.0f, sigma);
  std::vector<float> v(n);
  for (auto& e : v) e = d(rng);
  return v;
}

// bf16 words ordered as integers: the distance is the number of bf16 values between them.
inline int32_t bf16_key(uint16_t v) { return (v & 0x8000u) ? -int32_t(v & 0x7FFFu) : int32_t(v); }
inline uint32_t max_ulp(const std::vector<uint16_t>& got, const std::vector<uint16_t>& ref) {
  CHECK_EQ(got.size(), ref.size());
  uint32_t worst = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const int32_t d = bf16_key(got[i]) - bf16_key(ref[i]);
    worst = std::max(worst, uint32_t(d < 0 ? -d : d));
  }
  return worst;
}

}  // namespace k2test
```

- [ ] **Step 5: Write the failing test** - `tests/kernels/k2/k2_shapes_test.cc`:

```cpp
// The EXISTING kernel sources at K2's shapes: gemv.cl (7 production + 1 test shape),
// gemv_bf16.cl (router at {16,16}, lm_head at {64,1}), embed_gather.cl and argmax.cl at
// hidden 2560 / vocab 250,624 with every id eligible. Same bars as gemv_test / gemv_bf16_test.
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#include "check.h"
#include "gemv_harness.h"
#include "gemv_ref.h"
#include "k2_dev.h"
#include "kernels/k2/k2_kernels.h"
#include "runtime/control.h"

using k2test::Dev;

static void int4_case(Dev& d, uint32_t K, uint32_t N) {
  const common::Int4Gptq w = common::Int4Gptq::random(K, N, 4242 + N);
  const std::vector<uint16_t> x = random_bf16(K, 17 + K);
  std::vector<float> ref;
  gemv_ref(w, x, 1, ref);
  const GemvResult r = run_gemv(d.ctx, d.q, d.fence, w, x, {1, K, N, 1, 0}, 0);
  const double err = max_abs_err(r.out, ref), tol = tol_for(ref);
  std::printf("gemv K=%u N=%u  max_abs_err=%.3g tol=%.3g\n", K, N, err, tol);
  CHECK(err <= tol);
}

static void bf16_case(Dev& d, uint32_t K, uint32_t N) {
  const std::vector<uint16_t> rows = random_bf16(size_t(N) * K, 99 + N, -0.05f, 0.05f);
  const std::vector<uint16_t> x = random_bf16(K, 5);
  std::vector<float> ref;
  gemv_bf16_ref(rows.data(), K, N, x, 1, ref);
  const common::Bf16Tiled t = common::Bf16Tiled::from_rowmajor(rows.data(), K, N);
  const GemvResult r = run_gemv_bf16(d.ctx, d.q, d.fence, t, x, 1, 0, kernels::gemv_bf16_tiling(N));
  const double err = max_abs_err(r.out, ref), tol = tol_for(ref);
  std::printf("gemv_bf16 K=%u N=%u  max_abs_err=%.3g tol=%.3g\n", K, N, err, tol);
  CHECK(err <= tol);
}

// embed_gather at the last id, and argmax where the maximum sits in the LAST row - a vocab-edge
// mask (VOCAB_USED < VOCAB) would hide it - and a tie that must go to the lower index.
static void token_case(Dev& d) {
  constexpr uint32_t H = 2560, V = 250624, G = (V + 1023) / 1024;
  std::vector<uint16_t> embed(size_t(V) * H);
  for (uint32_t k = 0; k < H; ++k) embed[size_t(V - 1) * H + k] = uint16_t(k * 7 + 1);
  l0::Mem em = d.up(embed);
  l0::Mem ctrl(d.ctx, l0::MemKind::Shared, sizeof(runtime::Control));
  std::memset(ctrl.ptr(), 0, sizeof(runtime::Control));
  runtime::Control* c = ctrl.as<runtime::Control>();
  c->n_active = 1;
  c->cur_token[0] = V - 1;
  l0::Mem resid(d.ctx, l0::MemKind::Device, H * 2);
  l0::Kernel& ke = d.kernel(kernels::k2::embed_gather(), "embed_gather", 256);
  ke.arg_ptr(0, ctrl.ptr());
  ke.arg_ptr(1, em.ptr());
  ke.arg_ptr(2, resid.ptr());

  std::vector<float> logits(V, -1.0f);
  logits[V - 1] = 5.0f;
  logits[1000] = 5.0f;          // tie: 1000 < 250623 must win
  logits[V - 2] = 4.5f;
  l0::Mem lm = d.up(logits);
  l0::Mem part(d.ctx, l0::MemKind::Device, size_t(G) * 2 * 4);
  l0::Kernel& k1 = d.kernel(kernels::k2::argmax_stage1(), "argmax_stage1", 256);
  k1.arg_ptr(0, lm.ptr());
  k1.arg_ptr(1, part.ptr());
  l0::Kernel& k2 = d.kernel(kernels::k2::argmax_stage2(), "argmax_stage2", 256);
  k2.arg_ptr(0, ctrl.ptr());
  k2.arg_ptr(1, part.ptr());
  d.run({{&ke, 1, 1}, {&k1, G, 1}, {&k2, 1, 1}});
  const std::vector<uint16_t> row = d.down<uint16_t>(resid, H);
  for (uint32_t k = 0; k < H; ++k) CHECK_EQ(row[k], uint16_t(k * 7 + 1));
  CHECK_EQ(c->out_token[0], uint32_t(1000));
  CHECK_EQ(c->pos, uint32_t(1));

  logits[1000] = -1.0f;         // now the last row is the unique maximum
  d.imm.copy(lm.ptr(), logits.data(), logits.size() * 4);
  c->pos = 0;
  d.run({{&k1, G, 1}, {&k2, 1, 1}});
  CHECK_EQ(c->out_token[0], V - 1);
  std::puts("embed_gather + argmax at vocab 250624: OK");
}

int main() {
  Dev d;
  const std::pair<uint32_t, uint32_t> shapes[] = {{2560, 10240}, {2560, 9280}, {4096, 2560},
                                                  {2560, 12288}, {6144, 2560}, {2560, 1536},
                                                  {768, 2560},   {2560, 1024}};
  for (const auto& [K, N] : shapes) int4_case(d, K, N);
  bf16_case(d, 2560, 128);
  bf16_case(d, 2560, 250624);
  token_case(d);
  std::puts("k2_shapes_test OK");
  return 0;
}
```

- [ ] **Step 6: Register** - append to `tests/CMakeLists.txt`:

```cmake
# --- spec 4 plan 8c: K2 kernels ----------------------------------------------------------
set(B70_K2_SHAPE_KERNELS
  kernel_gemv_M1_K2560_N10240_S1_L0 kernel_gemv_M1_K2560_N9280_S1_L0
  kernel_gemv_M1_K4096_N2560_S1_L0 kernel_gemv_M1_K2560_N12288_S1_L0
  kernel_gemv_M1_K6144_N2560_S1_L0 kernel_gemv_M1_K2560_N1536_S1_L0
  kernel_gemv_M1_K768_N2560_S1_L0 kernel_gemv_M1_K2560_N1024_S1_L0
  kernel_gemv_bf16_M1_K2560_N128_C16_S16 kernel_gemv_bf16_M1_K2560_N250624
  kernel_k2_embed_gather_M1 kernel_k2_argmax_stage1_M1 kernel_k2_argmax_stage2)
add_executable(k2_shapes_test kernels/k2/k2_shapes_test.cc)
target_include_directories(k2_shapes_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests
                           ${CMAKE_SOURCE_DIR}/tests/kernels ${CMAKE_SOURCE_DIR}/tests/kernels/k2)
target_link_libraries(k2_shapes_test PRIVATE b70_l0)
b70_target_kernel_dir(k2_shapes_test)
add_dependencies(k2_shapes_test ${B70_K2_SHAPE_KERNELS})
add_test(NAME k2_shapes_test COMMAND k2_shapes_test)
```

- [ ] **Step 7: Build and run** - `tools/box.sh sync && tools/box.sh build 2>&1 | tail -3 && ZE_AFFINITY_MASK=1 tools/box.sh test '^k2_shapes_test$'`. Expected: eight `gemv` lines and two `gemv_bf16` lines inside tolerance, `embed_gather + argmax at vocab 250624: OK`, `k2_shapes_test OK`. (The lm_head case uploads 1.28 GB; allow a minute.)

- [ ] **Step 8: Prove the 27B's binaries did not move** - `tools/box.sh test '^(gemv_test|gemv_bf16_test|prep_test|embed_gather_test|argmax_test|kernel_table_test)$'`. Expected: all pass.

- [ ] **Step 9: Commit**

```bash
git add src/kernels/k2/CMakeLists.txt src/kernels/k2/k2_kernels.h src/kernels/CMakeLists.txt tests/kernels/k2/k2_dev.h tests/kernels/k2/k2_shapes_test.cc tests/CMakeLists.txt
git commit -m "feat(kernels): K2 variant scaffold; gemv/gemv_bf16/fold/embed/argmax at K2 shapes

Claude-Session: "
```

---

### Task 2: The grouped RMSNorm and SiLU·mul per slot

**Files:**
- Create: `src/kernels/k2/k2_norm.cl`, `src/kernels/k2/k2_silu.cl`, `tests/kernels/k2/k2_ref.h`, `tests/kernels/k2/k2_prep_test.cc`
- Modify: `src/kernels/k2/CMakeLists.txt` (append), `tests/CMakeLists.txt` (append)

**Interfaces:**
- Consumes: Task 1's harness and names; `prep_ref::res_fold(partials, resid, sumsq, M, K, S_PREV, G)` from `tests/kernels/prep_ref.h` (K and G parametric).
- Produces:
  - `k2_norm_finish(float* sumsq [10], ushort* resid [2560], float* norm_w [2560], ushort* x_out [2560])`, grid (10, 1), work-group 256.
  - `k2_silu_mul(float* in [slots][2·INTER], ushort* out [slots][INTER], uint slot_base)`, grid (INTER/256, slots), work-group 256; variants `k2_silu_mul_I6144`, `k2_silu_mul_I768`.
  - `k2_ref::norm_finish`, `k2_ref::silu_mul`, `k2_ref::f32`, `k2_ref::rne`.

- [ ] **Step 1: Write the references** - `tests/kernels/k2/k2_ref.h` (Tasks 3 and 4 append to it):

```cpp
#pragma once
// Host references for src/kernels/k2/*.cl except attention (k2_attn_ref.h). Each function is
// its kernel's op chain in its kernel's order; the two are edited together.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "common/bf16.h"

namespace k2_ref {

inline float f32(uint16_t h) { return common::bf16_to_f32(h); }
inline uint16_t rne(float f) { return common::f32_to_bf16(f); }
inline float silu_f32(float x) { return x / (1.0f + std::exp(-x)); }
inline float sigmoid_f32(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// k2_norm_finish: stage B of K2RMSNorm(2 groups of 1280, plain weight), after prep.cl's fold
// wrote sumsq[0..9] (chunk g = elements [256g, 256g + 256); chunks 0-4 are group 0).
//   total = Σ_{g in the group, ascending} sumsq[g]
//   rstd  = 1 / sqrt(total / 1280 + 1e-6)
//   x[k]  = rne(f32(resid[k]) · rstd · w[k])
inline void norm_finish(const float* sumsq, const uint16_t* resid, const float* w, uint16_t* x) {
  for (uint32_t grp = 0; grp < 2; ++grp) {
    float total = 0.0f;
    for (uint32_t g = grp * 5; g < grp * 5 + 5; ++g) total += sumsq[g];
    const float rstd = 1.0f / std::sqrt(total / 1280.0f + 1e-6f);
    for (uint32_t k = grp * 1280; k < (grp + 1) * 1280; ++k) x[k] = rne(f32(resid[k]) * rstd * w[k]);
  }
}

// k2_silu_mul for one slot: gate‖up interleaved in 16-column blocks.
inline void silu_mul(const float* in, uint16_t* out, uint32_t inter, uint32_t slot) {
  const size_t fused = 2 * size_t(inter);
  for (uint32_t k = 0; k < inter; ++k) {
    const size_t gflat = size_t(slot) * fused + size_t(k / 16) * 32 + (k % 16);
    const uint16_t g_b = rne(in[gflat]), u_b = rne(in[gflat + 16]);
    const uint16_t s_b = rne(silu_f32(f32(g_b)));
    out[size_t(slot) * inter + k] = rne(f32(s_b) * f32(u_b));
  }
}

}  // namespace k2_ref
```

- [ ] **Step 2: Write the failing test** - `tests/kernels/k2/k2_prep_test.cc`:

```cpp
// k2_norm.cl (with prep.cl's fold at K2's grid) and k2_silu.cl against k2_ref.h.
// Bars: the norm pair is bit-exact (no exp anywhere); silu is 2 bf16 ulp on random inputs
// (OpenCL allows 3 ulp on fp32 exp) and bit-exact where every gate is 30.0f, since
// 1 + exp(-30) == 1.0f in fp32 on any conforming implementation.
#include <cmath>
#include <cstdio>
#include <vector>

#include "k2_dev.h"
#include "k2_ref.h"
#include "kernels/k2/k2_kernels.h"
#include "prep_ref.h"

using k2test::Dev;

static void norm_case(Dev& d, uint32_t s_prev, uint32_t seed) {
  constexpr uint32_t K = 2560, G = 10;
  const std::vector<float> partials = k2test::rand_f32(K, seed, 0.5f);
  std::vector<uint16_t> resid = k2test::rand_bf16(K, seed + 1, -1.0f, 1.0f);
  // Group 1 at 64x group 0's scale: a norm over the WHOLE row would crush group 0, so the
  // bit-exact comparison against a grouped reference cannot pass by accident.
  for (uint32_t k = 1280; k < K; ++k) resid[k] = k2_ref::rne(k2_ref::f32(resid[k]) * 64.0f);
  std::vector<float> w = k2test::rand_f32(K, seed + 2, 0.3f);
  for (float& v : w) v = k2_ref::f32(k2_ref::rne(1.0f + v));   // bf16-valued, as load_k2 widens

  std::vector<uint16_t> r_ref = resid, x_ref(K);
  std::vector<float> sumsq_ref(G);
  prep_ref::res_fold(partials.data(), r_ref.data(), sumsq_ref.data(), 1, K, s_prev, G);
  k2_ref::norm_finish(sumsq_ref.data(), r_ref.data(), w.data(), x_ref.data());

  l0::Mem pm = d.up(partials), rm = d.up(resid), wm = d.up(w);
  l0::Mem sm(d.ctx, l0::MemKind::Device, G * 4), xm(d.ctx, l0::MemKind::Device, K * 2);
  l0::Kernel& ka = d.kernel(kernels::k2::res_fold(s_prev), "prep_res_fold", 256);
  ka.arg_ptr(0, pm.ptr());
  ka.arg_ptr(1, rm.ptr());
  ka.arg_ptr(2, sm.ptr());
  l0::Kernel& kb = d.kernel(kernels::k2::norm_finish(), "k2_norm_finish", 256);
  kb.arg_ptr(0, sm.ptr());
  kb.arg_ptr(1, rm.ptr());
  kb.arg_ptr(2, wm.ptr());
  kb.arg_ptr(3, xm.ptr());
  d.run({{&ka, G, 1}, {&kb, G, 1}});
  CHECK(d.down<uint16_t>(rm, K) == r_ref);
  CHECK(d.down<uint16_t>(xm, K) == x_ref);
  for (uint32_t grp = 0; grp < 2; ++grp) {   // each group really is unit-RMS before the weight
    double ss = 0;
    for (uint32_t k = grp * 1280; k < (grp + 1) * 1280; ++k) {
      const double v = double(k2_ref::f32(x_ref[k])) / double(w[k]);
      ss += v * v;
    }
    CHECK(std::fabs(std::sqrt(ss / 1280.0) - 1.0) < 0.02);
  }
  std::printf("norm SP%u: resid and x bit-exact, both groups unit-RMS\n", s_prev);
}

static void silu_case(Dev& d, uint32_t inter, uint32_t slots, uint32_t slot_base,
                      uint32_t grid_slots, bool exact) {
  const size_t fused = 2 * size_t(inter);
  std::vector<float> in = k2test::rand_f32(slots * fused, inter + slot_base, 2.0f);
  if (exact)
    for (size_t i = 0; i < in.size(); ++i)
      if (((i % fused) / 16) % 2 == 0) in[i] = 30.0f;   // gate columns: even 16-blocks
  std::vector<uint16_t> ref(slots * size_t(inter), 0);
  for (uint32_t s = slot_base; s < slot_base + grid_slots; ++s)
    k2_ref::silu_mul(in.data(), ref.data(), inter, s);

  l0::Mem im = d.up(in);
  l0::Mem om = d.up(std::vector<uint16_t>(ref.size(), 0));
  l0::Kernel& k = d.kernel(kernels::k2::silu_mul(inter), "k2_silu_mul", 256);
  k.arg_ptr(0, im.ptr());
  k.arg_ptr(1, om.ptr());
  k.arg(2, slot_base);
  d.run({{&k, inter / 256, grid_slots}});
  const std::vector<uint16_t> got = d.down<uint16_t>(om, ref.size());
  const uint32_t ulp = k2test::max_ulp(got, ref);   // untouched slots are 0 on both sides
  std::printf("silu I%u slots %u base %u grid %u %s: max %u bf16 ulp\n", inter, slots, slot_base,
              grid_slots, exact ? "exact" : "random", ulp);
  if (exact) CHECK(got == ref); else CHECK(ulp <= 2);
}

int main() {
  Dev d;
  norm_case(d, 0, 11);
  norm_case(d, 1, 23);
  silu_case(d, 6144, 1, 0, 1, false);
  silu_case(d, 6144, 1, 0, 1, true);
  silu_case(d, 768, 8, 5, 1, false);   // one launch per slot (spec §3.3)
  silu_case(d, 768, 8, 0, 8, false);   // all slots in one launch, same binary
  silu_case(d, 768, 8, 0, 8, true);
  std::puts("k2_prep_test OK");
  return 0;
}
```

- [ ] **Step 3: Register** - append to `tests/CMakeLists.txt`:

```cmake
add_executable(k2_prep_test kernels/k2/k2_prep_test.cc)
target_include_directories(k2_prep_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests
                           ${CMAKE_SOURCE_DIR}/tests/kernels ${CMAKE_SOURCE_DIR}/tests/kernels/k2)
target_link_libraries(k2_prep_test PRIVATE b70_l0)
b70_target_kernel_dir(k2_prep_test)
add_dependencies(k2_prep_test kernel_prep_res_fold_M1_K2560_SP0_G10
  kernel_prep_res_fold_M1_K2560_SP1_G10 kernel_k2_norm_finish
  kernel_k2_silu_mul_I6144 kernel_k2_silu_mul_I768)
add_test(NAME k2_prep_test COMMAND k2_prep_test)
```

- [ ] **Step 4: Confirm it fails** - `tools/box.sh sync && tools/box.sh build 2>&1 | grep -m2 "No rule\|kernel_k2_norm_finish"`. Expected: CMake reports no target `kernel_k2_norm_finish`.

- [ ] **Step 5: Write `src/kernels/k2/k2_norm.cl`**

```c
// k2_norm.cl - stage B of K2-Horizon's RMSNorm (spec 4 §3.2):
//   K2HorizonRMSNorm(hidden 2560, n_groups 2, eps 1e-6): fp32; each of the two 1280-wide
//   groups has its OWN mean of squares; x·rsqrt(mean + eps); then the PLAIN weight (no 1 + w).
//
// Stage A is prep.cl's prep_res_fold compiled at K = 2560, FOLD_G = 10 (plan 8c Task 1): it
// folds the previous linear's partials into `resid` and writes one fp32 Σx² per 256-element
// chunk. Chunks 0-4 are group 0 and 5-9 group 1, so the grouping is entirely in which chunks
// this kernel sums - nothing in prep.cl knows about it.
//
//   total = Σ_{g = 5γ … 5γ+4} sumsq[g]            ascending g, γ = w / 5
//   rstd  = 1 / sqrt(total / 1280 + 1e-6)          never rsqrt (the 27B's ruling, docs/12)
//   x_out[k] = rne_bf16(f32(resid[k]) · rstd · norm_w[k])
//
// tests/kernels/k2/k2_ref.h::norm_finish repeats this chain; edit the two together.

#define K 2560
#define NGROUPS 2
#define GROUP_K (K / NGROUPS)
#define FOLD_G 10
#define CHUNK (K / FOLD_G)
#define CHUNKS_PER_GROUP (FOLD_G / NGROUPS)
#define WG 256
#if CHUNK != WG
#error "k2_norm: one chunk must be exactly one work-group (10 x 256 = 2560)"
#endif

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }
inline ushort rne_bf16(float f) {
  uint u = as_uint(f);
  uint rounding = ((u >> 16) & 1u) + 0x7FFFu;
  return (ushort)((u + rounding) >> 16);
}

__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void k2_norm_finish(__global const float* restrict sumsq,
                             __global const ushort* restrict resid,
                             __global const float* restrict norm_w,
                             __global ushort* restrict x_out) {
  const uint w = get_group_id(0);
  const uint k = w * CHUNK + get_local_id(0);
  const uint g0 = (w / CHUNKS_PER_GROUP) * CHUNKS_PER_GROUP;
  float total = 0.0f;
  for (uint g = g0; g < g0 + CHUNKS_PER_GROUP; ++g) total += sumsq[g];
  const float rstd = 1.0f / sqrt(total / (float)GROUP_K + 1e-6f);
  x_out[k] = rne_bf16(bf16f(resid[k]) * rstd * norm_w[k]);
}
```

- [ ] **Step 6: Write `src/kernels/k2/k2_silu.cl`**

```c
// k2_silu.cl - SiLU(gate) · up over a gate‖up GEMV output interleaved in 16-column blocks
// (common::cols_interleave16), for K2's dense MLP (INTER 6144) and its experts (INTER 768).
//
// Row y of the grid is slot `slot_base + y`. Launched with grid y = 1 it serves ONE slot -
// spec §3.3's per-slot launch; the same binary at grid y = 8 serves all eight at once.
//
//   gflat = slot·2·INTER + (k/16)·32 + k%16 ;  uflat = gflat + 16
//   g_b = rne_bf16(in[gflat]) ;  u_b = rne_bf16(in[uflat])       (each linear's bf16 output)
//   s_b = rne_bf16(silu_f32(f32(g_b)))                          (torch silu on bf16)
//   out[slot·INTER + k] = rne_bf16(f32(s_b) · f32(u_b))         (bf16 × bf16)
//
// silu keeps plain `exp` (3 ulp allowed) and is the only op with slack.
// tests/kernels/k2/k2_ref.h::silu_mul repeats this chain.

#ifndef INTER
#error "k2_silu: INTER (6144 dense, 768 expert) must be defined"
#endif
#if INTER % 256 != 0
#error "k2_silu: INTER must be a multiple of the 256-wide work-group"
#endif
#define FUSED (2 * INTER)
#define WG 256

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }
inline ushort rne_bf16(float f) {
  uint u = as_uint(f);
  uint rounding = ((u >> 16) & 1u) + 0x7FFFu;
  return (ushort)((u + rounding) >> 16);
}
inline float silu_f32(float x) { return x / (1.0f + exp(-x)); }

__attribute__((reqd_work_group_size(WG, 1, 1)))
__kernel void k2_silu_mul(__global const float* restrict in,
                          __global ushort* restrict out,
                          uint slot_base) {
  const uint k = get_group_id(0) * WG + get_local_id(0);
  const size_t slot = (size_t)slot_base + get_group_id(1);
  const size_t gflat = slot * FUSED + (size_t)(k / 16) * 32 + (k % 16);
  const ushort g_b = rne_bf16(in[gflat]);
  const ushort u_b = rne_bf16(in[gflat + 16]);
  const ushort s_b = rne_bf16(silu_f32(bf16f(g_b)));
  out[slot * INTER + k] = rne_bf16(bf16f(s_b) * bf16f(u_b));
}
```

- [ ] **Step 7: Register the variants** - append to `src/kernels/k2/CMakeLists.txt`:

```cmake
add_ocloc_kernel(k2_norm_finish SOURCE ${K2_DIR}/k2_norm.cl)
foreach(I 6144 768)
  add_ocloc_kernel(k2_silu_mul_I${I} SOURCE ${K2_DIR}/k2_silu.cl DEFINES INTER=${I})
endforeach()
```

- [ ] **Step 8: Build and run** - `tools/box.sh sync && tools/box.sh build 2>&1 | tail -2 && ZE_AFFINITY_MASK=1 tools/box.sh test '^k2_prep_test$'`. Expected: `norm SP0/SP1 … bit-exact`, five silu lines (exact cases 0 ulp, random ≤ 2), `k2_prep_test OK`.

- [ ] **Step 9: Commit**

```bash
git add src/kernels/k2/k2_norm.cl src/kernels/k2/k2_silu.cl src/kernels/k2/CMakeLists.txt tests/kernels/k2/k2_ref.h tests/kernels/k2/k2_prep_test.cc tests/CMakeLists.txt
git commit -m "feat(kernels): K2 grouped RMSNorm stage B and per-slot SiLU·mul, with references

Claude-Session: "
```

---

### Task 3: Router top-k

**Files:**
- Create: `src/kernels/k2/k2_route.cl`, `tests/kernels/k2/k2_route_test.cc`
- Modify: `tests/kernels/k2/k2_ref.h` (append), `src/kernels/k2/CMakeLists.txt` (append), `tests/CMakeLists.txt` (append)

**Interfaces:**
- Consumes: Task 1's defines `K2_TOPK_SUM_T4`, `K2_TOPK_SUM_T8`, `K2_TOPK_TIE_LOW`; Task 2's `k2_ref.h`.
- Produces:
  - `k2_router_topk(float* logits, float* bias, float* route)`, grid (1, 1), work-group 1. `route` is fp32 `[TOPK][2]`: `route[2t] = (float)id` and `route[2t+1] = f32(bf16 weight)` for t in ascending-id order.
  - Variant `k2_router_topk_E64_T4`: `SRC_OFF = 9216` (the v_router columns of the sparse attention GEMV's partials), `BIAS_OFF = 100` (floats into the route block).
  - Variant `k2_router_topk_E100_T8`: `SRC_OFF = 0` (the padded router GEMV's own output), `BIAS_OFF = 0`.
  - `k2_ref::router_topk(...)`.

- [ ] **Step 1: Append the reference** to `tests/kernels/k2/k2_ref.h` (inside `namespace k2_ref`, before its closing brace):

```cpp
// k2_router_topk (spec §3.2's router, calc_router_weights / K2HorizonSparseMoeBlock):
//   s[e]   = sigmoid_f32(f32(rne(logits[src_off + e])))    logits are the linear's bf16 output
//   sel[e] = s[e] + bias[bias_off + e]                      bias only selects
//   picks  = TOPK best sel, descending; equal sel -> lower id first iff tie_low
//   sum    = Σ s[picks] in pick order: sum_mode 0 left to right, 1 pairwise, 2 four lanes (k=8)
//   w      = rne((s / sum) · 2.5)
//   route  = {id, f32(w)} ascending by id
inline void router_topk(const float* logits, uint32_t src_off, const float* bias,
                        uint32_t bias_off, uint32_t n_exp, uint32_t topk, int sum_mode,
                        bool tie_low, float* route) {
  std::vector<float> s(n_exp), sel(n_exp);
  std::vector<bool> used(n_exp, false);
  for (uint32_t e = 0; e < n_exp; ++e) {
    s[e] = sigmoid_f32(f32(rne(logits[src_off + e])));
    sel[e] = s[e] + bias[bias_off + e];
  }
  std::vector<uint32_t> pick(topk);
  for (uint32_t t = 0; t < topk; ++t) {
    uint32_t best = n_exp;
    for (uint32_t e = 0; e < n_exp; ++e) {
      if (used[e]) continue;
      if (best == n_exp || (tie_low ? sel[e] > sel[best] : sel[e] >= sel[best])) best = e;
    }
    used[best] = true;
    pick[t] = best;
  }
  float sum = 0.0f;
  if (sum_mode == 0) {
    for (uint32_t t = 0; t < topk; ++t) sum += s[pick[t]];
  } else if (sum_mode == 1) {
    std::vector<float> lvl(topk);
    for (uint32_t t = 0; t < topk; ++t) lvl[t] = s[pick[t]];
    for (uint32_t w = topk; w > 1; w /= 2)
      for (uint32_t i = 0; i < w / 2; ++i) lvl[i] = lvl[2 * i] + lvl[2 * i + 1];
    sum = lvl[0];
  } else {
    sum = (s[pick[0]] + s[pick[4]]) + (s[pick[1]] + s[pick[5]]) + (s[pick[2]] + s[pick[6]]) +
          (s[pick[3]] + s[pick[7]]);
  }
  std::vector<uint32_t> asc = pick;
  std::sort(asc.begin(), asc.end());
  for (uint32_t t = 0; t < topk; ++t) {
    route[2 * t] = float(asc[t]);
    route[2 * t + 1] = f32(rne((s[asc[t]] / sum) * 2.5f));
  }
}
```

- [ ] **Step 2: Write the failing test** - `tests/kernels/k2/k2_route_test.cc`:

```cpp
// k2_route.cl against k2_ref::router_topk, both routers, 200 random trials each plus a
// constructed tie at the k-th cut (spec §8 risk 2). Bars: ids exact and ascending; weights
// within 1 bf16 ulp (sigmoid's exp is the only slack).
#include <cstdio>
#include <vector>

#include "k2_dev.h"
#include "k2_ref.h"
#include "kernels/k2/k2_kernels.h"

using k2test::Dev;

static void trials(Dev& d, uint32_t n_exp, uint32_t topk, uint32_t src_off, uint32_t src_n,
                   uint32_t bias_off, int sum_mode) {
  l0::Mem lm(d.ctx, l0::MemKind::Device, src_n * 4);
  l0::Mem bm(d.ctx, l0::MemKind::Device, 164 * 4);
  l0::Mem rm(d.ctx, l0::MemKind::Device, topk * 2 * 4);
  l0::Kernel& k = d.kernel(kernels::k2::router_topk(n_exp, topk), "k2_router_topk", 1);
  k.arg_ptr(0, lm.ptr());
  k.arg_ptr(1, bm.ptr());
  k.arg_ptr(2, rm.ptr());
  uint32_t worst_ulp = 0;
  for (uint32_t trial = 0; trial <= 200; ++trial) {
    std::vector<float> logits = k2test::rand_f32(src_n, 1000 + trial, 2.0f);
    std::vector<float> bias = k2test::rand_f32(164, 5000 + trial, 0.02f);
    const bool tie = trial == 200;
    if (tie) {   // topk-1 clear winners, ids 3 and 7 tied for the last place, the rest far below
      for (uint32_t e = 0; e < n_exp; ++e) {
        logits[src_off + e] = e >= 10 && e < 10 + topk - 1 ? 8.0f : -8.0f;
        bias[bias_off + e] = 0.0f;
      }
      logits[src_off + 3] = logits[src_off + 7] = 1.0f;
    }
    std::vector<float> ref(topk * 2);
    k2_ref::router_topk(logits.data(), src_off, bias.data(), bias_off, n_exp, topk, sum_mode,
                        K2_TOPK_TIE_LOW != 0, ref.data());
    d.imm.copy(lm.ptr(), logits.data(), logits.size() * 4);
    d.imm.copy(bm.ptr(), bias.data(), bias.size() * 4);
    d.run({{&k, 1, 1}});
    const std::vector<float> got = d.down<float>(rm, topk * 2);
    for (uint32_t t = 0; t < topk; ++t) {
      CHECK(got[2 * t] == ref[2 * t]);
      if (t > 0) CHECK(got[2 * t] > got[2 * t - 2]);
      const int32_t du = k2test::bf16_key(k2_ref::rne(got[2 * t + 1])) -
                         k2test::bf16_key(k2_ref::rne(ref[2 * t + 1]));
      worst_ulp = std::max(worst_ulp, uint32_t(du < 0 ? -du : du));
    }
    if (tie) {
      const float want = K2_TOPK_TIE_LOW ? 3.0f : 7.0f, other = K2_TOPK_TIE_LOW ? 7.0f : 3.0f;
      bool has_want = false, has_other = false;
      for (uint32_t t = 0; t < topk; ++t) {
        has_want |= got[2 * t] == want;
        has_other |= got[2 * t] == other;
      }
      CHECK(has_want && !has_other);
    }
  }
  std::printf("router E%u T%u sum_mode %d: 201 trials, ids exact, weights max %u bf16 ulp\n",
              n_exp, topk, sum_mode, worst_ulp);
  CHECK(worst_ulp <= 1);
}

int main() {
  Dev d;
  trials(d, 64, 4, 9216, 9280, 100, K2_TOPK_SUM_T4);
  trials(d, 100, 8, 0, 128, 0, K2_TOPK_SUM_T8);
  std::puts("k2_route_test OK");
  return 0;
}
```

- [ ] **Step 3: Register** - append to `tests/CMakeLists.txt`:

```cmake
add_executable(k2_route_test kernels/k2/k2_route_test.cc)
target_include_directories(k2_route_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests
                           ${CMAKE_SOURCE_DIR}/tests/kernels/k2)
target_compile_definitions(k2_route_test PRIVATE K2_TOPK_SUM_T4=${K2_TOPK_SUM_T4}
                           K2_TOPK_SUM_T8=${K2_TOPK_SUM_T8} K2_TOPK_TIE_LOW=${K2_TOPK_TIE_LOW})
target_link_libraries(k2_route_test PRIVATE b70_l0)
b70_target_kernel_dir(k2_route_test)
add_dependencies(k2_route_test kernel_k2_router_topk_E64_T4 kernel_k2_router_topk_E100_T8)
add_test(NAME k2_route_test COMMAND k2_route_test)
```

- [ ] **Step 4: Confirm it fails** - build; expected: no target `kernel_k2_router_topk_E64_T4`.

- [ ] **Step 5: Write `src/kernels/k2/k2_route.cl`**

```c
// k2_route.cl - both routers of a sparse K2 layer (spec 4 §3.2, §3.3 option A):
//   MoVA: top 4 of 64, logits = the v_router columns of the attention GEMV (int4);
//   MoE:  top 8 of 100, logits = the bf16 router GEMV's first 100 of 128 rows.
//
//   s[e]   = sigmoid_f32(f32(rne_bf16(logits[SRC_OFF + e])))
//   sel[e] = s[e] + bias[BIAS_OFF + e]                 the bias selects, it never weights
//   picks  = TOPK largest sel, descending; equal sel -> lower id first (K2_TOPK_TIE_LOW)
//   sum    = Σ s over picks IN PICK ORDER               (K2_TOPK_SUM: torch's reduction order)
//   w      = rne_bf16((s / sum) · 2.5)                  fp32 divide, fp32 scale, one rounding
//   route[2t], route[2t+1] = (float)id, f32(w)          t ascending BY ID
//
// Ascending id is the slot order: slot t runs the t-th smallest id, and the combines sum slots
// in ascending t - the reference's expert loop order. The id travels as a float, exact below
// 2^24 (argmax.cl's precedent). One work-item: a 100-element scalar walk has no order to state.
// tests/kernels/k2/k2_ref.h::router_topk repeats this chain.

#ifndef N_EXP
#error "k2_route: N_EXP must be defined"
#endif
#ifndef TOPK
#error "k2_route: TOPK must be defined"
#endif
#ifndef SRC_OFF
#error "k2_route: SRC_OFF must be defined"
#endif
#ifndef BIAS_OFF
#error "k2_route: BIAS_OFF must be defined"
#endif
#ifndef K2_TOPK_SUM
#error "k2_route: K2_TOPK_SUM must be defined (plan 8a facts §6)"
#endif
#ifndef K2_TOPK_TIE_LOW
#error "k2_route: K2_TOPK_TIE_LOW must be defined (plan 8a facts §6)"
#endif
#if K2_TOPK_SUM == 2 && TOPK != 8
#error "k2_route: the four-lane sum is defined for TOPK 8 only"
#endif
#define ROUTE_SCALE 2.5f

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }
inline ushort rne_bf16(float f) {
  uint u = as_uint(f);
  uint rounding = ((u >> 16) & 1u) + 0x7FFFu;
  return (ushort)((u + rounding) >> 16);
}

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void k2_router_topk(__global const float* restrict logits,
                             __global const float* restrict bias,
                             __global float* restrict route) {
  float s[N_EXP];
  float sel[N_EXP];
  uchar used[N_EXP];
  for (uint e = 0; e < N_EXP; ++e) {
    const float l = bf16f(rne_bf16(logits[SRC_OFF + e]));
    s[e] = 1.0f / (1.0f + exp(-l));
    sel[e] = s[e] + bias[BIAS_OFF + e];
    used[e] = 0;
  }
  uint pick[TOPK];
  for (uint t = 0; t < TOPK; ++t) {
    uint best = N_EXP;
    for (uint e = 0; e < N_EXP; ++e) {
      if (used[e]) continue;
#if K2_TOPK_TIE_LOW
      if (best == N_EXP || sel[e] > sel[best]) best = e;
#else
      if (best == N_EXP || sel[e] >= sel[best]) best = e;
#endif
    }
    used[best] = 1;
    pick[t] = best;
  }
  float sum = 0.0f;
#if K2_TOPK_SUM == 0
  for (uint t = 0; t < TOPK; ++t) sum += s[pick[t]];
#elif K2_TOPK_SUM == 1
  float lvl[TOPK];
  for (uint t = 0; t < TOPK; ++t) lvl[t] = s[pick[t]];
  for (uint w = TOPK; w > 1; w /= 2)
    for (uint i = 0; i < w / 2; ++i) lvl[i] = lvl[2 * i] + lvl[2 * i + 1];
  sum = lvl[0];
#else
  sum = (s[pick[0]] + s[pick[4]]) + (s[pick[1]] + s[pick[5]]) + (s[pick[2]] + s[pick[6]]) +
        (s[pick[3]] + s[pick[7]]);
#endif
  for (uint a = 1; a < TOPK; ++a) {   // insertion sort: ascending id
    const uint v = pick[a];
    int b = (int)a - 1;
    while (b >= 0 && pick[b] > v) {
      pick[b + 1] = pick[b];
      --b;
    }
    pick[b + 1] = v;
  }
  for (uint t = 0; t < TOPK; ++t) {
    route[2 * t] = (float)pick[t];
    route[2 * t + 1] = bf16f(rne_bf16((s[pick[t]] / sum) * ROUTE_SCALE));
  }
}
```

- [ ] **Step 6: Register the variants** - append to `src/kernels/k2/CMakeLists.txt`:

```cmake
# BIAS_OFF is in FLOATS into the route block (model::K2Horizon::kRouteOffMovaBias / 4 = 100).
add_ocloc_kernel(k2_router_topk_E64_T4 SOURCE ${K2_DIR}/k2_route.cl
                 DEFINES N_EXP=64 TOPK=4 SRC_OFF=9216 BIAS_OFF=100
                         K2_TOPK_SUM=${K2_TOPK_SUM_T4} K2_TOPK_TIE_LOW=${K2_TOPK_TIE_LOW})
add_ocloc_kernel(k2_router_topk_E100_T8 SOURCE ${K2_DIR}/k2_route.cl
                 DEFINES N_EXP=100 TOPK=8 SRC_OFF=0 BIAS_OFF=0
                         K2_TOPK_SUM=${K2_TOPK_SUM_T8} K2_TOPK_TIE_LOW=${K2_TOPK_TIE_LOW})
```

- [ ] **Step 7: Build and run** - `tools/box.sh sync && tools/box.sh build 2>&1 | tail -2 && ZE_AFFINITY_MASK=1 tools/box.sh test '^k2_route_test$'`. Expected: two `router …: 201 trials, ids exact` lines, `k2_route_test OK`.

- [ ] **Step 8: Commit**

```bash
git add src/kernels/k2/k2_route.cl src/kernels/k2/CMakeLists.txt tests/kernels/k2/k2_ref.h tests/kernels/k2/k2_route_test.cc tests/CMakeLists.txt
git commit -m "feat(kernels): K2 router top-k - selection-only bias, ascending-id slots, bf16 weights

Claude-Session: "
```

---

### Task 4: The expert slot GEMV and the two combines

**Files:**
- Create: `src/kernels/k2/k2_experts.cl`, `tests/kernels/k2/k2_experts_test.cc`
- Modify: `tests/kernels/k2/k2_ref.h` (append), `src/kernels/k2/CMakeLists.txt` (append), `tests/CMakeLists.txt` (append)

**Interfaces:**
- Consumes: Task 3's route layout; `src/kernels/gemv.cl`'s layout-0 arithmetic (copied, not included); `common::Int4Gptq::random`; Task 1's `gemv_M1_K2560_N1024_S1_L0`, `gemv_M1_K2560_N1536_S1_L0` and `gemv_M1_K768_N2560_S1_L0`.
- Produces:
  - `k2_gemv_slot(uint* w [E][K/8][N], half* scales [E][K/64][N], float* route [TOPK][2], ushort* x, float* out [TOPK][N], uint slot_base)`, grid (N/64, slots), work-group 64.
    - `X_PER_SLOT = 0`: x is `[K]`. Variants `k2_gemv_slot_K2560_N1024` (value experts, EXPERTS 64) and `k2_gemv_slot_K2560_N1536` (MoE gate‖up, EXPERTS 100).
    - `X_PER_SLOT = 1`: x is `[TOPK][K]`. Variant `k2_gemv_slot_K768_N2560` (MoE down, EXPERTS 100).
  - `k2_value_combine(float* slots [4][1024], float* route [4][2], ushort* vbuf [1024])`, grid (16, 1), work-group 64.
  - `k2_moe_combine(float* down_slots [8][2560], float* route [8][2], float* io [2560])`, grid (10, 1), work-group 256. `io` holds the shared expert's down partials on entry and the MoE block's output (a bf16 word widened to fp32) on exit.
  - `k2_ref::value_combine`, `k2_ref::moe_combine`.

- [ ] **Step 1: Append the references** to `tests/kernels/k2/k2_ref.h` (inside the namespace):

```cpp
// k2_value_combine (MoVA, combine_routed_experts with activation = silu):
//   per element k, acc_b = +0; for t ascending:
//     e_b = rne(slots[t][k]); a_b = rne(silu(f32(e_b))); p_b = rne(f32(a_b) · w_t)
//     acc_b = rne(f32(acc_b) + f32(p_b))                  index_add_ in bf16
inline void value_combine(const float* slots, const float* route, uint16_t* vbuf) {
  for (uint32_t k = 0; k < 1024; ++k) {
    uint16_t acc = 0;
    for (uint32_t t = 0; t < 4; ++t) {
      const uint16_t e_b = rne(slots[size_t(t) * 1024 + k]);
      const uint16_t a_b = rne(silu_f32(f32(e_b)));
      const uint16_t p_b = rne(f32(a_b) * route[2 * t + 1]);
      acc = rne(f32(acc) + f32(p_b));
    }
    vbuf[k] = acc;
  }
}

// k2_moe_combine (K2HorizonSparseMoeBlock): the routed sum in ascending id, then + shared.
//   acc_b = Σ_t rne(f32(rne(down[t][k])) · w_t), each add rounded, from +0
//   io[k] = f32(rne(f32(acc_b) + f32(rne(io[k]))))       io held the shared expert's partials
inline void moe_combine(const float* down, const float* route, float* io) {
  for (uint32_t k = 0; k < 2560; ++k) {
    uint16_t acc = 0;
    for (uint32_t t = 0; t < 8; ++t) {
      const uint16_t d_b = rne(down[size_t(t) * 2560 + k]);
      const uint16_t p_b = rne(f32(d_b) * route[2 * t + 1]);
      acc = rne(f32(acc) + f32(p_b));
    }
    io[k] = f32(rne(f32(acc) + f32(rne(io[k]))));
  }
}
```

- [ ] **Step 2: Write the failing test** - `tests/kernels/k2/k2_experts_test.cc`:

```cpp
// k2_experts.cl. Bars (spec §4):
//   k2_gemv_slot - BITWISE equal to gemv.cl run on the selected expert's own buffer, per slot,
//                  in both launch forms (one launch per slot, one launch for all slots);
//                  an out-of-range id writes NaN instead of reading past the buffer.
//   k2_moe_combine - bit-exact (no exp).
//   k2_value_combine - 4 bf16 ulp on random slots (one silu per term, four rounded adds);
//                      bit-exact where every slot value is 30.0f (silu(30) == 30 exactly).
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "common/int4.h"
#include "k2_dev.h"
#include "k2_ref.h"
#include "kernels/k2/k2_kernels.h"

using k2test::Dev;

struct Flat {
  uint32_t K, N, E;
  std::vector<uint32_t> words;
  std::vector<uint16_t> scales;
  std::vector<common::Int4Gptq> each;
};

static Flat make_experts(uint32_t K, uint32_t N, uint32_t E, uint32_t seed) {
  Flat f{K, N, E, {}, {}, {}};
  for (uint32_t e = 0; e < E; ++e) {
    f.each.push_back(common::Int4Gptq::random(K, N, seed + e));
    f.words.insert(f.words.end(), f.each.back().qweight.begin(), f.each.back().qweight.end());
    f.scales.insert(f.scales.end(), f.each.back().scales.begin(), f.each.back().scales.end());
  }
  return f;
}

static void slot_case(Dev& d, uint32_t K, uint32_t N, bool x_per_slot, uint32_t experts,
                      const std::vector<uint32_t>& ids) {
  const uint32_t topk = uint32_t(ids.size());
  const Flat ex = make_experts(K, N, 10, 300 + N);
  const std::vector<uint16_t> x = k2test::rand_bf16((x_per_slot ? topk : 1) * size_t(K), 9 + K, -1, 1);
  std::vector<float> route(topk * 2);
  for (uint32_t t = 0; t < topk; ++t) route[2 * t] = float(ids[t]), route[2 * t + 1] = 0.5f;
  l0::Mem wm = d.up(ex.words), sm = d.up(ex.scales), rm = d.up(route), xm = d.up(x);
  const std::vector<float> canary(size_t(topk) * N, 7.0f);
  l0::Mem per = d.up(canary), all = d.up(canary);

  // (1) one launch per slot, each a separate Kernel like runtime capture's sites.
  for (uint32_t t = 0; t < topk; ++t) {
    l0::Kernel& k = d.kernel(kernels::k2::gemv_slot(K, N), "k2_gemv_slot", 64);
    k.arg_ptr(0, wm.ptr());
    k.arg_ptr(1, sm.ptr());
    k.arg_ptr(2, rm.ptr());
    k.arg_ptr(3, xm.ptr());
    k.arg_ptr(4, per.ptr());
    k.arg(5, t);
    d.run({{&k, N / 64, 1}});
  }
  // (2) one launch for every slot.
  {
    l0::Kernel& k = d.kernel(kernels::k2::gemv_slot(K, N), "k2_gemv_slot", 64);
    k.arg_ptr(0, wm.ptr());
    k.arg_ptr(1, sm.ptr());
    k.arg_ptr(2, rm.ptr());
    k.arg_ptr(3, xm.ptr());
    k.arg_ptr(4, all.ptr());
    k.arg(5, uint32_t(0));
    d.run({{&k, N / 64, topk}});
  }
  const std::vector<float> got_per = d.down<float>(per, size_t(topk) * N);
  const std::vector<float> got_all = d.down<float>(all, size_t(topk) * N);
  CHECK(std::memcmp(got_per.data(), got_all.data(), got_per.size() * 4) == 0);

  // (3) the reference: gemv.cl itself on expert ids[t]'s standalone buffer.
  for (uint32_t t = 0; t < topk; ++t) {
    const common::Int4Gptq& w = ex.each[ids[t]];
    l0::Mem ew = d.up(w.qweight), es = d.up(w.scales);
    const std::vector<uint16_t> xs(x.begin() + (x_per_slot ? t * size_t(K) : 0),
                                   x.begin() + (x_per_slot ? t * size_t(K) : 0) + K);
    l0::Mem exm = d.up(xs);
    l0::Mem om(d.ctx, l0::MemKind::Device, size_t(N) * 4);
    l0::Kernel& g = d.kernel(kernels::k2::gemv(K, N), "gemv", 64);
    g.arg_ptr(0, ew.ptr());
    g.arg_ptr(1, es.ptr());
    g.arg_ptr(2, exm.ptr());
    g.arg_ptr(3, om.ptr());
    d.run({{&g, N / 64, 1}});
    const std::vector<float> ref = d.down<float>(om, N);
    CHECK(std::memcmp(ref.data(), got_per.data() + size_t(t) * N, size_t(N) * 4) == 0);
  }

  // (4) an id past the model's expert count: NaN, never an out-of-bounds read.
  route[0] = float(experts);
  d.imm.copy(rm.ptr(), route.data(), route.size() * 4);
  l0::Kernel& kb = d.kernel(kernels::k2::gemv_slot(K, N), "k2_gemv_slot", 64);
  kb.arg_ptr(0, wm.ptr());
  kb.arg_ptr(1, sm.ptr());
  kb.arg_ptr(2, rm.ptr());
  kb.arg_ptr(3, xm.ptr());
  kb.arg_ptr(4, per.ptr());
  kb.arg(5, uint32_t(0));
  d.run({{&kb, N / 64, 1}});
  const std::vector<float> bad = d.down<float>(per, N);
  for (uint32_t n = 0; n < N; ++n) CHECK(std::isnan(bad[n]));
  std::printf("gemv_slot K=%u N=%u X%d: %u slots bitwise == gemv, both launch forms; bad id -> NaN\n",
              K, N, int(x_per_slot), topk);
}

static void value_case(Dev& d, bool exact) {
  std::vector<float> slots = k2test::rand_f32(4 * 1024, exact ? 1 : 2, 2.0f);
  if (exact) std::fill(slots.begin(), slots.end(), 30.0f);
  const std::vector<float> route = {3.0f, 0.5f, 17.0f, 0.75f, 40.0f, 0.625f, 63.0f, 0.125f};
  std::vector<uint16_t> ref(1024);
  k2_ref::value_combine(slots.data(), route.data(), ref.data());
  l0::Mem sm = d.up(slots), rm = d.up(route);
  l0::Mem vm(d.ctx, l0::MemKind::Device, 1024 * 2);
  l0::Kernel& k = d.kernel(kernels::k2::value_combine(), "k2_value_combine", 64);
  k.arg_ptr(0, sm.ptr());
  k.arg_ptr(1, rm.ptr());
  k.arg_ptr(2, vm.ptr());
  d.run({{&k, 16, 1}});
  const std::vector<uint16_t> got = d.down<uint16_t>(vm, 1024);
  const uint32_t ulp = k2test::max_ulp(got, ref);
  std::printf("value_combine %s: max %u bf16 ulp\n", exact ? "exact" : "random", ulp);
  if (exact) CHECK(got == ref); else CHECK(ulp <= 4);
}

static void moe_case(Dev& d) {
  const std::vector<float> down = k2test::rand_f32(8 * 2560, 3, 1.0f);
  std::vector<float> io = k2test::rand_f32(2560, 4, 1.0f);
  std::vector<float> route(16);
  for (uint32_t t = 0; t < 8; ++t)
    route[2 * t] = float(t * 11), route[2 * t + 1] = k2_ref::f32(k2_ref::rne(0.05f + 0.1f * t));
  std::vector<float> ref = io;
  k2_ref::moe_combine(down.data(), route.data(), ref.data());
  l0::Mem dm = d.up(down), rm = d.up(route), im = d.up(io);
  l0::Kernel& k = d.kernel(kernels::k2::moe_combine(), "k2_moe_combine", 256);
  k.arg_ptr(0, dm.ptr());
  k.arg_ptr(1, rm.ptr());
  k.arg_ptr(2, im.ptr());
  d.run({{&k, 10, 1}});
  const std::vector<float> got = d.down<float>(im, 2560);
  CHECK(std::memcmp(got.data(), ref.data(), got.size() * 4) == 0);
  std::puts("moe_combine: bit-exact");
}

int main() {
  Dev d;
  slot_case(d, 2560, 1024, false, 64, {1, 4, 7, 9});
  slot_case(d, 2560, 1536, false, 100, {0, 1, 3, 4, 5, 6, 8, 9});
  slot_case(d, 768, 2560, true, 100, {0, 2, 3, 4, 5, 7, 8, 9});
  value_case(d, false);
  value_case(d, true);
  moe_case(d);
  std::puts("k2_experts_test OK");
  return 0;
}
```

- [ ] **Step 3: Register** - append to `tests/CMakeLists.txt`:

```cmake
add_executable(k2_experts_test kernels/k2/k2_experts_test.cc)
target_include_directories(k2_experts_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests
                           ${CMAKE_SOURCE_DIR}/tests/kernels/k2)
target_link_libraries(k2_experts_test PRIVATE b70_l0)
b70_target_kernel_dir(k2_experts_test)
add_dependencies(k2_experts_test kernel_k2_gemv_slot_K2560_N1024 kernel_k2_gemv_slot_K2560_N1536
  kernel_k2_gemv_slot_K768_N2560 kernel_k2_value_combine kernel_k2_moe_combine
  kernel_gemv_M1_K2560_N1024_S1_L0 kernel_gemv_M1_K2560_N1536_S1_L0 kernel_gemv_M1_K768_N2560_S1_L0)
add_test(NAME k2_experts_test COMMAND k2_experts_test)
```

- [ ] **Step 4: Confirm it fails** - build; expected: no target `kernel_k2_gemv_slot_K2560_N1024`.

- [ ] **Step 5: Write `src/kernels/k2/k2_experts.cl`**

```c
// k2_experts.cl - routed experts under a replayed list (spec 4 §3.3, option A).
//
// k2_gemv_slot: one int4 GEMV of ONE routed expert. The launch's arguments are frozen; the
// expert is not. Slot `slot = slot_base + get_group_id(1)` reads its id from the router buffer
// ON THE DEVICE and indexes the layer's flat expert buffer at id × stride. Launched with grid
// y = 1 it is one slot (spec §3.3); grid y = TOPK runs every slot in one launch, same binary.
//
// The arithmetic is gemv.cl's layout-0 path at M = 1, S = 1, statement for statement (dot8 is
// copied verbatim from gemv.cl's non-DEQ_SHIFT branch): the same 16-lane subgroup over output
// columns, the same ascending g and j, the same `gacc` per group times its scale. That is what
// lets k2_experts_test hold every slot BITWISE equal to gemv.cl run on the expert alone.
//
//   id    = (uint)route[2·slot]                  exact below 2^24
//   w_id  = w + id·(K/8)·N ;  s_id = scales + id·(K/64)·N
//   x_s   = X_PER_SLOT ? x + slot·K : x           down reads its own slot's SiLU·mul
//   out[slot·N + n] = Σ_g scale(g, n) · Σ_j dot8(w_id[(8g + j)·N + n], x_s[64g + 8j .. +8])
//
// An id >= EXPERTS cannot come from k2_router_topk; if one ever arrives the slot writes NaN
// rather than reading past the buffer, so the corruption surfaces in the next norm.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#pragma OPENCL EXTENSION cl_intel_subgroups_short : enable

#define SG 16
#define WG_SLOT 64
#define GROUP 64

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }
inline ushort rne_bf16(float f) {
  uint u = as_uint(f);
  uint rounding = ((u >> 16) & 1u) + 0x7FFFu;
  return (ushort)((u + rounding) >> 16);
}
inline float silu_f32(float x) { return x / (1.0f + exp(-x)); }

#ifdef SLOT_KERNEL
#define G (K / GROUP)
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
__attribute__((reqd_work_group_size(WG_SLOT, 1, 1)))
__kernel void k2_gemv_slot(__global const uint* restrict w,
                           __global const half* restrict scales,
                           __global const float* restrict route,
                           __global const ushort* restrict x,
                           __global float* restrict out,
                           uint slot_base) {
  const uint lane = get_sub_group_local_id();
  const uint n_tile = get_group_id(0) * (WG_SLOT / SG) + get_sub_group_id();
  const uint n = n_tile * SG + lane;
  const uint slot = slot_base + get_group_id(1);
  const uint id = (uint)route[2 * slot];
  if (id >= EXPERTS) {
    out[(size_t)slot * N + n] = NAN;
    return;
  }
  const size_t wbase = (size_t)id * (K / 8) * N;
  const size_t sbase = (size_t)id * G * N;
#if X_PER_SLOT
  __global const ushort* xs = x + (size_t)slot * K;
#else
  __global const ushort* xs = x;
#endif
  float acc = 0.f;
  for (uint g = 0; g < G; ++g) {
    __global const uint* wp = w + wbase + (size_t)(g * 8) * N + n;
    uint wv[8];
    for (int j = 0; j < 8; ++j) wv[j] = wp[(size_t)j * N];
    const float scale = (float)scales[sbase + (size_t)g * N + n];
    float gacc = 0.f;
    for (int j = 0; j < 8; ++j) {
      ushort8 xv = vload8(0, xs + g * GROUP + j * 8);
      gacc += dot8(wv[j], xv);
    }
    acc += gacc * scale;
  }
  out[(size_t)slot * N + n] = acc;
}
#endif  // SLOT_KERNEL

#ifdef COMBINE_KERNELS
// k2_value_combine - MoVA's mixed value (combine_routed_experts, activation = silu), grid
// (16, 1), work-group 64, one element per work-item. Slots are ascending id, so ascending t is
// the reference's index_add_ order.
__attribute__((reqd_work_group_size(64, 1, 1)))
__kernel void k2_value_combine(__global const float* restrict slots,
                               __global const float* restrict route,
                               __global ushort* restrict vbuf) {
  const uint k = get_group_id(0) * 64 + get_local_id(0);
  ushort acc = 0;
  for (uint t = 0; t < 4; ++t) {
    const ushort e_b = rne_bf16(slots[(size_t)t * 1024 + k]);
    const ushort a_b = rne_bf16(silu_f32(bf16f(e_b)));
    const ushort p_b = rne_bf16(bf16f(a_b) * route[2 * t + 1]);
    acc = rne_bf16(bf16f(acc) + bf16f(p_b));
  }
  vbuf[k] = acc;
}

// k2_moe_combine - the MoE block's output, grid (10, 1), work-group 256, in place on `io`,
// which holds the shared expert's down partials on entry. Each work-item reads and writes only
// its own element, so in place is safe. The result is a bf16 word widened to fp32: the next
// prep_res_fold (S_PREV 1) rounds it to itself exactly and adds it to the residual.
__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void k2_moe_combine(__global const float* restrict down_slots,
                             __global const float* restrict route,
                             __global float* restrict io) {
  const uint k = get_group_id(0) * 256 + get_local_id(0);
  ushort acc = 0;
  for (uint t = 0; t < 8; ++t) {
    const ushort d_b = rne_bf16(down_slots[(size_t)t * 2560 + k]);
    const ushort p_b = rne_bf16(bf16f(d_b) * route[2 * t + 1]);
    acc = rne_bf16(bf16f(acc) + bf16f(p_b));
  }
  const ushort sh_b = rne_bf16(io[k]);
  io[k] = bf16f(rne_bf16(bf16f(acc) + bf16f(sh_b)));
}
#endif  // COMBINE_KERNELS
```

- [ ] **Step 6: Register the variants** - append to `src/kernels/k2/CMakeLists.txt`:

```cmake
add_ocloc_kernel(k2_gemv_slot_K2560_N1024 SOURCE ${K2_DIR}/k2_experts.cl
                 DEFINES SLOT_KERNEL=1 K=2560 N=1024 X_PER_SLOT=0 EXPERTS=64)
add_ocloc_kernel(k2_gemv_slot_K2560_N1536 SOURCE ${K2_DIR}/k2_experts.cl
                 DEFINES SLOT_KERNEL=1 K=2560 N=1536 X_PER_SLOT=0 EXPERTS=100)
add_ocloc_kernel(k2_gemv_slot_K768_N2560 SOURCE ${K2_DIR}/k2_experts.cl
                 DEFINES SLOT_KERNEL=1 K=768 N=2560 X_PER_SLOT=1 EXPERTS=100)
add_ocloc_kernel(k2_value_combine SOURCE ${K2_DIR}/k2_experts.cl DEFINES COMBINE_KERNELS=1)
add_ocloc_kernel(k2_moe_combine SOURCE ${K2_DIR}/k2_experts.cl DEFINES COMBINE_KERNELS=1)
```

- [ ] **Step 7: Build and run** - `tools/box.sh sync && tools/box.sh build 2>&1 | tail -2 && ZE_AFFINITY_MASK=1 tools/box.sh test '^k2_experts_test$'`. Expected: three `gemv_slot … bitwise == gemv` lines, two `value_combine` lines (exact: 0 ulp), `moe_combine: bit-exact`, `k2_experts_test OK`.
  **If a slot is not bitwise equal to `gemv`:** that is one defect (a codegen difference between two compilations of the same arithmetic, or an indexing error). Print the first differing (slot, n) and both values, and report. Do not relax the bar to a tolerance: spec §4 names bitwise.

- [ ] **Step 8: Commit**

```bash
git add src/kernels/k2/k2_experts.cl src/kernels/k2/CMakeLists.txt tests/kernels/k2/k2_ref.h tests/kernels/k2/k2_experts_test.cc tests/CMakeLists.txt
git commit -m "feat(kernels): K2 expert slot GEMV (id read on device) and the value/MoE combines

Claude-Session: "
```

---

### Task 5: The K2 attention trio

**Files:**
- Create: `src/kernels/k2/k2_attn.cl`, `tests/kernels/k2/k2_attn_ref.h`, `tests/kernels/k2/k2_attn_test.cc`
- Modify: `src/kernels/k2/CMakeLists.txt` (append), `tests/CMakeLists.txt` (append)

**Interfaces:**
- Consumes: `runtime::Control` (`pos` at `CTRL_POS`, `n_active` at `CTRL_NACT`); plan 8b's `loader::k2_rope_table` for the test's RoPE rows (link `b70_k2_loader`); Task 1's `K2_SCORE_SCALE` and `K2_SOFTPLUS_BETA`.
- Produces:
  - `k2_attn_prep(uint* ctrl, float* partials, ushort* vbuf [1024], float* rope [L][2][64], float* attn_q [32][128], float* attn_gate [32][128], ushort* kv_k, ushort* kv_v)`, grid (40, 1), work-group 128. Variants `k2_attn_prep_dense` (v from partials at 9216) and `k2_attn_prep_sparse` (v from `vbuf`).
  - `k2_attn_decode(uint* ctrl, float* attn_q, ushort* kv_k, ushort* kv_v, float* attn_part [32][L/64][130])`, grid (8, L/64), work-group 128 = 8 subgroups × 16 lanes.
  - `k2_attn_reduce(uint* ctrl, float* attn_part, float* attn_gate, ushort* attn_out [4096])`, grid (32, 1), work-group 128.
  - Variants at L = 16384 (production) and L = 4096 (test). Each layer's `kv_k`/`kv_v` argument points at that layer's slice `[L][8][128]` bf16.

**What is the 27B's attn.cl, and what changes.** The decode kernel's structure and its four stated orders are attn.cl's:
- one work-group per (kv-head, 64-position block);
- register-packed GQA;
- the lane dot tree;
- the online softmax in waves;
- the early-out;
- the ascending block merge.

What K2 changes:
- **Head shape:** head_dim 128 and GQA 4 (32 q-heads, 8 kv-heads). The work-group is 128 wide (8 subgroups × 16 lanes), so a wave covers 8 positions and a 64-position block takes 8 waves.
- **No q/k norm, full rotary:** RoPE covers all 128 dims with rotate_half halves (pairs `i`, `i+64`).
- **RoPE in torch's order:** the reference rotates in bf16 with bf16 cos/sin, `rne(rne(x·c) + rne(rot·s))`.
- **Score scale:** `K2_SCORE_SCALE`.
- **Gate:** softplus (β = ln 2, threshold 20) instead of sigmoid.
- **v source:** v comes from the dense row's columns or from MoVA's `vbuf`, fixed per variant.

- [ ] **Step 1: Write the reference** - `tests/kernels/k2/k2_attn_ref.h`:

```cpp
#pragma once
// Host reference for src/kernels/k2/k2_attn.cl at M = 1 - the same chain in the same order;
// edit the two together. Every order statement of k2_attn.cl's header is implemented here.
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "common/bf16.h"

namespace k2_attn_ref {

constexpr uint32_t kQ = 32, kKv = 8, kGqa = 4, kHd = 128, kHalf = 64;
constexpr uint32_t kSg = 16, kWaveP = 8, kPerLane = 8, kBlock = 64, kWaves = 8, kPart = 130;
constexpr uint32_t kKOff = 4096, kGateOff = 5120, kVOff = 9216;

inline float f32(uint16_t h) { return common::bf16_to_f32(h); }
inline uint16_t rne(float f) { return common::f32_to_bf16(f); }
inline float ninf() { return -std::numeric_limits<float>::infinity(); }
inline float softplus_f32(float x, float beta) {
  const float bx = x * beta;
  return bx > 20.0f ? x : std::log1p(std::exp(bx)) / beta;
}

// prep: q, k roped (rotate_half over 128, bf16 chain); attn_gate = f32(rne(gate)); kv at pos.
inline void prep(uint32_t pos, const float* partials, const uint16_t* vbuf, bool dense,
                 const float* rope, float* attn_q, float* attn_gate, uint16_t* kv_k,
                 uint16_t* kv_v) {
  const float* cs = rope + size_t(pos) * 2 * kHalf;
  auto roped = [&](size_t base, uint32_t i) -> uint16_t {
    const uint32_t ii = i < kHalf ? i : i - kHalf;
    const float c = cs[ii], s = cs[kHalf + ii];
    const float xi = f32(rne(partials[base + i]));
    const float xo = f32(rne(partials[base + (i < kHalf ? i + kHalf : i - kHalf)]));
    const float rot = i < kHalf ? -xo : xo;
    return rne(f32(rne(xi * c)) + f32(rne(rot * s)));
  };
  for (uint32_t h = 0; h < kQ; ++h)
    for (uint32_t i = 0; i < kHd; ++i) {
      attn_q[h * kHd + i] = f32(roped(size_t(h) * kHd, i));
      attn_gate[h * kHd + i] = f32(rne(partials[kGateOff + size_t(h) * kHd + i]));
    }
  for (uint32_t j = 0; j < kKv; ++j)
    for (uint32_t i = 0; i < kHd; ++i) {
      const size_t slot = (size_t(pos) * kKv + j) * kHd + i;
      kv_k[slot] = roped(kKOff + size_t(j) * kHd, i);
      kv_v[slot] = dense ? rne(partials[kVOff + size_t(j) * kHd + i]) : vbuf[size_t(j) * kHd + i];
    }
}

// decode: per (kv-head j, block), per q-head of its GQA group, waves of 8 positions.
inline void decode(uint32_t pos, uint32_t max_len, float scale, const float* attn_q,
                   const uint16_t* kv_k, const uint16_t* kv_v, float* attn_part) {
  const uint32_t nblocks = max_len / kBlock;
  for (uint32_t j = 0; j < kKv; ++j)
    for (uint32_t blk = 0; blk < nblocks; ++blk) {
      const uint32_t bstart = blk * kBlock;
      if (bstart > pos) continue;                        // the early-out: nothing written
      for (uint32_t ql = 0; ql < kGqa; ++ql) {
        const uint32_t qh = j * kGqa + ql;
        const float* q = attn_q + size_t(qh) * kHd;
        float mx = ninf(), sm = 0.0f;
        std::vector<float> acc(kHd, 0.0f);
        for (uint32_t w = 0; w < kWaves; ++w) {
          float sc[kWaveP];
          for (uint32_t s = 0; s < kWaveP; ++s) {
            const uint32_t p = bstart + w * kWaveP + s;
            float lane[kSg];
            for (uint32_t l = 0; l < kSg; ++l) {
              float a = 0.0f;
              if (p <= pos)
                for (uint32_t t = 0; t < kPerLane; ++t) {
                  const uint32_t d = l + kSg * t;
                  a = std::fma(q[d], f32(kv_k[(size_t(p) * kKv + j) * kHd + d]), a);
                }
              lane[l] = a;
            }
            for (uint32_t stride = kSg / 2; stride > 0; stride >>= 1)
              for (uint32_t l = 0; l < stride; ++l) lane[l] += lane[l + stride];
            sc[s] = p <= pos ? lane[0] * scale : ninf();
          }
          float nmx = mx;
          for (uint32_t s = 0; s < kWaveP; ++s) nmx = std::fmax(nmx, sc[s]);
          float resc;
          if (nmx > ninf()) {
            resc = std::exp(mx - nmx);
            float ssum = 0.0f;
            for (uint32_t s = 0; s < kWaveP; ++s) {
              sc[s] = std::exp(sc[s] - nmx);
              ssum += sc[s];
            }
            sm = std::fma(sm, resc, ssum);
            mx = nmx;
          } else {
            resc = 1.0f;
            for (uint32_t s = 0; s < kWaveP; ++s) sc[s] = 0.0f;
          }
          for (uint32_t d = 0; d < kHd; ++d) {
            float tsum = 0.0f;
            for (uint32_t s = 0; s < kWaveP; ++s) {
              const uint32_t ps = bstart + w * kWaveP + s;
              const float v = ps <= pos ? f32(kv_v[(size_t(ps) * kKv + j) * kHd + d]) : 0.0f;
              tsum = std::fma(sc[s], v, tsum);
            }
            acc[d] = std::fma(acc[d], resc, tsum);
          }
        }
        float* out = attn_part + (size_t(qh) * nblocks + blk) * kPart;
        out[0] = mx;
        out[1] = sm;
        for (uint32_t d = 0; d < kHd; ++d) out[2 + d] = acc[d];
      }
    }
}

// reduce: ascending block merge, divide, then rne(f32(rne(o)) · f32(rne(softplus(gate)))).
inline void reduce(uint32_t pos, uint32_t max_len, float beta, const float* attn_part,
                   const float* attn_gate, uint16_t* attn_out) {
  const uint32_t nblocks = max_len / kBlock;
  uint32_t nb = pos / kBlock + 1;
  if (nb > nblocks) nb = nblocks;
  for (uint32_t h = 0; h < kQ; ++h)
    for (uint32_t d = 0; d < kHd; ++d) {
      float mx = ninf(), sm = 0.0f, acc = 0.0f;
      for (uint32_t b = 0; b < nb; ++b) {
        const float* p = attn_part + (size_t(h) * nblocks + b) * kPart;
        const float nmx = std::fmax(mx, p[0]);
        const float a = std::exp(mx - nmx), bs = std::exp(p[0] - nmx);
        sm = std::fma(sm, a, p[1] * bs);
        acc = std::fma(acc, a, p[2 + d] * bs);
        mx = nmx;
      }
      const uint16_t o_b = rne(acc / sm);
      const uint16_t sp_b = rne(softplus_f32(attn_gate[h * kHd + d], beta));
      attn_out[h * kHd + d] = rne(f32(o_b) * f32(sp_b));
    }
}

}  // namespace k2_attn_ref
```

- [ ] **Step 2: Write the failing test** - `tests/kernels/k2/k2_attn_test.cc`:

```cpp
// k2_attn.cl vs k2_attn_ref.h at max_len 4096. Bars, as attn_test's (docs in attn_test.cc):
//   kv_k, kv_v over the WHOLE caches, attn_q, attn_gate - bit-exact (no exp before them);
//   attn_part - relative error <= 1e-3 where written, canary where the early-out skipped;
//   attn_out - (a) relative error <= 8e-3 floored at the RMS, and (b) <= 2 bf16 ulp wherever
//              |ref| >= rms/8 (exp's 3 ulp can move each of the two final roundings one ulp);
//   replay - a second run of the same list is bitwise identical.
// Softplus threshold (spec §8 risk 6): head 5's gate is 200 in one case, so β·x = 138.6 > 20
// and exp(β·x) overflows fp32 - without the threshold branch that head's output is inf.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "k2_attn_ref.h"
#include "k2_dev.h"
#include "kernels/k2/k2_kernels.h"
#include "loader/k2_loader.h"
#include "runtime/control.h"

using k2test::Dev;
namespace ar = k2_attn_ref;
constexpr float kCanary = 12345.0f;

static void attn_out_bars(const std::vector<uint16_t>& got, const std::vector<uint16_t>& ref) {
  double ss = 0;
  for (uint16_t v : ref) ss += double(ar::f32(v)) * ar::f32(v);
  const double rms = std::sqrt(ss / double(ref.size()));
  double worst_rel = 0;
  uint32_t worst_gated = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double g = ar::f32(got[i]), r = ar::f32(ref[i]);
    CHECK(std::isfinite(g));
    worst_rel = std::max(worst_rel, std::fabs(g - r) / std::max(std::fabs(r), rms));
    if (std::fabs(r) >= rms / 8) {
      const int32_t du = k2test::bf16_key(got[i]) - k2test::bf16_key(ref[i]);
      worst_gated = std::max(worst_gated, uint32_t(du < 0 ? -du : du));
    }
  }
  std::printf("    attn_out: worst rel %.3e (bar 8e-3), worst gated %u bf16 ulp (bar 2)\n", worst_rel,
              worst_gated);
  CHECK(worst_rel <= 8e-3);
  CHECK(worst_gated <= 2);
}

static void attn_case(Dev& d, bool dense, uint32_t pos, float hot_gate) {
  constexpr uint32_t L = 4096, NB = L / 64;
  const size_t kv_elems = size_t(L) * 8 * 128, part_elems = size_t(32) * NB * 130;
  const std::vector<float> rope = loader::k2_rope_table(L);
  std::vector<float> partials = k2test::rand_f32(dense ? 10240 : 9280, 7 + pos, 1.0f);
  if (hot_gate != 0.0f)
    for (uint32_t i = 0; i < 128; ++i) partials[5120 + 5 * 128 + i] = hot_gate;
  const std::vector<uint16_t> vbuf = k2test::rand_bf16(1024, 8 + pos, -1, 1);
  std::vector<uint16_t> kv_k = k2test::rand_bf16(kv_elems, 9 + pos, -1, 1);
  std::vector<uint16_t> kv_v = k2test::rand_bf16(kv_elems, 10 + pos, -1, 1);

  std::vector<float> q_ref(32 * 128), g_ref(32 * 128), part_ref(part_elems, kCanary);
  std::vector<uint16_t> k_ref = kv_k, v_ref = kv_v, out_ref(4096);
  ar::prep(pos, partials.data(), vbuf.data(), dense, rope.data(), q_ref.data(), g_ref.data(),
           k_ref.data(), v_ref.data());
  ar::decode(pos, L, K2_SCORE_SCALE, q_ref.data(), k_ref.data(), v_ref.data(), part_ref.data());
  ar::reduce(pos, L, K2_SOFTPLUS_BETA, part_ref.data(), g_ref.data(), out_ref.data());

  l0::Mem ctrl(d.ctx, l0::MemKind::Shared, sizeof(runtime::Control));
  std::memset(ctrl.ptr(), 0, sizeof(runtime::Control));
  ctrl.as<runtime::Control>()->pos = pos;
  ctrl.as<runtime::Control>()->n_active = 1;
  l0::Mem pm = d.up(partials), vm = d.up(vbuf), rm = d.up(rope), km = d.up(kv_k), vvm = d.up(kv_v);
  l0::Mem qm(d.ctx, l0::MemKind::Device, 32 * 128 * 4), gm(d.ctx, l0::MemKind::Device, 32 * 128 * 4);
  l0::Mem partm = d.up(std::vector<float>(part_elems, kCanary));
  l0::Mem om(d.ctx, l0::MemKind::Device, 4096 * 2);

  auto bind = [](l0::Kernel& k, std::initializer_list<void*> ptrs) {
    uint32_t i = 0;
    for (void* p : ptrs) k.arg_ptr(i++, p);
  };
  l0::Kernel& kp = d.kernel(kernels::k2::attn_prep(dense), "k2_attn_prep", 128);
  bind(kp, {ctrl.ptr(), pm.ptr(), vm.ptr(), rm.ptr(), qm.ptr(), gm.ptr(), km.ptr(), vvm.ptr()});
  l0::Kernel& kd = d.kernel(kernels::k2::attn_decode(L), "k2_attn_decode", 128);
  bind(kd, {ctrl.ptr(), qm.ptr(), km.ptr(), vvm.ptr(), partm.ptr()});
  l0::Kernel& kr = d.kernel(kernels::k2::attn_reduce(L), "k2_attn_reduce", 128);
  bind(kr, {ctrl.ptr(), partm.ptr(), gm.ptr(), om.ptr()});
  d.run({{&kp, 40, 1}, {&kd, 8, NB}, {&kr, 32, 1}});

  CHECK(d.down<uint16_t>(km, kv_elems) == k_ref);
  CHECK(d.down<uint16_t>(vvm, kv_elems) == v_ref);
  CHECK(d.down<float>(qm, 32 * 128) == q_ref);
  CHECK(d.down<float>(gm, 32 * 128) == g_ref);
  const std::vector<float> part = d.down<float>(partm, part_elems);
  double worst = 0;
  for (size_t i = 0; i < part_elems; ++i) {
    if (part_ref[i] == kCanary) { CHECK(part[i] == kCanary); continue; }
    if (!std::isfinite(part_ref[i])) { CHECK(std::memcmp(&part[i], &part_ref[i], 4) == 0); continue; }
    worst = std::max(worst, double(std::fabs(part[i] - part_ref[i])) /
                                std::max(double(std::fabs(part_ref[i])), 1e-30));
  }
  const std::vector<uint16_t> out = d.down<uint16_t>(om, 4096);
  std::printf("  %s pos %u gate %.0f: kv/q/gate bit-exact, attn_part worst rel %.3e\n",
              dense ? "dense" : "sparse", pos, double(hot_gate), worst);
  CHECK(worst <= 1e-3);
  attn_out_bars(out, out_ref);

  // Replay: the same three launches again over the same inputs give the same bytes.
  d.imm.copy(km.ptr(), kv_k.data(), kv_elems * 2);
  d.imm.copy(vvm.ptr(), kv_v.data(), kv_elems * 2);
  d.run({{&kp, 40, 1}, {&kd, 8, NB}, {&kr, 32, 1}});
  CHECK(d.down<uint16_t>(om, 4096) == out);
  CHECK(d.down<float>(partm, part_elems) == part);
}

int main() {
  Dev d;
  attn_case(d, true, 0, 0.0f);
  attn_case(d, true, 63, 0.0f);       // last position of block 0
  attn_case(d, true, 64, 200.0f);     // first position of block 1, softplus threshold branch
  attn_case(d, false, 1000, 0.0f);    // MoVA v from vbuf, 16 blocks
  attn_case(d, true, 4095, 0.0f);     // the last slot of the cache
  std::puts("k2_attn_test OK");
  return 0;
}
```

- [ ] **Step 3: Register** - append to `tests/CMakeLists.txt`:

```cmake
add_executable(k2_attn_test kernels/k2/k2_attn_test.cc)
target_include_directories(k2_attn_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests
                           ${CMAKE_SOURCE_DIR}/tests/kernels/k2)
target_compile_definitions(k2_attn_test PRIVATE K2_SCORE_SCALE=${K2_SCORE_SCALE}
                           K2_SOFTPLUS_BETA=${K2_SOFTPLUS_BETA})
target_link_libraries(k2_attn_test PRIVATE b70_k2_loader b70_l0)
b70_target_kernel_dir(k2_attn_test)
add_dependencies(k2_attn_test kernel_k2_attn_prep_dense kernel_k2_attn_prep_sparse
  kernel_k2_attn_decode_L4096_B${ATTN_BLOCK} kernel_k2_attn_reduce_L4096_B${ATTN_BLOCK})
add_test(NAME k2_attn_test COMMAND k2_attn_test)
set_tests_properties(k2_attn_test PROPERTIES TIMEOUT 900)
```

- [ ] **Step 4: Confirm it fails** - build; expected: no target `kernel_k2_attn_prep_dense`.

- [ ] **Step 5: Write `src/kernels/k2/k2_attn.cl`**

```c
// k2_attn.cl - K2-Horizon's decode attention (spec 4 §3.2), all 48 layers:
//   k2_attn_prep    RoPE (rotate_half, all 128 dims) on q and k, gate staging, the KV write
//   k2_attn_decode  one 64-position block per work-group, early-out past the context
//   k2_attn_reduce  merge the blocks, divide, softplus gate
// It is src/kernels/attn.cl's trio re-dimensioned - read that file's header for why each
// structure is what it is; the orders below are its orders at K2's shape, and
// tests/kernels/k2/k2_attn_ref.h implements every one of them.
//
//   k2_attn_prep(ctrl, partials, vbuf, rope, attn_q, attn_gate, kv_k, kv_v)   grid (40, 1), WG 128
//     partials   fp32  the fused attention GEMV: q [0,4096) k [4096,5120) gate [5120,9216)
//                      v [9216,10240) (dense rows only)
//     vbuf       bf16 [1024] MoVA's mixed value (sparse); bound but unread by the dense variant
//     rope       fp32 [MAXLEN][2][64] bf16-valued cos, sin (loader::k2_rope_table)
//     attn_q     fp32 [32][128]    f32 of the bf16 roped q
//     attn_gate  fp32 [32][128]    f32(rne_bf16(gate columns))
//     kv_k, kv_v bf16 [MAXLEN][8][128]  this layer's slices
//   k2_attn_decode(ctrl, attn_q, kv_k, kv_v, attn_part)                         grid (8, MAXLEN/64)
//     attn_part  fp32 [32][MAXLEN/64][130]   {mx, sm, acc[128]}
//   k2_attn_reduce(ctrl, attn_part, attn_gate, attn_out)                        grid (32, 1)
//     attn_out   bf16 [4096]
//
// RoPE, as the reference computes it (bf16 q and k, bf16 cos/sin, rotate_half):
//   x_i = f32(rne(partials[i])) ;  rot_i = i < 64 ? -x_{i+64} : x_{i-64} ;  c, s = rope[pos][·][i mod 64]
//   out_i = rne(f32(rne(x_i · c)) + f32(rne(rot_i · s)))
// The orders (identical text in k2_attn_ref.h):
//  1. score dot: subgroup s of the 8 owns one position; its lane l accumulates d = l + 16t,
//     t = 0..7 ascending, explicit fma; the 16 lanes collapse pairwise (8, 4, 2, 1);
//     score = lane total · K2_SCORE_SCALE.
//  2. online softmax wave (8 positions, 8 waves per block), exactly attn.cl's order 3.
//  3. block merge, ascending block, exactly attn.cl's order 4.
//  4. out = rne(f32(rne(acc/sm)) · f32(rne(softplus(gate)))),
//     softplus(x) = β·x > 20 ? x : log1p(exp(β·x)) / β,  β = K2_SOFTPLUS_BETA.
// Rounding: prep has no exp, so kv, attn_q and attn_gate are bit-reproducible on the host;
// the softmax, the merge and the softplus carry exp's 3 ulp, which is all attn_out's
// tolerance is for.

#ifndef MAXLEN
#define MAXLEN 16384
#endif
#ifndef CTRL_POS
#error "k2_attn: CTRL_POS must be defined (src/kernels/CMakeLists.txt CTRL_DEFINES)"
#endif
#ifndef CTRL_NACT
#error "k2_attn: CTRL_NACT must be defined"
#endif
#ifndef ATTN_BLOCK
#error "k2_attn: ATTN_BLOCK must be defined; it must equal kernels::k2::kAttnBlock"
#endif
#ifndef V_FROM_PARTIALS
#error "k2_attn: V_FROM_PARTIALS must be defined (1 dense, 0 sparse)"
#endif
#ifndef K2_SCORE_SCALE
#error "k2_attn: K2_SCORE_SCALE must be defined (plan 8a facts §6 Q2)"
#endif
#ifndef K2_SOFTPLUS_BETA
#error "k2_attn: K2_SOFTPLUS_BETA must be defined"
#endif

#define Q_HEADS 32
#define KV_HEADS 8
#define GQA 4
#define HD 128
#define HALF 64
#define K_OFF 4096
#define GATE_OFF 5120
#define V_OFF 9216
#define SG 16
#define WG_HD HD                 /* prep, decode and reduce all run 128-wide work-groups */
#define WAVE_P (WG_HD / SG)      /* 8 subgroups -> 8 positions per wave */
#define PER_LANE (HD / SG)       /* 8 elements of the 128-dim dot per lane */
#define WAVES (ATTN_BLOCK / WAVE_P)
#define NBLOCKS (MAXLEN / ATTN_BLOCK)
#define PART (2 + HD)
#if ATTN_BLOCK % WAVE_P != 0 || MAXLEN % ATTN_BLOCK != 0
#error "k2_attn: MAXLEN must be a multiple of ATTN_BLOCK, and ATTN_BLOCK of 8"
#endif

inline float bf16f(ushort h) { return as_float(((uint)h) << 16); }
inline ushort rne_bf16(float f) {
  uint u = as_uint(f);
  uint rounding = ((u >> 16) & 1u) + 0x7FFFu;
  return (ushort)((u + rounding) >> 16);
}
inline float softplus_f32(float x) {
  const float bx = x * K2_SOFTPLUS_BETA;
  return bx > 20.0f ? x : log1p(exp(bx)) / K2_SOFTPLUS_BETA;
}

__attribute__((reqd_work_group_size(WG_HD, 1, 1)))
__kernel void k2_attn_prep(__global const uint* restrict ctrl,
                           __global const float* restrict partials,
                           __global const ushort* restrict vbuf,
                           __global const float* restrict rope,
                           __global float* restrict attn_q, __global float* restrict attn_gate,
                           __global ushort* restrict kv_k, __global ushort* restrict kv_v) {
  const uint wg = get_group_id(0);
  const uint i = get_local_id(0);
  if (ctrl[CTRL_NACT] == 0u) return;
  const uint pos = ctrl[CTRL_POS];
  const bool is_q = wg < Q_HEADS;
  const uint h = is_q ? wg : wg - Q_HEADS;
  const size_t base = is_q ? (size_t)h * HD : (size_t)K_OFF + (size_t)h * HD;

  __global const float* restrict cs = rope + (size_t)pos * 2 * HALF;
  const uint ii = i < HALF ? i : i - HALF;
  const float c = cs[ii], s = cs[HALF + ii];
  const float xi = bf16f(rne_bf16(partials[base + i]));
  const float xo = bf16f(rne_bf16(partials[base + (i < HALF ? i + HALF : i - HALF)]));
  const float rot = i < HALF ? -xo : xo;
  const ushort out = rne_bf16(bf16f(rne_bf16(xi * c)) + bf16f(rne_bf16(rot * s)));

  if (is_q) {
    attn_q[(size_t)h * HD + i] = bf16f(out);
    attn_gate[(size_t)h * HD + i] = bf16f(rne_bf16(partials[GATE_OFF + (size_t)h * HD + i]));
  } else {
    const size_t slot = ((size_t)pos * KV_HEADS + h) * HD + i;
    kv_k[slot] = out;
#if V_FROM_PARTIALS
    kv_v[slot] = rne_bf16(partials[V_OFF + (size_t)h * HD + i]);
#else
    kv_v[slot] = vbuf[(size_t)h * HD + i];
#endif
  }
}

__attribute__((reqd_work_group_size(WG_HD, 1, 1)))
__attribute__((intel_reqd_sub_group_size(SG)))
__kernel void k2_attn_decode(__global const uint* restrict ctrl,
                             __global const float* restrict attn_q,
                             __global const ushort* restrict kv_k,
                             __global const ushort* restrict kv_v,
                             __global float* restrict attn_part) {
  const uint j = get_group_id(0);
  const uint blk = get_group_id(1);
  const uint lid = get_local_id(0);
  const uint sgid = lid / SG;
  const uint lane = lid % SG;
  __local float qpack[GQA * HD];
  __local float dot_red[WG_HD];

  if (ctrl[CTRL_NACT] == 0u) return;
  const uint pos = ctrl[CTRL_POS];
  const uint bstart = blk * ATTN_BLOCK;
  if (bstart > pos) return;                    // uniform across the work-group

  for (uint qhl = 0; qhl < GQA; ++qhl)
    qpack[qhl * HD + lid] = attn_q[(size_t)(j * GQA + qhl) * HD + lid];
  barrier(CLK_LOCAL_MEM_FENCE);

  float mx[GQA];
  float sm[GQA];
  float acc[GQA];
  for (uint qhl = 0; qhl < GQA; ++qhl) {
    mx[qhl] = -INFINITY;
    sm[qhl] = 0.0f;
    acc[qhl] = 0.0f;
  }
  for (uint w = 0; w < WAVES; ++w) {
    const uint p = bstart + w * WAVE_P + sgid;
    ushort kreg[PER_LANE];
    ushort vreg[WAVE_P];
    for (uint t = 0; t < PER_LANE; ++t) {
      const uint d = lane + SG * t;
      kreg[t] = p <= pos ? kv_k[((size_t)p * KV_HEADS + j) * HD + d] : (ushort)0;
    }
    for (uint s = 0; s < WAVE_P; ++s) {       // masked slots load +0: never 0 · garbage
      const uint ps = bstart + w * WAVE_P + s;
      vreg[s] = ps <= pos ? kv_v[((size_t)ps * KV_HEADS + j) * HD + lid] : (ushort)0;
    }
    for (uint qhl = 0; qhl < GQA; ++qhl) {
      float a = 0.0f;
      if (p <= pos)
        for (uint t = 0; t < PER_LANE; ++t)
          a = fma(qpack[qhl * HD + lane + SG * t], bf16f(kreg[t]), a);
      dot_red[lid] = a;
      barrier(CLK_LOCAL_MEM_FENCE);
      for (uint stride = SG / 2; stride > 0; stride >>= 1) {
        if (lane < stride) dot_red[sgid * SG + lane] += dot_red[sgid * SG + lane + stride];
        barrier(CLK_LOCAL_MEM_FENCE);
      }
      float sc[WAVE_P];
      for (uint s = 0; s < WAVE_P; ++s) {
        const uint ps = bstart + w * WAVE_P + s;
        sc[s] = ps <= pos ? dot_red[s * SG] * K2_SCORE_SCALE : -INFINITY;
      }
      float nmx = mx[qhl];
      for (uint s = 0; s < WAVE_P; ++s) nmx = fmax(nmx, sc[s]);
      float resc;
      if (nmx > -INFINITY) {
        resc = exp(mx[qhl] - nmx);
        float ssum = 0.0f;
        for (uint s = 0; s < WAVE_P; ++s) {
          sc[s] = exp(sc[s] - nmx);
          ssum += sc[s];
        }
        sm[qhl] = fma(sm[qhl], resc, ssum);
        mx[qhl] = nmx;
      } else {
        resc = 1.0f;
        for (uint s = 0; s < WAVE_P; ++s) sc[s] = 0.0f;
      }
      float tsum = 0.0f;
      for (uint s = 0; s < WAVE_P; ++s) tsum = fma(sc[s], bf16f(vreg[s]), tsum);
      acc[qhl] = fma(acc[qhl], resc, tsum);
      barrier(CLK_LOCAL_MEM_FENCE);           // closes dot_red's reuse by the next head
    }
  }
  for (uint qhl = 0; qhl < GQA; ++qhl) {
    __global float* restrict out =
        attn_part + ((size_t)(j * GQA + qhl) * NBLOCKS + blk) * PART;
    out[2 + lid] = acc[qhl];
    if (lid == 0) {
      out[0] = mx[qhl];
      out[1] = sm[qhl];
    }
  }
}

__attribute__((reqd_work_group_size(WG_HD, 1, 1)))
__kernel void k2_attn_reduce(__global const uint* restrict ctrl,
                             __global const float* restrict attn_part,
                             __global const float* restrict attn_gate,
                             __global ushort* restrict attn_out) {
  const uint h = get_group_id(0);
  const uint d = get_local_id(0);
  __local float hmx[NBLOCKS], hsm[NBLOCKS];
  if (ctrl[CTRL_NACT] == 0u) return;
  const uint pos = ctrl[CTRL_POS];
  uint nb = pos / ATTN_BLOCK + 1;
  if (nb > NBLOCKS) nb = NBLOCKS;            // a violated pos < MAXLEN is a wrong answer, not an overrun
  for (uint b = d; b < nb; b += WG_HD) {
    __global const float* restrict p = attn_part + ((size_t)h * NBLOCKS + b) * PART;
    hmx[b] = p[0];
    hsm[b] = p[1];
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  float mx = -INFINITY, sm = 0.0f, acc = 0.0f;
  for (uint b = 0; b < nb; ++b) {
    const float bmx = hmx[b], bsm = hsm[b];
    const float bacc = attn_part[((size_t)h * NBLOCKS + b) * PART + 2 + d];
    const float nmx = fmax(mx, bmx);
    const float a = exp(mx - nmx);
    const float bs = exp(bmx - nmx);
    sm = fma(sm, a, bsm * bs);
    acc = fma(acc, a, bacc * bs);
    mx = nmx;
  }
  const ushort o_b = rne_bf16(acc / sm);
  const ushort sp_b = rne_bf16(softplus_f32(attn_gate[(size_t)h * HD + d]));
  attn_out[(size_t)h * HD + d] = rne_bf16(bf16f(o_b) * bf16f(sp_b));
}
```

- [ ] **Step 6: Register the variants** - append to `src/kernels/k2/CMakeLists.txt`:

```cmake
set(K2_ATTN_DEFINES ${CTRL_DEFINES} ATTN_BLOCK=${ATTN_BLOCK} K2_SCORE_SCALE=${K2_SCORE_SCALE}
    K2_SOFTPLUS_BETA=${K2_SOFTPLUS_BETA})
add_ocloc_kernel(k2_attn_prep_dense SOURCE ${K2_DIR}/k2_attn.cl
                 DEFINES ${K2_ATTN_DEFINES} V_FROM_PARTIALS=1)
add_ocloc_kernel(k2_attn_prep_sparse SOURCE ${K2_DIR}/k2_attn.cl
                 DEFINES ${K2_ATTN_DEFINES} V_FROM_PARTIALS=0)
foreach(L 16384 4096)   # 16384: production (max_len); 4096: k2_attn_test's buffers
  add_ocloc_kernel(k2_attn_decode_L${L}_B${ATTN_BLOCK} SOURCE ${K2_DIR}/k2_attn.cl
                   DEFINES ${K2_ATTN_DEFINES} V_FROM_PARTIALS=1 MAXLEN=${L})
  add_ocloc_kernel(k2_attn_reduce_L${L}_B${ATTN_BLOCK} SOURCE ${K2_DIR}/k2_attn.cl
                   DEFINES ${K2_ATTN_DEFINES} V_FROM_PARTIALS=1 MAXLEN=${L})
endforeach()
```

- [ ] **Step 7: Build and run** - `tools/box.sh sync && tools/box.sh build 2>&1 | tail -2 && ZE_AFFINITY_MASK=1 tools/box.sh test '^k2_attn_test$'`. Expected: five case lines, each with `kv/q/gate bit-exact`, `attn_part` ≤ 1e-3 and both `attn_out` bars; `k2_attn_test OK`.

- [ ] **Step 8: Run every K2 kernel test together, and the 27B's kernel tests** - `ZE_AFFINITY_MASK=1 tools/box.sh test '^(k2_shapes_test|k2_prep_test|k2_route_test|k2_experts_test|k2_attn_test|gemv_test|gemv_bf16_test|prep_test|attn_test|embed_gather_test|argmax_test)$'`. Expected: 11/11 pass.

- [ ] **Step 9: Commit**

```bash
git add src/kernels/k2/k2_attn.cl src/kernels/k2/CMakeLists.txt tests/kernels/k2/k2_attn_ref.h tests/kernels/k2/k2_attn_test.cc tests/CMakeLists.txt
git commit -m "feat(kernels): K2 attention trio - GQA 4 at head_dim 128, full rotate_half RoPE, softplus gate

Claude-Session: "
```

---

## Self-review

**Spec coverage** (§3.3 "new kernels" and "new variants", §4 "Kernel tests"):
- grouped RMSNorm → Task 2 (stage B new; stage A is prep.cl's fold at K2's grid).
- softplus gate → Task 5, inside `k2_attn_reduce`, including the threshold branch test (§8 risk 6).
- router top-k k ∈ {4, 8} with selection bias, ascending id and bf16 weights → Task 3, including the tie at the cut (§8 risk 2).
- expert slot bitwise vs a plain GEMV → Task 4. The combines → Task 4.
- attention at ratio 4 / head_dim 128 → Task 5.
- int4 GEMV at every fused and per-expert shape, bf16 GEMV for the router and lm_head, embed at hidden 2560, argmax at vocab 250,624 → Task 1.

**Deliberate differences from spec §3.3's text:**
1. The softplus gate is folded into `k2_attn_reduce`, so each layer has one launch fewer than §3.3's table: dense 12, sparse 45, **total 2067** rather than ~2,115. The 27B applies its sigmoid gate the same way. The spec's table is updated in the commit that adds these plans.
2. RoPE and the cache write stay one launch (`k2_attn_prep`), as §3.3 lists.
3. The slot kernels take `slot_base + group id` rather than a bare slot argument. The per-slot launches spec §3.3 specifies are unchanged; the batched form costs nothing to keep available.

**Placeholders:** none. The defines' values depend on plan 8a §6, and their defaults are stated.

**Types and names:** `kernels::k2::{gemv, gemv_bf16, res_fold, norm_finish, silu_mul, router_topk, gemv_slot, value_combine, moe_combine, attn_prep, attn_decode, attn_reduce, embed_gather, argmax_stage1, argmax_stage2}` are the names plan 8e's capture calls. Argument orders are listed in each task's Interfaces.
