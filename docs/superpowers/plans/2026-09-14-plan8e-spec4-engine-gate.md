# Spec 4 / Stages 4-5 - K2 decode list, engine, gates and the recorded row Implementation Plan

**Status (2026-09-14): written, NOT dispatched.** Implementation waits for the operator's ruling on the prefill GEMM direction; these plans may be adjusted after it.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** K2-Horizon decodes through one captured, replayed Level Zero list: 2067 launches and 28 modules. It passes replay determinism and the tie-aware golden gate. `b70-decode` dispatches on `model_type`, and a `--bench` row at depth 4096 / tg 256 is recorded. The 27B's full suite and both golden gates are green afterwards, and `spec4-done` is tagged if all five spec §2 bars are met.

**Architecture:** `src/runtime/k2/` sits beside `src/runtime/`, and every file there stays byte-identical.
- `K2Buffers` owns control, the 48-layer KV and the decode scratch.
- `runtime::k2::build` walks `model::K2Horizon` and appends every launch in spec §3.3's order, one Kernel per site, arguments frozen at append.
- `K2Engine` is `runtime::Engine`'s contract (`reset` / `ingest` / `generate`) over that list.
- The only host-visible state per token is `runtime::Control`, as for the 27B.
- Routing decisions never reach the host: the router buffers are read only by kernels, and by the golden gate after the fence as a diagnostic.

**Tech Stack:** C++17, Level Zero via `src/l0`, plans 8b (`load_k2`) and 8c (`kernels::k2::*`), `tests/golden/golden_common.h` (verbatim decision rule), `tools/bench_decode.sh`.

**Spec:** `docs/superpowers/specs/2026-09-14-spec4-k2-horizon-decode-core-design.md` (§2 bars and stopping rule, §3.3 launch order, §3.4 memory, §4 gates, §5 stages 4-5)

## Global Constraints

- Branch `spec1.7-codex-exp`; never push. Commits end with `Claude-Session: `.
- Build/test only via `tools/box.sh` (JOBS 44). Long runs detached (`setsid nohup … > $HOME/<name>.log 2>&1 < /dev/null &`) and polled; WiFi drops several times an hour.
- **The 27B is untouched:** no edit to `src/runtime/*.{h,cc}`, `src/loader/loader.cc`, `src/model/qwen35.*`, existing kernels or tests. The one edit to an existing source is `src/cli/b70_decode.cc`'s `main`: three lines of `model_type` dispatch, which spec §3.1 names. CMake edits are appended lines.
- **No host decision may depend on a device-produced value.** The capture is a straight walk; nothing between replays reads a router buffer except test diagnostics.
- Record grade requires zero containers and zero DRM holders; `tools/bench_decode.sh` measures it. Never upgrade a grade after the fact.
- One defect, one fix, one measurement. Every number labelled measured / derived / estimated.
- GPU work on device 1 (`ZE_AFFINITY_MASK=1`) except the recorded bench row, which runs on device 0 like every series row (box-idle protocol).

---

## File structure

| path | responsibility |
|---|---|
| `src/runtime/k2/k2_buffers.h` / `.cc` | `runtime::k2::K2Buffers`: control, KV [48][L][8][128], scratch; sizes, zeroing |
| `src/runtime/k2/k2_capture.h` / `.cc` | `runtime::k2::build`: the 2067-launch walk |
| `src/runtime/k2/k2_engine.h` / `.cc` | `runtime::k2::K2Engine`: reset, ingest, generate, taps, route readback |
| `src/runtime/k2/CMakeLists.txt`, `src/runtime/CMakeLists.txt` (append) | `b70_k2_runtime` |
| `tests/runtime/k2_buffers_test.cc` | allocation sizes at 16384, rejection of a bad max_len |
| `tests/runtime/k2_replay_determinism_test.cc` | capture counts; three sessions bitwise identical |
| `tests/golden/k2_golden_gate_test.cc` | spec §2 bar 1 + the per-layer route diagnostic |
| `src/cli/k2_decode.h` / `.cc`, `src/cli/b70_decode.cc` (main only), `src/cli/CMakeLists.txt` (append) | `model_type` dispatch, `--ids`/`--n`, `--bench` |
| `docs/BENCHMARKS.md`, `docs/superpowers/specs/2026-09-14-spec4-k2-horizon-decode-core-design.md` (§10 close) | the row and the close |

---

### Task 1: `K2Buffers`

**Files:**
- Create: `src/runtime/k2/k2_buffers.h`, `src/runtime/k2/k2_buffers.cc`, `src/runtime/k2/CMakeLists.txt`, `tests/runtime/k2_buffers_test.cc`
- Modify: `src/runtime/CMakeLists.txt` (append), `tests/CMakeLists.txt` (append)

**Interfaces:**
- Consumes: `runtime::Control` (`src/runtime/control.h`), `model::K2Horizon`, `l0::Mem`, `l0::CmdList::fill`.
- Produces: `runtime::k2::K2Buffers(l0::Context&, uint32_t max_len)` with public `l0::Mem` fields in this exact order: `control, kv_k, kv_v, resid, x, partials, norm_sumsq, attn_q, attn_gate, attn_part, attn_out, vslots, vbuf, route_mova, route_moe, router_logits, gu_slots, mid, down_slots, logits, argmax_part`. It also has `uint32_t max_len`, `size_t kv_layer_bytes() const`, `size_t persistent_bytes() const`, `size_t scratch_bytes() const`, `void zero_persistent(l0::CmdList&)` and `static constexpr uint32_t kAttnBlock = 64, kArgmaxGroups = 245`.

- [ ] **Step 1: Write the failing test** - `tests/runtime/k2_buffers_test.cc`:

```cpp
// K2Buffers at max_len 16384 (spec §3.4). The exact byte counts are derived here in the
// comments from model::K2Horizon, not read back from the struct under test.
#include <cstdio>
#include <stdexcept>
#include "check.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "runtime/control.h"
#include "runtime/k2/k2_buffers.h"

int main() {
  l0::Context ctx(0);
  runtime::k2::K2Buffers b(ctx, 16384);
  CHECK_EQ(b.max_len, uint32_t(16384));
  CHECK_EQ(b.kv_layer_bytes(), size_t(33554432));        // 16384 x 8 x 128 x 2
  CHECK_EQ(b.kv_k.size(), size_t(1610612736));           // x 48 layers
  CHECK_EQ(b.kv_v.size(), size_t(1610612736));
  CHECK_EQ(b.control.size(), sizeof(runtime::Control));
  CHECK_EQ(b.attn_part.size(), size_t(32) * 256 * 130 * 4);
  CHECK_EQ(b.partials.size(), size_t(12288) * 4);        // largest S = 1 GEMV: dense gate||up
  CHECK_EQ(b.x.size(), size_t(6144) * 2);                // largest bf16 prep output: dense SiLU·mul
  CHECK_EQ(b.route_moe.size(), size_t(45) * 8 * 2 * 4);
  CHECK_EQ(b.route_mova.size(), size_t(45) * 4 * 2 * 4);
  CHECK_EQ(b.logits.size(), size_t(250624) * 4);
  CHECK_EQ(b.argmax_part.size(), size_t(245) * 2 * 4);   // ceil(250624 / 1024) groups
  CHECK_EQ(b.persistent_bytes(), size_t(3221225600ull)); // 128 + 2 x 1,610,612,736
  CHECK_EQ(b.scratch_bytes(), size_t(5538480));
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  b.zero_persistent(imm);
  CHECK_EQ(b.control.as<runtime::Control>()->pos, uint32_t(0));
  bool threw = false;
  try {
    runtime::k2::K2Buffers bad(ctx, 1000);   // not a multiple of the 64-position attention block
  } catch (const std::runtime_error&) {
    threw = true;
  }
  CHECK(threw);
  std::printf("k2_buffers_test OK: persistent %.3f GB, scratch %.3f MB\n", b.persistent_bytes() / 1e9,
              b.scratch_bytes() / 1e6);
  return 0;
}
```

- [ ] **Step 2: Register** - append to `src/runtime/CMakeLists.txt`:

```cmake
# Spec 4: K2-Horizon's decode runtime, beside this directory's qwen3_5 one (no shared code
# besides l0, Control and the CapturedStep struct).
add_subdirectory(k2)
```

Create `src/runtime/k2/CMakeLists.txt`:

```cmake
add_library(b70_k2_runtime STATIC k2_buffers.cc k2_capture.cc k2_engine.cc)
target_include_directories(b70_k2_runtime PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(b70_k2_runtime PUBLIC b70_l0 b70_k2_model b70_k2_loader)
b70_target_kernel_dir(b70_k2_runtime)
```

Until Task 2, create `src/runtime/k2/k2_capture.cc` and `k2_engine.cc` each containing only `// plan 8e Task 2.`, so the library builds.

Append to `tests/CMakeLists.txt`:

```cmake
add_executable(k2_buffers_test runtime/k2_buffers_test.cc)
target_include_directories(k2_buffers_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(k2_buffers_test PRIVATE b70_k2_runtime)
add_test(NAME k2_buffers_test COMMAND k2_buffers_test)
```

- [ ] **Step 3: Confirm it fails** - `tools/box.sh sync && tools/box.sh build 2>&1 | grep -m1 "k2_buffers.h"`. Expected: `No such file or directory`.

- [ ] **Step 4: Write the header** - `src/runtime/k2/k2_buffers.h`:

```cpp
#pragma once
#include <cstddef>
#include <cstdint>

#include "l0/context.h"
#include "l0/memory.h"

namespace l0 {
class CmdList;
}

namespace runtime::k2 {

// Every device allocation K2's captured decode list touches besides the weights (spec §3.4).
// `max_len` is declared FIRST: every size below reads it, and member order is init order.
struct K2Buffers {
  static constexpr uint32_t kAttnBlock = 64;       // == kernels::k2::kAttnBlock
  static constexpr uint32_t kArgmaxGroups = 245;   // ceil(250624 / 1024)

  K2Buffers(l0::Context& ctx, uint32_t max_len);

  uint32_t max_len;
  // --- persistent: survives a token boundary; zeroed by reset ---
  l0::Mem control;        // Shared, runtime::Control
  l0::Mem kv_k, kv_v;     // bf16 [48][max_len][8][128] each = 1,610,612,736 B at 16384
  // --- scratch: written before it is read, every step ---
  l0::Mem resid;          // bf16 [2560]    the residual stream
  l0::Mem x;              // bf16 [6144]    norm output, then SiLU·mul outputs (dense 6144)
  l0::Mem partials;       // fp32 [12288]   S = 1 GEMV outputs (dense gate||up is the largest)
  l0::Mem norm_sumsq;     // fp32 [10]      prep_res_fold -> k2_norm_finish
  l0::Mem attn_q;         // fp32 [32][128]
  l0::Mem attn_gate;      // fp32 [32][128]
  l0::Mem attn_part;      // fp32 [32][max_len/64][130]
  l0::Mem attn_out;       // bf16 [4096]
  l0::Mem vslots;         // fp32 [4][1024]  value-expert slot outputs
  l0::Mem vbuf;           // bf16 [1024]     MoVA's mixed value, written to kv_v
  l0::Mem route_mova;     // fp32 [45][4][2] {id, weight}, ascending id, per sparse layer
  l0::Mem route_moe;      // fp32 [45][8][2]
  l0::Mem router_logits;  // fp32 [128]      the bf16 router GEMV (rows 100..127 are padding)
  l0::Mem gu_slots;       // fp32 [8][1536]  expert gate||up slot outputs
  l0::Mem mid;            // bf16 [8][768]   expert SiLU·mul outputs, read by the down slots
  l0::Mem down_slots;     // fp32 [8][2560]
  l0::Mem logits;         // fp32 [250624]
  l0::Mem argmax_part;    // fp32 [245][2]

  size_t kv_layer_bytes() const { return size_t(max_len) * 8 * 128 * 2; }
  size_t persistent_bytes() const;
  size_t scratch_bytes() const;
  void zero_persistent(l0::CmdList& imm);
};

}  // namespace runtime::k2
```

- [ ] **Step 5: Implement** - `src/runtime/k2/k2_buffers.cc`:

```cpp
#include "runtime/k2/k2_buffers.h"

#include <stdexcept>
#include <string>

#include "l0/cmdlist.h"
#include "model/k2_horizon.h"
#include "runtime/control.h"

namespace runtime::k2 {
namespace {
using model::K2Horizon;
using l0::MemKind;

uint32_t checked(uint32_t max_len) {
  if (max_len == 0 || max_len % K2Buffers::kAttnBlock != 0)
    throw std::runtime_error("runtime::k2::K2Buffers: max_len " + std::to_string(max_len) +
                             " must be a positive multiple of " +
                             std::to_string(K2Buffers::kAttnBlock) + " (attn_decode's block grid)");
  return max_len;
}
}  // namespace

K2Buffers::K2Buffers(l0::Context& ctx, uint32_t max_len_)
    : max_len(checked(max_len_)),
      control(ctx, MemKind::Shared, sizeof(Control)),
      kv_k(ctx, MemKind::Device, size_t(K2Horizon::kLayers) * kv_layer_bytes()),
      kv_v(ctx, MemKind::Device, size_t(K2Horizon::kLayers) * kv_layer_bytes()),
      resid(ctx, MemKind::Device, size_t(K2Horizon::kHidden) * 2),
      x(ctx, MemKind::Device, size_t(K2Horizon::kDenseInter) * 2),
      partials(ctx, MemKind::Device, size_t(2) * K2Horizon::kDenseInter * 4),
      norm_sumsq(ctx, MemKind::Device, 10 * 4),
      attn_q(ctx, MemKind::Device, size_t(K2Horizon::kQHeads) * K2Horizon::kHeadDim * 4),
      attn_gate(ctx, MemKind::Device, size_t(K2Horizon::kQHeads) * K2Horizon::kHeadDim * 4),
      attn_part(ctx, MemKind::Device, size_t(K2Horizon::kQHeads) * (max_len / kAttnBlock) * 130 * 4),
      attn_out(ctx, MemKind::Device, size_t(K2Horizon::kQHeads) * K2Horizon::kHeadDim * 2),
      vslots(ctx, MemKind::Device, size_t(K2Horizon::kValueTopK) * 1024 * 4),
      vbuf(ctx, MemKind::Device, 1024 * 2),
      route_mova(ctx, MemKind::Device, size_t(45) * K2Horizon::kValueTopK * 2 * 4),
      route_moe(ctx, MemKind::Device, size_t(45) * K2Horizon::kTopK * 2 * 4),
      router_logits(ctx, MemKind::Device, 128 * 4),
      gu_slots(ctx, MemKind::Device, size_t(K2Horizon::kTopK) * 2 * K2Horizon::kMoeInter * 4),
      mid(ctx, MemKind::Device, size_t(K2Horizon::kTopK) * K2Horizon::kMoeInter * 2),
      down_slots(ctx, MemKind::Device, size_t(K2Horizon::kTopK) * K2Horizon::kHidden * 4),
      logits(ctx, MemKind::Device, size_t(K2Horizon::kVocab) * 4),
      argmax_part(ctx, MemKind::Device, size_t(kArgmaxGroups) * 2 * 4) {}

size_t K2Buffers::persistent_bytes() const { return control.size() + kv_k.size() + kv_v.size(); }

size_t K2Buffers::scratch_bytes() const {
  size_t n = 0;
  for (const l0::Mem* m : {&resid, &x, &partials, &norm_sumsq, &attn_q, &attn_gate, &attn_part,
                           &attn_out, &vslots, &vbuf, &route_mova, &route_moe, &router_logits,
                           &gu_slots, &mid, &down_slots, &logits, &argmax_part})
    n += m->size();
  return n;
}

void K2Buffers::zero_persistent(l0::CmdList& imm) {
  imm.fill(control.ptr(), 0u, control.size());
  imm.fill(kv_k.ptr(), 0u, kv_k.size());
  imm.fill(kv_v.ptr(), 0u, kv_v.size());
}

}  // namespace runtime::k2
```

- [ ] **Step 6: Build and run** - `tools/box.sh sync && tools/box.sh build 2>&1 | tail -2 && ZE_AFFINITY_MASK=1 tools/box.sh test '^k2_buffers_test$'`. Expected: `k2_buffers_test OK: persistent 3.221 GB, scratch 5.538 MB`.

- [ ] **Step 7: Commit**

```bash
git add src/runtime/k2 src/runtime/CMakeLists.txt tests/runtime/k2_buffers_test.cc tests/CMakeLists.txt
git commit -m "feat(runtime): K2Buffers - 48-layer bf16 KV and the K2 decode scratch

Claude-Session: "
```

---

### Task 2: The capture, `K2Engine`, and replay determinism

**Files:**
- Create: `src/runtime/k2/k2_capture.h`, `src/runtime/k2/k2_engine.h`, `tests/runtime/k2_replay_determinism_test.cc`
- Modify: `src/runtime/k2/k2_capture.cc`, `src/runtime/k2/k2_engine.cc` (replace the stubs), `tests/CMakeLists.txt` (append)

**Interfaces:**
- Consumes: `runtime::CapturedStep` (`src/runtime/capture.h`: `list, kernel_count, modules, kernels, labels`); `loader::K2LoadedModel` and `load_k2` (plan 8b); `kernels::k2::*` names and argument orders (plan 8c); Task 1's `K2Buffers`.
- Produces:
  - `runtime::k2::kLaunchesPerToken = 2067`, `runtime::k2::kModules = 28`
  - `runtime::CapturedStep runtime::k2::build(l0::Context&, const loader::K2LoadedModel&, K2Buffers&, l0::Mem* debug_resid = nullptr)`
  - `class runtime::k2::K2Engine { K2Engine(l0::Context&, loader::K2LoadedModel, uint32_t max_len, bool debug_resid = false); void reset(); void ingest(const std::vector<uint32_t>&); std::vector<uint32_t> generate(uint32_t n, const std::function<void(uint32_t)>& = {}); double last_tok_per_s() const; double last_gen_ms() const; double last_fence_ms() const; std::vector<uint16_t> read_debug_resid(); std::vector<float> read_routes(); const loader::K2LoadedModel& model() const; K2Buffers& buffers(); const CapturedStep& step() const; uint32_t pos() const; uint32_t max_len() const; }`
  - `read_routes()` returns `route_mova` [45][4][2] followed by `route_moe` [45][8][2].

**The walk** (spec §3.3; the launch counts are this plan's):

| block | launches, in order |
|---|---|
| embed | `k2_embed_gather` - **1** |
| dense layer 0-2 | fold (SP0 at L0, else SP1) · `k2_norm_finish` · gemv AttnDense (2560→10240) · `k2_attn_prep_dense` · `k2_attn_decode` · `k2_attn_reduce` · gemv OProj · fold SP1 · `k2_norm_finish` · gemv DenseGateUp · `k2_silu_mul_I6144` · gemv DenseDown - **12** |
| sparse layer 3-47 | fold SP1 · `k2_norm_finish` · gemv AttnSparse (2560→9280) · `k2_router_topk_E64_T4` · 4 × `k2_gemv_slot_K2560_N1024` · `k2_value_combine` · `k2_attn_prep_sparse` · `k2_attn_decode` · `k2_attn_reduce` · gemv OProj · fold SP1 · `k2_norm_finish` · `gemv_bf16` MoeRouter · `k2_router_topk_E100_T8` · 8 × (`k2_gemv_slot_K2560_N1536` · `k2_silu_mul_I768` · `k2_gemv_slot_K768_N2560`) · gemv SharedGateUp · `k2_silu_mul_I768` · gemv SharedDown · `k2_moe_combine` - **45** |
| head | fold SP1 · `k2_norm_finish` · `gemv_bf16` LmHead · `k2_argmax_stage1` · `k2_argmax_stage2` - **5** |
| **total** | 1 + 3 × 12 + 45 × 45 + 5 = **2067** launches, **28** distinct modules |

**Dataflow invariants the walk relies on** (each is a buffer written before any read in the same step):
- `x` holds the norm output until the last consumer of that output: the router GEMV, the 8 gate‖up slots and the shared gate‖up.
- The shared expert's SiLU·mul then reuses `x[0..768)`.
- `partials` holds the attention GEMV output until `k2_attn_prep` and the MoVA top-k have read it.
- Every mixer and MLP output lands in `partials[0..2560)` for the next fold (SP1). `k2_moe_combine` writes the MoE output there in place over the shared expert's partials.

- [ ] **Step 1: Write the failing test** - `tests/runtime/k2_replay_determinism_test.cc`:

```cpp
// Spec §2 bar 2: the K2 list replays bitwise. Three sessions from reset - A, B, and C with a
// different prompt ingested and reset away first - must give identical ids, taps, routes and
// KV caches. A proves nothing alone; A == B proves replay; A == C proves reset and that no
// step reads scratch it has not written.
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
#include "check.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "loader/k2_loader.h"
#include "runtime/k2/k2_engine.h"

namespace {
uint64_t fnv1a(l0::CmdList& imm, const l0::Mem& m) {
  uint64_t h = 1469598103934665603ull;
  std::vector<uint8_t> chunk(size_t(256) << 20);
  for (size_t off = 0; off < m.size(); off += chunk.size()) {
    const size_t n = std::min(chunk.size(), m.size() - off);
    imm.copy(chunk.data(), static_cast<const uint8_t*>(m.ptr()) + off, n);
    for (size_t i = 0; i < n; ++i) h = (h ^ chunk[i]) * 1099511628211ull;
  }
  return h;
}

struct Session {
  std::vector<uint32_t> ids;
  std::vector<uint16_t> taps;
  std::vector<float> routes;
  uint64_t kv_k = 0, kv_v = 0;
};

std::vector<uint32_t> read_ids(const std::string& path) {
  std::ifstream f(path);
  CHECK(bool(f));
  std::vector<uint32_t> v;
  for (long long x; f >> x;) v.push_back(uint32_t(x));
  CHECK(!v.empty());
  return v;
}
}  // namespace

int main(int argc, char** argv) {
  CHECK(argc > 2);   // <snapshot> <tests/golden/k2/prompts>
  const std::vector<uint32_t> prompt = read_ids(std::string(argv[2]) + "/prose.ids");
  const std::vector<uint32_t> other = read_ids(std::string(argv[2]) + "/code.ids");
  l0::Context ctx(0);
  runtime::k2::K2Engine eng(ctx, loader::load_k2(ctx, argv[1], 16384), 16384, /*debug_resid=*/true);
  std::printf("captured: %zu kernels, %zu modules\n", eng.step().kernel_count, eng.step().modules.size());
  CHECK_EQ(eng.step().kernel_count, runtime::k2::kLaunchesPerToken);
  CHECK_EQ(eng.step().modules.size(), runtime::k2::kModules);
  CHECK_EQ(eng.step().labels.size(), runtime::k2::kLaunchesPerToken);
  l0::CmdList imm = l0::CmdList::immediate(ctx);

  auto session = [&](const std::vector<uint32_t>* detour) {
    if (detour) {
      eng.reset();
      eng.ingest(*detour);
      eng.generate(4);
    }
    eng.reset();
    CHECK_EQ(eng.pos(), uint32_t(0));
    Session s;
    eng.ingest(prompt);
    s.ids = eng.generate(8, [&](uint32_t) {});
    s.taps = eng.read_debug_resid();
    s.routes = eng.read_routes();
    s.kv_k = fnv1a(imm, eng.buffers().kv_k);
    s.kv_v = fnv1a(imm, eng.buffers().kv_v);
    CHECK_EQ(eng.pos(), uint32_t(prompt.size() + 8));
    return s;
  };
  const Session a = session(nullptr), b = session(nullptr), c = session(&other);
  for (const Session* s : {&b, &c}) {
    CHECK(s->ids == a.ids);
    CHECK(s->taps == a.taps);
    CHECK(s->routes == a.routes);
    CHECK_EQ(s->kv_k, a.kv_k);
    CHECK_EQ(s->kv_v, a.kv_v);
  }
  for (uint16_t w : a.taps) CHECK((w & 0x7F80u) != 0x7F80u);   // finite bf16 everywhere
  std::printf("k2_replay_determinism_test OK: ids");
  for (uint32_t id : a.ids) std::printf(" %u", id);
  std::printf("; 3 sessions bitwise identical\n");
  return 0;
}
```

Append to `tests/CMakeLists.txt`:

```cmake
get_property(B70_K2_KERNELS DIRECTORY ${CMAKE_SOURCE_DIR}/src/kernels/k2 PROPERTY BUILDSYSTEM_TARGETS)
add_executable(k2_replay_determinism_test runtime/k2_replay_determinism_test.cc)
target_include_directories(k2_replay_determinism_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(k2_replay_determinism_test PRIVATE b70_k2_runtime)
add_dependencies(k2_replay_determinism_test ${B70_K2_KERNELS})
add_test(NAME k2_replay_determinism_test COMMAND k2_replay_determinism_test
  ${B70_K2_SNAPSHOT} ${CMAKE_SOURCE_DIR}/tests/golden/k2/prompts)
set_tests_properties(k2_replay_determinism_test PROPERTIES LABELS "checkpoint;k2" TIMEOUT 1800)
```

- [ ] **Step 2: Confirm it fails** - `tools/box.sh sync && tools/box.sh build 2>&1 | grep -m1 "k2_engine.h"`. Expected: missing header.

- [ ] **Step 3: Write `src/runtime/k2/k2_capture.h`**

```cpp
#pragma once
#include <cstddef>

#include "l0/context.h"
#include "l0/memory.h"
#include "loader/k2_loader.h"
#include "runtime/capture.h"      // CapturedStep - model-agnostic; build() there is the 27B's
#include "runtime/k2/k2_buffers.h"

namespace runtime::k2 {

inline constexpr size_t kLaunchesPerToken = 2067;   // 1 + 3 x 12 + 45 x 45 + 5 (plan 8e's table)
inline constexpr size_t kModules = 28;

// K2's decode step, captured once: a straight walk of model::K2Horizon with no conditional on
// runtime state. Throws on any mismatch it can see at capture (a missing binary, a buffer
// smaller than its slice arithmetic, max_len disagreeing with the model's, a launch count other
// than kLaunchesPerToken).
// debug_resid: bf16 [48][2560]; a copy of `resid` after each layer's last kernel (the tap
// semantics of runtime/capture.h's `debug_resid`, unchanged).
CapturedStep build(l0::Context& ctx, const loader::K2LoadedModel& m, K2Buffers& b,
                   l0::Mem* debug_resid = nullptr);

}  // namespace runtime::k2
```

- [ ] **Step 4: Implement the capture** - `src/runtime/k2/k2_capture.cc`:

```cpp
#include "runtime/k2/k2_capture.h"

#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "kernels/k2/k2_kernels.h"
#include "model/k2_horizon.h"
#include "runtime/control.h"

// k2_capture.cc - K2-Horizon's 2067-launch decode step (spec 4 §3.3, plan 8e's table). Every
// binding names the kernel it honours; argument orders are plan 8c's Interfaces, and a reader
// can check each launch against its .cl in isolation.
namespace runtime::k2 {
namespace {

using model::K2ExpertId;
using model::K2Horizon;
using model::K2LinearId;
namespace kk = kernels::k2;

void require(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error("runtime::k2::build: " + what);
}
uint8_t* at(const l0::Mem& m, size_t off) { return static_cast<uint8_t*>(m.ptr()) + off; }

class Capture {
 public:
  Capture(l0::Context& ctx, const loader::K2LoadedModel& m, K2Buffers& b, l0::Mem* tap)
      : ctx_(ctx), m_(m), b_(b), tap_(tap), step_{l0::CmdList::regular(ctx), 0, {}, {}, {}} {}

  CapturedStep run() {
    require(m_.max_len == b_.max_len, "model max_len " + std::to_string(m_.max_len) +
                                          " != buffers max_len " + std::to_string(b_.max_len));
    require(b_.kv_k.size() == b_.kv_layer_bytes() * K2Horizon::kLayers, "kv_k is not 48 slices");
    require(b_.kv_v.size() == b_.kv_layer_bytes() * K2Horizon::kLayers, "kv_v is not 48 slices");
    if (tap_) require(tap_->size() >= size_t(K2Horizon::kLayers) * K2Horizon::kHidden * 2,
                      "debug_resid is smaller than [48][2560] bf16");
    embed();
    uint32_t sparse = 0;
    for (uint32_t l = 0; l < K2Horizon::kLayers; ++l) {
      layer_ = int(l);
      if (K2Horizon::is_dense(l)) dense_layer(l); else sparse_layer(l, sparse++);
      if (tap_)
        step_.list.copy(at(*tap_, size_t(l) * K2Horizon::kHidden * 2), b_.resid.ptr(),
                        size_t(K2Horizon::kHidden) * 2);
    }
    layer_ = -1;
    head();
    require(sparse == 45, "sparse layer count is not 45");
    require(step_.kernel_count == kLaunchesPerToken,
            "the walk appended " + std::to_string(step_.kernel_count) + " launches, expected " +
                std::to_string(kLaunchesPerToken));
    require(step_.modules.size() == kModules, "the walk opened " +
                                                  std::to_string(step_.modules.size()) +
                                                  " modules, expected " + std::to_string(kModules));
    step_.list.close();
    return std::move(step_);
  }

 private:
  l0::Kernel& kernel(const std::string& variant, const char* entry, uint32_t wg) {
    auto it = step_.modules.find(variant);
    if (it == step_.modules.end())
      it = step_.modules.emplace(variant, std::make_unique<l0::Module>(ctx_, kernels::path(variant))).first;
    step_.kernels.push_back(std::make_unique<l0::Kernel>(*it->second, entry));
    step_.kernels.back()->group_size(wg);
    pending_ = (layer_ < 0 ? std::string("--") : "L" + std::to_string(layer_)) + " " + entry + " " + variant;
    return *step_.kernels.back();
  }
  void launch(l0::Kernel& k, uint32_t gx, uint32_t gy) {
    step_.list.launch(k, gx, gy, 1, nullptr);
    step_.labels.push_back(pending_);
    ++step_.kernel_count;
  }
  static void bind(l0::Kernel& k, std::initializer_list<const void*> ptrs) {
    uint32_t i = 0;
    for (const void* p : ptrs) k.arg_ptr(i++, p);
  }

  const loader::DeviceWeight& lin(uint32_t l, K2LinearId id) const { return m_.linears.at({l, id}); }
  void* kv_k(uint32_t l) { return at(b_.kv_k, size_t(l) * b_.kv_layer_bytes()); }
  void* kv_v(uint32_t l) { return at(b_.kv_v, size_t(l) * b_.kv_layer_bytes()); }

  void embed() {   // embed_gather.cl: (ctrl, embed, resid), grid (1, M)
    l0::Kernel& k = kernel(kk::embed_gather(), "embed_gather", 256);
    bind(k, {b_.control.ptr(), m_.embed.ptr(), b_.resid.ptr()});
    launch(k, 1, 1);
  }
  // prep.cl prep_res_fold (partials, resid, sumsq), grid (10, 1); k2_norm.cl (sumsq, resid, w, x).
  void norm(uint32_t s_prev, const void* w) {
    l0::Kernel& a = kernel(kk::res_fold(s_prev), "prep_res_fold", 256);
    bind(a, {b_.partials.ptr(), b_.resid.ptr(), b_.norm_sumsq.ptr()});
    launch(a, kk::kFoldG, 1);
    l0::Kernel& f = kernel(kk::norm_finish(), "k2_norm_finish", 256);
    bind(f, {b_.norm_sumsq.ptr(), b_.resid.ptr(), w, b_.x.ptr()});
    launch(f, kk::kFoldG, 1);
  }
  // gemv.cl (w, scales, x, out), grid (N/64, S = 1).
  void gemv(const loader::DeviceWeight& w, const void* x, const l0::Mem& out) {
    const model::GemvShape& s = w.shape;
    require(w.kind == model::WeightKind::Int4 && s.S == 1 && s.layout == 0 && w.scales,
            "K2 gemv needs an int4 layout-0 S=1 weight with scales");
    require(out.size() >= size_t(s.N) * 4, "gemv output smaller than [N] fp32");
    l0::Kernel& k = kernel(kk::gemv(s.K, s.N), "gemv", 64);
    bind(k, {w.mem.ptr(), w.scales->ptr(), x, out.ptr()});
    launch(k, s.N / 64, 1);
  }
  // gemv_bf16.cl (w, x, out), grid (N / cols).
  void gemv_bf16(const loader::DeviceWeight& w, const void* x, const l0::Mem& out) {
    const model::GemvShape& s = w.shape;
    const kernels::GemvBf16Tiling t = kernels::gemv_bf16_tiling(s.N);
    require(w.kind == model::WeightKind::Bf16 && out.size() >= size_t(s.N) * 4,
            "gemv_bf16 needs a bf16 weight and an [N] fp32 output");
    l0::Kernel& k = kernel(kk::gemv_bf16(s.K, s.N), "gemv_bf16", t.cols * t.ksplit);
    bind(k, {w.mem.ptr(), x, out.ptr()});
    launch(k, s.N / t.cols, 1);
  }
  // k2_silu.cl (in, out, slot_base), grid (INTER/256, 1).
  void silu(uint32_t inter, const l0::Mem& in, const l0::Mem& out, uint32_t slot) {
    l0::Kernel& k = kernel(kk::silu_mul(inter), "k2_silu_mul", 256);
    bind(k, {in.ptr(), out.ptr()});
    k.arg(2, slot);
    launch(k, inter / 256, 1);
  }
  // k2_route.cl (logits, bias, route), grid (1, 1).
  void topk(const std::string& variant, const l0::Mem& logits, const void* bias, const void* route) {
    l0::Kernel& k = kernel(variant, "k2_router_topk", 1);
    bind(k, {logits.ptr(), bias, route});
    launch(k, 1, 1);
  }
  // k2_experts.cl k2_gemv_slot (w, scales, route, x, out, slot_base), grid (N/64, 1).
  void slot(const loader::K2ExpertWeights& e, const void* route, const l0::Mem& x,
            const l0::Mem& out, uint32_t s) {
    l0::Kernel& k = kernel(kk::gemv_slot(e.shape.K, e.shape.N), "k2_gemv_slot", 64);
    bind(k, {e.weights.ptr(), e.scales.ptr(), route, x.ptr(), out.ptr()});
    k.arg(5, s);
    launch(k, e.shape.N / 64, 1);
  }
  // k2_attn.cl: prep (ctrl, partials, vbuf, rope, attn_q, attn_gate, kv_k, kv_v) grid (40, 1);
  // decode (ctrl, attn_q, kv_k, kv_v, attn_part) grid (8, L/64); reduce (ctrl, attn_part,
  // attn_gate, attn_out) grid (32, 1); then o_proj into partials.
  void attention(uint32_t l, bool dense) {
    l0::Kernel& p = kernel(kk::attn_prep(dense), "k2_attn_prep", 128);
    bind(p, {b_.control.ptr(), b_.partials.ptr(), b_.vbuf.ptr(), m_.rope.ptr(), b_.attn_q.ptr(),
             b_.attn_gate.ptr(), kv_k(l), kv_v(l)});
    launch(p, K2Horizon::kQHeads + K2Horizon::kKvHeads, 1);
    l0::Kernel& d = kernel(kk::attn_decode(b_.max_len), "k2_attn_decode", 128);
    bind(d, {b_.control.ptr(), b_.attn_q.ptr(), kv_k(l), kv_v(l), b_.attn_part.ptr()});
    launch(d, K2Horizon::kKvHeads, b_.max_len / K2Buffers::kAttnBlock);
    l0::Kernel& r = kernel(kk::attn_reduce(b_.max_len), "k2_attn_reduce", 128);
    bind(r, {b_.control.ptr(), b_.attn_part.ptr(), b_.attn_gate.ptr(), b_.attn_out.ptr()});
    launch(r, K2Horizon::kQHeads, 1);
    gemv(lin(l, K2LinearId::OProj), b_.attn_out.ptr(), b_.partials);
  }

  void dense_layer(uint32_t l) {
    norm(l == 0 ? 0u : 1u, at(m_.norms[l], K2Horizon::kNormsOffInput));
    gemv(lin(l, K2LinearId::AttnDense), b_.x.ptr(), b_.partials);
    attention(l, true);
    norm(1, at(m_.norms[l], K2Horizon::kNormsOffPost));
    gemv(lin(l, K2LinearId::DenseGateUp), b_.x.ptr(), b_.partials);
    silu(K2Horizon::kDenseInter, b_.partials, b_.x, 0);
    gemv(lin(l, K2LinearId::DenseDown), b_.x.ptr(), b_.partials);
  }

  void sparse_layer(uint32_t l, uint32_t s) {
    const void* bias = m_.route_bias.at(l).ptr();
    const void* rmova = at(b_.route_mova, size_t(s) * K2Horizon::kValueTopK * 2 * 4);
    const void* rmoe = at(b_.route_moe, size_t(s) * K2Horizon::kTopK * 2 * 4);
    const loader::K2ExpertWeights& value = m_.experts.at({l, K2ExpertId::Value});
    const loader::K2ExpertWeights& gu = m_.experts.at({l, K2ExpertId::MoeGateUp});
    const loader::K2ExpertWeights& down = m_.experts.at({l, K2ExpertId::MoeDown});

    norm(1, at(m_.norms[l], K2Horizon::kNormsOffInput));
    gemv(lin(l, K2LinearId::AttnSparse), b_.x.ptr(), b_.partials);
    topk(kk::router_topk(K2Horizon::kValueExperts, K2Horizon::kValueTopK), b_.partials, bias, rmova);
    for (uint32_t t = 0; t < K2Horizon::kValueTopK; ++t) slot(value, rmova, b_.x, b_.vslots, t);
    {   // k2_experts.cl k2_value_combine (slots, route, vbuf), grid (16, 1)
      l0::Kernel& k = kernel(kk::value_combine(), "k2_value_combine", 64);
      bind(k, {b_.vslots.ptr(), rmova, b_.vbuf.ptr()});
      launch(k, 16, 1);
    }
    attention(l, false);

    norm(1, at(m_.norms[l], K2Horizon::kNormsOffPost));
    gemv_bf16(lin(l, K2LinearId::MoeRouter), b_.x.ptr(), b_.router_logits);
    topk(kk::router_topk(K2Horizon::kExperts, K2Horizon::kTopK), b_.router_logits, bias, rmoe);
    for (uint32_t t = 0; t < K2Horizon::kTopK; ++t) {
      slot(gu, rmoe, b_.x, b_.gu_slots, t);
      silu(K2Horizon::kMoeInter, b_.gu_slots, b_.mid, t);
      slot(down, rmoe, b_.mid, b_.down_slots, t);
    }
    gemv(lin(l, K2LinearId::SharedGateUp), b_.x.ptr(), b_.partials);
    silu(K2Horizon::kMoeInter, b_.partials, b_.x, 0);
    gemv(lin(l, K2LinearId::SharedDown), b_.x.ptr(), b_.partials);
    {   // k2_experts.cl k2_moe_combine (down_slots, route, io = partials), grid (10, 1)
      l0::Kernel& k = kernel(kk::moe_combine(), "k2_moe_combine", 256);
      bind(k, {b_.down_slots.ptr(), rmoe, b_.partials.ptr()});
      launch(k, 10, 1);
    }
  }

  void head() {
    norm(1, m_.final_norm.ptr());
    gemv_bf16(m_.linears.at({loader::kTopLevel, K2LinearId::LmHead}), b_.x.ptr(), b_.logits);
    l0::Kernel& a1 = kernel(kk::argmax_stage1(), "argmax_stage1", 256);   // (logits, part)
    bind(a1, {b_.logits.ptr(), b_.argmax_part.ptr()});
    launch(a1, K2Buffers::kArgmaxGroups, 1);
    l0::Kernel& a2 = kernel(kk::argmax_stage2(), "argmax_stage2", 256);   // (ctrl, part)
    bind(a2, {b_.control.ptr(), b_.argmax_part.ptr()});
    launch(a2, 1, 1);
  }

  l0::Context& ctx_;
  const loader::K2LoadedModel& m_;
  K2Buffers& b_;
  l0::Mem* tap_;
  CapturedStep step_;
  int layer_ = -1;
  std::string pending_;
};

}  // namespace

CapturedStep build(l0::Context& ctx, const loader::K2LoadedModel& m, K2Buffers& b,
                   l0::Mem* debug_resid) {
  return Capture(ctx, m, b, debug_resid).run();
}

}  // namespace runtime::k2
```

- [ ] **Step 5: Write `src/runtime/k2/k2_engine.h`**

```cpp
#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "l0/cmdlist.h"
#include "l0/context.h"
#include "l0/fence.h"
#include "l0/memory.h"
#include "l0/queue.h"
#include "loader/k2_loader.h"
#include "runtime/control.h"
#include "runtime/k2/k2_buffers.h"
#include "runtime/k2/k2_capture.h"

namespace runtime::k2 {

// runtime::Engine's decode contract for K2-Horizon (spec §3.1), and nothing else: a token is a
// replay of the one captured list. The host writes at most cur_token[0], submits, waits on the
// fence and reads out_token/cur_token - no branch depends on a device value, so a replay is
// bitwise reproducible (k2_replay_determinism_test).
class K2Engine {
 public:
  K2Engine(l0::Context& ctx, loader::K2LoadedModel model, uint32_t max_len, bool debug_resid = false);
  void reset();                                   // zeroes control, kv_k, kv_v; never scratch
  void ingest(const std::vector<uint32_t>& ids);  // one replay per id
  std::vector<uint32_t> generate(uint32_t n, const std::function<void(uint32_t)>& on_token = {});
  double last_tok_per_s() const { return last_tok_per_s_; }
  double last_gen_ms() const { return last_gen_ms_; }
  double last_fence_ms() const { return last_fence_ms_; }
  std::vector<uint16_t> read_debug_resid();       // bf16 [48][2560]; throws unless debug_resid
  std::vector<float> read_routes();               // route_mova [45][4][2] then route_moe [45][8][2]
  const loader::K2LoadedModel& model() const { return model_; }
  K2Buffers& buffers() { return buffers_; }
  const CapturedStep& step() const { return step_; }
  uint32_t pos() const { return control_->pos; }
  uint32_t max_len() const { return buffers_.max_len; }

 private:
  void replay();
  l0::Context& ctx_;
  loader::K2LoadedModel model_;
  K2Buffers buffers_;
  std::unique_ptr<l0::Mem> tap_;
  CapturedStep step_;
  l0::Queue queue_;
  l0::Fence fence_;
  l0::CmdList imm_;
  Control* control_;
  double last_tok_per_s_ = 0.0, last_gen_ms_ = 0.0, last_fence_ms_ = 0.0;
};

}  // namespace runtime::k2
```

- [ ] **Step 6: Implement** - `src/runtime/k2/k2_engine.cc`:

```cpp
#include "runtime/k2/k2_engine.h"

#include <chrono>
#include <stdexcept>
#include <string>
#include <utility>

#include "model/k2_horizon.h"

namespace runtime::k2 {
namespace {
using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}
constexpr size_t kTapBytes = size_t(model::K2Horizon::kLayers) * model::K2Horizon::kHidden * 2;

uint32_t checked(const loader::K2LoadedModel& m, uint32_t max_len) {
  if (m.max_len != max_len)
    throw std::runtime_error("runtime::k2::K2Engine: model loaded with max_len " +
                             std::to_string(m.max_len) + ", engine asked for " +
                             std::to_string(max_len) + " - the RoPE table, the KV and the "
                             "attention variants' MAXLEN are one number");
  return max_len;
}
}  // namespace

K2Engine::K2Engine(l0::Context& ctx, loader::K2LoadedModel model, uint32_t max_len, bool debug_resid)
    : ctx_(ctx),
      model_(std::move(model)),
      buffers_(ctx, checked(model_, max_len)),
      tap_(debug_resid ? std::make_unique<l0::Mem>(ctx, l0::MemKind::Device, kTapBytes) : nullptr),
      step_(build(ctx, model_, buffers_, tap_.get())),
      queue_(ctx),
      fence_(queue_),
      imm_(l0::CmdList::immediate(ctx)),
      control_(buffers_.control.as<Control>()) {
  reset();
}

void K2Engine::reset() { buffers_.zero_persistent(imm_); }

void K2Engine::replay() {
  const size_t pos = control_->pos, n = control_->n_active;
  if (pos + n > size_t(buffers_.max_len))
    throw std::runtime_error("runtime::k2::K2Engine: pos " + std::to_string(pos) + " + n_active " +
                             std::to_string(n) + " exceeds max_len " +
                             std::to_string(buffers_.max_len));
  queue_.execute(step_.list, &fence_);
  fence_.wait();
}

void K2Engine::ingest(const std::vector<uint32_t>& ids) {
  control_->n_active = 1;
  for (uint32_t id : ids) {
    control_->cur_token[0] = id;
    replay();
  }
}

std::vector<uint32_t> K2Engine::generate(uint32_t n, const std::function<void(uint32_t)>& on_token) {
  std::vector<uint32_t> out;
  out.reserve(n);
  control_->n_active = 1;
  double fence_ms = 0.0;
  const Clock::time_point t0 = Clock::now();
  for (uint32_t i = 0; i < n; ++i) {
    const uint32_t id = control_->cur_token[0];
    const Clock::time_point f0 = Clock::now();
    replay();
    fence_ms += ms_since(f0);
    out.push_back(id);
    if (on_token) on_token(id);
  }
  last_gen_ms_ = ms_since(t0);
  last_fence_ms_ = fence_ms;
  last_tok_per_s_ = (n != 0 && last_gen_ms_ > 0.0) ? double(n) * 1000.0 / last_gen_ms_ : 0.0;
  return out;
}

std::vector<uint16_t> K2Engine::read_debug_resid() {
  if (!tap_) throw std::runtime_error("runtime::k2::K2Engine: constructed without debug_resid");
  std::vector<uint16_t> out(kTapBytes / 2);
  imm_.copy(out.data(), tap_->ptr(), kTapBytes);
  return out;
}

std::vector<float> K2Engine::read_routes() {
  const size_t a = buffers_.route_mova.size() / 4, b = buffers_.route_moe.size() / 4;
  std::vector<float> out(a + b);
  imm_.copy(out.data(), buffers_.route_mova.ptr(), a * 4);
  imm_.copy(out.data() + a, buffers_.route_moe.ptr(), b * 4);
  return out;
}

}  // namespace runtime::k2
```

- [ ] **Step 7: Build, then run the determinism test detached** (the load is ~5 min, then three sessions):
  `tools/box.sh sync && tools/box.sh build 2>&1 | tail -3`
  `ssh -o BatchMode=yes user@box 'cd ~/b70-inference-server/build && ZE_AFFINITY_MASK=1 setsid nohup ctest -R "^k2_replay_determinism_test$" --output-on-failure > $HOME/k2-replay.log 2>&1 < /dev/null &'`
  Poll `tail -8 $HOME/k2-replay.log`. Expected: `captured: 2067 kernels, 28 modules`, then `k2_replay_determinism_test OK: ids … ; 3 sessions bitwise identical`.
  **Two ways this can fail before any numerics:**
  - A throw naming a missing binary means a name disagrees between `kernels::k2` and the CMake. Fix the one name.
  - A launch count other than 2067 means the walk and plan 8e's table disagree. Fix the walk, not the constant.

- [ ] **Step 8: Commit**

```bash
git add src/runtime/k2 tests/runtime/k2_replay_determinism_test.cc tests/CMakeLists.txt
git commit -m "feat(runtime): K2 decode capture (2067 launches, 28 modules) and K2Engine; replay determinism

Claude-Session: "
```

---

### Task 3: The golden gate (spec §2 bar 1)

**Files:**
- Create: `tests/golden/k2_golden_gate_test.cc`
- Modify: `tests/CMakeLists.txt` (append)

**Interfaces:**
- Consumes: `golden_common.h` verbatim (`Golden`, `compare_bf16`, `compare_f32`, `golden_decision`, `argmax_full`, `print_top5`, `exists`, `kBar`); Task 2's `K2Engine`; plan 8d's golden files (`resid.L*`, `mixer.L*`, `logits`, `tokens`, `route.{mova,moe}.ids.L*`, `rope.{cos,sin}`); plan 8b's `k2_rope_table`.
- Produces: `k2_golden_gate_test <oracle-out-k2> <tests/golden/k2/prompts> <snapshot>`. Exit 0 iff, on all three prompts, every determined row is element-exact and every undetermined row's engine id is in the oracle's argmax set. Exit 77 if the golden files are absent.

- [ ] **Step 1: Write the test** - `tests/golden/k2_golden_gate_test.cc`:

```cpp
// Spec 4 §2 bar 1 / §4: K2-Horizon against its CPU oracle, 3 prompts x 32 greedy tokens.
// The DECISION RULE is golden_common.h's, unchanged: a determined row (unique golden argmax)
// must be element-exact; an undetermined row must land inside the argmax set; the walk
// teacher-forces the golden id after any difference. Tokens gate; everything else here is
// diagnosis: per-layer tap cosines, the reference's RoPE rows, and - new for K2 - per sparse
// layer, the first forward whose engine expert ids differ from the oracle's.
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "golden_common.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "loader/k2_loader.h"
#include "model/k2_horizon.h"
#include "runtime/control.h"
#include "runtime/k2/k2_engine.h"

namespace {
using model::K2Horizon;
constexpr uint32_t kMaxLen = 16384, kGen = 32, kHid = 2560, kV = K2Horizon::kVocab;
const char* const kPrompts[] = {"prose", "code", "cjk"};

std::vector<uint32_t> read_k2_ids(const std::string& path) {
  std::ifstream f(path);
  if (!f) {
    std::fprintf(stderr, "cannot open prompt ids: %s\n", path.c_str());
    std::exit(1);
  }
  std::vector<uint32_t> ids;
  for (long long v; f >> v;) {
    CHECK(v >= 0 && v < (long long)kV);
    ids.push_back(uint32_t(v));
  }
  return ids;
}

struct Verdict {
  std::string name;
  uint32_t n_determined = 0, det_exact = 0, n_tie = 0, tie_agree = 0, tie_member = 0;
  int first_bad = -1;
  double tap_min_cos = 2.0;
  uint32_t tap_min_layer = 0, route_layers_diff = 0;
};
}  // namespace

int main(int argc, char** argv) {
  CHECK(argc > 3);
  const std::string gdir = argv[1], pdir = argv[2], snap = argv[3];
  for (const char* p : kPrompts)
    if (!golden::exists(gdir + "/" + p + ".golden.safetensors")) {
      std::printf("SKIP: %s/%s.golden.safetensors is absent (plan 8d builds it on the box)\n",
                  gdir.c_str(), p);
      return 77;
    }
  l0::Context ctx(0);
  runtime::k2::K2Engine eng(ctx, loader::load_k2(ctx, snap, kMaxLen), kMaxLen, /*debug_resid=*/true);
  CHECK_EQ(eng.step().kernel_count, runtime::k2::kLaunchesPerToken);
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  const std::vector<float> rope = loader::k2_rope_table(kMaxLen);
  runtime::Control* ctrl = eng.buffers().control.as<runtime::Control>();
  std::vector<Verdict> verdicts;
  std::vector<double> sa, sb;

  for (const char* pname : kPrompts) {
    Verdict v;
    v.name = pname;
    const std::vector<uint32_t> ids = read_k2_ids(pdir + "/" + pname + ".ids");
    golden::Golden g(gdir + "/" + pname + ".golden.safetensors");
    const uint32_t T = uint32_t(ids.size());
    CHECK(T > 0 && T <= 64);
    CHECK_EQ(g.dim("resid.L0", 2, 0), uint64_t(T));
    CHECK_EQ(g.dim("logits", 2, 0), uint64_t(T + kGen));
    CHECK_EQ(g.dim("logits", 2, 1), uint64_t(kV));
    std::printf("\n================ %s: %u prompt ids, %u generated ================\n", pname, T, kGen);

    eng.reset();
    std::vector<std::vector<uint16_t>> tap(T), emb(T, std::vector<uint16_t>(kHid));
    std::vector<std::vector<float>> routes;   // one per forward: T prompt rows, then kGen
    for (uint32_t t = 0; t < T; ++t) {
      eng.ingest({ids[t]});
      tap[t] = eng.read_debug_resid();
      routes.push_back(eng.read_routes());
      imm.copy(emb[t].data(),
               static_cast<const uint8_t*>(eng.model().embed.ptr()) + size_t(ids[t]) * kHid * 2,
               size_t(kHid) * 2);
    }

    // 1. taps: expected = bf16(resid.L{l-1} + mixer.L{l}), embed row standing in for L-1.
    std::printf("  layer kind     min cos      at t\n");
    std::vector<uint16_t> expect(kHid);
    for (uint32_t l = 0; l < K2Horizon::kLayers; ++l) {
      const uint16_t* mix = g.bf16("mixer.L" + std::to_string(l), size_t(T) * kHid);
      const uint16_t* prev = l == 0 ? nullptr
                                    : g.bf16("resid.L" + std::to_string(l - 1), size_t(T) * kHid);
      double lmin = 2.0;
      uint32_t at = 0;
      for (uint32_t t = 0; t < T; ++t) {
        const uint16_t* base = prev ? prev + size_t(t) * kHid : emb[t].data();
        for (uint32_t k = 0; k < kHid; ++k)
          expect[k] = common::f32_to_bf16(common::bf16_to_f32(base[k]) +
                                          common::bf16_to_f32(mix[size_t(t) * kHid + k]));
        const golden::Metric m =
            golden::compare_bf16(tap[t].data() + size_t(l) * kHid, expect.data(), kHid, sa, sb);
        if (m.cos < lmin) { lmin = m.cos; at = t; }
      }
      if (lmin < v.tap_min_cos) { v.tap_min_cos = lmin; v.tap_min_layer = l; }
      std::printf("    %2u  %-6s  %.9f  %4u%s\n", l, K2Horizon::is_dense(l) ? "dense" : "sparse",
                  lmin, at, lmin < golden::kBar ? "   **LOW**" : "");
    }

    // 2. RoPE: the engine's table rows vs the reference's bf16 cos/sin (dims i and i+64 share i).
    {
      const uint16_t* gc = g.bf16("rope.cos", size_t(T + kGen) * 128);
      const uint16_t* gs = g.bf16("rope.sin", size_t(T + kGen) * 128);
      size_t diff = 0;
      for (uint32_t p = 0; p < T + kGen; ++p)
        for (uint32_t i = 0; i < 128; ++i) {
          diff += rope[(size_t(p) * 2 + 0) * 64 + i % 64] != common::bf16_to_f32(gc[size_t(p) * 128 + i]);
          diff += rope[(size_t(p) * 2 + 1) * 64 + i % 64] != common::bf16_to_f32(gs[size_t(p) * 128 + i]);
        }
      std::printf("  rope: %zu of %u cos/sin elements differ from the reference's rows\n", diff,
                  (T + kGen) * 256);
    }

    // 3. the gate: golden_common.h's decision rule, verbatim semantics.
    const int32_t* gtok = g.i32("tokens", kGen);
    const float* glog = g.f32("logits", size_t(T + kGen) * kV);
    std::vector<float> dec(kV);
    auto read_logits = [&] { imm.copy(dec.data(), eng.buffers().logits.ptr(), size_t(kV) * 4); };
    read_logits();
    bool forced = false;
    std::printf("  greedy %u:  step  engine  golden  det?  logit-cos\n", kGen);
    for (uint32_t p = 0; p < kGen; ++p) {
      const float* grow = glog + size_t(p == 0 ? T - 1 : T + p - 1) * kV;
      const golden::Metric lm = golden::compare_f32(dec.data(), grow, kV, sa, sb);
      const golden::GoldenDecision gd = golden::golden_decision(grow, kV, kV);
      CHECK_EQ(gd.set[0], uint32_t(gtok[p]));
      const uint32_t id = ctrl->cur_token[0];
      CHECK_EQ(id, golden::argmax_full(dec.data(), kV));
      const bool ok = id == uint32_t(gtok[p]);
      if (gd.determined()) {
        ++v.n_determined;
        if (ok) ++v.det_exact;
        else if (v.first_bad < 0) v.first_bad = int(p);
      } else {
        ++v.n_tie;
        if (ok) ++v.tie_agree;
        else if (gd.contains(id)) ++v.tie_member;
        else if (v.first_bad < 0) v.first_bad = int(p);
      }
      std::printf("            %4u  %6u  %6u  %-4s  %.9f%s%s\n", p, id, uint32_t(gtok[p]),
                  gd.determined() ? "yes" : "TIE", lm.cos,
                  ok ? "" : (gd.determined() ? "  <== MISMATCH (determined - GATE)" : "  <== tie row"),
                  forced ? "  [teacher-forced]" : "");
      if (!ok && !forced) {
        golden::print_top5("engine", dec.data(), kV, kV);
        golden::print_top5("golden", grow, kV, kV);
      }
      if (!ok) forced = true;
      if (forced) eng.ingest({uint32_t(gtok[p])});
      else CHECK_EQ(eng.generate(1)[0], id);
      routes.push_back(eng.read_routes());
      read_logits();
    }

    // 4. routes (spec §4's new diagnostic): per sparse layer, the first forward whose ids differ.
    constexpr size_t kMovaFloats = 45 * 4 * 2;
    for (uint32_t s = 0; s < 45; ++s) {
      const uint32_t l = 3 + s;
      for (int kind = 0; kind < 2; ++kind) {
        const uint32_t k = kind == 0 ? K2Horizon::kValueTopK : K2Horizon::kTopK;
        const size_t base = kind == 0 ? size_t(s) * 4 * 2 : kMovaFloats + size_t(s) * 8 * 2;
        const std::string name = std::string("route.") + (kind == 0 ? "mova" : "moe") + ".ids.L" +
                                 std::to_string(l);
        const int32_t* gid = g.i32(name, size_t(T + kGen) * k);
        int first = -1;
        uint32_t rows = 0;
        for (size_t r = 0; r < routes.size(); ++r) {
          bool same = true;
          for (uint32_t t = 0; t < k; ++t)
            same &= uint32_t(routes[r][base + size_t(t) * 2]) == uint32_t(gid[r * k + t]);
          if (!same) {
            ++rows;
            if (first < 0) first = int(r);
          }
        }
        if (rows != 0) {
          ++v.route_layers_diff;
          std::printf("  %s: %u of %zu forwards differ, first at %d (%s)\n", name.c_str(), rows,
                      routes.size(), first, first < int(T) ? "prompt" : "generated");
        }
      }
    }
    std::printf("  %s: %u/%u determined exact, %u undetermined (%u agree + %u other member); "
                "worst tap cos %.6f at L%u; %u (layer, router) pairs with a routing difference\n",
                pname, v.det_exact, v.n_determined, v.n_tie, v.tie_agree, v.tie_member,
                v.tap_min_cos, v.tap_min_layer, v.route_layers_diff);
    verdicts.push_back(v);
  }

  uint32_t det = 0, exact = 0, ties = 0, agree = 0, member = 0;
  bool pass = true;
  for (const Verdict& v : verdicts) {
    det += v.n_determined; exact += v.det_exact;
    ties += v.n_tie; agree += v.tie_agree; member += v.tie_member;
    pass &= v.first_bad < 0;
  }
  std::printf("\nTOTAL: %u/%u determined rows exact, %u undetermined (%u agree + %u other member) - %s\n",
              exact, det, ties, agree, member, pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
```

- [ ] **Step 2: Register** - append to `tests/CMakeLists.txt`:

```cmake
set(B70_K2_ORACLE_DIR "${CMAKE_SOURCE_DIR}/oracle-out-k2"
  CACHE PATH "The K2 CPU-oracle golden set (tools/oracle/golden_k2.sh OUT_DIR)")
add_executable(k2_golden_gate_test golden/k2_golden_gate_test.cc)
target_include_directories(k2_golden_gate_test PRIVATE ${CMAKE_SOURCE_DIR}/src
                           ${CMAKE_SOURCE_DIR}/tests ${CMAKE_SOURCE_DIR}/tests/golden)
target_link_libraries(k2_golden_gate_test PRIVATE b70_k2_runtime b70_loader b70_model)
add_dependencies(k2_golden_gate_test ${B70_K2_KERNELS})
add_test(NAME k2_golden_gate_test COMMAND k2_golden_gate_test
  ${B70_K2_ORACLE_DIR} ${CMAKE_SOURCE_DIR}/tests/golden/k2/prompts ${B70_K2_SNAPSHOT})
set_tests_properties(k2_golden_gate_test PROPERTIES
  LABELS "checkpoint;golden;k2" SKIP_RETURN_CODE 77 TIMEOUT 3600)
```

- [ ] **Step 3: Build and run detached** (plan 8d's golden sets must exist on the box):
  `tools/box.sh sync && tools/box.sh build 2>&1 | tail -2`
  `ssh -o BatchMode=yes user@box 'cd ~/b70-inference-server/build && ZE_AFFINITY_MASK=1 setsid nohup ctest -R "^k2_golden_gate_test$" -V > $HOME/k2-gate.log 2>&1 < /dev/null &'`
  Poll `grep -E "TOTAL|: [0-9]+/[0-9]+ determined|SKIP" $HOME/k2-gate.log`. Expected: `TOTAL: <n>/<n> determined rows exact … - PASS`.
  **If it FAILs:** that is the stopping rule's memo (spec §2), not a tuning pass. Pull the log (`tools/box.sh pull` is not needed; `scp` the log to the scratchpad). The report quotes:
  1. the first determined mismatch's step and top-5 rows;
  2. the worst tap layer, and whether it is the first layer below `kBar`;
  3. the first routing difference (layer, router, forward) that precedes the token mismatch;
  4. the RoPE mismatch count.

  Name the one defect those point at, and propose one fix. Do not change a define or a kernel on a guess.

- [ ] **Step 4: Commit** (the test; the log's numbers go into Task 5's records)

```bash
git add tests/golden/k2_golden_gate_test.cc tests/CMakeLists.txt
git commit -m "test(golden): k2_golden_gate_test - tie-aware gate plus per-layer routing diagnostic

Claude-Session: "
```

---

### Task 4: `b70-decode` dispatch and the bench row

**Files:**
- Create: `src/cli/k2_decode.h`, `src/cli/k2_decode.cc`
- Modify: `src/cli/b70_decode.cc` (`main` only, plus one include), `src/cli/CMakeLists.txt` (append)

**Interfaces:**
- Consumes: `loader::resolve_snapshot`, `common::json::parse`, `loader::load_k2`, `runtime::k2::K2Engine`, `model::K2Horizon`.
- Produces:
  - `bool k2cli::is_k2(const std::string& model_arg)` - true iff the resolved snapshot's config.json says `k2_horizon`; false if it cannot be read, so the 27B path reports its own error.
  - `int k2cli::run(int argc, char** argv)` - `b70-decode <k2-snapshot> --ids <file> --n <N> [--device N] [--max-len L]` and `b70-decode <k2-snapshot> --bench [--depth 4096] [--tg 256] [--device N]`.
  - Bench row `| b70-decode k2 <sha> | <depth> | <tg> | <t/s> | <ms/token> |`. `tools/bench_decode.sh` parses it unchanged (`| b70-decode ` prefix, t/s in column 5).

- [ ] **Step 1: Write `src/cli/k2_decode.h`**

```cpp
#pragma once
#include <string>

// b70-decode's K2-Horizon half (spec 4 §3.1): dispatched from main() on config.json's model_type.
namespace k2cli {
bool is_k2(const std::string& model_arg);
int run(int argc, char** argv);
}  // namespace k2cli
```

- [ ] **Step 2: Write `src/cli/k2_decode.cc`**

```cpp
#include "cli/k2_decode.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/json.h"
#include "l0/context.h"
#include "loader/k2_loader.h"
#include "loader/snapshot.h"
#include "model/k2_horizon.h"
#include "runtime/k2/k2_engine.h"

namespace k2cli {
namespace {
using model::K2Horizon;
constexpr double kDeviceGBs = 590.0;   // GB/s, measured (docs/01-hardware.md)

uint32_t u32(const char* flag, const std::string& s) {
  if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos || s.size() > 9)
    throw std::runtime_error(std::string(flag) + " expects a non-negative integer, got '" + s + "'");
  return uint32_t(std::stoul(s));
}

std::vector<uint32_t> read_ids(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open --ids file '" + path + "'");
  std::vector<uint32_t> ids;
  for (std::string w; f >> w;) {
    const uint32_t id = u32("--ids", w);
    if (id >= K2Horizon::kVocab) throw std::runtime_error("--ids: id " + w + " is outside K2's vocabulary");
    ids.push_back(id);
  }
  if (ids.empty()) throw std::runtime_error("--ids file '" + path + "' is empty");
  return ids;
}
}  // namespace

bool is_k2(const std::string& model_arg) {
  try {
    const std::string snap = loader::resolve_snapshot(model_arg);
    std::ifstream cf(snap + "config.json");
    if (!cf) return false;
    std::stringstream cs;
    cs << cf.rdbuf();
    const common::json::Value cfg = common::json::parse(cs.str());
    const common::json::Value* mt = cfg.find("model_type");
    return mt && mt->is_string() && mt->str() == "k2_horizon";
  } catch (const std::exception&) {
    return false;
  }
}

int run(int argc, char** argv) {
  const std::string path = argv[1];
  std::string ids_path;
  uint32_t n = 0, depth = 4096, tg = 256, max_len = 16384, device = l0::Context::kFromEnv;
  bool bench = false, have_n = false;
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    auto value = [&](const char* flag) -> std::string {
      if (++i >= argc) throw std::runtime_error(std::string(flag) + " needs a value");
      return argv[i];
    };
    if (a == "--ids") ids_path = value("--ids");
    else if (a == "--n") { n = u32("--n", value("--n")); have_n = true; }
    else if (a == "--bench") bench = true;
    else if (a == "--depth") depth = u32("--depth", value("--depth"));
    else if (a == "--tg") tg = u32("--tg", value("--tg"));
    else if (a == "--device") device = u32("--device", value("--device"));
    else if (a == "--max-len") max_len = u32("--max-len", value("--max-len"));
    else throw std::runtime_error("K2-Horizon supports --ids/--n and --bench [--depth --tg], plus "
                                  "--device and --max-len; got '" + a + "' (prefill and --profile are later specs)");
  }
  if (bench == !ids_path.empty())
    throw std::runtime_error("choose exactly one of --ids <file> --n <N> or --bench");
  if (!bench && !have_n) throw std::runtime_error("--ids needs --n");
  if (bench && depth + tg > max_len) throw std::runtime_error("--depth + --tg exceeds --max-len");

  l0::Context ctx(device);
  std::fprintf(stderr, "device: %s\n", ctx.name().c_str());
  runtime::k2::K2Engine eng(ctx, loader::load_k2(ctx, path, max_len), max_len);
  std::fprintf(stderr, "engine: %zu kernels, %zu modules, max_len %u, %.2f GB KV + %.1f MB scratch\n",
               eng.step().kernel_count, eng.step().modules.size(), max_len,
               eng.buffers().persistent_bytes() / 1e9, eng.buffers().scratch_bytes() / 1e6);

  std::vector<uint32_t> ids;
  if (bench) {
    // Synthetic ingest ids: BOS, then an LCG over the ordinary vocabulary [2, 250000). Content
    // does not change a decode token's bytes read - every token reads 4/64 value and 8/100 MoE
    // experts per sparse layer whichever they are - so the row is a property of depth and tg.
    ids.resize(depth);
    ids[0] = K2Horizon::kBos;
    uint64_t s = 0x9E3779B97F4A7C15ull;
    for (uint32_t i = 1; i < depth; ++i) {
      s = s * 6364136223846793005ull + 1442695040888963407ull;
      ids[i] = 2 + uint32_t((s >> 33) % 249998);
    }
  } else {
    ids = read_ids(ids_path);
  }
  const auto t0 = std::chrono::steady_clock::now();
  eng.ingest(ids);
  const double ingest_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  std::fprintf(stderr, "ingest: %zu ids in %.1f ms (%.2f ms/token), pos %u\n", ids.size(), ingest_ms,
               ingest_ms / double(ids.size()), eng.pos());

  if (!bench) {
    eng.generate(n, [](uint32_t id) {
      std::printf("%u\n", id);
      std::fflush(stdout);
    });
    std::fprintf(stderr, "generate: %u ids, %.2f t/s\n", n, eng.last_tok_per_s());
    return 0;
  }
  eng.generate(tg);
  const double ms = eng.last_gen_ms() / double(tg);
  const double gb = double(eng.model().report.read_per_token) / 1e9;
  std::fprintf(stderr,
               "generate: %u ids, %.2f t/s, %.3f ms/token (%.1f%% inside the fence)\n"
               "  launches per token: %zu; per-launch share of the step: %.3f us/launch (derived)\n"
               "  MBU: %.2f t/s x %.3f GB = %.0f GB/s of %.0f GB/s measured = %.1f%% (read/token counts active experts only)\n",
               tg, eng.last_tok_per_s(), ms, 100.0 * eng.last_fence_ms() / eng.last_gen_ms(),
               eng.step().kernel_count, 1000.0 * ms / double(eng.step().kernel_count),
               eng.last_tok_per_s(), gb, eng.last_tok_per_s() * gb, kDeviceGBs,
               100.0 * eng.last_tok_per_s() * gb / kDeviceGBs);
  const char* sha = std::getenv("B70_GIT_SHA");
  if (sha == nullptr || *sha == '\0') sha = "unknown";
  std::printf("| b70-decode k2 %s | %u | %u | %.2f | %.2f |\n", sha, depth, tg,
              eng.last_tok_per_s(), ms);
  return 0;
}

}  // namespace k2cli
```

- [ ] **Step 3: Dispatch in `main`** - in `src/cli/b70_decode.cc`, add `#include "cli/k2_decode.h"` after `#include "runtime/engine.h"`, and change `main` to:

```cpp
int main(int argc, char** argv) {
  try {
    // Spec 4 §3.1: the one model-type dispatch. The model is argv[1] in every documented
    // invocation; a K2-Horizon snapshot goes to its own engine, everything else is unchanged.
    if (argc > 1 && argv[1][0] != '-' && k2cli::is_k2(argv[1])) return k2cli::run(argc, argv);
    return run(argc, argv);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "b70-decode: %s\n", e.what());
    return 1;
  }
}
```

Append to `src/cli/CMakeLists.txt`:

```cmake
# Spec 4: K2-Horizon, reached through main()'s model_type dispatch.
target_sources(b70-decode PRIVATE k2_decode.cc)
target_link_libraries(b70-decode PRIVATE b70_k2_runtime)
get_property(K2_KERNELS DIRECTORY ${CMAKE_SOURCE_DIR}/src/kernels/k2 PROPERTY BUILDSYSTEM_TARGETS)
add_dependencies(b70-decode ${K2_KERNELS})
```

- [ ] **Step 4: Build; check both halves of the dispatch** -
  - `tools/box.sh sync && tools/box.sh build 2>&1 | tail -2`
  - `tools/box.sh test '^b70_cli_reject'`. Expected: the 27B CLI's rejection tests all still pass.
  - `ZE_AFFINITY_MASK=1 tools/box.sh run './build/src/cli/b70-decode /home/user/.cache/huggingface/hub/models--urakozz--IFM-K2-Horizon-MoVA-36B-A4B-W4A16-AutoRound-GPTQ/snapshots/c0fd9971ad6d3bb4caedf726e798a02b8df9bcb5 --ids tests/golden/k2/prompts/prose.ids --n 8'`. This is detached if the WiFi is flaky; the load is minutes. Expected: 8 ids on stdout, **equal to the first 8 `tokens` of `oracle-out-k2/prose.golden.safetensors`** (Task 3's log prints them in its greedy table).

- [ ] **Step 5: Commit**

```bash
git add src/cli/k2_decode.h src/cli/k2_decode.cc src/cli/b70_decode.cc src/cli/CMakeLists.txt
git commit -m "feat(cli): b70-decode dispatches K2-Horizon on model_type; --ids and --bench

Claude-Session: "
```

---

### Task 5: Bar 5 (the 27B untouched), the recorded row, records, tag

**Files:**
- Modify: `docs/BENCHMARKS.md`, `docs/superpowers/specs/2026-09-14-spec4-k2-horizon-decode-core-design.md` (append §10)

- [ ] **Step 1: Pre-register the row expectation** - append to the spec a `## 10. Close (pre-registration)` section *before* measuring, and commit it. It copies §5's derived roofline (156 t/s weights-only; ~129 t/s with depth-4096 KV) and the **estimated ~95 t/s**. It adds one derived line from plan 8a §3's launch probe: `<slope µs> × 2067 = <ms> per token of launch cost`. The row is recorded, not gated.
  `git commit -m "docs(spec4): pre-register the K2 decode row before measuring" -m "Claude-Session: "`

- [ ] **Step 2: The full suite, detached** - 27B and K2 together. The 27B's `golden_gate_test` and `prefill_gate_test` are in it; allow over an hour.
  `ssh -o BatchMode=yes user@box 'cd ~/b70-inference-server/build && ZE_AFFINITY_MASK=1 setsid nohup ctest --output-on-failure -V > $HOME/spec4-suite.log 2>&1 < /dev/null &'`
  Poll `grep -E "tests passed|tests failed|captured:|TOTAL:" $HOME/spec4-suite.log`. Expected:
  - `100% tests passed`;
  - `captured: 774 kernels, 19 modules` from the 27B's `replay_determinism_test`;
  - `captured: 2067 kernels, 28 modules` from K2's;
  - three `TOTAL:` lines: the 27B decode gate `93/93`, the 27B prefill gate `93/93`, K2 `… PASS`.

  Then prove the protected directories are unchanged since spec 4 began: `git diff --stat f9cde87..HEAD -- src/runtime/*.h src/runtime/*.cc src/loader/loader.cc src/loader/loader.h src/model/qwen35.h src/model/qwen35.cc src/kernels/*.cl tests/golden/golden_gate_test.cc`. Expected: empty.

- [ ] **Step 3: The recorded row** - prove idleness, then run the harness on device 0 (the series device):
  `ssh -o BatchMode=yes user@box 'docker ps -q | wc -l; for f in /proc/*/fdinfo/*; do grep -l drm-driver "$f" >/dev/null 2>&1 && echo "$f"; done | head'`. Expected: `0` and no fd lines. Otherwise tell the operator and wait.
  `MODEL=/home/user/.cache/huggingface/hub/models--urakozz--IFM-K2-Horizon-MoVA-36B-A4B-W4A16-AutoRound-GPTQ/snapshots/c0fd9971ad6d3bb4caedf726e798a02b8df9bcb5 tools/bench_decode.sh --depth 4096 --tg 256 2>&1 | tee /private/tmp/claude-501/k2-bench.log`
  Each run ingests 4096 ids one replay at a time, so allow several minutes per run. Expected: three `| b70-decode k2 <sha> | 4096 | 256 | … |` rows, and a median line carrying the harness's grade word.

- [ ] **Step 4: Records**
  - **`docs/BENCHMARKS.md`:** a K2-Horizon section with the three rows, the median, the grade word as printed, the stderr decomposition (fence share, launches per token, MBU), the checkpoint and sha. Put it beside spec §5's derived expectation and plan 8a §3's launch-cost line. Label every number measured / derived / estimated.
  - **Spec §10 (append under the pre-registration):**
    - bars 1-5 each with its evidence line: gate TOTAL, determinism line, kernel test list, load test line, suite tally with 774/19 and both 27B gates;
    - the row against the pre-registered estimate;
    - any deviation from this plan and why;
    - the stopping-rule verdict.

- [ ] **Step 5: Commit, then tag only if all five bars are met**

```bash
git add docs/BENCHMARKS.md docs/superpowers/specs/2026-09-14-spec4-k2-horizon-decode-core-design.md
git commit -m "docs(spec4): close - K2-Horizon decode core, bars and the recorded row

Claude-Session: "
git tag spec4-done   # ONLY if bars 1-5 are all met; otherwise no tag, and §10 is the memo to the operator
```

Never push the branch or the tag.

---

## Self-review

**Spec coverage:**
- §2 bar 1 → Task 3.
- Bar 2 → Task 2.
- Bar 3 → plan 8c, re-run in Task 5's suite.
- Bar 4 → plan 8b's load test, in the suite.
- Bar 5 → Task 5 Step 2, with the protected-path diff.
- The recorded row with t/s, ms/token, launches per token and fence share → Tasks 4 and 5.
- §3.3's decode list and option A routing → Task 2's walk.
- §3.4 memory → Task 1: 3.221 GB KV + 5.5 MB scratch + plan 8b's 21.810 GB weights = **~25.04 GB, derived**, against the spec's ~25.1 estimate.
- §4's per-layer routing diagnostic → Task 3 part 4.
- §5 stage 5's `model_type` dispatch → Task 4.
- The stopping rule → Task 5 Step 5.

**Numbers fixed by this plan:** 2067 launches and 28 modules. Both are asserted at capture and in the determinism test, and they replace spec §3.3's "~2,115": the softplus gate lives in `k2_attn_reduce`. The spec row changes in the same commit as these plans.

**Placeholders:** the `<…>` in Task 5 are measurements.

**Types:** every name used here is defined in plans 8b (`K2LoadedModel`, `K2ExpertWeights`, `load_k2`, `k2_rope_table`), 8c (`kernels::k2::*`) or earlier tasks of this plan. Argument orders follow plan 8c's Interfaces.
