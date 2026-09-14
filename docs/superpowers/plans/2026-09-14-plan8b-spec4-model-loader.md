# Spec 4 / Stage 1 - K2-Horizon model table and loader Implementation Plan

**Status (2026-09-14): written, NOT dispatched.** Implementation waits for the operator's ruling on the prefill GEMM direction; these plans may be adjusted after it.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `model::K2Horizon` describes the checkpoint as data, and `loader::load_k2()` puts every tensor on one B70 in the layouts plan 8c's kernels read. That includes one flat buffer per (layer, expert group), with expert `e` at `e × stride`.

**Architecture:** A second model beside qwen3_5 (spec 4 §3.1, operator ruling "no generalisation"). New files only, plus append-only CMake lines. The K2 loader reuses the loader's **public** pieces unchanged: `resolve_snapshot`, `QuantConfig::parse`, `SafetensorsSet`, `assert_quant_invariants`, `LinearSrc::classify`, `check_align` and the `common/repack.h` helpers. `src/loader/loader.cc` is not edited; its anonymous-namespace helpers are re-spelled in `k2_loader.cc` at K2's much smaller scope.

**Tech Stack:** C++17, Level Zero via `src/l0`, `tests/check.h` (`CHECK`, `CHECK_EQ`), ctest via `tools/box.sh`.

**Spec:** `docs/superpowers/specs/2026-09-14-spec4-k2-horizon-decode-core-design.md` (§1 bytes, §3.1 structure, §3.2 semantics, §3.4 memory, §4 load test)

## Global Constraints

- Branch `spec1.7-codex-exp`; never push. Every commit message ends with `Claude-Session: `.
- The Mac never compiles: `tools/box.sh sync`, `tools/box.sh build`, `tools/box.sh test <regex>`, `tools/box.sh run '<cmd>'` (JOBS 44). Anything longer than about a minute on the box launches detached (`setsid nohup … > $HOME/<name>.log 2>&1 < /dev/null &`) and is polled.
- **The 27B is untouched:** no edit to `src/loader/{loader,quant,safetensors,snapshot}.{h,cc}`, `src/loader/small_layout.h`, `src/model/qwen35.{h,cc}`, `src/runtime/*`, any existing test. CMake edits are appended lines only.
- Build flags are `-Wall -Wextra -Werror` with gcc 13. **Never bind a `const TensorInfo&` to the return of a function that takes a temporary `std::string`** (gcc 13's `-Wdangling-reference` fires; copy the `TensorInfo` instead - the 27B loader's own note).
- Checkpoint snapshot for every test: `/home/user/.cache/huggingface/hub/models--urakozz--IFM-K2-Horizon-MoVA-36B-A4B-W4A16-AutoRound-GPTQ/snapshots/c0fd9971ad6d3bb4caedf726e798a02b8df9bcb5`.
- Plan 8a's facts doc (`docs/k2-stage0-facts-2026-09-14.md`) must exist with §1-§2 committed: if §1 says `QuantConfig::parse` threw, stop and report - this plan assumes it accepts K2's config.
- GPU work on device 1: `ZE_AFFINITY_MASK=1`.

---

## File structure

| path | responsibility |
|---|---|
| `src/model/k2_horizon.h` / `.cc` | constants, the per-layer linear / expert-group / small-tensor tables, name helpers, strides |
| `src/model/CMakeLists.txt` (append) | `b70_k2_model` |
| `tests/model/k2_horizon_test.cc` | the table against the checkpoint header facts, host only |
| `src/loader/k2_loader.h` / `.cc` | `K2LoadedModel`, `k2_rope_table`, `load_k2` |
| `src/loader/CMakeLists.txt` (append) | `b70_k2_loader` |
| `tests/loader/k2_rope_test.cc` | the RoPE table's construction, host only |
| `tests/loader/k2_load_checkpoint_test.cc` | the real checkpoint: exact byte buckets, zero unconsumed, readbacks |
| `tests/CMakeLists.txt` (append) | the three tests |

---

### Task 1: `model::K2Horizon` - the model as data

**Files:**
- Create: `src/model/k2_horizon.h`, `src/model/k2_horizon.cc`, `tests/model/k2_horizon_test.cc`
- Modify: `src/model/CMakeLists.txt` (append), `tests/CMakeLists.txt` (append)

**Interfaces:**
- Consumes: `model::GemvShape{K,N,S,layout}`, `model::WeightKind{Int4,Bf16}`, `model::Fuse{Single,Concat,Interleave16}` from `src/model/qwen35.h`.
- Produces (used by Tasks 2-3 and by plans 8c, 8e):
  - `enum class model::K2LayerKind { Dense, Sparse }`
  - `enum class model::K2LinearId { AttnDense, AttnSparse, OProj, DenseGateUp, DenseDown, MoeRouter, SharedGateUp, SharedDown, LmHead, kCount }`
  - `enum class model::K2ExpertId { Value, MoeGateUp, MoeDown, kCount }`
  - `struct model::K2Linear { K2LinearId id; GemvShape shape; WeightKind kind; Fuse fuse; std::vector<std::string> parts; uint32_t pad_n; }`
  - `struct model::K2ExpertGroup { K2ExpertId id; GemvShape shape; Fuse fuse; std::vector<std::string> parts; uint32_t count; }`
  - `enum class model::K2SmallBlock { Norms, Route }`; `struct model::K2SmallTensor { std::string name; uint32_t elems; const char* dtype; K2SmallBlock block; uint32_t offset; }`
  - `struct model::K2LayerDesc { uint32_t index; K2LayerKind kind; std::vector<K2Linear> linears; std::vector<K2ExpertGroup> experts; std::vector<K2SmallTensor> small; }`
  - `struct model::K2Horizon` constants (below), `static bool is_dense(uint32_t)`, `static std::vector<K2LayerDesc> layers()`, `static const K2Linear& lm_head()`, `static std::string layer_prefix(uint32_t)` → `"model.layers.<n>."`, `static std::string expert_part(const std::string& tmpl, uint32_t e)`, `static uint64_t expert_weight_stride(const GemvShape&)`, `static uint64_t expert_scale_stride(const GemvShape&)`.

- [ ] **Step 1: Write the failing test** - `tests/model/k2_horizon_test.cc`:

```cpp
// model::K2Horizon against the checkpoint's own headers (snapshot c0fd997, read
// 2026-09-14 - every literal below is a header shape, not a derivation from the
// table under test).
#include <cstdio>
#include <set>
#include <string>
#include "check.h"
#include "model/k2_horizon.h"

int main() {
  using model::Fuse;
  using model::K2ExpertId;
  using model::K2Horizon;
  using model::K2LayerKind;
  using model::K2LinearId;
  using model::WeightKind;

  const auto layers = K2Horizon::layers();
  CHECK_EQ(layers.size(), size_t(48));
  for (uint32_t l = 0; l < 48; ++l) {
    CHECK_EQ(layers[l].index, l);
    CHECK((layers[l].kind == K2LayerKind::Dense) == (l < 3));
    CHECK_EQ(K2Horizon::is_dense(l), l < 3);
  }

  // Dense layer: 4 linears in execution order, no experts, the norms only.
  const auto& d = layers[0];
  CHECK_EQ(d.linears.size(), size_t(4));
  CHECK(d.linears[0].id == K2LinearId::AttnDense && d.linears[1].id == K2LinearId::OProj &&
        d.linears[2].id == K2LinearId::DenseGateUp && d.linears[3].id == K2LinearId::DenseDown);
  CHECK(d.experts.empty());
  CHECK_EQ(d.small.size(), size_t(2));

  // Sparse layer: 5 linears, 3 expert groups, norms + the two router biases.
  const auto& s = layers[47];
  CHECK_EQ(s.linears.size(), size_t(5));
  CHECK(s.linears[0].id == K2LinearId::AttnSparse && s.linears[1].id == K2LinearId::OProj &&
        s.linears[2].id == K2LinearId::MoeRouter && s.linears[3].id == K2LinearId::SharedGateUp &&
        s.linears[4].id == K2LinearId::SharedDown);
  CHECK_EQ(s.experts.size(), size_t(3));
  CHECK_EQ(s.small.size(), size_t(4));

  struct Want { K2LinearId id; uint32_t K, N; WeightKind kind; Fuse fuse; size_t parts; uint32_t pad; };
  const Want wants[] = {
      {K2LinearId::AttnDense, 2560, 10240, WeightKind::Int4, Fuse::Concat, 4, 0},
      {K2LinearId::AttnSparse, 2560, 9280, WeightKind::Int4, Fuse::Concat, 4, 0},
      {K2LinearId::OProj, 4096, 2560, WeightKind::Int4, Fuse::Single, 1, 0},
      {K2LinearId::DenseGateUp, 2560, 12288, WeightKind::Int4, Fuse::Interleave16, 2, 0},
      {K2LinearId::DenseDown, 6144, 2560, WeightKind::Int4, Fuse::Single, 1, 0},
      {K2LinearId::MoeRouter, 2560, 128, WeightKind::Bf16, Fuse::Single, 1, 100},
      {K2LinearId::SharedGateUp, 2560, 1536, WeightKind::Int4, Fuse::Interleave16, 2, 0},
      {K2LinearId::SharedDown, 768, 2560, WeightKind::Int4, Fuse::Single, 1, 0},
  };
  for (const Want& w : wants) {
    const auto& rows = (w.id == K2LinearId::AttnDense || w.id == K2LinearId::DenseGateUp ||
                        w.id == K2LinearId::DenseDown) ? d.linears : s.linears;
    bool found = false;
    for (const auto& r : rows) {
      if (r.id != w.id) continue;
      found = true;
      CHECK_EQ(r.shape.K, w.K);
      CHECK_EQ(r.shape.N, w.N);
      CHECK_EQ(r.shape.S, uint32_t(1));
      CHECK_EQ(r.shape.layout, uint32_t(0));
      CHECK(r.kind == w.kind && r.fuse == w.fuse);
      CHECK_EQ(r.parts.size(), w.parts);
      CHECK_EQ(r.pad_n, w.pad);
      if (r.kind == WeightKind::Int4) {   // every int4 shape is kernel-legal (spec §3.3)
        CHECK_EQ(r.shape.K % 64, uint32_t(0));
        CHECK_EQ(r.shape.N % 64, uint32_t(0));
      }
    }
    CHECK(found);
  }
  // The fused attention column map: q 4096 | k 1024 | gate 4096 | v or v_router.
  CHECK_EQ(K2Horizon::kQOff, uint32_t(0));
  CHECK_EQ(K2Horizon::kKOff, uint32_t(4096));
  CHECK_EQ(K2Horizon::kGateOff, uint32_t(5120));
  CHECK_EQ(K2Horizon::kVOff, uint32_t(9216));
  CHECK(s.linears[0].parts[3] == "self_attn.v_router" && d.linears[0].parts[3] == "self_attn.v_proj");
  CHECK(d.linears[0].parts[2] == "self_attn.gate_proj");

  const auto& lm = K2Horizon::lm_head();
  CHECK(lm.id == K2LinearId::LmHead && lm.kind == WeightKind::Bf16);
  CHECK_EQ(lm.shape.N, uint32_t(250624));
  CHECK_EQ(lm.shape.N % 64, uint32_t(0));

  // Expert groups and their strides (spec §3.4, derived from the checkpoint).
  for (const auto& g : s.experts) {
    const uint64_t ws = K2Horizon::expert_weight_stride(g.shape);
    const uint64_t ss = K2Horizon::expert_scale_stride(g.shape);
    if (g.id == K2ExpertId::Value) {
      CHECK(g.count == 64 && g.shape.K == 2560 && g.shape.N == 1024 && g.fuse == Fuse::Single);
      CHECK_EQ(ws, uint64_t(1310720));
      CHECK_EQ(ss, uint64_t(81920));
    } else if (g.id == K2ExpertId::MoeGateUp) {
      CHECK(g.count == 100 && g.shape.K == 2560 && g.shape.N == 1536 && g.fuse == Fuse::Interleave16);
      CHECK_EQ(ws, uint64_t(1966080));
      CHECK_EQ(ss, uint64_t(122880));
    } else {
      CHECK(g.id == K2ExpertId::MoeDown);
      CHECK(g.count == 100 && g.shape.K == 768 && g.shape.N == 2560 && g.fuse == Fuse::Single);
      CHECK_EQ(ws, uint64_t(983040));
      CHECK_EQ(ss, uint64_t(61440));
    }
  }
  CHECK(K2Horizon::expert_part("mlp.experts.{e}.up_proj", 99) == "mlp.experts.99.up_proj");
  CHECK(K2Horizon::expert_part("self_attn.v_experts.{e}", 0) == "self_attn.v_experts.0");
  CHECK(K2Horizon::layer_prefix(7) == "model.layers.7.");

  // Small blocks: every byte has exactly one owner.
  for (const auto& l : {d, s}) {
    std::set<std::pair<int, uint32_t>> seen;
    uint32_t norms = 0, route = 0;
    for (const auto& t : l.small) {
      CHECK(seen.insert({int(t.block), t.offset}).second);
      (t.block == model::K2SmallBlock::Norms ? norms : route) += t.elems * 4;
    }
    CHECK_EQ(norms, K2Horizon::kNormsBytes);
    CHECK_EQ(route, l.kind == K2LayerKind::Sparse ? K2Horizon::kRouteBytes : uint32_t(0));
  }
  CHECK_EQ(K2Horizon::kNormsOffPost, uint32_t(10240));
  CHECK_EQ(K2Horizon::kRouteOffMovaBias, uint32_t(400));
  CHECK_EQ(K2Horizon::kEos[1], uint32_t(250019));
  std::puts("k2_horizon_test OK");
  return 0;
}
```

- [ ] **Step 2: Register the test and library**

Append to `src/model/CMakeLists.txt`:

```cmake
# Spec 4: K2-Horizon as data, BESIDE qwen3_5 (operator ruling: no generalisation).
# It shares GemvShape / WeightKind / Fuse from qwen35.h and nothing else.
add_library(b70_k2_model STATIC k2_horizon.cc)
target_include_directories(b70_k2_model PUBLIC ${CMAKE_SOURCE_DIR}/src)
```

Append to `tests/CMakeLists.txt`:

```cmake
# --- spec 4: K2-Horizon ---------------------------------------------------------
add_executable(k2_horizon_test model/k2_horizon_test.cc)
target_include_directories(k2_horizon_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(k2_horizon_test PRIVATE b70_k2_model)
add_test(NAME k2_horizon_test COMMAND k2_horizon_test)
```

- [ ] **Step 3: Confirm it fails to build** - `tools/box.sh sync && tools/box.sh build 2>&1 | grep -m3 -E "error|k2_horizon"`. Expected: `fatal error: model/k2_horizon.h: No such file or directory`.

- [ ] **Step 4: Write the header** - `src/model/k2_horizon.h`:

```cpp
#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "model/qwen35.h"   // GemvShape, WeightKind, Fuse - the only shared vocabulary

// K2-Horizon (`K2HorizonForCausalLM`, modeling_k2_horizon.py @ snapshot c0fd997) as data:
// which linears exist, how each fused device weight is assembled, the routed expert groups
// and every small tensor. Spec 4 §3.1 - a second model BESIDE qwen3_5. It knows nothing of
// Level Zero, kernels or I/O; `loader::load_k2` executes it and plan 8e's capture walks it.
namespace model {

enum class K2LayerKind { Dense, Sparse };   // layers 0-2 / 3-47

// Linears that are not routed experts. Every int4 row is GPTQ layout 0, S = 1.
enum class K2LinearId {
  AttnDense,     // q ‖ k ‖ gate ‖ v           2560 -> 10240
  AttnSparse,    // q ‖ k ‖ gate ‖ v_router    2560 -> 9280 (MoVA: v is routed)
  OProj,         // 4096 -> 2560
  DenseGateUp,   // gate ‖ up, interleave16    2560 -> 12288
  DenseDown,     // 6144 -> 2560
  MoeRouter,     // bf16, 2560 -> 100, zero-padded to 128 (gemv_bf16 tiles N in 16s)
  SharedGateUp,  // gate ‖ up, interleave16    2560 -> 1536
  SharedDown,    // 768 -> 2560
  LmHead,        // bf16, 2560 -> 250624, top level
  kCount
};

// Routed experts. One flat device buffer per (layer, group); expert e at e × stride.
enum class K2ExpertId { Value, MoeGateUp, MoeDown, kCount };

struct K2Linear {
  K2LinearId id;
  GemvShape shape;
  WeightKind kind;
  Fuse fuse;
  std::vector<std::string> parts;   // layer-relative checkpoint prefixes, in column order
  uint32_t pad_n = 0;               // the unpadded N when the kernel's N is larger; 0 = none
};

struct K2ExpertGroup {
  K2ExpertId id;
  GemvShape shape;                  // ONE expert's fused shape
  Fuse fuse;
  std::vector<std::string> parts;   // templates: "{e}" is replaced by the expert id
  uint32_t count;
};

// Every K2 small tensor is widened to fp32 verbatim - no `1 + w` (K2's RMSNorm multiplies by
// the plain weight, spec §3.1's first trap) - so its source dtype is the whole bake:
// "BF16" through common::bf16_to_f32, "F16" (the MoVA router bias) through common::f16_to_f32.
enum class K2SmallBlock { Norms, Route };
struct K2SmallTensor {
  std::string name;   // layer-relative
  uint32_t elems;
  const char* dtype;  // "BF16" | "F16", as safetensors spells it
  K2SmallBlock block;
  uint32_t offset;    // bytes within the block
};

struct K2LayerDesc {
  uint32_t index;
  K2LayerKind kind;
  std::vector<K2Linear> linears;        // per-token execution order
  std::vector<K2ExpertGroup> experts;   // empty for dense layers
  std::vector<K2SmallTensor> small;
};

struct K2Horizon {
  static constexpr uint32_t kLayers = 48, kDenseLayers = 3;
  static constexpr uint32_t kHidden = 2560, kNormGroups = 2;          // RMSNorm over 2 × 1280
  static constexpr uint32_t kQHeads = 32, kKvHeads = 8, kHeadDim = 128;
  static constexpr uint32_t kDenseInter = 6144, kMoeInter = 768;
  static constexpr uint32_t kExperts = 100, kTopK = 8;
  static constexpr uint32_t kValueExperts = 64, kValueTopK = 4;
  static constexpr uint32_t kVocab = 250624;   // fully used (250,000 + 626 added): no mask
  static constexpr uint32_t kBos = 0;
  static constexpr std::array<uint32_t, 2> kEos = {1, 250019};
  static constexpr double kRopeTheta = 1e7;
  static constexpr float kRouterScale = 2.5f;

  // The fused attention GEMV's column map - the SAME in both rows, so k2_attn.cl has one map:
  // q [0, 4096) · k [4096, 5120) · gate [5120, 9216) · v (dense) or v_router (sparse) from 9216.
  static constexpr uint32_t kQOff = 0, kKOff = 4096, kGateOff = 5120, kVOff = 9216;

  // Small blocks, bytes. Norms (every layer): input w fp32 [2560] ‖ post w fp32 [2560].
  // Route (sparse layers): MoE gate bias fp32 [100] ‖ MoVA v_router bias fp32 [64].
  static constexpr uint32_t kNormsOffInput = 0, kNormsOffPost = 10240, kNormsBytes = 20480;
  static constexpr uint32_t kRouteOffMoeBias = 0, kRouteOffMovaBias = 400, kRouteBytes = 656;

  static bool is_dense(uint32_t layer) { return layer < kDenseLayers; }
  static std::vector<K2LayerDesc> layers();          // all 48, fully populated
  static const K2Linear& lm_head();
  static std::string layer_prefix(uint32_t layer);   // "model.layers.<n>."
  static std::string expert_part(const std::string& tmpl, uint32_t e);   // throws if no "{e}"
  static uint64_t expert_weight_stride(const GemvShape& s) { return uint64_t(s.K / 8) * s.N * 4; }
  static uint64_t expert_scale_stride(const GemvShape& s) { return uint64_t(s.K / 64) * s.N * 2; }
};

}  // namespace model
```

- [ ] **Step 5: Write the tables** - `src/model/k2_horizon.cc`:

```cpp
#include "model/k2_horizon.h"

#include <stdexcept>

namespace model {
namespace {

using K2 = K2Horizon;

// Linears, by K2LinearId ordinal. Shapes are the checkpoint headers (qweight [K/8][N]).
const std::vector<K2Linear>& linear_table() {
  static const std::vector<K2Linear> t = {
      {K2LinearId::AttnDense, {2560, 10240, 1, 0}, WeightKind::Int4, Fuse::Concat,
       {"self_attn.q_proj", "self_attn.k_proj", "self_attn.gate_proj", "self_attn.v_proj"}, 0},
      // MoVA: the 64 int4 router logits ride in v's place (4096 + 1024 + 4096 + 64).
      {K2LinearId::AttnSparse, {2560, 9280, 1, 0}, WeightKind::Int4, Fuse::Concat,
       {"self_attn.q_proj", "self_attn.k_proj", "self_attn.gate_proj", "self_attn.v_router"}, 0},
      {K2LinearId::OProj, {4096, 2560, 1, 0}, WeightKind::Int4, Fuse::Single,
       {"self_attn.o_proj"}, 0},
      {K2LinearId::DenseGateUp, {2560, 12288, 1, 0}, WeightKind::Int4, Fuse::Interleave16,
       {"mlp.gate_proj", "mlp.up_proj"}, 0},
      {K2LinearId::DenseDown, {6144, 2560, 1, 0}, WeightKind::Int4, Fuse::Single,
       {"mlp.down_proj"}, 0},
      // bf16 by the quantiser's 45 `dynamic` exclusions; 100 real rows, 28 zero rows.
      {K2LinearId::MoeRouter, {2560, 128, 1, 0}, WeightKind::Bf16, Fuse::Single, {"mlp.gate"}, 100},
      {K2LinearId::SharedGateUp, {2560, 1536, 1, 0}, WeightKind::Int4, Fuse::Interleave16,
       {"mlp.shared_experts.gate_proj", "mlp.shared_experts.up_proj"}, 0},
      {K2LinearId::SharedDown, {768, 2560, 1, 0}, WeightKind::Int4, Fuse::Single,
       {"mlp.shared_experts.down_proj"}, 0},
      {K2LinearId::LmHead, {2560, 250624, 1, 0}, WeightKind::Bf16, Fuse::Single, {"lm_head"}, 0},
  };
  return t;
}

const K2Linear& row(K2LinearId id) {
  const size_t i = static_cast<size_t>(id);
  const auto& t = linear_table();
  if (i >= t.size() || t[i].id != id)
    throw std::out_of_range("K2Horizon: no linear row for ordinal " + std::to_string(i));
  return t[i];
}

const std::vector<K2ExpertGroup>& expert_groups() {
  static const std::vector<K2ExpertGroup> v = {
      {K2ExpertId::Value, {2560, 1024, 1, 0}, Fuse::Single, {"self_attn.v_experts.{e}"},
       K2::kValueExperts},
      {K2ExpertId::MoeGateUp, {2560, 1536, 1, 0}, Fuse::Interleave16,
       {"mlp.experts.{e}.gate_proj", "mlp.experts.{e}.up_proj"}, K2::kExperts},
      {K2ExpertId::MoeDown, {768, 2560, 1, 0}, Fuse::Single, {"mlp.experts.{e}.down_proj"},
       K2::kExperts},
  };
  return v;
}

const std::vector<K2SmallTensor>& dense_small() {
  static const std::vector<K2SmallTensor> v = {
      {"input_layernorm.weight", K2::kHidden, "BF16", K2SmallBlock::Norms, K2::kNormsOffInput},
      {"post_attention_layernorm.weight", K2::kHidden, "BF16", K2SmallBlock::Norms,
       K2::kNormsOffPost},
  };
  return v;
}

const std::vector<K2SmallTensor>& sparse_small() {
  static const std::vector<K2SmallTensor> v = [] {
    std::vector<K2SmallTensor> s = dense_small();
    // The router biases are used ONLY for selection (spec §3.2); both land fp32.
    s.push_back({"mlp.gate.bias", K2::kExperts, "BF16", K2SmallBlock::Route, K2::kRouteOffMoeBias});
    s.push_back({"self_attn.v_router.bias", K2::kValueExperts, "F16", K2SmallBlock::Route,
                 K2::kRouteOffMovaBias});
    return s;
  }();
  return v;
}

}  // namespace

std::vector<K2LayerDesc> K2Horizon::layers() {
  std::vector<K2LayerDesc> out;
  out.reserve(kLayers);
  for (uint32_t i = 0; i < kLayers; ++i) {
    if (is_dense(i)) {
      out.push_back({i, K2LayerKind::Dense,
                     {row(K2LinearId::AttnDense), row(K2LinearId::OProj),
                      row(K2LinearId::DenseGateUp), row(K2LinearId::DenseDown)},
                     {},
                     dense_small()});
    } else {
      out.push_back({i, K2LayerKind::Sparse,
                     {row(K2LinearId::AttnSparse), row(K2LinearId::OProj),
                      row(K2LinearId::MoeRouter), row(K2LinearId::SharedGateUp),
                      row(K2LinearId::SharedDown)},
                     expert_groups(),
                     sparse_small()});
    }
  }
  return out;
}

const K2Linear& K2Horizon::lm_head() { return row(K2LinearId::LmHead); }

std::string K2Horizon::layer_prefix(uint32_t layer) {
  return "model.layers." + std::to_string(layer) + ".";
}

std::string K2Horizon::expert_part(const std::string& tmpl, uint32_t e) {
  const size_t at = tmpl.find("{e}");
  if (at == std::string::npos)
    throw std::invalid_argument("K2Horizon::expert_part: template '" + tmpl + "' has no {e}");
  return tmpl.substr(0, at) + std::to_string(e) + tmpl.substr(at + 3);
}

}  // namespace model
```

- [ ] **Step 6: Build and run** - `tools/box.sh sync && tools/box.sh build 2>&1 | tail -2 && tools/box.sh test '^k2_horizon_test$'`. Expected: `k2_horizon_test OK`, ctest `100% tests passed`.

- [ ] **Step 7: Commit**

```bash
git add src/model/k2_horizon.h src/model/k2_horizon.cc src/model/CMakeLists.txt tests/model/k2_horizon_test.cc tests/CMakeLists.txt
git commit -m "feat(model): K2-Horizon model table beside qwen3_5 (spec 4 stage 1)

Claude-Session: "
```

---

### Task 2: `loader::k2_rope_table` - K2's RoPE as the reference builds it

**Files:**
- Create: `src/loader/k2_loader.h` (the full header; `load_k2` is implemented in Task 3), `src/loader/k2_rope.cc`, `tests/loader/k2_rope_test.cc`
- Modify: `src/loader/CMakeLists.txt` (append), `tests/CMakeLists.txt` (append)

**Interfaces:**
- Consumes: `model::K2Horizon::{kHeadDim, kRopeTheta}`, `common::f32_to_bf16`, `common::bf16_to_f32`.
- Produces: `std::vector<float> loader::k2_rope_table(uint32_t max_len)` - fp32 `[max_len][2][64]`, cos at `[p][0][i]`, sin at `[p][1][i]`, **every value a bf16 word widened** (the reference casts cos/sin to the hidden dtype before rotating). Plan 8c's `k2_attn_prep` reads exactly this layout.

**The construction, and why it is not the 27B's.** The 27B loader computes each angle in double. K2's reference does all of it in fp32 (`compute_default_rope_parameters` and `K2HorizonRotaryEmbedding.forward`), and at 16k positions the difference is not small. `inv_freq[1]` carries ≈ 0.5 ulp of fp32 error, about 3e-8 relative, so the angle at p = 16383 is off by ≈ 4e-4 rad against a double-computed one. So this table mirrors the reference's fp32 steps:

- `inv[i] = 1 / powf(1e7, (2i)/128)`;
- `ang = float(p) · inv[i]`;
- `cos`/`sin` of that fp32 angle, rounded to bf16.

torch's fp32 `cos` is not guaranteed to be correctly rounded, and neither is the host's `double` cos narrowed to float. The bf16 rounding absorbs sub-ulp disagreement at almost every entry. Plan 8d records the oracle's own cos/sin rows so the gate can report any residue.

- [ ] **Step 1: Write the failing test** - `tests/loader/k2_rope_test.cc`:

```cpp
// loader::k2_rope_table: layout, the fp32 construction, and bf16-valued entries.
#include <cmath>
#include <cstdio>
#include "check.h"
#include "common/bf16.h"
#include "loader/k2_loader.h"

int main() {
  const uint32_t L = 16384, H = 64;
  const std::vector<float> t = loader::k2_rope_table(L);
  CHECK_EQ(t.size(), size_t(L) * 2 * H);
  auto cosv = [&](uint32_t p, uint32_t i) { return t[(size_t(p) * 2 + 0) * H + i]; };
  auto sinv = [&](uint32_t p, uint32_t i) { return t[(size_t(p) * 2 + 1) * H + i]; };
  for (uint32_t i = 0; i < H; ++i) {           // position 0: the identity rotation
    CHECK(cosv(0, i) == 1.0f);
    CHECK(sinv(0, i) == 0.0f);
  }
  // Every entry is exactly a bf16 word: widening its own rounding changes nothing.
  for (size_t j = 0; j < t.size(); ++j)
    CHECK(common::bf16_to_f32(common::f32_to_bf16(t[j])) == t[j]);
  // The construction, re-spelled independently at three probes.
  for (uint32_t p : {1u, 4097u, 16383u})
    for (uint32_t i : {0u, 1u, 31u, 63u}) {
      const float inv = 1.0f / std::pow(1e7f, float(2 * i) / 128.0f);
      const float ang = float(p) * inv;
      CHECK(cosv(p, i) == common::bf16_to_f32(common::f32_to_bf16(float(std::cos(double(ang))))));
      CHECK(sinv(p, i) == common::bf16_to_f32(common::f32_to_bf16(float(std::sin(double(ang))))));
    }
  // i = 0 has inv = 1 exactly, so the angle is the position itself.
  CHECK(cosv(1, 0) == common::bf16_to_f32(common::f32_to_bf16(float(std::cos(1.0)))));
  std::puts("k2_rope_test OK");
  return 0;
}
```

- [ ] **Step 2: Register** - append to `src/loader/CMakeLists.txt`:

```cmake
# Spec 4: the K2-Horizon loader. It reuses this directory's PUBLIC pieces (snapshot,
# safetensors, quant) unchanged and does not touch loader.cc.
add_library(b70_k2_loader STATIC k2_rope.cc k2_loader.cc)
target_include_directories(b70_k2_loader PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(b70_k2_loader PUBLIC b70_loader b70_k2_model b70_l0)
```

Append to `tests/CMakeLists.txt`:

```cmake
add_executable(k2_rope_test loader/k2_rope_test.cc)
target_include_directories(k2_rope_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(k2_rope_test PRIVATE b70_k2_loader)
add_test(NAME k2_rope_test COMMAND k2_rope_test)
```

Create an empty-bodied `src/loader/k2_loader.cc` now, so the library links before Task 3:

```cpp
#include "loader/k2_loader.h"
// load_k2 lands in plan 8b Task 3.
```

- [ ] **Step 3: Write the header** - `src/loader/k2_loader.h`:

```cpp
#pragma once
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "l0/context.h"
#include "l0/memory.h"
#include "loader/loader.h"      // DeviceWeight, kTopLevel - reused, not re-declared
#include "model/k2_horizon.h"

namespace loader {

// One routed expert group of one layer: every expert's GPTQ layout-0 words and scales,
// concatenated in id order. Expert e's words start at e × w_stride and its scales at
// e × s_stride (bytes). Flat is required, not preferred: a frozen launch argument cannot hand
// a kernel a different buffer per token, only an offset inside one (spec §3.4).
struct K2ExpertWeights {
  l0::Mem weights;          // u32 [count][K/8][N]
  l0::Mem scales;           // f16 [count][K/64][N]
  model::GemvShape shape;   // ONE expert
  uint32_t count;
  size_t w_stride, s_stride;
};

struct K2LoadReport {
  size_t int4_bytes = 0, scale_bytes = 0;                 // non-expert int4 linears
  size_t expert_int4_bytes = 0, expert_scale_bytes = 0;   // every expert of every layer
  size_t bf16_linear_bytes = 0, pad_bytes = 0;            // MoE routers: real rows / zero rows
  size_t lm_head_bytes = 0, embed_bytes = 0;
  size_t small_bytes = 0;      // norm blocks + route blocks + final norm, fp32
  size_t rope_bytes = 0;
  // Spec §4: what ONE decode token reads - every non-expert weight, lm_head, the small blocks,
  // and only the ACTIVE experts (4 of 64 value, 8 of 100 gate‖up and down). Never computed
  // against expert bytes a token does not read.
  size_t read_per_token = 0;
  size_t unconsumed = 0;       // checkpoint tensors nothing loaded, qzeros/g_idx excepted
  double seconds = 0;
  size_t total() const {
    return int4_bytes + scale_bytes + expert_int4_bytes + expert_scale_bytes + bf16_linear_bytes +
           pad_bytes + lm_head_bytes + embed_bytes + small_bytes + rope_bytes;
  }
};

struct K2LoadedModel {
  K2LoadedModel(l0::Mem embed_, l0::Mem final_norm_, l0::Mem rope_, uint32_t max_len_)
      : embed(std::move(embed_)), final_norm(std::move(final_norm_)), rope(std::move(rope_)),
        max_len(max_len_) {}
  std::map<std::pair<uint32_t, model::K2LinearId>, DeviceWeight> linears;  // lm_head: kTopLevel
  std::map<std::pair<uint32_t, model::K2ExpertId>, K2ExpertWeights> experts;
  std::vector<l0::Mem> norms;               // [48] fp32: input w [2560] ‖ post w [2560]
  std::map<uint32_t, l0::Mem> route_bias;   // sparse layers: fp32 moe bias [100] ‖ mova bias [64]
  l0::Mem embed;        // bf16 [250624][2560] row-major, gathered one row per token
  l0::Mem final_norm;   // fp32 plain w [2560]
  l0::Mem rope;         // fp32 [max_len][2][64], bf16-valued (k2_rope_table)
  K2LoadReport report;
  uint32_t max_len;
};

// K2's RoPE table: fp32 [max_len][2][64], cos at [p][0][i], sin at [p][1][i]. Built with the
// reference's fp32 steps (inv = 1/powf(1e7, 2i/128); ang = float(p)·inv) and rounded to bf16,
// because the reference casts cos/sin to the hidden dtype before it rotates.
std::vector<float> k2_rope_table(uint32_t max_len);

// Loads the K2-Horizon checkpoint (resolve_snapshot rules) onto ctx's device. Throws unless
// config.json's model_type is "k2_horizon"; asserts the quant invariants, every shape against
// model::K2Horizon, and zero unconsumed tensors (qzeros / g_idx are dropped by name).
K2LoadedModel load_k2(l0::Context& ctx, const std::string& snapshot_or_repo,
                      uint32_t max_len = 16384);

}  // namespace loader
```

- [ ] **Step 4: Confirm the test fails to link** - `tools/box.sh sync && tools/box.sh build 2>&1 | grep -m2 "undefined reference"`. Expected: `undefined reference to 'loader::k2_rope_table(unsigned int)'`.

- [ ] **Step 5: Implement** - `src/loader/k2_rope.cc`:

```cpp
#include <cmath>

#include "common/bf16.h"
#include "loader/k2_loader.h"

namespace loader {

// K2HorizonRotaryEmbedding, step for step, in fp32 (modeling_k2_horizon.py):
//   inv_freq = 1.0 / (base ** (arange(0, 128, 2).float() / 128))    fp32 pow, fp32 divide
//   freqs    = inv_freq @ position                                    one fp32 product
//   cos, sin = emb.cos(), emb.sin()  (× attention_scaling = 1.0)      fp32
//   .to(hidden dtype)                                                 bf16
// cat(freqs, freqs) makes dims i and i + 64 share angle i, so 64 angles per position suffice.
std::vector<float> k2_rope_table(uint32_t max_len) {
  const uint32_t half = model::K2Horizon::kHeadDim / 2;
  const float base = float(model::K2Horizon::kRopeTheta);   // 1e7 is exact in fp32
  std::vector<float> inv(half);
  for (uint32_t i = 0; i < half; ++i)
    inv[i] = 1.0f / std::pow(base, float(2 * i) / float(model::K2Horizon::kHeadDim));
  std::vector<float> t(size_t(max_len) * 2 * half);
  for (uint32_t p = 0; p < max_len; ++p)
    for (uint32_t i = 0; i < half; ++i) {
      const float ang = float(p) * inv[i];
      t[(size_t(p) * 2 + 0) * half + i] =
          common::bf16_to_f32(common::f32_to_bf16(float(std::cos(double(ang)))));
      t[(size_t(p) * 2 + 1) * half + i] =
          common::bf16_to_f32(common::f32_to_bf16(float(std::sin(double(ang)))));
    }
  return t;
}

}  // namespace loader
```

- [ ] **Step 6: Build and run** - `tools/box.sh sync && tools/box.sh build 2>&1 | tail -2 && tools/box.sh test '^k2_rope_test$'`. Expected: `k2_rope_test OK`.

- [ ] **Step 7: Commit**

```bash
git add src/loader/k2_loader.h src/loader/k2_loader.cc src/loader/k2_rope.cc src/loader/CMakeLists.txt tests/loader/k2_rope_test.cc tests/CMakeLists.txt
git commit -m "feat(loader): K2 RoPE table built with the reference's fp32 steps, bf16-rounded

Claude-Session: "
```

---

### Task 3: `loader::load_k2` - the checkpoint on the device, every byte accounted for

**Files:**
- Modify: `src/loader/k2_loader.cc` (replace the Task-2 stub)
- Create: `tests/loader/k2_load_checkpoint_test.cc`
- Modify: `tests/CMakeLists.txt` (append)

**Interfaces:**
- Consumes: Task 1's table, Task 2's header and `k2_rope_table`; `loader::resolve_snapshot`, `QuantConfig::parse`, `SafetensorsSet{tensors(), data(), bytes()}`, `assert_quant_invariants`, `LinearSrc::classify` (fields `kind, K, N, qweight, scales, weight, name`), `check_align`; `common::{Part, ColSource, cols_concat, cols_interleave16, repack_int4_layout0_cols, repack_bf16_tiled, bf16_to_f32, f16_to_f32}`; `l0::CmdList::immediate(ctx).copy`.
- Produces: `loader::load_k2(ctx, snapshot, max_len)` → `K2LoadedModel`, consumed by plans 8c (tests read real expert bytes) and 8e (capture binds every buffer).

**The exact byte buckets** (derived from the checkpoint headers; recomputed independently in Python while writing this plan). The test asserts each with `CHECK_EQ`, not a tolerance:

| bucket | bytes |
|---|---:|
| `int4_bytes` (non-expert) | 1,028,997,120 |
| `scale_bytes` (non-expert) | 64,312,320 |
| `expert_int4_bytes` | 17,045,913,600 |
| `expert_scale_bytes` | 1,065,369,600 |
| `bf16_linear_bytes` (45 routers × 100 × 2560 × 2) | 23,040,000 |
| `pad_bytes` (45 × 28 × 2560 × 2) | 6,451,200 |
| `lm_head_bytes` | 1,283,194,880 |
| `embed_bytes` | 1,283,194,880 |
| `small_bytes` (48 × 20,480 + 45 × 656 + 10,240) | 1,022,800 |
| `rope_bytes` at max_len 16384 | 8,388,608 |
| **`total()`** | **21,809,885,008** |
| **`read_per_token`** | **3,785,731,920** |

`read_per_token` is spec §1's 3.779 GB of checkpoint bytes plus the 28 zero rows of each router (6.45 MB) and the norms and biases widened to fp32 (0.52 MB), itemised, not hidden.

- [ ] **Step 1: Write the failing test** - `tests/loader/k2_load_checkpoint_test.cc`:

```cpp
// load_k2 against the real checkpoint (spec 4 §2 bar 4): every tensor accounted for, every
// bucket exact, and device bytes read back against the mmapped source at the edges that a
// wrong stride or a wrong fuse would break.
#include <cstdio>
#include <cstring>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include "check.h"
#include "common/bf16.h"
#include "l0/cmdlist.h"
#include "l0/context.h"
#include "loader/k2_loader.h"
#include "loader/quant.h"
#include "loader/safetensors.h"
#include "loader/snapshot.h"

namespace {
using model::K2ExpertId;
using model::K2Horizon;
using model::K2LinearId;

std::vector<uint8_t> readback(l0::CmdList& imm, const l0::Mem& m, size_t off, size_t bytes) {
  std::vector<uint8_t> v(bytes);
  imm.copy(v.data(), static_cast<const uint8_t*>(m.ptr()) + off, bytes);
  return v;
}
}  // namespace

int main(int argc, char** argv) {
  CHECK(argc > 1);
  const std::string snap = loader::resolve_snapshot(argv[1]);
  l0::Context ctx(0);
  loader::K2LoadedModel m = loader::load_k2(ctx, snap, 16384);
  const loader::K2LoadReport& r = m.report;

  CHECK_EQ(r.unconsumed, size_t(0));
  CHECK_EQ(m.max_len, uint32_t(16384));
  CHECK_EQ(m.linears.size(), size_t(3 * 4 + 45 * 5 + 1));
  CHECK_EQ(m.experts.size(), size_t(45 * 3));
  CHECK_EQ(m.norms.size(), size_t(48));
  CHECK_EQ(m.route_bias.size(), size_t(45));

  CHECK_EQ(r.int4_bytes, size_t(1028997120));
  CHECK_EQ(r.scale_bytes, size_t(64312320));
  CHECK_EQ(r.expert_int4_bytes, size_t(17045913600ull));
  CHECK_EQ(r.expert_scale_bytes, size_t(1065369600));
  CHECK_EQ(r.bf16_linear_bytes, size_t(23040000));
  CHECK_EQ(r.pad_bytes, size_t(6451200));
  CHECK_EQ(r.lm_head_bytes, size_t(1283194880));
  CHECK_EQ(r.embed_bytes, size_t(1283194880));
  CHECK_EQ(r.small_bytes, size_t(1022800));
  CHECK_EQ(r.rope_bytes, size_t(8388608));
  CHECK_EQ(r.total(), size_t(21809885008ull));
  CHECK_EQ(r.read_per_token, size_t(3785731920ull));

  loader::SafetensorsSet set(snap);
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  auto src = [&](const std::string& name) {
    const loader::TensorInfo t = set.tensors().at(name);
    return std::make_pair(set.data(t), set.bytes(t));
  };

  // A Single-fuse expert is its checkpoint bytes verbatim, at e × stride. Test the LAST expert
  // of the LAST sparse layer (a stride error walks off the end) and a middle one.
  for (auto [layer, e, gid, part] : {std::make_tuple(47u, 63u, K2ExpertId::Value,
                                                     std::string("self_attn.v_experts.63")),
                                     std::make_tuple(10u, 37u, K2ExpertId::MoeDown,
                                                     std::string("mlp.experts.37.down_proj"))}) {
    const loader::K2ExpertWeights& ew = m.experts.at({layer, gid});
    const std::string base = K2Horizon::layer_prefix(layer) + part;
    const auto [qw, qwb] = src(base + ".qweight");
    const auto [sc, scb] = src(base + ".scales");
    CHECK_EQ(qwb, ew.w_stride);
    CHECK_EQ(scb, ew.s_stride);
    CHECK(readback(imm, ew.weights, e * ew.w_stride, ew.w_stride) ==
          std::vector<uint8_t>(qw, qw + qwb));
    CHECK(readback(imm, ew.scales, e * ew.s_stride, ew.s_stride) ==
          std::vector<uint8_t>(sc, sc + scb));
  }

  // Interleave16: fused column 16 of expert 99 is up_proj column 0; column 0 is gate_proj's.
  {
    const loader::K2ExpertWeights& ew = m.experts.at({3, K2ExpertId::MoeGateUp});
    const uint32_t N = ew.shape.N, rows = ew.shape.K / 8, Np = 768;
    const std::vector<uint8_t> dev = readback(imm, ew.weights, 99 * ew.w_stride, ew.w_stride);
    const auto [gate, gb] = src("model.layers.3.mlp.experts.99.gate_proj.qweight");
    const auto [up, ub] = src("model.layers.3.mlp.experts.99.up_proj.qweight");
    (void)gb; (void)ub;
    for (uint32_t row : {0u, rows - 1}) {
      uint32_t d0, d16, g0, u0;
      std::memcpy(&d0, dev.data() + (size_t(row) * N + 0) * 4, 4);
      std::memcpy(&d16, dev.data() + (size_t(row) * N + 16) * 4, 4);
      std::memcpy(&g0, gate + (size_t(row) * Np + 0) * 4, 4);
      std::memcpy(&u0, up + (size_t(row) * Np + 0) * 4, 4);
      CHECK_EQ(d0, g0);
      CHECK_EQ(d16, u0);
    }
  }

  // Concat: AttnSparse's v_router columns start at 9216.
  {
    const loader::DeviceWeight& w = m.linears.at({20, K2LinearId::AttnSparse});
    const auto [vr, vrb] = src("model.layers.20.self_attn.v_router.qweight");
    (void)vrb;
    const std::vector<uint8_t> dev = readback(imm, w.mem, (0 * 9280 + K2Horizon::kVOff) * 4, 4);
    uint32_t got, want;
    std::memcpy(&got, dev.data(), 4);
    std::memcpy(&want, vr, 4);
    CHECK_EQ(got, want);
  }

  // Small blocks: plain w and F16 bias widened verbatim.
  {
    const auto [nw, nwb] = src("model.layers.5.post_attention_layernorm.weight");
    (void)nwb;
    const std::vector<uint8_t> dev = readback(imm, m.norms[5], K2Horizon::kNormsOffPost, 4 * 2560);
    for (uint32_t i : {0u, 1279u, 1280u, 2559u}) {
      uint16_t w16; float f;
      std::memcpy(&w16, nw + size_t(i) * 2, 2);
      std::memcpy(&f, dev.data() + size_t(i) * 4, 4);
      CHECK(f == common::bf16_to_f32(w16));
    }
    const auto [vb, vbb] = src("model.layers.30.self_attn.v_router.bias");
    (void)vbb;
    const std::vector<uint8_t> rb =
        readback(imm, m.route_bias.at(30), K2Horizon::kRouteOffMovaBias, 4 * 64);
    for (uint32_t i : {0u, 63u}) {
      uint16_t h; float f;
      std::memcpy(&h, vb + size_t(i) * 2, 2);
      std::memcpy(&f, rb.data() + size_t(i) * 4, 4);
      CHECK(f == common::f16_to_f32(h));
    }
  }

  // MoE router: rows 100..127 are zero on the device (the pad top-8 never reads).
  {
    const loader::DeviceWeight& w = m.linears.at({3, K2LinearId::MoeRouter});
    CHECK_EQ(w.mem.size(), size_t(128) * 2560 * 2);
    // Tile of output row 100 (n_tile 6, lane 4), k-octet 0: all zero after the repack.
    const std::vector<uint8_t> dev = readback(imm, w.mem, 0, w.mem.size());
    const size_t K8 = 320;
    for (size_t k8 = 0; k8 < K8; ++k8)
      for (size_t k = 0; k < 8; ++k) {
        uint16_t v;
        std::memcpy(&v, dev.data() + (((6 * K8 + k8) * 8 + k) * 16 + 4) * 2, 2);
        CHECK_EQ(v, uint16_t(0));
      }
  }

  std::printf("k2_load_checkpoint_test OK: %.3f GB resident, %.3f GB read per token, %.1f s\n",
              r.total() / 1e9, r.read_per_token / 1e9, r.seconds);
  return 0;
}
```

- [ ] **Step 2: Register** - append to `tests/CMakeLists.txt` (after the `B70_TEST_SNAPSHOT` block, so the pattern matches the 27B's checkpoint tests):

```cmake
set(B70_K2_SNAPSHOT
  "${_b70_hf_home}/hub/models--urakozz--IFM-K2-Horizon-MoVA-36B-A4B-W4A16-AutoRound-GPTQ/snapshots/c0fd9971ad6d3bb4caedf726e798a02b8df9bcb5"
  CACHE PATH "The K2-Horizon snapshot the k2-labelled tests load")
add_executable(k2_load_checkpoint_test loader/k2_load_checkpoint_test.cc)
target_include_directories(k2_load_checkpoint_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/tests)
target_link_libraries(k2_load_checkpoint_test PRIVATE b70_k2_loader)
add_test(NAME k2_load_checkpoint_test COMMAND k2_load_checkpoint_test ${B70_K2_SNAPSHOT})
set_tests_properties(k2_load_checkpoint_test PROPERTIES LABELS "checkpoint;k2" TIMEOUT 1200)
```

- [ ] **Step 3: Confirm it fails** - `tools/box.sh sync && tools/box.sh build 2>&1 | grep -m1 "undefined reference.*load_k2"`. Expected: the undefined reference.

- [ ] **Step 4: Implement** - `src/loader/k2_loader.cc`:

```cpp
#include "loader/k2_loader.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>

#include "common/bf16.h"
#include "common/json.h"
#include "common/repack.h"
#include "l0/cmdlist.h"
#include "loader/quant.h"
#include "loader/safetensors.h"
#include "loader/snapshot.h"

namespace loader {
namespace {

using model::K2Horizon;

bool ends_with(const std::string& s, const char* suf) {
  const size_t n = std::strlen(suf);
  return s.size() >= n && s.compare(s.size() - n, n, suf) == 0;
}

// Every tensor a load reads is named here; what is left at the end, minus qzeros/g_idx, is
// the report's `unconsumed` - spec §2 bar 4's "every checkpoint tensor accounted for".
struct Consumed {
  std::set<std::string> names;
  void linear(const LinearSrc& s) {
    if (s.kind == WKind::Int4) {
      names.insert(s.name + ".qweight");
      names.insert(s.name + ".scales");
    } else {
      names.insert(s.name + ".weight");
    }
  }
};

l0::Mem upload(l0::Context& ctx, l0::CmdList& imm, const void* src, size_t bytes) {
  l0::Mem m(ctx, l0::MemKind::Device, bytes);
  imm.copy(m.ptr(), src, bytes);
  return m;
}

// A plain tensor by full name, dtype and element count enforced. Returned BY VALUE: callers
// pass temporary names, and gcc 13's -Wdangling-reference cannot see the referent is in the map.
TensorInfo take(const SafetensorsSet& set, Consumed& c, const std::string& name, const char* dtype,
                uint64_t elems) {
  auto it = set.tensors().find(name);
  if (it == set.tensors().end())
    throw std::runtime_error("K2 checkpoint has no tensor '" + name + "'");
  const TensorInfo t = it->second;
  if (t.dtype != dtype)
    throw std::runtime_error("'" + name + "': dtype " + t.dtype + ", expected " + dtype);
  uint64_t n = 1;
  for (uint64_t d : t.shape) n *= d;
  if (n != elems)
    throw std::runtime_error("'" + name + "': " + std::to_string(n) + " elements, expected " +
                             std::to_string(elems));
  c.names.insert(name);
  return t;
}

// 16-bit words widened to fp32 verbatim: "BF16" or "F16" - the source dtype IS the bake.
void widen(const SafetensorsSet& set, const TensorInfo& t, const std::string& name, float* dst) {
  const uint8_t* p = set.data(t);
  check_align(p, 2, name);
  const uint16_t* w = reinterpret_cast<const uint16_t*>(p);
  const size_t n = set.bytes(t) / 2;
  const bool f16 = t.dtype == "F16";
  for (size_t i = 0; i < n; ++i) dst[i] = f16 ? common::f16_to_f32(w[i]) : common::bf16_to_f32(w[i]);
}

// Classify every part of one fused linear and check it against the table's kind, K and N.
std::vector<LinearSrc> sources(const SafetensorsSet& set, Consumed& c, const std::string& prefix,
                               const std::vector<std::string>& parts, model::WeightKind kind,
                               const model::GemvShape& sh, uint32_t n_want) {
  std::vector<LinearSrc> v;
  uint32_t n_sum = 0;
  for (const std::string& part : parts) {
    LinearSrc s = LinearSrc::classify(set, prefix + part);   // checks alignment and dtypes
    const bool int4 = s.kind == WKind::Int4;
    if (int4 != (kind == model::WeightKind::Int4))
      throw std::runtime_error(prefix + part + " is " + (int4 ? "int4" : "bf16") +
                               ", but model::K2Horizon says otherwise");
    if (s.K != sh.K)
      throw std::runtime_error(prefix + part + ": K=" + std::to_string(s.K) + ", table says " +
                               std::to_string(sh.K));
    n_sum += s.N;
    c.linear(s);
    v.push_back(s);
  }
  if (n_sum != n_want)
    throw std::runtime_error(prefix + parts[0] + ": parts sum to N=" + std::to_string(n_sum) +
                             ", table says " + std::to_string(n_want));
  return v;
}

std::vector<common::ColSource> columns(model::Fuse fuse, const std::vector<LinearSrc>& srcs,
                                       const std::string& id) {
  std::vector<common::Part> parts;
  for (const LinearSrc& s : srcs) parts.push_back({s.qweight, s.scales, s.N});
  if (fuse == model::Fuse::Interleave16) {
    if (parts.size() != 2 || parts[0].N != parts[1].N || parts[0].N % 16 != 0)
      throw std::runtime_error(id + ": interleave16 needs two parts of equal, 16-divisible N");
    return common::cols_interleave16(parts[0], parts[1]);
  }
  return common::cols_concat(parts);
}

struct Staging {
  std::vector<uint32_t> words;
  std::vector<uint16_t> scales;
  std::vector<uint16_t> bf_src, bf_tiled;
};

DeviceWeight load_linear(l0::Context& ctx, l0::CmdList& imm, const SafetensorsSet& set,
                         Consumed& c, const std::string& prefix, const model::K2Linear& fl,
                         Staging& st, K2LoadReport& rep) {
  const model::GemvShape& sh = fl.shape;
  const std::string id = prefix + fl.parts[0];
  const uint32_t n_want = fl.pad_n != 0 ? fl.pad_n : sh.N;
  const std::vector<LinearSrc> srcs = sources(set, c, prefix, fl.parts, fl.kind, sh, n_want);
  if (fl.kind == model::WeightKind::Int4) {
    if (sh.layout != 0 || fl.pad_n != 0 || sh.K % 64 != 0 || sh.N % 64 != 0)
      throw std::runtime_error(id + ": K2 int4 linears are layout 0, unpadded, K and N % 64 == 0");
    const size_t words = size_t(sh.K / 8) * sh.N, scales = size_t(sh.K / 64) * sh.N;
    if (st.words.size() < words) st.words.resize(words);
    if (st.scales.size() < scales) st.scales.resize(scales);
    common::repack_int4_layout0_cols(sh.K, sh.N, columns(fl.fuse, srcs, id), st.words.data(),
                                     st.scales.data());
    rep.int4_bytes += words * 4;
    rep.scale_bytes += scales * 2;
    l0::Mem w = upload(ctx, imm, st.words.data(), words * 4);
    auto s = std::make_unique<l0::Mem>(upload(ctx, imm, st.scales.data(), scales * 2));
    return {std::move(w), std::move(s), sh, fl.kind};
  }
  // bf16: parts stacked row-major, zero rows to shape.N, then the canonical tiles.
  const size_t elems = size_t(sh.N) * sh.K;
  st.bf_src.assign(elems, 0);
  size_t row = 0;
  for (const LinearSrc& s : srcs) {
    std::memcpy(st.bf_src.data() + row * sh.K, s.weight, size_t(s.N) * sh.K * 2);
    row += s.N;
  }
  if (st.bf_tiled.size() < elems) st.bf_tiled.resize(elems);
  common::repack_bf16_tiled(st.bf_src.data(), sh.K, sh.N, st.bf_tiled.data());
  (fl.id == model::K2LinearId::LmHead ? rep.lm_head_bytes : rep.bf16_linear_bytes) +=
      size_t(n_want) * sh.K * 2;
  rep.pad_bytes += size_t(sh.N - n_want) * sh.K * 2;
  return {upload(ctx, imm, st.bf_tiled.data(), elems * 2), nullptr, sh, fl.kind};
}

// One layer's routed group: each expert repacked into staging at e × stride, one upload each
// for words and scales. Staging peaks at 100 × 1,966,080 B for gate‖up.
K2ExpertWeights load_experts(l0::Context& ctx, l0::CmdList& imm, const SafetensorsSet& set,
                             Consumed& c, const std::string& prefix,
                             const model::K2ExpertGroup& g, Staging& st, K2LoadReport& rep) {
  const model::GemvShape& sh = g.shape;
  const size_t ws = K2Horizon::expert_weight_stride(sh), ss = K2Horizon::expert_scale_stride(sh);
  const size_t all_words = size_t(g.count) * (ws / 4), all_scales = size_t(g.count) * (ss / 2);
  if (st.words.size() < all_words) st.words.resize(all_words);
  if (st.scales.size() < all_scales) st.scales.resize(all_scales);
  for (uint32_t e = 0; e < g.count; ++e) {
    std::vector<std::string> parts;
    for (const std::string& t : g.parts) parts.push_back(K2Horizon::expert_part(t, e));
    const std::vector<LinearSrc> srcs =
        sources(set, c, prefix, parts, model::WeightKind::Int4, sh, sh.N);
    common::repack_int4_layout0_cols(sh.K, sh.N, columns(g.fuse, srcs, prefix + parts[0]),
                                     st.words.data() + size_t(e) * (ws / 4),
                                     st.scales.data() + size_t(e) * (ss / 2));
  }
  rep.expert_int4_bytes += size_t(g.count) * ws;
  rep.expert_scale_bytes += size_t(g.count) * ss;
  return {upload(ctx, imm, st.words.data(), size_t(g.count) * ws),
          upload(ctx, imm, st.scales.data(), size_t(g.count) * ss), sh, g.count, ws, ss};
}

// One small block: the table's entries for `block`, widened into place; every byte owned.
l0::Mem load_block(l0::Context& ctx, l0::CmdList& imm, const SafetensorsSet& set, Consumed& c,
                   const std::string& prefix, const std::vector<model::K2SmallTensor>& table,
                   model::K2SmallBlock block, size_t bytes, K2LoadReport& rep) {
  std::vector<float> buf(bytes / 4, 0.0f);
  size_t filled = 0;
  for (const model::K2SmallTensor& t : table) {
    if (t.block != block) continue;
    if (size_t(t.offset) + size_t(t.elems) * 4 > bytes)
      throw std::runtime_error(prefix + t.name + " overruns its " + std::to_string(bytes) +
                               "-byte block");
    const TensorInfo info = take(set, c, prefix + t.name, t.dtype, t.elems);
    widen(set, info, prefix + t.name, buf.data() + t.offset / 4);
    filled += size_t(t.elems) * 4;
  }
  if (filled != bytes)
    throw std::runtime_error(prefix + ": small table fills " + std::to_string(filled) + " of " +
                             std::to_string(bytes) + " B - every byte of a block needs an owner");
  rep.small_bytes += bytes;
  return upload(ctx, imm, buf.data(), bytes);
}

}  // namespace

K2LoadedModel load_k2(l0::Context& ctx, const std::string& snapshot_or_repo, uint32_t max_len) {
  const auto t0 = std::chrono::steady_clock::now();
  const std::string snap = resolve_snapshot(snapshot_or_repo);
  std::ifstream cf(snap + "config.json");
  if (!cf) throw std::runtime_error("cannot read " + snap + "config.json");
  std::stringstream cs;
  cs << cf.rdbuf();
  const common::json::Value cfg = common::json::parse(cs.str());
  const std::string model_type = cfg.at("model_type").str();
  if (model_type != "k2_horizon")
    throw std::runtime_error("load_k2: " + snap + " is model_type '" + model_type +
                             "', not k2_horizon");
  const QuantConfig qc = QuantConfig::parse(cfg);
  SafetensorsSet set(snap);
  const QuantScan scan = assert_quant_invariants(set);
  if (!qc.desc_act_declared && scan.g_idx_tensors != 0)
    throw std::runtime_error("load_k2: config declares no desc_act but the checkpoint ships " +
                             std::to_string(scan.g_idx_tensors) + " g_idx tensors");

  Consumed c;
  l0::CmdList imm = l0::CmdList::immediate(ctx);
  const TensorInfo emb = take(set, c, "model.embed_tokens.weight", "BF16",
                              uint64_t(K2Horizon::kVocab) * K2Horizon::kHidden);
  std::vector<float> fnorm(K2Horizon::kHidden);
  widen(set, take(set, c, "model.norm.weight", "BF16", K2Horizon::kHidden), "model.norm.weight",
        fnorm.data());
  const std::vector<float> rope = k2_rope_table(max_len);
  K2LoadedModel m(upload(ctx, imm, set.data(emb), set.bytes(emb)),
                  upload(ctx, imm, fnorm.data(), fnorm.size() * 4),
                  upload(ctx, imm, rope.data(), rope.size() * 4), max_len);
  m.report.embed_bytes = set.bytes(emb);
  m.report.small_bytes += fnorm.size() * 4;
  m.report.rope_bytes = rope.size() * 4;

  Staging st;
  m.norms.reserve(K2Horizon::kLayers);
  for (const model::K2LayerDesc& ld : K2Horizon::layers()) {
    const std::string lp = K2Horizon::layer_prefix(ld.index);
    for (const model::K2Linear& fl : ld.linears)
      m.linears.emplace(std::make_pair(ld.index, fl.id),
                        load_linear(ctx, imm, set, c, lp, fl, st, m.report));
    for (const model::K2ExpertGroup& g : ld.experts)
      m.experts.emplace(std::make_pair(ld.index, g.id),
                        load_experts(ctx, imm, set, c, lp, g, st, m.report));
    m.norms.push_back(load_block(ctx, imm, set, c, lp, ld.small, model::K2SmallBlock::Norms,
                                 K2Horizon::kNormsBytes, m.report));
    if (ld.kind == model::K2LayerKind::Sparse)
      m.route_bias.emplace(ld.index,
                           load_block(ctx, imm, set, c, lp, ld.small, model::K2SmallBlock::Route,
                                      K2Horizon::kRouteBytes, m.report));
  }
  m.linears.emplace(std::make_pair(kTopLevel, model::K2LinearId::LmHead),
                    load_linear(ctx, imm, set, c, "", K2Horizon::lm_head(), st, m.report));

  K2LoadReport& r = m.report;
  std::string names;
  for (const auto& [name, info] : set.tensors()) {
    (void)info;
    if (c.names.count(name) || ends_with(name, ".qzeros") || ends_with(name, ".g_idx")) continue;
    if (r.unconsumed++ < 5) names += (names.empty() ? "" : ", ") + name;
  }
  size_t active = 0;
  for (const auto& [key, ew] : m.experts) {
    const size_t k = key.second == model::K2ExpertId::Value ? K2Horizon::kValueTopK : K2Horizon::kTopK;
    active += k * (ew.w_stride + ew.s_stride);
  }
  r.read_per_token = r.int4_bytes + r.scale_bytes + r.bf16_linear_bytes + r.pad_bytes +
                     r.lm_head_bytes + r.small_bytes + active;
  r.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

  const double gb = 1e9;
  std::printf(
      "load_k2: %s\n"
      "  quant       int4 g%u sym (%s), %zu dynamic exclusion rules, %zu subnormal f16 scales\n"
      "  tensors     %zu in the index; %zu unconsumed%s%s\n"
      "  int4        %13zu B %7.3f GB   (+ experts %13zu B %7.3f GB)\n"
      "  scales      %13zu B %7.3f GB   (+ experts %13zu B %7.3f GB)\n"
      "  routers     %13zu B %7.3f GB   bf16 real rows (+ %zu B zero pad)\n"
      "  lm_head     %13zu B %7.3f GB   bf16\n"
      "  embed       %13zu B %7.3f GB   (gathered, not per-token)\n"
      "  small       %13zu B %7.3f GB   (norms, router biases, final norm; fp32)\n"
      "  rope        %13zu B %7.3f GB   (max_len %u)\n"
      "  total       %13zu B %7.3f GB\n"
      "  read/token  %13zu B %7.3f GB   (active experts only: 4/64 value, 8/100 MoE)\n"
      "  load        %.1f s\n",
      snap.c_str(), qc.group_size, qc.quant_method.c_str(), qc.dynamic_rule_count,
      scan.subnormal_scales, set.tensors().size(), r.unconsumed, r.unconsumed ? ": " : "",
      names.c_str(), r.int4_bytes, r.int4_bytes / gb, r.expert_int4_bytes,
      r.expert_int4_bytes / gb, r.scale_bytes, r.scale_bytes / gb, r.expert_scale_bytes,
      r.expert_scale_bytes / gb, r.bf16_linear_bytes, r.bf16_linear_bytes / gb, r.pad_bytes,
      r.lm_head_bytes, r.lm_head_bytes / gb, r.embed_bytes, r.embed_bytes / gb, r.small_bytes,
      r.small_bytes / gb, r.rope_bytes, r.rope_bytes / gb, max_len, r.total(), r.total() / gb,
      r.read_per_token, r.read_per_token / gb, r.seconds);
  if (r.unconsumed != 0)
    throw std::runtime_error("load_k2: " + std::to_string(r.unconsumed) +
                             " checkpoint tensors were not loaded: " + names);
  return m;
}

}  // namespace loader
```

- [ ] **Step 5: Build, then run the load test detached** (it loads 21.8 GB; allow ~5 min):
  `tools/box.sh sync && tools/box.sh build 2>&1 | tail -2`
  `ssh -o BatchMode=yes user@box 'cd ~/b70-inference-server/build && ZE_AFFINITY_MASK=1 setsid nohup ctest -R "^k2_load_checkpoint_test$" --output-on-failure > $HOME/k2-load.log 2>&1 < /dev/null &'`
  Poll: `ssh -o BatchMode=yes user@box 'tail -20 $HOME/k2-load.log'`. Expected: the `load_k2:` report, `k2_load_checkpoint_test OK: 21.810 GB resident, 3.786 GB read per token`, ctest `100% tests passed`.
  **If a bucket differs:** the printed report is the breakdown. A different number is a finding about the table or the loader - fix the one defect it names; never edit the expected constant to match.

- [ ] **Step 6: Run the 27B's loader and model tests to prove nothing moved** - `tools/box.sh test '^(qwen35_test|quant_test|safetensors_test|dequant_fixture_test)$'`. Expected: all pass (these are the host-only 27B tests; the full 27B suite runs at plan 8e's end).

- [ ] **Step 7: Commit**

```bash
git add src/loader/k2_loader.cc tests/loader/k2_load_checkpoint_test.cc tests/CMakeLists.txt
git commit -m "feat(loader): load_k2 - K2-Horizon on one B70, flat expert buffers, exact byte buckets

Claude-Session: "
```

---

## Self-review

**Spec coverage:** §3.1 model table + loader → Tasks 1, 3; the two traps (plain `w`, 2-group norm) → Task 1's small table stores the plain weight, and the group split is kernel-side (plan 8c); §3.4 flat per-(layer, group) expert buffers with the stated strides → Task 1 test + Task 3; KV is not the loader's (plan 8e allocates it); §4 load test (unconsumed 0, shapes asserted, read/token redefined for MoE) → Task 3; §1 bytes → Task 3's exact buckets. RoPE table builder "K2 gets its own table, fp32 [max_len][2][64]" → Task 2.
**Deviation from the spec's words, deliberate:** §3.1 says the loader pieces are "factored out of `load()`"; this plan re-spells ~120 lines of K2-sized helpers instead, so `loader.cc` stays byte-identical and bar 5 has nothing to prove about the loader. The spec's §3.1 row is updated in the same commit as these plans.
**Placeholders:** none. **Types:** `K2LinearId`, `K2ExpertId`, `K2ExpertWeights{weights, scales, shape, count, w_stride, s_stride}`, `K2LoadedModel{linears, experts, norms, route_bias, embed, final_norm, rope, report, max_len}` are used with these exact names in plans 8c-8e.
