#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include "model/model_desc.h"
#include "runtime/buffer_sizes.h"
#include "runtime/prefill_backend.h"

// Spec 6 §10 (max_len auto): the device memory an engine will hold at a max_len,
// computed before a byte of it is allocated, and the largest max_len that fits.
//
// Pure host arithmetic over runtime/buffer_sizes.h - the functions the allocations
// themselves are sized by - plus the loaded weights' byte count, which only the
// loader knows. Nothing here reads a device; the device total and the weights come
// in as numbers, so tests/runtime/memory_plan_test.cc runs on any host and
// tests/runtime/memory_plan_box_test.cc asserts on the card that every planned
// component equals what the engine allocated.
namespace runtime {

// The five figures Engine::memory_line() prints, in its order. `model` is the
// loader's total (the RoPE table included); `kv` the main model's K and V caches;
// `decode_state` everything else the decode lists use (the rest of
// PersistentBuffers, DecodeScratch, and with MTP the head's buffers and the prefill
// hidden rows); `prefill_scratch` PrefillScratch, eager and lazy; `int8` the h8
// path's Int8State.
//
// `draft_vocab` (spec 8 §11, `--draft-vocab`): the compact draft head and its id table,
// which the loader allocates OUTSIDE LoadReport::total() (so `model` reads the same with
// and without it), sized by loader::draft_vocab_bytes. 0 when off.
struct MemoryComponents {
  size_t model = 0, kv = 0, decode_state = 0, prefill_scratch = 0, int8 = 0;
  size_t draft_vocab = 0;
  size_t total() const { return model + kv + decode_state + prefill_scratch + int8 + draft_vocab; }
};

// "<label>: model 18.116 GB, kv 8.590 GB, decode state 0.590 GB, prefill scratch
// 0.700 GB, int8 0.085 GB, total 28.081 GB of 32.530 GB" - memory_line()'s format
// (GB = 1e9 B), one formatter for the measured line and the planned one. With a draft
// vocabulary a "draft vocab <GB>," term precedes the total; without one the line is
// exactly the above.
std::string format_memory(const char* label, const MemoryComponents& c, size_t device_bytes);

// What a session's prefill allocates, which is a property of the backend and the
// attention path (PrefillScratch's lazy accessors, Engine::prepare_prefill):
//
//   l0-int8 (default)   Int8State (xq, xs, w8, every int4 linear's column scales); the L0
//                       slab only with MTP (step_mtp_kv's bf16 linears walk it).
//   l0                  the slab ([intermediate][1024] bf16), every int4 linear.
//   sycl-tla            the bf16 dequant scratch and, its attention always being the
//                       composed path, pf_s / pf_p.
//   composed attention  (B70_PREFILL_ATTN=composed on an L0 backend) pf_s / pf_p:
//                       [GQA group][kC][max_len] fp32 + bf16 - 73,728 B per position
//                       on Qwen3.8, the only prefill scratch that scales with max_len.
//                       The default flash path builds neither.
//
// `prefill = false` is a decode-only engine (b70-decode without --pp / --prefill):
// ruling R7 keeps every prefill allocation lazy, so it plans none of them.
struct PrefillPath {
  bool prefill = true;
  PrefillBackend backend = PrefillBackend::L0Int8;
  bool composed_attn = false;
};

// Spec 8 §11 (`--draft-vocab`): a draft vocabulary of `rows` ids, its compact head in
// the lm_head's form (`int8`, else bf16). `rows` = 0 is off. With MTP it adds the
// loader's compact head + id table (`MemoryComponents::draft_vocab`,
// loader::draft_vocab_bytes) and MtpBuffers' compact logits (in decode_state,
// MtpDims::sizes). Neither depends on max_len, but both come out of what auto can give
// the context. cli::draft_vocab_plan reads it off a loaded model.
struct DraftVocabPlan {
  uint32_t rows = 0;
  bool int8 = true;
};

struct MemoryPlan : MemoryComponents {
  uint32_t max_len = 0;
  bool mtp = false;
  KvCache kv_cache = KvCache::Bf16;   // spec 12b: the form `kv` (and the MTP head's KV) is in
  // The finer split, each already counted in one of the five above.
  size_t rope = 0;           // in model: Qwen35::rope_table_bytes(max_len)
  size_t mtp_buffers = 0;    // in decode_state: MtpBuffers
  size_t mtp_hidden = 0;     // in decode_state: Engine::prefill's MTP hidden rows
  size_t prefill_lazy = 0;   // in prefill_scratch: the lazy buffers `path` builds
  // Spec 15c, a mixture-of-experts model: the loader's expert blocks, routers and shared
  // experts (loader::moe_bytes - already inside `model_bytes`, the loaded total), and the
  // decode MoE scratch (runtime::moe_scratch_layout - inside decode_state). 0 when dense.
  size_t moe_weights = 0;    // in model
  size_t moe_scratch = 0;    // in decode_state
};

// `kv` (spec 12b, `--kv-cache`): the KV cache's form - its term follows it (bf16: 64 KiB
// per position on Qwen3.8; int8: 32 KiB of rows + 256 B of fp16 scales), and so does the
// MTP head's own layer. The default is runtime::default_kv_cache(), what the engine
// allocates when nobody passes one, so a plan and an allocation in one process agree.
//
// `model_bytes`: everything loader::load() allocated EXCEPT the RoPE table -
// `report.total() - report.rope_bytes` - i.e. the weights, which do not depend on
// max_len. The MTP head's weights are in it when the model was loaded with the head
// (`mtp`); its buffers are planned here. `plan.model` adds the table back at
// `max_len`, so it equals memory_line()'s `model` once the table has that length.
MemoryPlan plan(const model::ModelDesc& desc, uint32_t max_len, bool mtp, size_t model_bytes,
                const PrefillPath& path = {}, const DraftVocabPlan& dv = {},
                KvCache kv = default_kv_cache());

// The max_len grid: the L0 prefill backends need max_len % 256 == 0 (spec 2.1 §3.1,
// prefill/attn.cc), which is also a whole number of decode attention blocks (64).
constexpr uint32_t kMaxLenQuantum = 256;
// Below this, auto gives up instead of serving a context nobody asked for.
constexpr uint32_t kMinAutoMaxLen = 4096;
// The default --mem-reserve-gb: device memory the plan does not count - the driver's
// own allocations, the kernel modules and command lists (decode, MTP draft / verify,
// prefill), the int8 sign tables, Int8State's per-weight colmax temporaries, the
// allocator's slack. An estimate, NOT a measurement: the box has to confirm or tune
// it (docs/superpowers/plans/box-validation-queue.md).
constexpr double kDefaultReserveGb = 1.5;

// The largest multiple of kMaxLenQuantum, at most `cap` (the trained context),
// whose plan total + `reserve_bytes` <= `device_bytes`. 0 when not even
// min(kMinAutoMaxLen, cap) fits. Throws std::invalid_argument when `cap` is below
// one quantum.
uint32_t max_len_that_fits(const model::ModelDesc& desc, bool mtp, size_t model_bytes,
                           size_t device_bytes, size_t reserve_bytes, uint32_t cap,
                           const PrefillPath& path = {}, const DraftVocabPlan& dv = {},
                           KvCache kv = default_kv_cache());

// One line for the startup log: the plan's five components, the finer split where it
// is non-zero, the reserve and the device total.
std::string describe(const MemoryPlan& p, size_t device_bytes, size_t reserve_bytes);

}  // namespace runtime
