#pragma once
// `--max-len auto|N`, shared by b70-serve and b70-decode
// (spec 6 §10). The order is forced by what each step needs:
//
//   1. parse                 before the device: "auto" or a positive integer
//   2. check_before_load     config.json's max_position_embeddings bounds an explicit N
//                            (and the 256 grid, for a CLI that prefills); auto needs it
//   3. load_len              auto loads at a small max_len - the RoPE table is the only
//                            thing load() sizes by it - because the plan needs the
//                            loaded bytes and load() needs a max_len
//   4. settle                after load, before the Engine: auto plans the largest
//                            max_len that fits and re-tables the model
//                            (loader::set_max_len); an explicit N is held to the same
//                            plan and refused with its breakdown instead of failing in
//                            an allocation later
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "kernels/kernels.h"
#include "l0/context.h"
#include "loader/loader.h"
#include "runtime/buffers.h"
#include "runtime/capture.h"
#include "runtime/memory_plan.h"

namespace cli {

struct MaxLenArg {
  bool is_auto = true;
  uint32_t value = 0;   // the explicit N; 0 with is_auto
};

inline MaxLenArg parse_max_len(const std::string& v) {
  if (v == "auto") return {true, 0};
  if (v.empty() || v.find_first_not_of("0123456789") != std::string::npos || v.size() > 10)
    throw std::runtime_error("--max-len expects auto or a positive integer, got '" + v + "'");
  const unsigned long long n = std::stoull(v);
  if (n > 0xFFFFFFFFull) throw std::runtime_error("--max-len is out of range: " + v);
  if (n == 0) throw std::runtime_error("--max-len 0 is not a model length");
  return {false, static_cast<uint32_t>(n)};
}

// Step 2. `trained` is loader::trained_context(snapshot) (0 = not declared).
// `require_quantum`: the CLI prefills, and the L0 prefill backends need max_len % 256
// (spec 2.1 §3.1) - refused here rather than at the first request.
inline void check_before_load(const MaxLenArg& a, uint32_t trained, bool require_quantum) {
  if (a.is_auto) {
    if (trained == 0)
      throw std::runtime_error("--max-len auto plans under config.json's max_position_embeddings,"
                               " and this checkpoint declares none: give --max-len N");
    return;
  }
  if (trained != 0 && a.value > trained)
    throw std::runtime_error("--max-len " + std::to_string(a.value) + " exceeds the trained context " +
                             std::to_string(trained) + " (config.json max_position_embeddings)");
  if (require_quantum && a.value % runtime::kMaxLenQuantum != 0)
    throw std::runtime_error("--max-len " + std::to_string(a.value) + " is not a multiple of " +
                             std::to_string(runtime::kMaxLenQuantum) +
                             " (the L0 prefill backends' grid, spec 2.1 §3.1)");
}

// Step 3: the max_len to hand loader::load().
inline uint32_t load_len(const MaxLenArg& a, uint32_t trained) {
  if (!a.is_auto) return a.value;
  return trained != 0 && trained < runtime::kMinAutoMaxLen ? trained : runtime::kMinAutoMaxLen;
}

// What the engine's prefill will allocate (runtime::PrefillPath): `prefill` false for a
// decode-only b70-decode run; the backend the engine will run (set_prefill_backend, else
// the default, l0).
inline runtime::PrefillPath prefill_path(bool prefill, runtime::PrefillBackend backend) {
  return {prefill, backend};
}

// 3o (operator 2026-11-10, accepted as a known FAIL): the l0-int8 prefill with int8 KV compounds two roundings
// (docs/probes/a4-int8-prefill-int8-kv-2026-11-10.md) - one of the 36 tool-call scenarios writes an invalid edit.
inline void warn_int8_prefill_kv8(runtime::PrefillBackend backend, runtime::KvCache kv) {
  if (backend == runtime::PrefillBackend::L0Int8 && kv == runtime::KvCache::Int8)
    std::fprintf(stderr, "warning: --prefill-backend l0-int8 with --kv-cache int8 is a known fail: ~3%% of tool calls "
                         "that copy exact text from a long prompt can come out wrong; use the l0 prefill or the bf16 "
                         "cache\n");
}

// Spec 12b: `--kv-cache bf16|int8` (default runtime::default_kv_cache(): B70_KV_CACHE, unset
// = bf16). int8 decodes through attention v2's kv8 twin and prefills through the flash
// attention.
inline runtime::KvCache parse_kv_cache_arg(const std::string& v) {
  return runtime::parse_kv_cache(v.c_str());
}

// Spec 8 §11: the draft vocabulary a loaded model carries (load(..., DraftVocabSpec)
// built it, before the plan), as the planner counts it: its compact head and id table are
// outside LoadReport::total(), so `weights` below does not hold them.
inline runtime::DraftVocabPlan draft_vocab_plan(const loader::LoadedModel& m) {
  if (!m.draft_vocab) return {};
  return {m.draft_vocab->size(), m.draft_vocab->head.kind == model::WeightKind::Int8};
}

// Step 4. Returns the max_len the Engine is built with; `m` is re-tabled to it for auto.
// Prints the choice and the plan's breakdown to stderr. `kv` (spec 12b): the KV cache's
// form - its KV term (and the MTP head's) is planned in it, so int8 roughly doubles what
// auto can give the context. `drafter` (spec 19c, `--spec dflash`): the DFlash drafter's
// term (runtime::dflash::drafter_plan - its weights and buffers, and the engine's verify
// slots and prefill taps for it); empty without one.
inline uint32_t settle(l0::Context& ctx, loader::LoadedModel& m, const MaxLenArg& a,
                       size_t reserve_bytes, const runtime::PrefillPath& path,
                       runtime::KvCache kv = runtime::default_kv_cache(),
                       const runtime::DrafterPlan& drafter = {}) {
  const model::ModelDesc& d = *m.desc;
  const bool mtp = m.mtp != nullptr;
  const runtime::DraftVocabPlan dv = draft_vocab_plan(m);
  const size_t weights = m.report.total() - m.report.rope_bytes;
  const size_t device = ctx.memory_bytes();
  const double gb = 1e9;
  uint32_t len = a.value;
  if (a.is_auto) {
    const uint32_t fit = runtime::max_len_that_fits(d, mtp, weights, device, reserve_bytes,
                                                    m.trained_max_len, path, dv, kv, drafter);
    if (fit == 0)
      throw std::runtime_error(
          "--max-len auto: not even " + std::to_string(runtime::kMinAutoMaxLen) +
          " positions fit - " +
          runtime::describe(runtime::plan(d, runtime::kMinAutoMaxLen, mtp, weights, path, dv, kv, drafter),
                            device, reserve_bytes));
    len = fit;
    loader::set_max_len(ctx, m, len);
    std::fprintf(stderr,
                 "max_len: auto -> %u (the largest multiple of %u that fits %.3f GB with a %.3f GB"
                 " reserve; trained context %u)\n",
                 len, runtime::kMaxLenQuantum, device / gb, reserve_bytes / gb, m.trained_max_len);
  } else {
    const runtime::MemoryPlan p = runtime::plan(d, len, mtp, weights, path, dv, kv, drafter);
    if (p.total() + reserve_bytes > device) {
      const uint32_t cap = m.trained_max_len != 0 ? m.trained_max_len : len;
      const uint32_t fit =
          cap < runtime::kMaxLenQuantum
              ? 0
              : runtime::max_len_that_fits(d, mtp, weights, device, reserve_bytes, cap, path, dv,
                                           kv, drafter);
      throw std::runtime_error(
          "--max-len " + std::to_string(len) + " does not fit: " +
          runtime::describe(p, device, reserve_bytes) + ". The largest that fits is " +
          std::to_string(fit) + " (--max-len auto)");
    }
    std::fprintf(stderr, "max_len: %u (--max-len)\n", len);
  }
  std::fprintf(stderr, "%s\n",
               runtime::describe(runtime::plan(d, len, mtp, weights, path, dv, kv, drafter), device,
                                 reserve_bytes)
                   .c_str());
  return len;
}

// Step 4 under --max-num-seqs (operator 2026-11-11): a card's `seqs` > 1 sequence slots (plan 13c), slot 0 the
// largest context that fits beside seqs - 1 slots of SlotDims::kShortLen (auto) or exactly the asked N, refused with
// the plan's numbers when it does not fit beside them. `m` is re-tabled at the slots' sum (one RoPE table, one KV
// allocation). Returns the layout; prints the choice and the plan's breakdown like settle(). No drafter and no draft
// vocabulary beside slots (refused before the load).
inline std::vector<uint32_t> settle_slots(l0::Context& ctx, loader::LoadedModel& m, const MaxLenArg& a, uint32_t seqs,
                                          size_t reserve_bytes, const runtime::PrefillPath& path, runtime::KvCache kv) {
  const model::ModelDesc& d = *m.desc;
  const bool mtp = m.mtp != nullptr;
  const size_t weights = m.report.total() - m.report.rope_bytes;
  const size_t device = ctx.memory_bytes();
  const double gb = 1e9;
  const uint32_t cap = m.trained_max_len != 0 ? m.trained_max_len : a.value;
  const uint32_t fit = cap < runtime::kMaxLenQuantum
                           ? 0
                           : runtime::slot0_len_that_fits(d, seqs, mtp, weights, device, reserve_bytes, cap, path, kv);
  const std::string beside = " beside " + std::to_string(seqs - 1) + " slot" + (seqs > 2 ? "s" : "") + " of " +
                             std::to_string(runtime::SlotDims::kShortLen) + " (--max-num-seqs " +
                             std::to_string(seqs) + ")";
  const uint32_t len = a.is_auto ? fit : a.value;
  if (len == 0)
    throw std::runtime_error(
        "--max-len auto: not even " + std::to_string(runtime::kMinAutoMaxLen) + " positions fit slot 0" + beside +
        " - " +
        runtime::describe(runtime::plan_slots(d, runtime::slot_layout(runtime::kMinAutoMaxLen, seqs), mtp, weights,
                                              path, {}, kv),
                          device, reserve_bytes) +
        ". A smaller --max-num-seqs");
  const std::vector<uint32_t> lens = runtime::slot_layout(len, seqs);
  const runtime::MemoryPlan p = runtime::plan_slots(d, lens, mtp, weights, path, {}, kv);
  if (!a.is_auto && p.total() + reserve_bytes > device)
    throw std::runtime_error("--max-len " + std::to_string(len) + " does not fit" + beside + ": " +
                             runtime::describe(p, device, reserve_bytes) + ". The largest slot 0 that fits beside them is " +
                             std::to_string(fit) + " (--max-len auto), or a smaller --max-num-seqs");
  if (a.is_auto)
    std::fprintf(stderr,
                 "max_len: auto -> slot 0 %u (the largest multiple of %u that fits%s in %.3f GB with a %.3f GB "
                 "reserve; trained context %u)\n",
                 len, runtime::kMaxLenQuantum, beside.c_str(), device / gb, reserve_bytes / gb, m.trained_max_len);
  else
    std::fprintf(stderr, "max_len: slot 0 %u (--max-len)%s\n", len, beside.c_str());
  uint32_t total = 0;
  for (uint32_t l : lens) total += l;
  loader::set_max_len(ctx, m, total);
  std::fprintf(stderr, "%s\n", runtime::describe(p, device, reserve_bytes).c_str());
  return lens;
}

// Plan 5d: the prefill chunk cap (2048 or 4096), in two steps around settle().
//
//   asked = runtime::prefill_chunk_cap_for(--prefill-chunk C, desc)   0 when no C was given;
//                                                                     refuses C > 4096 and a
//                                                                     MoE model above 2048
//   path.chunk_cap = planned_chunk_cap(asked)    what settle() plans max_len at
//   max_len = settle(..., path, ...)
//   cap = settle_chunk_cap(..., max_len, path, ..., asked)   what the engine is given
//
// An asked cap (b70-decode's --prefill-chunk) is planned and kept as it is: an explicit 4096
// shortens what auto gives the context. With nothing asked, kPrefillChunkCapRule decides:
// off, the default 2048; on, operator ruling (i) of 2026-10-27 (context first) - max_len is
// planned at 2048 and 4096 is taken only when the plan at that max_len has room for it
// (runtime::prefill_chunk_cap_that_fits). There is no environment variable or server flag for
// it (operator, 2026-11-03): the engine side chooses.
inline constexpr bool kPrefillChunkCapRule = false;
inline uint32_t planned_chunk_cap(uint32_t asked) {
  return asked != 0 ? asked : runtime::PrefillScratchDims::kCDefault;
}
// The same, with the model checked (an asked cap was checked by prefill_chunk_cap_for already;
// the default 2048 every model runs).
inline uint32_t planned_chunk_cap(uint32_t asked, const model::ModelDesc& d) {
  const uint32_t cap = planned_chunk_cap(asked);
  runtime::check_prefill_chunk_cap(d, cap);
  return cap;
}
inline uint32_t settle_chunk_cap(l0::Context& ctx, const loader::LoadedModel& m, uint32_t max_len,
                                 size_t reserve_bytes, const runtime::PrefillPath& path,
                                 runtime::KvCache kv, const runtime::DrafterPlan& drafter, uint32_t asked) {
  if (!path.prefill) return path.chunk_cap;
  const char* why = asked != 0 ? "--prefill-chunk" : nullptr;
  uint32_t cap = path.chunk_cap;
  if (why == nullptr && kPrefillChunkCapRule) {
    cap = runtime::prefill_chunk_cap_that_fits(*m.desc, m.mtp != nullptr, m.report.total() - m.report.rope_bytes,
                                               ctx.memory_bytes(), reserve_bytes, max_len, path,
                                               draft_vocab_plan(m), kv, drafter);
    why = cap == runtime::PrefillScratchDims::kCMax ? "the plan at this max_len has room for it"
                                                   : "4096 would not fit beside this max_len";
  }
  if (why) std::fprintf(stderr, "prefill chunk cap: %u (%s)\n", cap, why);   // the default says nothing
  return cap;
}

}  // namespace cli
