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
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <print>
#include <stdexcept>
#include <string>
#include <vector>

#include "kernels/kernels.h"
#include "l0/context.h"
#include "loader/loader.h"
#include "model/model_desc.h"
#include "runtime/buffers.h"
#include "runtime/capture.h"
#include "runtime/memory_plan.h"
#include "runtime/vram_bars.h"

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
                             " (the L0 prefill backends' grid)");
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

// PERF-114 (the owner's decision of 2026-11-21): on a dense model l0-int8 runs on the int8 context - its attention
// scores read the int8 keys - so it turns `--kv-cache int8` on and refuses an explicit bf16. `dense`: the checkpoint is
// not a MoE model (cli::int8_context_model). Returns the KV cache form to run.
inline runtime::KvCache int8_pass_kv(runtime::PrefillBackend backend, bool dense, const std::string& kv_cache_flag,
                                     runtime::KvCache kv) {
  if (backend != runtime::PrefillBackend::L0Int8 || !dense) return kv;
  if (!kv_cache_flag.empty() && runtime::parse_kv_cache(kv_cache_flag.c_str()) != runtime::KvCache::Int8)
    throw std::runtime_error("--prefill-backend l0-int8 runs on the int8 context (its attention reads the int8 KV "
                             "cache); drop --kv-cache bf16, or use the bf16 prompt pass (--prefill-backend l0)");
  return runtime::KvCache::Int8;
}

// 3o: a MoE model's l0-int8 (W4A8 experts) beside the int8 KV cache compounds two roundings - on Qwen3.8 with the MLP
// alone int8, ~3 % of the tool calls that copy exact text from a long prompt went wrong - so it is refused there.
inline void check_int8_prefill_kv8(runtime::PrefillBackend backend, runtime::KvCache kv, bool dense) {
  if (backend == runtime::PrefillBackend::L0Int8 && kv == runtime::KvCache::Int8 && !dense)
    throw std::runtime_error("--prefill-backend l0-int8 does not run with --kv-cache int8 on this model: the two "
                             "roundings together change tool calls that copy exact text from a long prompt; use the "
                             "bf16 prompt pass (--prefill-backend l0) or the bf16 cache");
}

// The startup notice while the int8 prompt pass is on - what it costs, measured against the bf16 prompt pass. Printed
// once, with the startup lines (b70-serve, b70-decode), for a dense model only: a MoE model's l0-int8 is its W4A8
// experts beside bf16 mixer linears (open-work 16), not what these numbers measured.
// Measured 2026-11-21 on Qwen3.8-27B (PERF-114's card turn: every linear int8, the int8 context, int8 attention scores),
// one card, served, MTP on, two interleaved rounds: 1.109 / 1.145 / 1.177 / 1.216x the earlier MLP-only int8 pass at
// 3.9k / 30k / 61k / 122k ids (1.40-1.47x l0); top-1 582 / 585 over 4k-64k against l0 on the bf16 cache; 0 of 12
// exact-copy requests (8k-99k) differ from l0; the golden gates 93 / 93 and the long gate 32 / 32.
inline void notice_int8_prefill(runtime::PrefillBackend backend, const model::ModelDesc& d) {
  if (backend != runtime::PrefillBackend::L0Int8 || d.is_moe()) return;
  std::print(stderr, "the int8 prompt pass is on: every prompt linear, the context (KV cache) and the prompt's "
                     "attention scores run in int8, 11-22 % faster than the earlier int8 pass (40-47 % faster than the "
                     "bf16 pass). Measured on Qwen3.8-27B against the bf16 pass: the most likely next token differs at "
                     "3 of 585 positions; 0 of 12 long exact-copy tool calls changed; the long-prompt gate passed. For "
                     "full quality, use the default bf16 pass (--prefill-backend l0).\n");
}

// Spec 12b: `--kv-cache bf16|int8` (default runtime::default_kv_cache(): B70_KV_CACHE, unset
// = bf16). int8 decodes through attention v2's kv8 twin and prefills through the flash
// attention.
inline runtime::KvCache parse_kv_cache_arg(const std::string& v) {
  return runtime::parse_kv_cache(v.c_str());
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
  const size_t weights = m.report.total() - m.report.rope_bytes;
  const size_t device = ctx.memory_bytes();
  const double gb = 1e9;
  uint32_t len = a.value;
  if (a.is_auto) {
    const uint32_t fit = runtime::max_len_that_fits(d, mtp, weights, device, reserve_bytes,
                                                    m.trained_max_len, path, kv, drafter);
    if (fit == 0)
      throw std::runtime_error(
          "--max-len auto: not even " + std::to_string(runtime::kMinAutoMaxLen) +
          " positions fit - " +
          runtime::describe(runtime::plan(d, runtime::kMinAutoMaxLen, mtp, weights, path, kv, drafter),
                            device, reserve_bytes));
    len = fit;
    loader::set_max_len(ctx, m, len);
    std::print(stderr,
               "max_len: auto -> {:d} (the largest multiple of {:d} that fits {:.3f} GB with a {:.3f} GB"
               " reserve; trained context {:d})\n",
               len, runtime::kMaxLenQuantum, device / gb, reserve_bytes / gb, m.trained_max_len);
  } else {
    const runtime::MemoryPlan p = runtime::plan(d, len, mtp, weights, path, kv, drafter);
    if (p.total() + reserve_bytes > device) {
      const uint32_t cap = m.trained_max_len != 0 ? m.trained_max_len : len;
      const uint32_t fit =
          cap < runtime::kMaxLenQuantum
              ? 0
              : runtime::max_len_that_fits(d, mtp, weights, device, reserve_bytes, cap, path,
                                           kv, drafter);
      throw std::runtime_error(
          "--max-len " + std::to_string(len) + " does not fit: " +
          runtime::describe(p, device, reserve_bytes) + ". The largest that fits is " +
          std::to_string(fit) + " (--max-len auto)");
    }
    std::print(stderr, "max_len: {:d} (--max-len)\n", len);
  }
  std::print(stderr, "{}\n",
             runtime::describe(runtime::plan(d, len, mtp, weights, path, kv, drafter), device,
                               reserve_bytes)
                 .c_str());
  return len;
}

// Step 4 under --max-num-seqs with the KV page pool (spec 31): a card's `seqs` > 1 sequences share the pages left
// beside their fixed state (runtime::pool_that_fits). --max-len auto lets one request use the whole pool (up to the
// trained context); an explicit --max-len L caps a request at L and is refused when the pool cannot hold one request
// of L. `m` is re-tabled at the pool's positions (runtime::Engine's max_len; every sequence counts from 0 up to the
// cap). Prints the pool's line and the plan's breakdown like settle().
inline runtime::PoolShape settle_pool(l0::Context& ctx, loader::LoadedModel& m, const MaxLenArg& a, uint32_t seqs,
                                      size_t reserve_bytes, const runtime::PrefillPath& path, runtime::KvCache kv,
                                      const runtime::DrafterPlan& drafter = {}) {
  const model::ModelDesc& d = *m.desc;
  const bool mtp = m.mtp != nullptr;
  const size_t weights = m.report.total() - m.report.rope_bytes;
  const size_t device = ctx.memory_bytes();
  const uint32_t limit = a.is_auto ? m.trained_max_len : a.value;   // check_before_load: auto has a trained context
  const std::string beside = " with --max-num-seqs " + std::to_string(seqs);
  const runtime::PoolShape s =
      runtime::pool_that_fits(d, seqs, mtp, weights, device, reserve_bytes, limit, path, kv, drafter);
  if (s.pages == 0) {
    const uint32_t least = std::min(runtime::kMinAutoMaxLen, limit / runtime::kMaxLenQuantum * runtime::kMaxLenQuantum);
    throw std::runtime_error(
        "not even " + std::to_string(least) + " positions of KV fit" + beside + " - " +
        runtime::describe(runtime::plan_pool(d, {seqs, runtime::kv_pages(least), least}, mtp, weights, path, kv, drafter),
                          device, reserve_bytes) +
        ". A smaller --max-num-seqs");
  }
  if (!a.is_auto && s.cap < a.value)
    throw std::runtime_error("--max-len " + std::to_string(a.value) + " does not fit" + beside + ": the KV left beside " +
                             std::to_string(seqs) + " requests' state holds " + std::to_string(s.positions()) +
                             " positions. --max-len auto, at most " + std::to_string(s.cap) +
                             ", or a smaller --max-num-seqs");
  std::print(stderr, "kv pool: {:d} pages of {:d} ({:d} tokens) shared by {:d} requests; one request up to {:d}{}\n",
             s.pages, runtime::kKvPage, s.positions(), seqs, s.cap, a.is_auto ? "" : " (--max-len)");
  loader::set_max_len(ctx, m, uint32_t(s.positions()), s.cap);   // the engine's max_len: the pool; a request <= the cap
  std::print(stderr, "{}\n",
             runtime::describe(runtime::plan_pool(d, s, mtp, weights, path, kv, drafter), device, reserve_bytes).c_str());
  return s;
}

// Plan 5d: the prefill chunk cap (2048 or 4096), in two steps around settle().
//
//   asked = runtime::prefill_chunk_cap_for(--prefill-chunk C, desc)   0 when no C was given;
//                                                                     refuses C > 4096
//   path.chunk_cap = planned_chunk_cap(asked)    what settle() plans max_len at
//   max_len = settle(..., path, ...)
//   cap = settle_chunk_cap(..., max_len, path, ..., asked)   what the engine is given
//
// An asked cap (b70-decode's --prefill-chunk) is planned and kept as it is: an explicit 4096
// shortens what auto gives the context. With nothing asked, every model takes ruling (i) of
// 2026-10-27 (context first; a MoE model: plan 2026-11-12 D2, the expert GEMMs and l0-int8's quantiser
// amortise over twice the rows; Qwen3.8: open-work 18, ~5.6 % faster cold,
// - max_len (and every request slot) is planned
// at 2048 and 4096 is taken only when the plan at those lengths has room for it
// (runtime::prefill_chunk_cap_that_fits, or runtime::plan_pool beside --max-num-seqs). Qwen3.8's
// hooked prefill keeps 2048-row chunks over a prompt's last 6144+ positions (runtime::kPrefillTail),
// so a prefix-cache hit after a turn still restarts at most 2047 rows back. There is no
// environment variable or server flag for it: the engine side chooses.
//
// The owner's ruling of 2026-11-24 reverses (i) for one case: --max-len auto on the bf16 prompt pass (l0) plans at
// 4096 (the four-argument form below) - Qwen3.8 one card, MTP on: max_len 157,184 at 2048 against 133,120 at 4096,
// prompts 4k 1,850 -> 1,974 t/s (+6.7 %), 30k 1,586 -> 1,685 (+6.2 %). The full context stays with l0-int8 (its int8
// cache), two cards, or an explicit --max-len N (4096 only when it fits beside N, as before). Ornith already ran
// 4096-row chunks at its auto context; Kolibri-1, Flash-Next and the two-card modes settle elsewhere, unchanged.
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
// The same for a one-card settle (one engine, its slots or its page pool; b70-serve, --parallel replica's cards,
// b70-decode): `a` and `backend` decide the 2026-11-24 rule above.
inline uint32_t planned_chunk_cap(uint32_t asked, const model::ModelDesc& d, const MaxLenArg& a,
                                  runtime::PrefillBackend backend) {
  if (asked != 0 || !a.is_auto || backend != runtime::PrefillBackend::L0) return planned_chunk_cap(asked, d);
  runtime::check_prefill_chunk_cap(d, runtime::PrefillScratchDims::kCMax);
  return runtime::PrefillScratchDims::kCMax;
}
// `max_len`: one request's, as settle() gave it (the pool's positions beside `pool`).
// Prints the card's VRAM chart (runtime/vram_bars.h) at the cap chosen; `card` of `cards` names it (--parallel replica
// settles each card on its own).
//
// Open-work 38, after the cap and the same way (context first): a MoE model's mixer linears are held in bf16
// (PrefillPath::dense_bf16, runtime::dense_slab_bytes) when the plan at these lengths and this cap has room for
// them, else every chunk unpacks them as before - bitwise the same prompt pass. Returns `path` at the cap chosen and
// with that choice: the engine is given both (set_prefill_chunk_cap, set_prefill_dense_bf16).
// `pool` (spec 31): the KV page pool settle_pool gave (--max-num-seqs N) - every plan here is then plan_pool's at it.
inline runtime::PrefillPath settle_chunk_cap(l0::Context& ctx, const loader::LoadedModel& m,
                                             uint32_t max_len, size_t reserve_bytes,
                                             const runtime::PrefillPath& path, runtime::KvCache kv,
                                             const runtime::DrafterPlan& drafter, uint32_t asked, uint32_t card = 0,
                                             uint32_t cards = 1, const runtime::PoolShape* pool = nullptr) {
  const size_t weights = m.report.total() - m.report.rope_bytes;
  const auto plan_at = [&](const runtime::PrefillPath& p) {
    return pool ? runtime::plan_pool(*m.desc, *pool, m.mtp != nullptr, weights, p, kv, drafter)
                : runtime::plan(*m.desc, max_len, m.mtp != nullptr, weights, p, kv, drafter);
  };
  const char* why = asked != 0 ? "--prefill-chunk" : nullptr;
  uint32_t cap = path.chunk_cap;
  if (path.prefill && why == nullptr) {
    if (!pool) {
      cap = runtime::prefill_chunk_cap_that_fits(*m.desc, m.mtp != nullptr, weights, ctx.memory_bytes(),
                                                 reserve_bytes, max_len, path, kv, drafter);
    } else {   // the sequences' plan, not one request's alone
      runtime::PrefillPath wide = path;
      wide.chunk_cap = runtime::PrefillScratchDims::kCMax;
      cap = plan_at(wide).total() + reserve_bytes <= ctx.memory_bytes()
                ? runtime::PrefillScratchDims::kCMax
                : runtime::PrefillScratchDims::kCDefault;
    }
    // planned_chunk_cap's 2026-11-24 rule: auto sized the context for 4096 (so it fits), and says so.
    const bool sized = path.chunk_cap == runtime::PrefillScratchDims::kCMax;
    why = sized ? "the context was sized for 4096-row chunks, the faster prompt pass; --prefill-backend l0-int8 "
                  "(the int8 cache) or two cards give the longer context"
          : cap == runtime::PrefillScratchDims::kCMax ? "the plan at this max_len has room for it"
                                                      : "4096 would not fit beside this max_len";
  }
  if (path.prefill && why) std::print(stderr, "prefill chunk cap: {:d} ({})\n", cap, why);   // the default says nothing
  runtime::PrefillPath at = path;
  at.chunk_cap = cap;
  if (const size_t dense = runtime::dense_slab_bytes(*m.desc); path.prefill && dense != 0) {
    runtime::PrefillPath with = at;
    with.dense_bf16 = true;
    at.dense_bf16 = plan_at(with).total() + reserve_bytes <= ctx.memory_bytes();
    std::print(stderr, "prefill dense weights: {} ({:.3f} GB in bf16 {})\n",
               at.dense_bf16 ? "kept in bf16" : "unpacked every chunk", dense / 1e9,
               at.dense_bf16 ? "fit beside this max_len" : "would not fit beside this max_len");
  }
  const runtime::MemoryPlan p = plan_at(at);
  std::print(stderr, "{}",
             runtime::vram_block({runtime::vram_parts(p, runtime::vram_state_name(p), "DFlash drafter"),
                                  reserve_bytes, ctx.memory_bytes()},
                                 card, cards, runtime::vram_utf8()));
  return at;
}

}  // namespace cli
