#pragma once
// Spec 16d: the split and the max_len of a two-card run - b70-decode --pp 2
// (cli/pipeline_decode.h, spec 16b / 16c, where this lived inline) and b70-serve --pp 2 (spec
// 16d) - over the LOADED model's bytes (runtime::pp_weights) and each device's memory:
//
//   --max-len auto    the largest multiple of 256 both devices hold with the reserve each, at
//                     the auto split (both auto: the split whose length is largest) or at
//                     --pipeline-split N; B70_DECODE_ATTN=v1 then takes its largest compiled
//                     length at or below it;
//   --max-len N       the auto split at N (or --pipeline-split N), refused with the plan's
//                     breakdown and the largest length that fits when a device cannot hold it.
//
// `x` (runtime::PpExtras): spec 16d's terms - the MTP head on device 1 (its weights, the
// embedding replica, its buffers, its prefill rows) and the prefix cache's block shadows.
// PpExtras{} is spec 16b / 16c's plan exactly (b70-decode without --mtp). Prints the lines
// b70-decode always printed ("max_len: ...", "split: ...", the plan).
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

#include "cli/max_len.h"
#include "cli/pipeline_args.h"
#include "loader/loader.h"
#include "runtime/pipeline_plan.h"

namespace cli {

struct PpSettled {
  uint32_t split = 0, max_len = 0;
};

// What a loaded model adds to the plan (spec 16d): its MTP head and draft vocabulary, and
// whether a prefix cache's block hook will run (b70-serve with the cache on).
inline runtime::PpExtras pp_extras(const loader::LoadedModel& full, bool hook) {
  runtime::PpExtras x;
  x.mtp = full.mtp != nullptr;
  if (full.draft_vocab) {
    x.draft_vocab = full.draft_vocab->size();
    x.draft_vocab_int8 = full.draft_vocab->head.kind == model::WeightKind::Int8;
  }
  x.hook = hook;
  return x;
}

inline PpSettled pp_settle(const model::ModelDesc& d, const runtime::PpWeights& w,
                           const std::array<size_t, runtime::kPpDevices>& dev, const MaxLenArg& max_len,
                           uint32_t trained, size_t reserve, runtime::KvCache kv,
                           const runtime::PrefillPath& pf, const runtime::PpExtras& x,
                           const PipelineArgs& pipe) {
  const double gb = 1e9;
  uint32_t split = pipe.split, len = max_len.value;
  if (!pipe.split_auto) runtime::require_split(d, split);

  if (max_len.is_auto) {
    const uint32_t cap = trained;
    if (pipe.split_auto) {
      const runtime::PpChoice c = runtime::pp_auto_split_and_len(d, w, dev, reserve, cap, kv, pf, x);
      split = c.split;
      len = c.max_len;
    } else {
      len = runtime::pp_max_len_that_fits(d, split, w, dev, reserve, cap, kv, pf, x);
    }
    if (len == 0) {
      const uint32_t at = std::min(runtime::kMinAutoMaxLen, cap);
      const uint32_t s = split != 0 ? split : runtime::pp_auto_split(d, w, at, kv, pf, x);
      throw std::runtime_error("--max-len auto: not even " + std::to_string(at) +
                               " positions fit on both devices - " +
                               runtime::pp_describe(runtime::pp_plan(d, s, at, w, kv, pf, x), dev, reserve));
    }
    const uint32_t fit = len;
    if (runtime::decode_attn() == runtime::DecodeAttn::V1) {
      len = v1_compiled_at_most(fit, d);
      if (len == 0)
        throw std::runtime_error("--max-len auto with B70_DECODE_ATTN=v1: no compiled v1 decode"
                                 " attention at or below " + std::to_string(fit));
    }
    std::fprintf(stderr,
                 "max_len: auto -> %u (the largest multiple of %u that fits BOTH devices, %.3f and"
                 " %.3f GB, with a %.3f GB reserve each; trained context %u%s)\n",
                 len, runtime::kMaxLenQuantum, dev[0] / gb, dev[1] / gb, reserve / gb, cap,
                 len != fit ? ", v1 decode attention's largest compiled length" : "");
  } else {
    if (pipe.split_auto) split = runtime::pp_auto_split(d, w, len, kv, pf, x);
    const runtime::PpPlan p = runtime::pp_plan(d, split, len, w, kv, pf, x);
    for (uint32_t i = 0; i < runtime::kPpDevices; ++i)
      if (p.dev[i].total() + reserve > dev[i]) {
        const uint32_t cap = trained != 0 ? trained : len;
        const uint32_t best =
            cap < runtime::kMaxLenQuantum
                ? 0
                : runtime::pp_max_len_that_fits(d, split, w, dev, reserve, cap, kv, pf, x);
        throw std::runtime_error("--max-len " + std::to_string(len) + " does not fit on device " +
                                 std::to_string(i) + ": " + runtime::pp_describe(p, dev, reserve) +
                                 ". The largest that fits at split " + std::to_string(split) +
                                 " is " + std::to_string(best) +
                                 " (--max-len auto); --mem-reserve-gb lowers the reserve");
      }
    std::fprintf(stderr, "max_len: %u (--max-len)\n", len);
  }
  if (pipe.split_auto)
    std::fprintf(stderr, "split: auto -> %u (the split whose heavier device holds the fewest bytes"
                 " at max_len %u)\n", split, len);
  else
    std::fprintf(stderr, "split: %u (--pipeline-split)\n", split);
  std::fprintf(stderr, "%s\n",
               runtime::pp_describe(runtime::pp_plan(d, split, len, w, kv, pf, x), dev, reserve).c_str());
  return {split, len};
}

}  // namespace cli
