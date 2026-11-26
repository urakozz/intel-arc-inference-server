#pragma once
#include <print>
#include "cli/max_len.h"
#include "runtime/inkling2/inkling2_sizes.h"

namespace cli::inkling2 {

inline bool is_inkling2(const std::string& path) {
  try {
    const auto dir = loader::resolve_snapshot(path);
    return model::is_inkling2_model_type(loader::model_type_of(loader::read_json(dir + "config.json")));
  } catch (const std::exception&) {
    return false;   // the chosen loader reports a bad path or config
  }
}

// The refusal for a --max-len one card cannot hold: what fits, and with the bf16 cache whether the int8 one would.
inline std::string no_fit(const model::Inkling2Desc& d, const loader::Inkling2WeightLayout& w, size_t capacity,
                          size_t reserve, bool draft, uint32_t chunk, uint32_t length, bool kv8) {
  const uint32_t bf16 = runtime::inkling2::max_len_that_fits(d, w, capacity, reserve, draft, chunk, false);
  const uint32_t int8 = runtime::inkling2::max_len_that_fits(d, w, capacity, reserve, draft, chunk, true);
  std::string msg = "Inkling does not fit one card at --max-len " + std::to_string(length) + " with the " +
                    (kv8 ? "int8" : "bf16") + " KV cache. ";
  if (!kv8 && int8 >= length)
    return msg + "Add --kv-cache int8 (it holds up to " + std::to_string(int8) + " positions) or use --max-len " +
           std::to_string(bf16) + " or less.";
  return msg + "Use --max-len auto or at most " + std::to_string(kv8 ? int8 : bf16) + ".";
}

inline int plan(const std::string& path, MaxLenArg length, const std::string& spec, bool kv8) {
  if (spec != "auto" && spec != "off" && spec != "mtp")
    throw std::runtime_error("Inkling --plan accepts --spec auto, mtp or off");
  const loader::Inkling2Checkpoint checkpoint(path);
  const auto& d = checkpoint.desc();
  const auto& w = checkpoint.layout();
  const bool draft = spec != "off" && checkpoint.has_drafter();
  constexpr size_t capacity = 32'000'000'000ull;
  constexpr size_t reserve = size_t(runtime::kDefaultReserveGb * 1e9);
  if (length.is_auto)
    length.value = runtime::inkling2::max_len_that_fits(d, w, capacity, reserve, draft, 0, kv8);
  if (length.value == 0)
    throw std::runtime_error("Inkling cannot fit the minimum context on one 32 GB card");
  const auto m = runtime::inkling2::sizes(d, w, length.value, draft, 0, kv8).memory();
  if (m.total() > capacity - reserve) throw std::runtime_error(no_fit(d, w, capacity, reserve, draft, 0, length.value, kv8));
  std::println("Inkling 2 Mini, max_len {}, drafter {}, KV cache {}", length.value, draft ? "on" : "off",
               kv8 ? "int8 (the full-attention visits)" : "bf16");
  std::println("{}", runtime::format_memory("32 GB card estimate", m, capacity));
  return 0;
}

}  // namespace cli::inkling2
