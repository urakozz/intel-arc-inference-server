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

inline int plan(const std::string& path, MaxLenArg length, const std::string& spec) {
  if (spec != "auto" && spec != "off" && spec != "mtp")
    throw std::runtime_error("Inkling --plan accepts --spec auto, mtp or off");
  const loader::Inkling2Checkpoint checkpoint(path);
  const auto& d = checkpoint.desc();
  const auto& w = checkpoint.layout();
  const bool draft = spec != "off" && checkpoint.has_drafter();
  constexpr size_t capacity = 32'000'000'000ull;
  constexpr size_t reserve = size_t(runtime::kDefaultReserveGb * 1e9);
  if (length.is_auto)
    length.value = runtime::inkling2::max_len_that_fits(d, w, capacity, reserve, draft);
  if (length.value == 0)
    throw std::runtime_error("Inkling cannot fit the minimum context on one 32 GB card");
  const auto m = runtime::inkling2::sizes(d, w, length.value, draft).memory();
  if (m.total() > capacity - reserve)
    throw std::runtime_error("Inkling does not fit one 32 GB card at this --max-len. Use --max-len auto or a shorter length.");
  std::println("Inkling 2 Mini, max_len {}, drafter {}", length.value, draft ? "on" : "off");
  std::println("{}", runtime::format_memory("32 GB card estimate", m, capacity));
  return 0;
}

}  // namespace cli::inkling2
