#pragma once
#include <cstdint>
#include <string>
#include "common/json.h"

// Spec 6 §10: the context a checkpoint was trained for, config.json's
// `max_position_embeddings` - in `text_config` for the multimodal checkpoints
// (Qwen3.8, Agnes and Ornith all say 262144 there), at the top level otherwise. It is
// the hard upper bound of any max_len (the RoPE angles beyond it are positions the
// model never saw) and the cap `--max-len auto` plans under. Host only: no device.
namespace loader {

// 0 when neither place declares it (the caller decides: an explicit max_len is then
// unchecked, auto refuses). Throws std::runtime_error when it is declared but is not a
// positive integer that fits uint32_t.
uint32_t trained_context(const common::json::Value& config);
// The same, read from `snapshot_dir` + "config.json" (resolve_snapshot's trailing-slash
// form). Throws when the file cannot be read or parsed.
uint32_t trained_context(const std::string& snapshot_dir);

}  // namespace loader
