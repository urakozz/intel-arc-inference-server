#pragma once
#include <string>

#include "common/json.h"

namespace loader {
// Resolves a model argument to a snapshot directory (with trailing '/').
// Accepts an absolute/relative directory containing config.json, or an HF
// repo id "<org>/<name>" resolved against the local cache
// ($HF_HOME or ~/.cache/huggingface, + /hub/models--<org>--<name>, revision
// from refs/main). Never downloads; throws naming the exact path on failure.
std::string resolve_snapshot(const std::string& arg);

// A JSON file (a snapshot's config.json and the like) parsed whole; throws naming the path when it cannot be read.
common::json::Value read_json(const std::string& path);
// config.json's "model_type", or "" when it has none.
std::string model_type_of(const common::json::Value& cfg);
}  // namespace loader
