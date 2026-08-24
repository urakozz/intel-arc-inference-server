#pragma once
#include <string>

namespace loader {
// Resolves a model argument to a snapshot directory (with trailing '/').
// Accepts an absolute/relative directory containing config.json, or an HF
// repo id "<org>/<name>" resolved against the local cache
// ($HF_HOME or ~/.cache/huggingface, + /hub/models--<org>--<name>, revision
// from refs/main). Never downloads; throws naming the exact path on failure.
std::string resolve_snapshot(const std::string& arg);
}  // namespace loader
