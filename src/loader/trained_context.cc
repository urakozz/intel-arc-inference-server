#include "loader/trained_context.h"

#include <cmath>
#include <stdexcept>

#include "loader/snapshot.h"

namespace loader {

uint32_t trained_context(const common::json::Value& config) {
  if (!config.is_object()) throw std::runtime_error("config.json is not a JSON object");
  const common::json::Value* text = config.find("text_config");
  for (const common::json::Value* scope : {text, &config}) {
    if (scope == nullptr || !scope->is_object()) continue;
    const common::json::Value* v = scope->find("max_position_embeddings");
    if (v == nullptr) continue;
    const double n = v->is_number() ? v->num() : -1.0;
    if (!(n >= 1.0 && n <= 4294967295.0 && std::floor(n) == n))
      throw std::runtime_error(std::string("config.json ") +
                               (scope == text ? "text_config." : "") +
                               "max_position_embeddings is not a positive integer");
    return static_cast<uint32_t>(n);
  }
  return 0;
}

uint32_t trained_context(const std::string& snapshot_dir) { return trained_context(read_json(snapshot_dir + "config.json")); }

}  // namespace loader
