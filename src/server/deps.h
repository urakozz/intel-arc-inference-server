#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

namespace server {

struct StreamerIface {
  virtual ~StreamerIface() = default;
  virtual std::string push(uint32_t id) = 0;
  virtual std::string flush() = 0;
};

struct TokIface {
  virtual ~TokIface() = default;
  virtual std::vector<uint32_t> encode(std::string_view text) = 0;
  virtual std::string decode(const std::vector<uint32_t>& ids) = 0;
  virtual std::unique_ptr<StreamerIface> streamer() = 0;
  virtual uint32_t vocab_used() = 0;
};

struct TemplateIface {
  virtual ~TemplateIface() = default;
  virtual std::string render(const nlohmann::json& messages, const nlohmann::json& tools,
                             bool enable_thinking) = 0;
};

struct Sampling {
  bool greedy = true;
  float temperature = 1.0f;
  uint32_t top_k = 20;
  float top_p = 0.95f;
  uint64_t seed = 0;
  bool has_seed = false;
};

struct EngineIface {
  virtual ~EngineIface() = default;
  virtual void reset() = 0;
  virtual void prefill(const std::vector<uint32_t>& ids) = 0;
  virtual uint32_t step(const Sampling& s) = 0;
  virtual uint32_t max_len() = 0;
  virtual uint32_t pos() = 0;
};

struct Deps {
  TokIface& tok;
  TemplateIface& tmpl;
  EngineIface& engine;
};

}  // namespace server
