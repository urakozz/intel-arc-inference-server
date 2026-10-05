#pragma once

#include <memory>
#include <string>

#include <nlohmann/json.hpp>

namespace chat {

class Template {
 public:
  explicit Template(const std::string& snapshot_dir);
  ~Template();

  Template(const Template&) = delete;
  Template& operator=(const Template&) = delete;

  std::string render(const nlohmann::json& messages, const nlohmann::json& tools,
                     bool enable_thinking) const;
  // Spec 18d: `kwargs` (an object, or null) are the request's chat_template_kwargs - template
  // variables beside enable_thinking, e.g. K2-Horizon's tool_call_format / reasoning_effort.
  std::string render(const nlohmann::json& messages, const nlohmann::json& tools,
                     bool enable_thinking, const nlohmann::json& kwargs) const;
  const std::string& bos_token() const;
  const std::string& eos_token() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace chat
