#pragma once

#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include <nlohmann/json.hpp>

namespace chat {

class Template {
 public:
  explicit Template(const std::string& snapshot_dir);
  // Spec 23e (b70-serve --chat-template FILE): the template's source from `template_file` instead of the
  // snapshot's chat_template.jinja / tokenizer_config.json string; BOS / EOS still from the snapshot's
  // tokenizer_config.json.
  Template(const std::string& snapshot_dir, const std::string& template_file);
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
  // Spec 23e: the sha256 (hex) of the template source as read (before any fallback / patch keyed on it) -
  // Bolt's serving prints it beside the published one (spec 25e).
  const std::string& source_sha256() const;
  // Spec 23e: the clock strftime_now reads (transformers' strftime_now: the host's local time at the render).
  // Default: the system clock at each render; a test fixes it.
  void set_clock(std::function<std::chrono::system_clock::time_point()> clock);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// SHA-256 of `bytes`, lower-case hex (FIPS 180-4).
std::string sha256_hex(const std::string& bytes);

}  // namespace chat
