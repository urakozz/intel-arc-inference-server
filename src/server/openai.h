#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "server/deps.h"
#include "server/toolcall.h"

namespace server {

struct Request {
  bool chat = false;
  nlohmann::json messages;
  nlohmann::json tools;
  bool enable_thinking = true;
  // Spec 18d: chat_template_kwargs as sent (an object; enable_thinking included). Which of
  // them reach the template, and which the output parser reads, is the model's
  // (server/chat_format.h).
  nlohmann::json template_kwargs = nlohmann::json::object();
  std::string prompt;
  std::optional<uint32_t> max_tokens;
  uint32_t min_tokens = 0;
  bool ignore_eos = false;
  bool stream = false;
  bool include_usage = false;
  std::vector<std::string> stop;
  bool return_token_ids = false;
  Sampling sampling;
  std::string model;
};

struct BadRequest : std::runtime_error {
  using std::runtime_error::runtime_error;
};

Request parse_request(const std::string& body, bool chat);

std::string error_body(const std::string& message, const std::string& type);

struct Usage {
  uint32_t prompt_tokens = 0;
  uint32_t completion_tokens = 0;
  uint32_t cached_tokens = 0;  // chat: usage.prompt_tokens_details.cached_tokens (plan 7c)
};

std::string completion_body(const Request& r, const std::string& id, uint64_t created,
                            const std::string& text, const std::string& finish_reason, Usage u,
                            const std::vector<uint32_t>* prompt_ids,
                            const std::vector<uint32_t>* out_ids,
                            const ParsedOutput* parsed = nullptr);  // chat: reasoning, calls
std::string stream_frame_role(const Request& r, const std::string& id, uint64_t created);
std::string stream_frame_text(const Request& r, const std::string& id, uint64_t created,
                              const std::string& text);
std::string stream_frame_reasoning(const Request& r, const std::string& id, uint64_t created,
                                   const std::string& text);
std::string stream_frame_tool_call(const Request& r, const std::string& id, uint64_t created,
                                   const ToolCall& call, uint32_t index);
std::string stream_frame_finish(const Request& r, const std::string& id, uint64_t created,
                                const std::string& finish_reason);
std::string stream_frame_usage(const Request& r, const std::string& id, uint64_t created,
                               Usage u);
inline const char* kDone = "data: [DONE]\n\n";
std::string models_body(const std::string& served_name, uint64_t created);

}  // namespace server
