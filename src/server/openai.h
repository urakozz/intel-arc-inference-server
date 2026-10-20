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
  // Spec 23e: the OpenAI top-level `reasoning_effort` as sent (null when absent). Which models read it, and how, is
  // the chat format's (Muse-Glimmer maps it onto chat_template_kwargs.reasoning_strength - spec 23 decision 3);
  // every other model ignores it, as before.
  nlohmann::json reasoning_effort;
};

struct BadRequest : std::runtime_error {
  using std::runtime_error::runtime_error;
};

Request parse_request(const std::string& body, bool chat);
// Spec 20e: with the model's sampling defaults - what a request that names no temperature / top_p /
// top_k samples with (b70-serve sets them for Kolibri-1 from generation_config.json; every other model
// keeps Sampling{}: greedy unless the request says otherwise, exactly as before). The two-argument form
// is this one with Sampling{}.
Request parse_request(const std::string& body, bool chat, const Sampling& defaults);
// generation_config.json's sampling fields as defaults: do_sample true -> sampled (greedy when it is
// false or absent, or temperature is 0), temperature / top_p / top_k / repetition_penalty where present, and for a field
// the file does not name Hugging Face's GenerationConfig default (temperature 1.0, top_k 50, top_p 1.0, penalty 1.0 -
// spec 24e; before it, Sampling{}'s top_k 20 / top_p 0.95, which no served model's file left to the default).
// Throws std::invalid_argument naming a field of the wrong type or range.
Sampling generation_sampling(const nlohmann::json& generation_config);
// Spec 24e: generation_config.json's `eos_token_id` - an integer or a non-empty list of integers (HF allows both); anything
// else throws std::invalid_argument naming the key. (b70-serve's eos_ids, which used to hold this, calls it.)
std::vector<uint32_t> generation_eos_ids(const nlohmann::json& generation_config);
// Spec 23e: generation_config.json's `stop_strings` - absent: none; a string, or a list of strings (Hugging Face
// allows both; openvino.genai's reader fixed the string form, d54d3768). Anything else throws std::invalid_argument
// naming the key. b70-serve merges them with every request's `stop` (Options::stop_strings).
std::vector<std::string> generation_stop_strings(const nlohmann::json& generation_config);

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
