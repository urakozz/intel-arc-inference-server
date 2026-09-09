#include "server/openai.h"

#include <cmath>
#include <limits>
#include <utility>

namespace server {
namespace {

using json = nlohmann::json;

[[noreturn]] void invalid(const std::string& message) {
  throw BadRequest(message);
}

bool is_uint(const json& value) {
  if (value.is_number_unsigned()) return value.get<uint64_t>() <= UINT32_MAX;
  return value.is_number_integer() && value.get<int64_t>() >= 0 &&
         static_cast<uint64_t>(value.get<int64_t>()) <= UINT32_MAX;
}

uint32_t uint_field(const json& object, const char* name, bool allow_zero) {
  const json& value = object.at(name);
  if (!is_uint(value) || (!allow_zero && value.get<uint64_t>() == 0)) {
    invalid(std::string(name) + (allow_zero ? " must be a non-negative integer"
                                             : " must be a positive integer"));
  }
  return value.get<uint32_t>();
}

bool bool_field(const json& object, const char* name) {
  const json& value = object.at(name);
  if (!value.is_boolean()) invalid(std::string(name) + " must be a boolean");
  return value.get<bool>();
}

float probability_field(const json& object, const char* name, bool allow_zero) {
  const json& value = object.at(name);
  if (!value.is_number()) invalid(std::string(name) + " must be a number");
  const double number = value.get<double>();
  if (!std::isfinite(number) || (allow_zero ? number < 0.0 : number <= 0.0) || number > 1.0) {
    invalid(std::string(name) + " must be in (0, 1]");
  }
  return static_cast<float>(number);
}

float temperature_field(const json& object) {
  const json& value = object.at("temperature");
  if (!value.is_number()) invalid("temperature must be a number");
  const double number = value.get<double>();
  if (!std::isfinite(number) || number < 0.0) invalid("temperature must be non-negative");
  return static_cast<float>(number);
}

json base(const Request& r, const std::string& id, uint64_t created, const char* object) {
  return {
      {"id", id},
      {"object", object},
      {"created", created},
      {"model", r.model},
  };
}

std::string frame(json body) {
  return "data: " + body.dump() + "\n\n";
}

}  // namespace

Request parse_request(const std::string& body, bool chat) {
  json request;
  try {
    request = json::parse(body);
  } catch (const json::parse_error&) {
    invalid("invalid JSON body");
  }
  if (!request.is_object()) invalid("request body must be an object");

  Request out;
  out.chat = chat;
  if (request.contains("model")) {
    if (!request.at("model").is_string()) invalid("model must be a string");
    out.model = request.at("model").get<std::string>();
  }

  if (request.contains("n")) {
    if (!is_uint(request.at("n")) || request.at("n").get<uint32_t>() != 1) {
      invalid("n must be 1");
    }
  }

  if (chat) {
    if (!request.contains("messages")) invalid("missing messages");
    const json& messages = request.at("messages");
    if (!messages.is_array()) invalid("messages must be an array");
    if (messages.empty()) invalid("messages must not be empty");
    for (const json& message : messages) {
      if (!message.is_object()) invalid("messages entries must be objects");
      if (!message.contains("role") || !message.at("role").is_string()) {
        invalid("message role must be a string");
      }
      if (!message.contains("content") || !message.at("content").is_string()) {
        invalid("content must be a string");
      }
    }
    out.messages = messages;
    out.tools = request.contains("tools") ? request.at("tools") : json(nullptr);
    if (request.contains("chat_template_kwargs")) {
      const json& kwargs = request.at("chat_template_kwargs");
      if (!kwargs.is_object()) invalid("chat_template_kwargs must be an object");
      if (kwargs.contains("enable_thinking")) {
        out.enable_thinking = bool_field(kwargs, "enable_thinking");
      }
    }
  } else {
    if (!request.contains("prompt")) invalid("missing prompt");
    if (!request.at("prompt").is_string()) invalid("prompt must be a string");
    out.prompt = request.at("prompt").get<std::string>();
  }

  if (request.contains("max_tokens")) out.max_tokens = uint_field(request, "max_tokens", false);
  if (request.contains("min_tokens")) out.min_tokens = uint_field(request, "min_tokens", true);
  if (request.contains("ignore_eos")) out.ignore_eos = bool_field(request, "ignore_eos");
  if (request.contains("stream")) out.stream = bool_field(request, "stream");
  if (request.contains("return_token_ids")) {
    out.return_token_ids = bool_field(request, "return_token_ids");
  }

  if (request.contains("stream_options")) {
    const json& stream_options = request.at("stream_options");
    if (!stream_options.is_object()) invalid("stream_options must be an object");
    if (stream_options.contains("include_usage")) {
      out.include_usage = bool_field(stream_options, "include_usage");
    }
  }

  if (request.contains("stop")) {
    const json& stop = request.at("stop");
    if (stop.is_string()) {
      out.stop.push_back(stop.get<std::string>());
    } else if (stop.is_array()) {
      if (stop.size() > 4) invalid("stop must contain at most 4 strings");
      for (const json& value : stop) {
        if (!value.is_string()) invalid("stop entries must be strings");
        out.stop.push_back(value.get<std::string>());
      }
    } else {
      invalid("stop must be a string or array of strings");
    }
  }

  if (request.contains("temperature")) {
    out.sampling.temperature = temperature_field(request);
    out.sampling.greedy = out.sampling.temperature == 0.0f;
  }
  if (request.contains("top_k")) out.sampling.top_k = uint_field(request, "top_k", true);
  if (request.contains("top_p")) out.sampling.top_p = probability_field(request, "top_p", false);
  if (request.contains("seed")) {
    const json& seed = request.at("seed");
    if (!seed.is_number_unsigned() && !(seed.is_number_integer() && seed.get<int64_t>() >= 0)) {
      invalid("seed must be a non-negative integer");
    }
    out.sampling.seed = seed.get<uint64_t>();
    out.sampling.has_seed = true;
  }
  return out;
}

std::string error_body(const std::string& message, const std::string& type) {
  return json{{"error", {{"message", message}, {"type", type}, {"param", nullptr}, {"code", nullptr}}}}
      .dump();
}

std::string completion_body(const Request& r, const std::string& id, uint64_t created,
                            const std::string& text, const std::string& finish_reason, Usage u,
                            const std::vector<uint32_t>* prompt_ids,
                            const std::vector<uint32_t>* out_ids) {
  json body = base(r, id, created, r.chat ? "chat.completion" : "text_completion");
  json choice = {{"index", 0}, {"finish_reason", finish_reason}};
  if (r.chat) {
    choice["message"] = {{"role", "assistant"}, {"content", text}};
  } else {
    choice["text"] = text;
    choice["logprobs"] = nullptr;
  }
  if (r.return_token_ids && out_ids != nullptr) choice["token_ids"] = *out_ids;
  body["choices"] = json::array({std::move(choice)});
  body["usage"] = {{"prompt_tokens", u.prompt_tokens},
                   {"completion_tokens", u.completion_tokens},
                   {"total_tokens", u.prompt_tokens + u.completion_tokens}};
  if (r.return_token_ids && prompt_ids != nullptr) body["prompt_token_ids"] = *prompt_ids;
  return body.dump();
}

std::string stream_frame_role(const Request& r, const std::string& id, uint64_t created) {
  if (!r.chat) return "";
  json body = base(r, id, created, "chat.completion.chunk");
  body["choices"] = json::array({{{"index", 0}, {"delta", {{"role", "assistant"}}},
                                   {"finish_reason", nullptr}}});
  return frame(std::move(body));
}

std::string stream_frame_text(const Request& r, const std::string& id, uint64_t created,
                              const std::string& text) {
  json body = base(r, id, created, r.chat ? "chat.completion.chunk" : "text_completion");
  if (r.chat) {
    body["choices"] = json::array(
        {{{"index", 0}, {"delta", {{"content", text}}}, {"finish_reason", nullptr}}});
  } else {
    body["choices"] = json::array(
        {{{"index", 0}, {"text", text}, {"logprobs", nullptr}, {"finish_reason", nullptr}}});
  }
  return frame(std::move(body));
}

std::string stream_frame_finish(const Request& r, const std::string& id, uint64_t created,
                                const std::string& finish_reason) {
  json body = base(r, id, created, r.chat ? "chat.completion.chunk" : "text_completion");
  if (r.chat) {
    body["choices"] = json::array(
        {{{"index", 0}, {"delta", json::object()}, {"finish_reason", finish_reason}}});
  } else {
    body["choices"] = json::array(
        {{{"index", 0}, {"text", ""}, {"logprobs", nullptr}, {"finish_reason", finish_reason}}});
  }
  return frame(std::move(body));
}

std::string stream_frame_usage(const Request& r, const std::string& id, uint64_t created, Usage u) {
  json body = base(r, id, created, r.chat ? "chat.completion.chunk" : "text_completion");
  body["choices"] = json::array();
  body["usage"] = {{"prompt_tokens", u.prompt_tokens},
                   {"completion_tokens", u.completion_tokens},
                   {"total_tokens", u.prompt_tokens + u.completion_tokens}};
  return frame(std::move(body));
}

std::string models_body(const std::string& served_name, uint64_t created) {
  return json{{"object", "list"},
              {"data", json::array({{{"id", served_name},
                                      {"object", "model"},
                                      {"created", created},
                                      {"owned_by", "b70"}}})}}
      .dump();
}

}  // namespace server
