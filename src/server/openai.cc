#include "server/openai.h"

#include <algorithm>
#include <cmath>
#include <iterator>
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

json usage_json(const Request& r, Usage u) {
  json usage = {{"prompt_tokens", u.prompt_tokens},
                {"completion_tokens", u.completion_tokens},
                {"total_tokens", u.prompt_tokens + u.completion_tokens}};
  if (r.chat) usage["prompt_tokens_details"] = {{"cached_tokens", u.cached_tokens}};
  return usage;
}

json tool_call_json(const ToolCall& call) {
  // (A string argument cut inside a character by the model's output must not throw out of the response body: replace.)
  return {{"id", call.id},
          {"type", "function"},
          {"function", {{"name", call.name},
                        {"arguments", call.arguments.dump(-1, ' ', false, json::error_handler_t::replace)}}}};
}

// Messages as OpenAI clients send them: content may be null (an assistant turn
// holding only tool calls) or a list of text parts; tool-call arguments arrive as a
// JSON string and the chat template iterates them as an object.
json normalize_message(const json& message) {
  json out = message;
  const json& content = message.at("content");
  if (content.is_array()) {
    for (const json& part : content) {
      // Spec 23e: an image / video / audio part is refused by name - no served model loads its encoder (Muse-Glimmer's
      // vision tower is out of scope, spec 23 §10). A text-bearing part of another type (input_text) stays text.
      static const char* const kMedia[] = {"image", "image_url", "input_image", "video", "video_url", "input_video",
                                           "audio", "input_audio", "file", "input_file"};
      const bool media = part.is_object() && part.contains("type") && part.at("type").is_string() &&
                         std::any_of(std::begin(kMedia), std::end(kMedia),
                                     [&](const char* m) { return part.at("type") == m; });
      if (media) {
        invalid("content part of type '" + part.at("type").get<std::string>() +
                "' is refused: this server serves text only - the vision encoder (images, video) and any audio "
                "encoder are not loaded");
      }
      if (!part.is_object() || !part.contains("text") || !part.at("text").is_string()) {
        invalid("content parts must be text parts");
      }
    }
  } else if (!content.is_string() && !content.is_null()) {
    invalid("content must be a string");
  }
  if (message.contains("tool_calls") && message.at("tool_calls").is_array()) {
    for (json& call : out.at("tool_calls")) {
      if (!call.is_object() || !call.contains("function") || !call.at("function").is_object()) continue;
      json& fn = call.at("function");
      if (fn.contains("arguments") && fn.at("arguments").is_string()) {
        const std::string text = fn.at("arguments").get<std::string>();
        json parsed = text.empty() ? json::object() : json::parse(text, nullptr, false);
        if (parsed.is_discarded() || !parsed.is_object()) {
          // Spec 23e (plan 23e Review Focus 5): name the call - a template that iterates the arguments (Muse's raises
          // on a string) must never see one, and the client must see which call it sent wrong.
          const std::string name = fn.contains("name") && fn.at("name").is_string() ? fn.at("name").get<std::string>() : "";
          const std::string id = call.contains("id") && call.at("id").is_string() ? call.at("id").get<std::string>() : "";
          invalid("tool call arguments must be a JSON object (or a JSON string holding one): the call '" + name +
                  "' (id '" + id + "')");
        }
        fn["arguments"] = std::move(parsed);
      }
    }
  }
  return out;
}

std::string frame(json body) {
  return "data: " + body.dump() + "\n\n";
}

}  // namespace

Request parse_request(const std::string& body, bool chat) { return parse_request(body, chat, Sampling{}); }

Sampling generation_sampling(const json& g) {
  // Spec 24e: a field the file does not name takes Hugging Face's GenerationConfig default (temperature 1.0, top_k 50,
  // top_p 1.0, repetition_penalty 1.0), not Sampling{}'s request defaults (top_k 20, top_p 0.95). Kolibri-1's and
  // Muse-Glimmer's files name all of temperature, top_p and top_k, so their resolved sampling is as before.
  Sampling s;
  s.temperature = 1.0f;
  s.top_k = 50;
  s.top_p = 1.0f;
  s.repetition_penalty = 1.0f;
  if (!g.is_object()) throw std::invalid_argument("generation_config.json is not an object");
  const auto number = [&](const char* key, double lo, double hi) {
    const json& v = g.at(key);
    if (!v.is_number() || !std::isfinite(v.get<double>()) || v.get<double>() < lo || v.get<double>() > hi)
      throw std::invalid_argument(std::string("generation_config.json ") + key + " is not a number in [" +
                                  std::to_string(lo) + ", " + std::to_string(hi) + "]");
    return v.get<double>();
  };
  if (g.contains("temperature")) s.temperature = static_cast<float>(number("temperature", 0.0, 1e6));
  if (g.contains("top_p")) {
    s.top_p = static_cast<float>(number("top_p", 0.0, 1.0));
    if (s.top_p <= 0.0f) throw std::invalid_argument("generation_config.json top_p must be in (0, 1]");
  }
  if (g.contains("top_k")) {
    const json& v = g.at("top_k");
    if (!is_uint(v)) throw std::invalid_argument("generation_config.json top_k is not a non-negative integer");
    s.top_k = v.get<uint32_t>();
  }
  if (g.contains("repetition_penalty")) {
    // HF accepts any strictly positive float; the server's request field is held to (0, 2] (vLLM's range), and so is this.
    const json& v = g.at("repetition_penalty");
    if (!v.is_number() || !std::isfinite(v.get<double>()) || v.get<double>() <= 0.0 || v.get<double>() > 2.0)
      throw std::invalid_argument("generation_config.json repetition_penalty is not a number in (0, 2]");
    s.repetition_penalty = static_cast<float>(v.get<double>());
  }
  bool sample = false;
  if (g.contains("do_sample")) {
    if (!g.at("do_sample").is_boolean()) throw std::invalid_argument("generation_config.json do_sample is not a boolean");
    sample = g.at("do_sample").get<bool>();
  }
  s.greedy = !sample || s.temperature == 0.0f;
  return s;
}

std::vector<uint32_t> generation_eos_ids(const json& g) {
  if (!g.is_object() || !g.contains("eos_token_id"))
    throw std::invalid_argument("generation_config.json has no eos_token_id");
  const auto add = [](const json& value, std::vector<uint32_t>& ids) {
    if (!value.is_number_unsigned() && !value.is_number_integer())
      throw std::invalid_argument("generation_config.json eos_token_id is not an integer");
    const int64_t id = value.get<int64_t>();
    if (id < 0 || static_cast<uint64_t>(id) > 0xFFFFFFFFull)
      throw std::invalid_argument("generation_config.json eos_token_id is out of range");
    ids.push_back(static_cast<uint32_t>(id));
  };
  std::vector<uint32_t> ids;
  const json& value = g.at("eos_token_id");
  if (value.is_array()) {
    for (const json& id : value) add(id, ids);
  } else {
    add(value, ids);
  }
  if (ids.empty()) throw std::invalid_argument("generation_config.json eos_token_id is empty");
  return ids;
}

std::vector<std::string> generation_stop_strings(const json& g) {
  std::vector<std::string> out;
  if (!g.is_object() || !g.contains("stop_strings") || g.at("stop_strings").is_null()) return out;
  const json& v = g.at("stop_strings");
  if (v.is_string()) {
    out.push_back(v.get<std::string>());
    return out;
  }
  if (!v.is_array())
    throw std::invalid_argument("generation_config.json stop_strings is neither a string nor a list of strings");
  for (const json& s : v) {
    if (!s.is_string())
      throw std::invalid_argument("generation_config.json stop_strings holds a non-string entry: " + s.dump());
    out.push_back(s.get<std::string>());
  }
  return out;
}

Request parse_request(const std::string& body, bool chat, const Sampling& defaults) {
  json request;
  try {
    request = json::parse(body);
  } catch (const json::parse_error&) {
    invalid("invalid JSON body");
  }
  if (!request.is_object()) invalid("request body must be an object");

  Request out;
  out.chat = chat;
  out.sampling = defaults;
  out.sampling.has_seed = false;
  out.sampling.seed = 0;
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
      if (!message.contains("content")) invalid("content must be a string");
    }
    out.messages = json::array();
    for (const json& message : messages) out.messages.push_back(normalize_message(message));
    out.tools = request.contains("tools") ? request.at("tools") : json(nullptr);
    if (request.contains("chat_template_kwargs")) {
      const json& kwargs = request.at("chat_template_kwargs");
      if (!kwargs.is_object()) invalid("chat_template_kwargs must be an object");
      if (kwargs.contains("enable_thinking")) {
        out.enable_thinking = bool_field(kwargs, "enable_thinking");
      }
      out.template_kwargs = kwargs;
    }
    if (request.contains("reasoning_effort")) out.reasoning_effort = request.at("reasoning_effort");
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
  if (request.contains("repetition_penalty")) {
    // vLLM's extension field (spec 24e): Hugging Face's multiplicative penalty over the prompt and the output; 1.0 is none.
    const json& v = request.at("repetition_penalty");
    if (!v.is_number() || !std::isfinite(v.get<double>()) || v.get<double>() <= 0.0 || v.get<double>() > 2.0)
      invalid("repetition_penalty must be a number in (0, 2]");
    out.sampling.repetition_penalty = static_cast<float>(v.get<double>());
  }
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
                            const std::vector<uint32_t>* out_ids,
                            const ParsedOutput* parsed) {
  json body = base(r, id, created, r.chat ? "chat.completion" : "text_completion");
  json choice = {{"index", 0}, {"finish_reason", finish_reason}};
  if (r.chat && parsed != nullptr) {
    json message = {{"role", "assistant"}, {"content", parsed->content}};
    if (parsed->content.empty() && !parsed->tool_calls.empty()) message["content"] = nullptr;
    if (!parsed->reasoning.empty()) message["reasoning_content"] = parsed->reasoning;
    if (!parsed->tool_calls.empty()) {
      json calls = json::array();
      for (const ToolCall& call : parsed->tool_calls) calls.push_back(tool_call_json(call));
      message["tool_calls"] = std::move(calls);
    }
    choice["message"] = std::move(message);
  } else if (r.chat) {
    choice["message"] = {{"role", "assistant"}, {"content", text}};
  } else {
    choice["text"] = text;
    choice["logprobs"] = nullptr;
  }
  if (r.return_token_ids && out_ids != nullptr) choice["token_ids"] = *out_ids;
  body["choices"] = json::array({std::move(choice)});
  body["usage"] = usage_json(r, u);
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

std::string stream_frame_reasoning(const Request& r, const std::string& id, uint64_t created,
                                   const std::string& text) {
  json body = base(r, id, created, "chat.completion.chunk");
  body["choices"] = json::array(
      {{{"index", 0}, {"delta", {{"reasoning_content", text}}}, {"finish_reason", nullptr}}});
  return frame(std::move(body));
}

std::string stream_frame_tool_call(const Request& r, const std::string& id, uint64_t created,
                                   const ToolCall& call, uint32_t index) {
  json body = base(r, id, created, "chat.completion.chunk");
  json entry = tool_call_json(call);
  entry["index"] = index;
  body["choices"] = json::array(
      {{{"index", 0}, {"delta", {{"tool_calls", json::array({std::move(entry)})}}},
        {"finish_reason", nullptr}}});
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
  body["usage"] = usage_json(r, u);
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
