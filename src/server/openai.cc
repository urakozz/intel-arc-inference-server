#include "server/openai.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <iterator>
#include <limits>
#include <utility>

#include "server/json_struct.h"

namespace server {
namespace {

using json = nlohmann::json;
using ojson = nlohmann::ordered_json;   // messages and tools: the body's key order

[[noreturn]] void invalid(const std::string& message) {
  throw BadRequest(message);
}

bool is_uint(const json& value) {
  if (value.is_number_unsigned()) return value.get<uint64_t>() <= UINT32_MAX;
  return value.is_number_integer() && value.get<int64_t>() >= 0 &&
         static_cast<uint64_t>(value.get<int64_t>()) <= UINT32_MAX;
}

// The request's fields as OpenAI clients send them, read by server/json_struct.h - Head, then (chat) the messages
// and ChatFields, then Tail, each in member order: the first fault is the one refused.
bool is_one(const uint32_t& n) { return n == 1; }
bool positive(const uint32_t& n) { return n > 0; }
bool at_most_4(const std::vector<std::string>& stop) { return stop.size() <= 4; }
bool non_negative(const double& t) { return std::isfinite(t) && t >= 0.0; }
bool in_unit(const double& p) { return std::isfinite(p) && p > 0.0 && p <= 1.0; }
// vLLM's rule for repetition_penalty: finite and > 0, no upper bound (a float).
bool penalty(const double& r) { return std::isfinite(r) && r > 0.0 && r <= FLT_MAX; }

using json_struct::Holds;
using json_struct::Says;

struct Head {
  std::optional<std::string> model;
  [[=Says("must be 1"), =Holds<uint32_t>(is_one)]] std::optional<uint32_t> n;
};
struct ChatTemplateKwargs {
  std::optional<bool> enable_thinking;
};
struct ChatFields {
  std::optional<ChatTemplateKwargs> chat_template_kwargs;
  json reasoning_effort;
};
struct StreamOptions {
  std::optional<bool> include_usage;
};
struct Tail {
  [[=Says("must be a positive integer"), =Holds<uint32_t>(positive)]] std::optional<uint32_t> max_tokens;
  std::optional<uint32_t> min_tokens;
  std::optional<bool> ignore_eos, stream, return_token_ids;
  std::optional<StreamOptions> stream_options;
  [[=Holds<std::vector<std::string>>(at_most_4, "must contain at most 4 strings")]]
  std::optional<std::vector<std::string>> stop;
  [[=Holds<double>(non_negative, "must be non-negative")]] std::optional<double> temperature;
  std::optional<uint32_t> top_k;
  [[=Holds<double>(in_unit, "must be in (0, 1]")]] std::optional<double> top_p;
  // vLLM's extension field (spec 24e): Hugging Face's multiplicative penalty over the prompt and the output; 1.0 is none.
  // Null is the same as absent. The check applies to every model's request (an engine that does not apply the penalty
  // still refuses a malformed value).
  [[=json_struct::null_is_absent, =Says("must be a finite number > 0"), =Holds<double>(penalty)]]
  std::optional<double> repetition_penalty;
  std::optional<uint64_t> seed;
};

// The response bodies and stream chunks (written with their keys sorted, json_struct::write).
struct Function {
  std::string name, arguments;
};
struct ToolCallOut {
  std::string id, type = "function";
  Function function{};
  std::optional<uint32_t> index{};   // stream chunks
};
struct Message {
  std::string role = "assistant";
  [[=json_struct::nullable]] std::optional<std::string> content{};   // null beside tool calls
  std::optional<std::string> reasoning_content{};
  std::optional<std::vector<ToolCallOut>> tool_calls{};
};
struct ChunkDelta {
  std::optional<std::string> role{}, content{}, reasoning_content{};
  std::optional<std::vector<ToolCallOut>> tool_calls{};
};
struct Choice {
  uint32_t index = 0;
  [[=json_struct::nullable]] std::optional<std::string> finish_reason{};   // null until the last chunk
  std::optional<Message> message{};          // chat completion
  std::optional<ChunkDelta> delta{};         // chat chunk
  std::optional<std::string> text{};         // completions
  std::optional<std::nullptr_t> logprobs{};  // completions: null
  std::optional<std::vector<uint32_t>> token_ids{};
};
struct CachedTokens {
  uint32_t cached_tokens;
};
struct UsageOut {
  uint32_t prompt_tokens, completion_tokens, total_tokens;
  std::optional<CachedTokens> prompt_tokens_details{};   // chat (plan 7c)
};
struct Body {
  std::string id, object;
  uint64_t created;
  std::string model;
  std::vector<Choice> choices{};
  std::optional<UsageOut> usage{};
  std::optional<std::vector<uint32_t>> prompt_token_ids{};
};

Body base(const Request& r, const std::string& id, uint64_t created, const char* object) {
  return {.id = id, .object = object, .created = created, .model = r.model};
}

UsageOut usage_out(const Request& r, Usage u) {
  UsageOut usage{u.prompt_tokens, u.completion_tokens, u.prompt_tokens + u.completion_tokens};
  if (r.chat) usage.prompt_tokens_details = CachedTokens{u.cached_tokens};
  return usage;
}

// The chunk's one choice: chat's delta, or the completion's text.
Choice chunk_choice(const Request& r, ChunkDelta delta, const std::string& text) {
  Choice c;
  if (r.chat) {
    c.delta = std::move(delta);
  } else {
    c.text = text;
    c.logprobs = nullptr;
  }
  return c;
}

ToolCallOut tool_call_out(const ToolCall& call) {
  // (A string argument cut inside a character by the model's output must not throw out of the response body: replace.)
  return {.id = call.id,
          .function = {call.name, call.arguments.dump(-1, ' ', false, json::error_handler_t::replace)}};
}

// Messages as OpenAI clients send them: content may be null (an assistant turn
// holding only tool calls) or a list of text parts; tool-call arguments arrive as a
// JSON string and the chat template iterates them as an object (parsed in its own key order, as vLLM's json.loads).
ojson normalize_message(const ojson& message) {
  ojson out = message;
  const ojson& content = message.at("content");
  if (content.is_array()) {
    for (const ojson& part : content) {
      // Spec 23e: an image / video / audio part is refused by name - no served model loads its encoder (spec 23 §10's
      // rule). A text-bearing part of another type (input_text) stays text.
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
    for (ojson& call : out.at("tool_calls")) {
      if (!call.is_object() || !call.contains("function") || !call.at("function").is_object()) continue;
      ojson& fn = call.at("function");
      if (fn.contains("arguments") && fn.at("arguments").is_string()) {
        const std::string text = fn.at("arguments").get<std::string>();
        ojson parsed = text.empty() ? ojson::object() : ojson::parse(text, nullptr, false);
        if (parsed.is_discarded() || !parsed.is_object()) {
          // Spec 23e (plan 23e Review Focus 5): name the call - a template that iterates the arguments (it may raise
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

std::string frame(const Body& body) {
  return "data: " + json_struct::write(body).dump() + "\n\n";
}

}  // namespace

Request parse_request(const std::string& body, bool chat) { return parse_request(body, chat, Sampling{}); }

Sampling generation_sampling(const json& g) {
  // Spec 24e: a field the file does not name takes Hugging Face's GenerationConfig default (temperature 1.0, top_k 50,
  // top_p 1.0, repetition_penalty 1.0), not Sampling{}'s request defaults (top_k 20, top_p 0.95). Kolibri-1's file
  // names all of temperature, top_p and top_k, so their resolved sampling is as before.
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
    // HF accepts any strictly positive float (no upper bound), as does vLLM's request field; so does this.
    const json& v = g.at("repetition_penalty");
    if (!v.is_number() || !std::isfinite(v.get<double>()) || v.get<double>() <= 0.0 || v.get<double>() > FLT_MAX)
      throw std::invalid_argument("generation_config.json repetition_penalty is not a finite number > 0");
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
  // Parsed in the body's key order; messages and tools are taken out as they are (the template prints that order).
  ojson request;
  try {
    request = ojson::parse(body);
  } catch (const json::parse_error&) {
    invalid("invalid JSON body");
  }
  if (!request.is_object()) invalid("request body must be an object");

  Request out;
  out.chat = chat;
  out.sampling = defaults;
  out.sampling.has_seed = false;
  out.sampling.seed = 0;
  Head head;
  json_struct::read(request, head, invalid);
  out.model = head.model.value_or("");

  if (chat) {
    if (!request.contains("messages")) invalid("missing messages");
    const ojson& messages = request.at("messages");
    if (!messages.is_array()) invalid("messages must be an array");
    if (messages.empty()) invalid("messages must not be empty");
    for (const ojson& message : messages) {
      if (!message.is_object()) invalid("messages entries must be objects");
      if (!message.contains("role") || !message.at("role").is_string()) {
        invalid("message role must be a string");
      }
      if (!message.contains("content")) invalid("content must be a string");
    }
    out.messages = ojson::array();
    for (const ojson& message : messages) out.messages.push_back(normalize_message(message));
    out.tools = request.contains("tools") ? std::move(request.at("tools")) : ojson(nullptr);
    ChatFields fields;
    json_struct::read(request, fields, invalid);
    if (fields.chat_template_kwargs) {
      out.enable_thinking = fields.chat_template_kwargs->enable_thinking.value_or(out.enable_thinking);
      out.template_kwargs = request.at("chat_template_kwargs");
    }
    out.reasoning_effort = std::move(fields.reasoning_effort);
  } else {
    if (!request.contains("prompt")) invalid("missing prompt");
    if (!request.at("prompt").is_string()) invalid("prompt must be a string");
    out.prompt = request.at("prompt").get<std::string>();
  }

  Tail tail;
  json_struct::read(request, tail, invalid);
  out.max_tokens = tail.max_tokens;
  out.min_tokens = tail.min_tokens.value_or(out.min_tokens);
  out.ignore_eos = tail.ignore_eos.value_or(out.ignore_eos);
  out.stream = tail.stream.value_or(out.stream);
  out.return_token_ids = tail.return_token_ids.value_or(out.return_token_ids);
  if (tail.stream_options) out.include_usage = tail.stream_options->include_usage.value_or(out.include_usage);
  out.stop = tail.stop.value_or(out.stop);
  if (tail.temperature) {
    out.sampling.temperature = static_cast<float>(*tail.temperature);
    out.sampling.greedy = out.sampling.temperature == 0.0f;
  }
  out.sampling.top_k = tail.top_k.value_or(out.sampling.top_k);
  if (tail.top_p) out.sampling.top_p = static_cast<float>(*tail.top_p);
  if (tail.repetition_penalty) out.sampling.repetition_penalty = static_cast<float>(*tail.repetition_penalty);
  if (tail.seed) {
    out.sampling.seed = *tail.seed;
    out.sampling.has_seed = true;
  }
  return out;
}

std::string error_body(const std::string& message, const std::string& type) {
  struct Error {
    std::string message, type;
    std::nullptr_t param, code;
  };
  struct ErrorBody {
    Error error;
  };
  return json_struct::write(ErrorBody{{message, type, nullptr, nullptr}}).dump();
}

std::string completion_body(const Request& r, const std::string& id, uint64_t created,
                            const std::string& text, const std::string& finish_reason, Usage u,
                            const std::vector<uint32_t>* prompt_ids,
                            const std::vector<uint32_t>* out_ids,
                            const ParsedOutput* parsed) {
  Body body = base(r, id, created, r.chat ? "chat.completion" : "text_completion");
  Choice choice{.finish_reason = finish_reason};
  if (r.chat && parsed != nullptr) {
    Message message{.content = parsed->content};
    if (parsed->content.empty() && !parsed->tool_calls.empty()) message.content.reset();
    if (!parsed->reasoning.empty()) message.reasoning_content = parsed->reasoning;
    if (!parsed->tool_calls.empty()) {
      message.tool_calls.emplace();
      for (const ToolCall& call : parsed->tool_calls) message.tool_calls->push_back(tool_call_out(call));
    }
    choice.message = std::move(message);
  } else if (r.chat) {
    choice.message = Message{.content = text};
  } else {
    choice.text = text;
    choice.logprobs = nullptr;
  }
  if (r.return_token_ids && out_ids != nullptr) choice.token_ids = *out_ids;
  body.choices.push_back(std::move(choice));
  body.usage = usage_out(r, u);
  if (r.return_token_ids && prompt_ids != nullptr) body.prompt_token_ids = *prompt_ids;
  return json_struct::write(body).dump();
}

std::string stream_frame_role(const Request& r, const std::string& id, uint64_t created) {
  if (!r.chat) return "";
  Body body = base(r, id, created, "chat.completion.chunk");
  body.choices.push_back(chunk_choice(r, ChunkDelta{.role = "assistant"}, ""));
  return frame(body);
}

std::string stream_frame_text(const Request& r, const std::string& id, uint64_t created,
                              const std::string& text) {
  Body body = base(r, id, created, r.chat ? "chat.completion.chunk" : "text_completion");
  body.choices.push_back(chunk_choice(r, ChunkDelta{.content = text}, text));
  return frame(body);
}

std::string stream_frame_reasoning(const Request& r, const std::string& id, uint64_t created,
                                   const std::string& text) {
  Body body = base(r, id, created, "chat.completion.chunk");
  body.choices.push_back(Choice{.delta = ChunkDelta{.reasoning_content = text}});
  return frame(body);
}

std::string stream_frame_tool_call(const Request& r, const std::string& id, uint64_t created,
                                   const ToolCall& call, uint32_t index) {
  Body body = base(r, id, created, "chat.completion.chunk");
  ToolCallOut entry = tool_call_out(call);
  entry.index = index;
  body.choices.push_back(Choice{.delta = ChunkDelta{.tool_calls = std::vector<ToolCallOut>{std::move(entry)}}});
  return frame(body);
}

std::string stream_frame_finish(const Request& r, const std::string& id, uint64_t created,
                                const std::string& finish_reason) {
  Body body = base(r, id, created, r.chat ? "chat.completion.chunk" : "text_completion");
  body.choices.push_back(chunk_choice(r, ChunkDelta{}, ""));
  body.choices[0].finish_reason = finish_reason;
  return frame(body);
}

std::string stream_frame_usage(const Request& r, const std::string& id, uint64_t created, Usage u) {
  Body body = base(r, id, created, r.chat ? "chat.completion.chunk" : "text_completion");
  body.usage = usage_out(r, u);
  return frame(body);
}

std::string models_body(const std::string& served_name, uint64_t created) {
  struct Model {
    std::string id, object;
    uint64_t created;
    std::string owned_by;
  };
  struct Models {
    std::string object;
    std::vector<Model> data;
  };
  return json_struct::write(Models{"list", {{served_name, "model", created, "b70"}}}).dump();
}

}  // namespace server
