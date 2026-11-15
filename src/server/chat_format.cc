#include "server/chat_format.h"

#include <stdexcept>

#include "server/json_struct.h"
#include "server/toolcall_bolt.h"
#include "server/toolcall_kolibri.h"

namespace server {
namespace {

using json = nlohmann::json;

bool ends_with(const std::string& s, const std::string& tail) {
  return s.size() >= tail.size() && s.compare(s.size() - tail.size(), tail.size(), tail) == 0;
}

// Bolt's template reads enable_thinking, medium_effort, truncate_history_thinking by truthiness; the output parser reads
// force_nonempty_content as `is True`. A non-boolean is the client's error, not a silent truthy / falsy reading.
struct BoltKwargs {
  std::optional<bool> enable_thinking, medium_effort, truncate_history_thinking, force_nonempty_content;
};

// Kolibri's template: its seven efforts (none disables thinking; minimal = low; xhigh, max = high), or null.
bool kolibri_effort(const std::string& e) {
  for (const char* k : {"none", "minimal", "low", "medium", "high", "xhigh", "max"})
    if (e == k) return true;
  return false;
}
struct KolibriKwargs {
  [[=json_struct::null_is_absent, =json_struct::Says("must be one of none, minimal, low, medium, high, xhigh, max"),
    =json_struct::Holds<std::string>(kolibri_effort)]]
  std::optional<std::string> reasoning_effort;
  std::optional<bool> preserve_thinking;
};

}  // namespace

ChatFormat ChatFormat::for_model_type(const std::string& model_type) {
  if (model_type == "kolibri1") return kolibri();
  if (model_type == "nemotron_h_omni") return bolt();
  return ChatFormat{};
}

const char* ChatFormat::name() const {
  switch (kind) {
    case Kind::Kolibri: return "kolibri1";
    case Kind::Bolt: return "bolt";
    default: return "qwen";
  }
}

nlohmann::ordered_json template_messages(const ChatFormat& format, const nlohmann::ordered_json& messages) {
  if (!format.string_content() || !messages.is_array()) return messages;
  nlohmann::ordered_json out = messages;
  for (nlohmann::ordered_json& m : out) {
    if (!m.is_object() || !m.contains("content") || !m.at("content").is_array()) continue;
    std::string text;
    bool first = true;
    for (const nlohmann::ordered_json& part : m.at("content")) {
      if (!part.is_object() || !part.contains("text") || !part.at("text").is_string()) continue;
      if (!first) text += "\n";
      text += part.at("text").get<std::string>();
      first = false;
    }
    m["content"] = text;
  }
  return out;
}

void check_template_kwargs(const ChatFormat& format, const json& kwargs) {
  const auto fail = [](const std::string& what) {
    throw std::invalid_argument("chat_template_kwargs." + what);
  };
  if (format.kind == ChatFormat::Kind::Bolt && kwargs.is_object()) {
    BoltKwargs k;
    json_struct::read(kwargs, k, fail);
  } else if (format.kind == ChatFormat::Kind::Kolibri && kwargs.is_object()) {
    KolibriKwargs k;
    json_struct::read(kwargs, k, fail);
  }
}

json request_template_kwargs(const ChatFormat& format, const json& kwargs, const json& reasoning_effort) {
  if (format.kind == ChatFormat::Kind::Bolt && !reasoning_effort.is_null()) {
    // Spec 25 decision 13 (OpenAI's levels onto the template's two switches); an explicit kwarg wins.
    const bool off = reasoning_effort == "none" || reasoning_effort == "minimal";
    const bool efficient = reasoning_effort == "low" || reasoning_effort == "medium";
    const bool full = reasoning_effort == "high" || reasoning_effort == "xhigh";
    if (!off && !efficient && !full)
      throw std::invalid_argument("reasoning_effort must be one of none, minimal, low, medium, high, xhigh (Nemotron 3.5 "
                                  "Bolt: none / minimal -> enable_thinking false, low / medium -> medium_effort true, "
                                  "high -> the default)");
    json out = kwargs.is_object() ? kwargs : json::object();
    if (off && !out.contains("enable_thinking")) out["enable_thinking"] = false;
    if (efficient && !out.contains("medium_effort")) out["medium_effort"] = true;
    return out;
  }
  if (format.kind == ChatFormat::Kind::Kolibri && !reasoning_effort.is_null()) {
    // Kolibri's template reads `reasoning_effort` itself (vLLM hands the top-level field to the template as this kwarg);
    // an explicit chat_template_kwargs.reasoning_effort wins.
    bool ok = false;
    for (const char* e : {"none", "minimal", "low", "medium", "high", "xhigh", "max"}) ok = ok || reasoning_effort == e;
    if (!ok)
      throw std::invalid_argument("reasoning_effort must be one of none, minimal, low, medium, high, xhigh, max (Kolibri-1)");
    json out = kwargs.is_object() ? kwargs : json::object();
    if (!out.contains("reasoning_effort")) out["reasoning_effort"] = reasoning_effort;
    return out;
  }
  return kwargs;
}

std::unique_ptr<OutputParser> make_output_parser(const ChatFormat& format, const std::string& prompt,
                                                 const json& kwargs, const json& tools) {
  // Kolibri: the model opens its reasoning itself; the parser reads it from the output.
  if (format.kind == ChatFormat::Kind::Kolibri) return std::make_unique<KolibriOutputStream>(tools);
  // Bolt: the prompt's end says whether the output opens inside reasoning (the ultra_v3 rules, spec 25e).
  if (format.kind == ChatFormat::Kind::Bolt) {
    const bool force = kwargs.is_object() && kwargs.contains("force_nonempty_content") &&
                       kwargs.at("force_nonempty_content").is_boolean() && kwargs.at("force_nonempty_content").get<bool>();
    return std::make_unique<BoltOutputStream>(ends_with(prompt, "<think>\n"), force, tools);
  }
  // Qwen: with thinking on, the prompt ends in "<think>\n" and the output opens with reasoning.
  return std::make_unique<OutputStream>(ends_with(prompt, "<think>\n"), tools);
}

}  // namespace server
