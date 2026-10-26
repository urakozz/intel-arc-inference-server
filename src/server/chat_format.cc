#include "server/chat_format.h"

#include <stdexcept>

#include "server/toolcall_bolt.h"
#include "server/toolcall_k2.h"
#include "server/toolcall_kolibri.h"
#include "server/toolcall_lfm2.h"

namespace server {
namespace {

using json = nlohmann::json;

bool ends_with(const std::string& s, const std::string& tail) {
  return s.size() >= tail.size() && s.compare(s.size() - tail.size(), tail.size(), tail) == 0;
}

// K2's template default (chat_template.jinja: tool_call_format | default('xml')).
K2CallFormat k2_call_format(const json& kwargs) {
  K2CallFormat format = K2CallFormat::Xml;
  if (!kwargs.is_object() || !kwargs.contains("tool_call_format")) return format;
  const json& v = kwargs.at("tool_call_format");
  if (!v.is_string() || !parse_k2_call_format(v.get<std::string>(), format))
    throw std::invalid_argument("chat_template_kwargs.tool_call_format must be one of xml, xml_typed, json");
  return format;
}

}  // namespace

ChatFormat ChatFormat::for_model_type(const std::string& model_type) {
  if (model_type == "k2_horizon") return k2_horizon();
  if (model_type == "kolibri1") return kolibri();
  if (model_type == "lfm2_moe") return lfm2();
  if (model_type == "nemotron_h_omni") return bolt();
  return ChatFormat{};
}

const char* ChatFormat::name() const {
  switch (kind) {
    case Kind::K2Horizon: return "k2_horizon";
    case Kind::Kolibri: return "kolibri1";
    case Kind::Lfm2: return "lfm2_moe";
    case Kind::Bolt: return "bolt";
    default: return "qwen";
  }
}

json template_messages(const ChatFormat& format, const json& messages) {
  if (!format.string_content() || !messages.is_array()) return messages;
  json out = messages;
  for (json& m : out) {
    if (!m.is_object() || !m.contains("content") || !m.at("content").is_array()) continue;
    std::string text;
    bool first = true;
    for (const json& part : m.at("content")) {
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
  if (format.kind == ChatFormat::Kind::Bolt && kwargs.is_object()) {
    // The template reads enable_thinking, medium_effort, truncate_history_thinking by truthiness; the output parser reads
    // force_nonempty_content as `is True`. A non-boolean is the client's error, not a silent truthy / falsy reading.
    for (const char* key : {"enable_thinking", "medium_effort", "truncate_history_thinking", "force_nonempty_content"})
      if (kwargs.contains(key) && !kwargs.at(key).is_boolean())
        throw std::invalid_argument(std::string("chat_template_kwargs.") + key + " must be a boolean");
    return;
  }
  if (format.kind == ChatFormat::Kind::Lfm2 && kwargs.is_object()) {
    // The template reads preserve_thinking | default(false); enable_thinking is accepted and ignored (decision 8).
    if (kwargs.contains("preserve_thinking") && !kwargs.at("preserve_thinking").is_boolean())
      throw std::invalid_argument("chat_template_kwargs.preserve_thinking must be a boolean");
    return;
  }
  if (format.kind == ChatFormat::Kind::Kolibri && kwargs.is_object()) {
    // The template's seven efforts (none disables thinking; minimal = low; xhigh, max = high).
    if (kwargs.contains("reasoning_effort")) {
      const json& v = kwargs.at("reasoning_effort");
      bool ok = v.is_null();
      for (const char* e : {"none", "minimal", "low", "medium", "high", "xhigh", "max"}) ok = ok || v == e;
      if (!ok)
        throw std::invalid_argument("chat_template_kwargs.reasoning_effort must be one of none, minimal, low, "
                                    "medium, high, xhigh, max");
    }
    if (kwargs.contains("preserve_thinking") && !kwargs.at("preserve_thinking").is_boolean())
      throw std::invalid_argument("chat_template_kwargs.preserve_thinking must be a boolean");
    return;
  }
  if (format.kind != ChatFormat::Kind::K2Horizon || !kwargs.is_object()) return;
  (void)k2_call_format(kwargs);
  if (kwargs.contains("reasoning_effort")) {
    const json& v = kwargs.at("reasoning_effort");
    if (!v.is_string() || (v != "high" && v != "medium" && v != "low"))
      throw std::invalid_argument("chat_template_kwargs.reasoning_effort must be one of high, medium, low");
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
                                  "high -> the default; spec 25 decision 13)");
    json out = kwargs.is_object() ? kwargs : json::object();
    if (off && !out.contains("enable_thinking")) out["enable_thinking"] = false;
    if (efficient && !out.contains("medium_effort")) out["medium_effort"] = true;
    return out;
  }
  return kwargs;
}

std::unique_ptr<OutputParser> make_output_parser(const ChatFormat& format, const std::string& prompt,
                                                 const json& kwargs, const json& tools) {
  if (format.kind == ChatFormat::Kind::K2Horizon) {
    // The generation prompt's think tag, by reasoning_effort (the template's last lines).
    std::string open;
    for (const char* tag : {"<ifm|think>", "<ifm|think_fast>", "<ifm|think_faster>"}) {
      if (ends_with(prompt, std::string(tag) + "\n")) open = tag;
    }
    return std::make_unique<K2OutputStream>(open, k2_call_format(kwargs), tools);
  }
  // Kolibri: the model opens its reasoning itself; the parser reads it from the output.
  if (format.kind == ChatFormat::Kind::Kolibri) return std::make_unique<KolibriOutputStream>(tools);
  // LFM2: the prompt ends in "<think>" and the output opens inside reasoning (spec 24e).
  if (format.kind == ChatFormat::Kind::Lfm2) return std::make_unique<Lfm2OutputStream>(tools, ends_with(prompt, "<think>"));
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
