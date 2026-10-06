#include "server/chat_format.h"

#include <stdexcept>

#include "server/toolcall_k2.h"
#include "server/toolcall_kolibri.h"

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
  return ChatFormat{};
}

const char* ChatFormat::name() const {
  switch (kind) {
    case Kind::K2Horizon: return "k2_horizon";
    case Kind::Kolibri: return "kolibri1";
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
  // Qwen: with thinking on, the prompt ends in "<think>\n" and the output opens with reasoning.
  return std::make_unique<OutputStream>(ends_with(prompt, "<think>\n"), tools);
}

}  // namespace server
