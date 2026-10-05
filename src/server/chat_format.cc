#include "server/chat_format.h"

#include <stdexcept>

#include "server/toolcall_k2.h"

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
  return model_type == "k2_horizon" ? k2_horizon() : ChatFormat{};
}

const char* ChatFormat::name() const { return kind == Kind::K2Horizon ? "k2_horizon" : "qwen"; }

void check_template_kwargs(const ChatFormat& format, const json& kwargs) {
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
  // Qwen: with thinking on, the prompt ends in "<think>\n" and the output opens with reasoning.
  return std::make_unique<OutputStream>(ends_with(prompt, "<think>\n"), tools);
}

}  // namespace server
