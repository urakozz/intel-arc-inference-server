#include "server/chat_format.h"

#include <stdexcept>

#include "server/json_struct.h"
#include "server/toolcall_inkling2.h"
#include "server/toolcall_kolibri.h"

namespace server {
namespace {

using json = nlohmann::json;

bool ends_with(const std::string& s, const std::string& tail) {
  return s.size() >= tail.size() && s.compare(s.size() - tail.size(), tail.size(), tail) == 0;
}

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

// Inkling's template: its seven effort names, or a number in [0, 0.99] (it raises outside), or null.
bool inkling_effort(const json& e) {
  if (e.is_null()) return true;
  if (e.is_number()) return e.get<double>() >= 0.0 && e.get<double>() <= 0.99;
  return e.is_string() && kolibri_effort(e.get<std::string>());
}
const char* const kInklingEfforts = "must be one of none, minimal, low, medium, high, xhigh, max, or a number from 0 to 0.99";

}  // namespace

ChatFormat ChatFormat::for_model_type(const std::string& model_type) {
  if (model_type == "kolibri1") return kolibri();
  if (model_type == "inkling2_loop") return inkling();
  return ChatFormat{};
}

const char* ChatFormat::name() const {
  switch (kind) {
    case Kind::Kolibri: return "kolibri1";
    case Kind::Inkling: return "inkling2";
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
  if (format.kind == ChatFormat::Kind::Kolibri && kwargs.is_object()) {
    KolibriKwargs k;
    json_struct::read(kwargs, k, fail);
  }
  if (format.kind == ChatFormat::Kind::Inkling && kwargs.is_object() && kwargs.contains("reasoning_effort") &&
      !inkling_effort(kwargs.at("reasoning_effort")))
    fail(std::string("reasoning_effort ") + kInklingEfforts);
}

json request_template_kwargs(const ChatFormat& format, const json& kwargs, const json& reasoning_effort, bool thinking) {
  if (format.kind == ChatFormat::Kind::Inkling) {
    if (!inkling_effort(reasoning_effort))
      throw std::invalid_argument(std::string("reasoning_effort ") + kInklingEfforts + " (Inkling 2)");
    json out = kwargs.is_object() ? kwargs : json::object();
    const bool off = !thinking || (out.contains("enable_thinking") && out.at("enable_thinking") == false);
    if (!out.contains("reasoning_effort") || out.at("reasoning_effort").is_null()) {
      if (!reasoning_effort.is_null()) out["reasoning_effort"] = reasoning_effort;
      else if (off) out["reasoning_effort"] = "none";
    }
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
                                                 const json& /*kwargs*/, const json& tools) {
  // Kolibri: the model opens its reasoning itself; the parser reads it from the output.
  if (format.kind == ChatFormat::Kind::Kolibri) return std::make_unique<KolibriOutputStream>(tools);
  // Inkling: the model opens its blocks; the parser reads them from the output.
  if (format.kind == ChatFormat::Kind::Inkling) return std::make_unique<Inkling2OutputStream>(tools);
  // Qwen: with thinking on, the prompt ends in "<think>\n" and the output opens with reasoning.
  return std::make_unique<OutputStream>(ends_with(prompt, "<think>\n"), tools);
}

}  // namespace server
