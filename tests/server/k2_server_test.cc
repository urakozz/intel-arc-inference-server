// Spec 18d: the server's K2-Horizon dispatch over a mock engine (no device, no checkpoint).
// The chat-format half of the wiring b70-serve uses for K2 (the engine half - K2Engine behind
// cli::k2::K2EngineAdapterT - is k2_serve_test's):
// Options::chat_format = ChatFormat::for_model_type("k2_horizon") makes
//   - the request's chat_template_kwargs reach the template (render_with_kwargs), checked
//     first (a bad tool_call_format / reasoning_effort is a 400);
//   - the output parsed as K2's: reasoning up to </ifm|think...> by the prompt's think tag,
//     tool calls in the request's tool_call_format, finish_reason tool_calls;
//   - both EOS ids [1, 250019] stop generation (Review Focus 5);
//   - streamed frames (one byte per token) carrying the same reasoning, content and calls as
//     the non-streamed body;
// and the default (Qwen) format still renders without kwargs and leaves K2's tags alone.
#include <string>
#include <vector>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "check.h"
#include "server/mock.h"
#include "server/server.h"

namespace {

using json = nlohmann::json;

// K2's template in the shape the server sees: the generation prompt ends in the think tag
// that reasoning_effort picks. Records what it was called with.
struct K2Template : server::TemplateIface {
  int plain_calls = 0, kwargs_calls = 0;
  json last_kwargs;
  std::string render(const json& messages, const json&, bool) override {
    ++plain_calls;
    return body(messages) + "<|ifm|im_start|>assistant\n<ifm|think>\n";
  }
  std::string render_with_kwargs(const json& messages, const json&, bool, const json& kwargs) override {
    ++kwargs_calls;
    last_kwargs = kwargs;
    const std::string effort = kwargs.value("reasoning_effort", std::string("high"));
    const std::string tag = effort == "low" ? "<ifm|think_faster>" : effort == "medium" ? "<ifm|think_fast>" : "<ifm|think>";
    return body(messages) + "<|ifm|im_start|>assistant\n" + tag + "\n";
  }
  static std::string body(const json& messages) {
    std::string text = "<|ifm|begin_of_text|>";
    for (const auto& m : messages) {
      text += "<|ifm|im_start|>" + m.at("role").get<std::string>() + "\n" + m.at("content").get<std::string>() +
              "<|ifm|im_end|>";
    }
    return text;
  }
};

struct Fixture {
  explicit Fixture(bool k2 = true) : options(make_options(k2)), server({tok, tmpl, engine}, options) {
    CHECK(server.start());
  }
  ~Fixture() { server.stop(); }
  static server::Options make_options(bool k2) {
    server::Options o;
    o.host = "127.0.0.1";
    o.port = 0;
    o.served_model = "k2";
    o.eos_ids = {1, 250019};   // generation_config.json's
    if (k2) o.chat_format = server::ChatFormat::for_model_type("k2_horizon");
    return o;
  }
  httplib::Client client() { return httplib::Client("127.0.0.1", server.bound_port()); }
  // The engine generates `text` one byte per token, then `eos`.
  void script(const std::string& text, uint32_t eos) {
    engine.script = tok.raw(text, 1);
    engine.script.push_back(eos);
  }

  MockTok tok;
  K2Template tmpl;
  MockEngine engine;
  server::Options options;
  server::Server server;
};

const json kTools = json::parse(R"([{"type":"function","function":{"name":"read","parameters":{"type":"object",
  "properties":{"filePath":{"type":"string"},"offset":{"type":"integer"}},"required":["filePath"]}}}])");

json body(const json& kwargs = json::object(), bool stream = false) {
  json b = {{"model", "k2"},
            {"messages", json::array({{{"role", "user"}, {"content", "Open a.cc"}}})},
            {"tools", kTools},
            {"stream", stream}};
  if (!kwargs.empty()) b["chat_template_kwargs"] = kwargs;
  return b;
}

const std::string kXmlOutput =
    "I should read it.</ifm|think>Reading.\n<ifm|tool_calls>\n<ifm|tool_call>read\n<ifm|arg_key>filePath"
    "</ifm|arg_key>\n<ifm|arg_value>/w/a.cc</ifm|arg_value>\n<ifm|arg_key>offset</ifm|arg_key>\n"
    "<ifm|arg_value>12</ifm|arg_value>\n</ifm|tool_call>\n</ifm|tool_calls>";

json post(Fixture& f, const json& b, int want_status = 200) {
  auto client = f.client();
  const auto response = client.Post("/v1/chat/completions", b.dump(), "application/json");
  CHECK(response);
  if (response->status != want_status) std::fprintf(stderr, "status %d: %s\n", response->status, response->body.c_str());
  CHECK_EQ(response->status, want_status);
  return json::parse(response->body);
}

void case_xml_call() {
  Fixture f;
  f.script(kXmlOutput, 250019);
  const json r = post(f, body());
  CHECK_EQ(f.tmpl.kwargs_calls, 1);
  CHECK_EQ(f.tmpl.plain_calls, 0);
  const json& choice = r.at("choices").at(0);
  CHECK_EQ(choice.at("finish_reason"), std::string("tool_calls"));
  const json& m = choice.at("message");
  CHECK_EQ(m.at("reasoning_content"), std::string("I should read it."));
  CHECK_EQ(m.at("content"), std::string("Reading.\n"));
  CHECK_EQ(m.at("tool_calls").size(), 1U);
  CHECK_EQ(m.at("tool_calls").at(0).at("function").at("name"), std::string("read"));
  CHECK_EQ(json::parse(m.at("tool_calls").at(0).at("function").at("arguments").get<std::string>()),
           json({{"filePath", "/w/a.cc"}, {"offset", 12}}));
  CHECK_EQ(r.at("usage").at("completion_tokens"), kXmlOutput.size());
}

void case_eos_one() {
  // <|ifm|endoftext|> (1) stops as <|ifm|im_end|> (250019) does.
  Fixture f;
  f.script("Done.</ifm|think>Paris.", 1);
  const json r = post(f, body());
  const json& choice = r.at("choices").at(0);
  CHECK_EQ(choice.at("finish_reason"), std::string("stop"));
  CHECK_EQ(choice.at("message").at("content"), std::string("Paris."));
  CHECK_EQ(choice.at("message").at("reasoning_content"), std::string("Done."));
}

void case_kwargs() {
  // tool_call_format json and reasoning_effort low reach the template; the parser follows.
  Fixture f;
  f.script("short</ifm|think_faster><ifm|tool_calls>\n<ifm|tool_call>{\"name\": \"read\", \"arguments\": "
           "{\"filePath\": \"/b\"}}</ifm|tool_call>\n</ifm|tool_calls>",
           250019);
  const json kwargs = {{"tool_call_format", "json"}, {"reasoning_effort", "low"}};
  const json r = post(f, body(kwargs));
  CHECK_EQ(f.tmpl.last_kwargs, kwargs);
  const json& m = r.at("choices").at(0).at("message");
  CHECK_EQ(m.at("reasoning_content"), std::string("short"));
  CHECK(m.at("content").is_null());
  CHECK_EQ(json::parse(m.at("tool_calls").at(0).at("function").at("arguments").get<std::string>()),
           json({{"filePath", "/b"}}));
  // Bad values are the client's: 400, nothing generated.
  Fixture g;
  const json bad = post(g, body({{"tool_call_format", "yaml"}}), 400);
  CHECK(bad.at("error").at("message").get<std::string>().find("tool_call_format") != std::string::npos);
  post(g, body({{"reasoning_effort", "max"}}), 400);
  CHECK_EQ(g.tmpl.kwargs_calls, 0);
}

void case_stream() {
  Fixture f;
  f.script(kXmlOutput, 250019);
  auto client = f.client();
  std::string sse;
  const auto response = client.Post("/v1/chat/completions", httplib::Headers(), body({}, true).dump(),
                                    "application/json", [&sse](const char* d, size_t n) {
                                      sse.append(d, n);
                                      return true;
                                    },
                                    nullptr);
  CHECK(response);
  CHECK_EQ(response->status, 200);
  std::string reasoning, content, finish;
  json calls = json::array();
  for (size_t at = 0, end; (end = sse.find("\n\n", at)) != std::string::npos; at = end + 2) {
    const std::string frame = sse.substr(at, end - at);
    if (frame == "data: [DONE]") continue;
    const json b = json::parse(frame.substr(6));
    const json& c = b.at("choices").at(0);
    const json& d = c.at("delta");
    if (d.contains("reasoning_content")) reasoning += d.at("reasoning_content").get<std::string>();
    if (d.contains("content")) content += d.at("content").get<std::string>();
    if (d.contains("tool_calls")) calls.push_back(d.at("tool_calls").at(0));
    if (!c.at("finish_reason").is_null()) finish = c.at("finish_reason");
  }
  CHECK_EQ(reasoning, std::string("I should read it."));
  CHECK_EQ(content, std::string("Reading.\n"));
  CHECK_EQ(calls.size(), 1U);
  CHECK_EQ(calls.at(0).at("index"), 0);
  CHECK_EQ(calls.at(0).at("function").at("name"), std::string("read"));
  CHECK_EQ(finish, std::string("tool_calls"));
}

void case_qwen_default() {
  // The default format: no kwargs reach the template, K2's tags are plain text, and
  // 250019 is just an id - only the configured EOS set stops (here it holds 250019).
  Fixture f(/*k2=*/false);
  f.script(kXmlOutput, 250019);
  const json r = post(f, body({{"tool_call_format", "json"}}));
  CHECK_EQ(f.tmpl.plain_calls, 1);
  CHECK_EQ(f.tmpl.kwargs_calls, 0);
  const json& m = r.at("choices").at(0).at("message");
  CHECK(!m.contains("tool_calls"));
  CHECK_EQ(m.at("content"), kXmlOutput);
}

}  // namespace

int main() {
  case_xml_call();
  case_eos_one();
  case_kwargs();
  case_stream();
  case_qwen_default();
  std::printf("k2_server_test OK: 5 cases\n");
  return 0;
}
