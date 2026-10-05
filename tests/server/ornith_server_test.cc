// Spec 15e Task 1, host only: the server over Ornith 1.5's chat template - the real
// renderer (chat::Template on the checkpoint's chat_template.jinja + tokenizer_config.json,
// vendored at tests/tokenizer/ornith/) behind the OpenAI endpoint, a scripted engine and
// the mock tokenizer. What b70-serve does for Ornith that it did not do for Qwen3.8 is
// the template; everything after the prompt (the tool-call parser, reasoning, EOS) is the
// same code, and this pins that the two meet:
//
//   1. a tools request with thinking off: the prompt the engine is given is exactly
//      transformers' render of the same request (tests/tokenizer/ornith_template_tools.txt,
//      tools/tokenizer/dump_agnes_template.py), and an Ornith-format tool call in the
//      output (`<tool_call>\n<function=...>\n<parameter=...>`, the format the template
//      advertises) comes back as an OpenAI tool call, finish_reason "tool_calls" - buffered
//      and streamed (SSE) in 5-byte pieces;
//   2. a history with an OpenAI-form tool call (arguments a JSON string) and two tool
//      results, thinking on: the prompt equals transformers' render of the same messages
//      (ornith_template_tool_response.txt, rendered from the object form the server
//      normalises to) and the output's reasoning and answer split at </think>.
//
// usage: ornith_server_test <tests/tokenizer>   (exit 77 if the vendored files are absent)
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "check.h"
#include "server/mock.h"
#include "server/server.h"
#include "tokenizer/chat_template.h"

namespace {

using json = nlohmann::json;

std::string slurp(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  CHECK(file.good());
  std::stringstream s;
  s << file.rdbuf();
  return s.str();
}

struct OrnithTemplate : server::TemplateIface {
  explicit OrnithTemplate(const std::string& dir) : tmpl(dir) {}
  std::string render(const json& messages, const json& tools, bool think) override {
    return tmpl.render(messages, tools, think);
  }
  chat::Template tmpl;
};

struct Fixture {
  explicit Fixture(const std::string& snap) : tmpl(snap), server({tok, tmpl, engine}, options()) {
    CHECK(server.start());
  }
  ~Fixture() { server.stop(); }
  static server::Options options() {
    server::Options o;
    o.host = "127.0.0.1";
    o.port = 0;
    o.served_model = "ornith";
    // Ornith's generation_config.json: [248046 <|im_end|>, 248044 <|endoftext|>].
    o.eos_ids = {248046, 248044};
    return o;
  }
  MockTok tok;
  OrnithTemplate tmpl;
  MockEngine engine;
  server::Server server;
};

std::string post(Fixture& f, const json& body) {
  httplib::Client client("127.0.0.1", f.server.bound_port());
  std::string raw;
  const auto response = client.Post("/v1/chat/completions", httplib::Headers(), body.dump(),
                                    "application/json",
                                    [&raw](const char* d, size_t n) {
                                      raw.append(d, n);
                                      return true;
                                    },
                                    nullptr);
  CHECK(response);
  CHECK_EQ(response->status, 200);
  return raw;
}

const char* kCall =
    "<tool_call>\n<function=read_file>\n<parameter=path>\nsrc/main.cc\n</parameter>\n"
    "</function>\n</tool_call>";

// The "tools" case of ornith_template_cases.json as an OpenAI request.
json tools_request(const json& c, bool stream) {
  json b = {{"model", "ornith"},        {"messages", c.at("messages")}, {"tools", c.at("tools")},
            {"max_tokens", 128},        {"stream", stream},
            {"chat_template_kwargs", {{"enable_thinking", c.at("think").get<bool>()}}}};
  return b;
}

const json& find_case(const json& cases, const std::string& name) {
  for (const json& c : cases)
    if (c.at("name") == name) return c;
  std::fprintf(stderr, "no case '%s'\n", name.c_str());
  std::exit(1);
}

void check_prompt(Fixture& f, const std::string& want_text, const char* what) {
  const std::vector<uint32_t> want = f.tok.encode(want_text);
  if (f.engine.last_prompt != want) {
    std::fprintf(stderr, "%s: the engine's prompt is not transformers' render (%zu vs %zu ids)\n",
                 what, f.engine.last_prompt.size(), want.size());
    std::exit(1);
  }
  std::printf("%s: prompt == transformers' render, %zu ids\n", what, want.size());
}

void tool_call_buffered(const std::string& dir, const json& cases) {
  Fixture f(dir + "/ornith");
  f.engine.script = f.tok.raw(kCall, 5);
  const json r = json::parse(post(f, tools_request(find_case(cases, "tools"), false)));
  check_prompt(f, slurp(dir + "/ornith_template_tools.txt"), "tools (buffered)");
  const json& choice = r.at("choices").at(0);
  CHECK_EQ(choice.at("finish_reason").get<std::string>(), std::string("tool_calls"));
  const json& calls = choice.at("message").at("tool_calls");
  CHECK_EQ(calls.size(), size_t(1));
  CHECK_EQ(calls[0].at("function").at("name").get<std::string>(), std::string("read_file"));
  CHECK(json::parse(calls[0].at("function").at("arguments").get<std::string>()) ==
        json({{"path", "src/main.cc"}}));
  std::printf("tools (buffered): read_file {\"path\": \"src/main.cc\"}, finish tool_calls\n");
}

void tool_call_streamed(const std::string& dir, const json& cases) {
  Fixture f(dir + "/ornith");
  f.engine.script = f.tok.raw(kCall, 5);
  const std::string raw = post(f, tools_request(find_case(cases, "tools"), true));
  check_prompt(f, slurp(dir + "/ornith_template_tools.txt"), "tools (streamed)");
  std::string name, args, finish;
  size_t begin = 0;
  while (begin < raw.size()) {
    const size_t end = raw.find("\n\n", begin);
    if (end == std::string::npos) break;
    const std::string frame = raw.substr(begin, end - begin);
    begin = end + 2;
    if (frame.rfind("data: {", 0) != 0) continue;
    const json j = json::parse(frame.substr(6));
    if (!j.contains("choices") || j.at("choices").empty()) continue;
    const json& c = j.at("choices").at(0);
    if (c.contains("finish_reason") && c.at("finish_reason").is_string())
      finish = c.at("finish_reason").get<std::string>();
    if (!c.at("delta").contains("tool_calls")) continue;
    for (const json& t : c.at("delta").at("tool_calls")) {
      const json& fn = t.at("function");
      if (fn.contains("name")) name += fn.at("name").get<std::string>();
      if (fn.contains("arguments")) args += fn.at("arguments").get<std::string>();
    }
  }
  CHECK_EQ(name, std::string("read_file"));
  CHECK(json::parse(args) == json({{"path", "src/main.cc"}}));
  CHECK_EQ(finish, std::string("tool_calls"));
  std::printf("tools (streamed): the same call from the SSE deltas\n");
}

void history_with_reasoning(const std::string& dir, const json& cases) {
  Fixture f(dir + "/ornith");
  json c = find_case(cases, "tool_response");
  // OpenAI clients send a tool call's arguments as a JSON STRING; the server normalises
  // them to the object the template iterates (and transformers was given).
  for (json& m : c.at("messages"))
    if (m.contains("tool_calls"))
      for (json& t : m.at("tool_calls"))
        t.at("function").at("arguments") = t.at("function").at("arguments").dump();
  f.engine.script = f.tok.raw("The filter matched.\n</think>\n\nAll 5 loader tests passed.", 7);
  const json r = json::parse(post(f, tools_request(c, false)));
  check_prompt(f, slurp(dir + "/ornith_template_tool_response.txt"), "tool_response");
  const json& msg = r.at("choices").at(0).at("message");
  CHECK_EQ(msg.at("reasoning_content").get<std::string>(), std::string("The filter matched."));
  CHECK_EQ(msg.at("content").get<std::string>(), std::string("All 5 loader tests passed."));
  CHECK_EQ(r.at("choices").at(0).at("finish_reason").get<std::string>(), std::string("stop"));
  std::printf("tool_response: reasoning and content split at </think>, finish stop\n");
}

}  // namespace

int main(int argc, char** argv) {
  const std::string dir = argc > 1 ? argv[1] : "tests/tokenizer";
  for (const char* f : {"/ornith/tokenizer_config.json", "/ornith/chat_template.jinja",
                        "/ornith_template_cases.json", "/ornith_template_tools.txt",
                        "/ornith_template_tool_response.txt"})
    if (!std::ifstream(dir + f).good()) {
      std::printf("SKIP: %s%s is absent\n", dir.c_str(), f);
      return 77;
    }
  const json cases = json::parse(slurp(dir + "/ornith_template_cases.json"));
  tool_call_buffered(dir, cases);
  tool_call_streamed(dir, cases);
  history_with_reasoning(dir, cases);
  std::puts("ornith_server_test OK");
  return 0;
}
