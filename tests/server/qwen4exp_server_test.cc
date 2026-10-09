// Spec 21e Task 1, host only: the server over Qwen3.8-Flash-Next's chat template - the real renderer
// (chat::Template on the checkpoint's own chat_template.jinja + tokenizer_config.json: the original's small files,
// which 21b's synthetic checkpoints carry; not vendored) behind the OpenAI endpoint, a scripted engine and the mock
// tokenizer, with the server::Options b70-serve's qwen4_exp dispatch sets (cli::qwen4exp::chat_options). Pinned:
//
//   1. ChatFormat::for_model_type("qwen4_exp") is Kind::Qwen, and the request's chat_template_kwargs do not reach
//      the template (template_kwargs() false: Qwen3.8's server path, reasoning_effort not forwarded);
//   2. a tools request with thinking off: the engine's prompt is transformers' render (agnes_template_tools.txt:
//      the template is Agnes's / Qwen3.8's byte for byte), and two parallel Qwen XML calls come back as two OpenAI
//      tool calls, finish_reason "tool_calls" - buffered and streamed (SSE) in 5-byte pieces;
//   3. a history whose assistant turn carries an OpenAI-form tool call (arguments a JSON string) and a tool result:
//      the prompt equals transformers' render of the object form (agnes_template_tool_call_history.txt);
//   4. reasoning: enable_thinking on - the prompt ends in "<think>\n" (agnes_template_thinking.txt) and the output
//      splits at </think> into reasoning_content and content; enable_thinking off - no split;
//   5. both EOS ids (248046 <|im_end|>, 248044 <|endoftext|>) stop with finish_reason "stop", nothing after them
//      streamed;
//   6. a request without sampling fields reaches the engine greedy (spec 21 decision 11's interim); one with a
//      temperature samples.
//
// usage: qwen4exp_server_test <tests/tokenizer> <snapshot dir>   (exit 77 without the snapshot's template files)
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "check.h"
#include "cli/qwen4exp_chat.h"
#include "server/chat_format.h"
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

struct Q4Template : server::TemplateIface {
  explicit Q4Template(const std::string& dir) : tmpl(dir) {}
  std::string render(const json& messages, const json& tools, bool think) override {
    return tmpl.render(messages, tools, think);
  }
  std::string render_with_kwargs(const json& messages, const json& tools, bool think, const json& kwargs) override {
    ++with_kwargs;
    return tmpl.render(messages, tools, think, kwargs);
  }
  chat::Template tmpl;
  int with_kwargs = 0;
};

// MockEngine, recording the Sampling every step was given.
struct RecordingEngine : MockEngine {
  std::vector<server::Sampling> seen;
  uint32_t step(const server::Sampling& s) override {
    seen.push_back(s);
    return MockEngine::step(s);
  }
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
    o.served_model = "qwen4exp";
    // generation_config.json's eos_token_id, as b70-serve reads it (eos_ids(snapshot_dir)).
    cli::qwen4exp::chat_options(o, {248046, 248044});
    return o;
  }
  MockTok tok;
  Q4Template tmpl;
  RecordingEngine engine;
  server::Server server;
};

std::string post(Fixture& f, const json& body) {
  httplib::Client client("127.0.0.1", f.server.bound_port());
  std::string raw;
  const auto response = client.Post("/v1/chat/completions", httplib::Headers(), body.dump(), "application/json",
                                    [&raw](const char* d, size_t n) {
                                      raw.append(d, n);
                                      return true;
                                    },
                                    nullptr);
  CHECK(response);
  CHECK_EQ(response->status, 200);
  return raw;
}

const json& find_case(const json& cases, const std::string& name) {
  for (const json& c : cases)
    if (c.at("name") == name) return c;
  std::fprintf(stderr, "no case '%s'\n", name.c_str());
  std::exit(1);
}

json request(const json& c, bool stream, bool think) {
  json b = {{"model", "qwen4exp"},
            {"messages", c.at("messages")},
            {"max_tokens", 128},
            {"stream", stream},
            {"chat_template_kwargs", {{"enable_thinking", think}, {"reasoning_effort", "low"}}}};
  if (c.at("tools").is_array()) b["tools"] = c.at("tools");
  return b;
}

void check_prompt(Fixture& f, const std::string& want_text, const char* what) {
  const std::vector<uint32_t> want = f.tok.encode(want_text);
  if (f.engine.last_prompt != want) {
    std::fprintf(stderr, "%s: the engine's prompt is not transformers' render (%zu vs %zu ids)\n", what,
                 f.engine.last_prompt.size(), want.size());
    std::exit(1);
  }
  std::printf("%s: prompt == transformers' render, %zu ids\n", what, want.size());
}

const char* kCalls =
    "<tool_call>\n<function=read_file>\n<parameter=path>\nsrc/main.cc\n</parameter>\n</function>\n</tool_call>\n"
    "<tool_call>\n<function=read_file>\n<parameter=path>\nsrc/util.h\n</parameter>\n</function>\n</tool_call>";

void chat_format() {
  const server::ChatFormat f = server::ChatFormat::for_model_type("qwen4_exp");
  CHECK(f.kind == server::ChatFormat::Kind::Qwen);
  CHECK(!f.template_kwargs());
  CHECK(!f.string_content());
  server::Options o;
  cli::qwen4exp::chat_options(o, {248046, 248044});
  CHECK(o.chat_format.kind == server::ChatFormat::Kind::Qwen);
  CHECK(!o.sampling_defaults.has_value());
  CHECK(o.eos_ids == std::vector<uint32_t>({248046, 248044}));
  std::printf("chat format: qwen4_exp -> %s, no template kwargs, no sampling defaults, eos [248046, 248044]\n",
              f.name());
}

void parallel_calls_buffered(const std::string& snap, const std::string& dir, const json& cases) {
  Fixture f(snap);
  f.engine.script = f.tok.raw(kCalls, 5);
  const json r = json::parse(post(f, request(find_case(cases, "tools"), false, false)));
  check_prompt(f, slurp(dir + "/agnes_template_tools.txt"), "tools (buffered)");
  CHECK_EQ(f.tmpl.with_kwargs, 0);   // the Qwen kind renders without the request's kwargs
  const json& choice = r.at("choices").at(0);
  CHECK_EQ(choice.at("finish_reason").get<std::string>(), std::string("tool_calls"));
  const json& calls = choice.at("message").at("tool_calls");
  CHECK_EQ(calls.size(), size_t(2));
  CHECK_EQ(calls[0].at("function").at("name").get<std::string>(), std::string("read_file"));
  CHECK(json::parse(calls[0].at("function").at("arguments").get<std::string>()) == json({{"path", "src/main.cc"}}));
  CHECK(json::parse(calls[1].at("function").at("arguments").get<std::string>()) == json({{"path", "src/util.h"}}));
  CHECK(calls[0].at("id") != calls[1].at("id"));
  std::printf("tools (buffered): two parallel read_file calls, finish tool_calls\n");
}

void parallel_calls_streamed(const std::string& snap, const std::string& dir, const json& cases) {
  Fixture f(snap);
  f.engine.script = f.tok.raw(kCalls, 5);
  const std::string raw = post(f, request(find_case(cases, "tools"), true, false));
  check_prompt(f, slurp(dir + "/agnes_template_tools.txt"), "tools (streamed)");
  std::vector<std::string> names(2), args(2);
  std::string finish;
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
    if (c.contains("finish_reason") && c.at("finish_reason").is_string()) finish = c.at("finish_reason").get<std::string>();
    if (!c.at("delta").contains("tool_calls")) continue;
    for (const json& t : c.at("delta").at("tool_calls")) {
      const size_t i = t.at("index").get<size_t>();
      CHECK(i < 2);
      const json& fn = t.at("function");
      if (fn.contains("name")) names[i] += fn.at("name").get<std::string>();
      if (fn.contains("arguments")) args[i] += fn.at("arguments").get<std::string>();
    }
  }
  CHECK_EQ(names[0], std::string("read_file"));
  CHECK_EQ(names[1], std::string("read_file"));
  CHECK(json::parse(args[0]) == json({{"path", "src/main.cc"}}));
  CHECK(json::parse(args[1]) == json({{"path", "src/util.h"}}));
  CHECK_EQ(finish, std::string("tool_calls"));
  std::printf("tools (streamed): the same two calls from the SSE deltas\n");
}

void history(const std::string& snap, const std::string& dir, const json& cases) {
  Fixture f(snap);
  json c = find_case(cases, "tool_call_history");
  // OpenAI clients send a call's arguments as a JSON STRING; the server normalises them to the object the template
  // iterates (and transformers was given).
  for (json& m : c.at("messages"))
    if (m.contains("tool_calls"))
      for (json& t : m.at("tool_calls")) t.at("function").at("arguments") = t.at("function").at("arguments").dump();
  f.engine.script = f.tok.raw("Yes, it returns 0.", 4);
  const json r = json::parse(post(f, request(c, false, false)));
  check_prompt(f, slurp(dir + "/agnes_template_tool_call_history.txt"), "tool_call_history");
  CHECK_EQ(r.at("choices").at(0).at("message").at("content").get<std::string>(), std::string("Yes, it returns 0."));
  CHECK_EQ(r.at("choices").at(0).at("finish_reason").get<std::string>(), std::string("stop"));
  std::printf("tool_call_history: OpenAI-form arguments render as transformers', finish stop\n");
}

void reasoning(const std::string& snap, const std::string& dir, const json& cases) {
  {
    Fixture f(snap);
    f.engine.script = f.tok.raw("19 * 21 = 399.\n</think>\n\n399", 6);
    const json r = json::parse(post(f, request(find_case(cases, "thinking"), false, true)));
    check_prompt(f, slurp(dir + "/agnes_template_thinking.txt"), "thinking on");
    const json& msg = r.at("choices").at(0).at("message");
    CHECK_EQ(msg.at("reasoning_content").get<std::string>(), std::string("19 * 21 = 399."));
    CHECK_EQ(msg.at("content").get<std::string>(), std::string("399"));
    std::printf("thinking on: the prompt ends in <think>, reasoning and content split at </think>\n");
  }
  {
    Fixture f(snap);
    f.engine.script = f.tok.raw("399", 2);
    const json r = json::parse(post(f, request(find_case(cases, "thinking"), false, false)));
    const std::string prompt = f.tmpl.tmpl.render(find_case(cases, "thinking").at("messages"), nullptr, false);
    CHECK(prompt.size() < 8 || prompt.compare(prompt.size() - 8, 8, "<think>\n") != 0);   // no open reasoning
    check_prompt(f, prompt, "thinking off");
    const json& msg = r.at("choices").at(0).at("message");
    CHECK(!msg.contains("reasoning_content") || msg.at("reasoning_content").is_null() ||
          msg.at("reasoning_content").get<std::string>().empty());
    CHECK_EQ(msg.at("content").get<std::string>(), std::string("399"));
    std::printf("thinking off: no reasoning split\n");
  }
}

void both_eos(const std::string& snap, const json& cases) {
  for (const uint32_t eos : {248046u, 248044u}) {
    Fixture f(snap);
    std::vector<uint32_t> s = f.tok.raw("Done.", 3);
    s.push_back(eos);
    const std::vector<uint32_t> after = f.tok.raw(" never streamed", 4);
    s.insert(s.end(), after.begin(), after.end());
    f.engine.script = s;
    const json r = json::parse(post(f, request(find_case(cases, "plain"), false, false)));
    CHECK_EQ(r.at("choices").at(0).at("finish_reason").get<std::string>(), std::string("stop"));
    CHECK_EQ(r.at("choices").at(0).at("message").at("content").get<std::string>(), std::string("Done."));
    std::printf("eos %u: finish stop, nothing after it\n", eos);
  }
}

void greedy_default(const std::string& snap, const json& cases) {
  {
    Fixture f(snap);
    f.engine.script = f.tok.raw("91", 2);
    post(f, request(find_case(cases, "plain"), false, false));
    CHECK(!f.engine.seen.empty());
    for (const server::Sampling& s : f.engine.seen) CHECK(s.greedy);
  }
  {
    Fixture f(snap);
    f.engine.script = f.tok.raw("97", 2);
    json b = request(find_case(cases, "plain"), false, false);
    b["temperature"] = 0.7;
    post(f, b);
    CHECK(!f.engine.seen.empty());
    for (const server::Sampling& s : f.engine.seen) CHECK(!s.greedy);
  }
  std::printf("sampling: no fields -> greedy (decision 11's interim); temperature 0.7 -> sampled\n");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: qwen4exp_server_test <tests/tokenizer> <snapshot dir>\n");
    return 2;
  }
  const std::string dir = argv[1], snap = argv[2];
  for (const std::string& f : {snap + "/tokenizer_config.json", snap + "/chat_template.jinja",
                              dir + "/agnes_template_cases.json", dir + "/agnes_template_tools.txt"})
    if (!std::ifstream(f).good()) {
      std::printf("SKIP: %s is absent\n", f.c_str());
      return 77;
    }
  const json cases = json::parse(slurp(dir + "/agnes_template_cases.json"));
  chat_format();
  parallel_calls_buffered(snap, dir, cases);
  parallel_calls_streamed(snap, dir, cases);
  history(snap, dir, cases);
  reasoning(snap, dir, cases);
  both_eos(snap, cases);
  greedy_default(snap, cases);
  std::puts("qwen4exp_server_test OK");
  return 0;
}
