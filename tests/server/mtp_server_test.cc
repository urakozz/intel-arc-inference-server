// Spec 8 §3.6 (plan 8c Task 2), host only: the server over a scripted speculative
// engine. Every request is served twice - --mtp 0 (step() only, today's server) and
// with bursts of 1..K+1 ids from step_many() - and the two responses must be the same
// bytes (modulo the `created` clock), with the engine left at the same position.
#include <cstdint>
#include <cstdio>
#include <regex>
#include <string>
#include <vector>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "check.h"
#include "server/mock.h"
#include "server/server.h"

namespace {

using json = nlohmann::json;

struct Fixture {
  Fixture() : server({tok, tmpl, engine}, options()) { CHECK(server.start()); }
  ~Fixture() { server.stop(); }
  static server::Options options() {
    server::Options o;
    o.host = "127.0.0.1";
    o.port = 0;
    o.served_model = "mock-model";
    o.eos_ids = {248046};
    return o;
  }
  MockTok tok;
  MockTemplate tmpl;
  MockEngine engine;
  server::Server server;
};

// Drop the wall-clock field and the random tool-call ids so two runs compare byte for
// byte otherwise.
std::string normalise(const std::string& raw, bool sse) {
  const std::string body = std::regex_replace(raw, std::regex("call_[0-9a-f]+"), "call_X");
  if (!sse) {
    json j = json::parse(body);
    j.erase("created");
    return j.dump();
  }
  std::string out;
  size_t begin = 0;
  while (begin < body.size()) {
    const size_t end = body.find("\n\n", begin);
    if (end == std::string::npos) break;
    std::string frame = body.substr(begin, end - begin);
    begin = end + 2;
    if (frame.rfind("data: {", 0) == 0) {
      json j = json::parse(frame.substr(6));
      j.erase("created");
      frame = "data: " + j.dump();
    }
    out += frame + "\n";
  }
  return out;
}

struct Run {
  std::string body;
  uint32_t pos = 0;
  std::vector<uint32_t> truncations;
  size_t burst_calls = 0;
};

// `script` is a function of the fixture's tokenizer (raw pieces are registered there).
template <class Script>
Run serve(uint32_t mtp, std::vector<uint32_t> bursts, const std::string& path, const json& body,
          Script script, bool think_tag = false) {
  Fixture f;
  f.tmpl.think_tag = think_tag;
  f.engine.script = script(f.tok);
  f.engine.mtp = mtp;
  f.engine.bursts = std::move(bursts);
  httplib::Client client("127.0.0.1", f.server.bound_port());
  std::string raw;
  const auto response = client.Post(path, httplib::Headers(), body.dump(), "application/json",
                                    [&raw](const char* d, size_t n) {
                                      raw.append(d, n);
                                      return true;
                                    },
                                    nullptr);
  CHECK(response);
  CHECK_EQ(response->status, 200);
  Run r;
  r.body = normalise(raw, body.value("stream", false));
  r.pos = f.engine.pos();
  r.truncations = f.engine.truncations;
  r.burst_calls = f.engine.burst_calls;
  return r;
}

json completion(uint32_t max_tokens, bool stream, std::vector<std::string> stop = {}) {
  json b = {{"model", "mock-model"}, {"prompt", "one two three"}, {"max_tokens", max_tokens},
            {"stream", stream}};
  if (!stop.empty()) b["stop"] = stop;
  if (stream) b["stream_options"] = {{"include_usage", true}};
  return b;
}

std::vector<uint32_t> words(MockTok& t, const std::string& text) { return t.encode(text); }

// The --mtp 0 run and runs at several burst patterns agree on everything.
template <class Script>
void same_as_plain(const char* what, const std::string& path, const json& body, Script script,
                   uint32_t expect_pos, bool think_tag = false) {
  const Run plain = serve(0, {1}, path, body, script, think_tag);
  CHECK_EQ(plain.burst_calls, 0U);   // --mtp 0 never calls step_many
  CHECK(plain.truncations.empty());
  CHECK_EQ(plain.pos, expect_pos);
  const std::vector<std::vector<uint32_t>> patterns = {{4}, {1, 4, 2, 3}, {2}, {3, 1}};
  for (const auto& pattern : patterns) {
    const Run spec = serve(3, pattern, path, body, script, think_tag);
    if (spec.body != plain.body) {
      std::fprintf(stderr, "%s: bursts differ from --mtp 0\nplain: %s\nmtp:   %s\n", what,
                   plain.body.c_str(), spec.body.c_str());
      std::exit(1);
    }
    CHECK(spec.burst_calls > 0);
    CHECK_EQ(spec.pos, plain.pos);
  }
  std::printf("%s: identical to --mtp 0 over 4 burst patterns, pos %u\n", what, plain.pos);
}

}  // namespace

int main() {
  const uint32_t prompt = 3;   // "one two three"
  // Review Focus 1: EOS at accepted position 2 of a 4-id burst. The server consumes
  // a, b and the EOS (as a run of step() calls would), drops c, and rewinds to 3 kept.
  {
    const auto script = [](MockTok& t) {
      std::vector<uint32_t> s = words(t, "a b");
      s.push_back(248046);
      for (uint32_t id : words(t, "c d e f")) s.push_back(id);
      return s;
    };
    const Run spec = serve(3, {4}, "/v1/completions", completion(50, false), script);
    CHECK_EQ(spec.pos, prompt + 3);
    CHECK_EQ(spec.truncations.size(), 1U);
    CHECK_EQ(spec.truncations[0], prompt + 3);
    CHECK_EQ(json::parse(spec.body).at("usage").at("completion_tokens"), 2);
    same_as_plain("EOS inside a burst", "/v1/completions", completion(50, false), script,
                  prompt + 3);
    same_as_plain("EOS inside a burst, streamed", "/v1/completions", completion(50, true), script,
                  prompt + 3);
  }
  // A stop string inside a burst.
  {
    const auto script = [](MockTok& t) { return words(t, "x y STOP z w v u"); };
    const Run spec = serve(3, {4}, "/v1/completions", completion(50, false, {"STOP"}), script);
    CHECK_EQ(spec.pos, prompt + 3);
    CHECK_EQ(spec.truncations.size(), 1U);
    CHECK_EQ(json::parse(spec.body).at("choices").at(0).at("text"), std::string("x y "));
    same_as_plain("stop string inside a burst", "/v1/completions",
                  completion(50, false, {"STOP"}), script, prompt + 3);
    same_as_plain("stop string inside a burst, streamed", "/v1/completions",
                  completion(50, true, {"STOP"}), script, prompt + 3);
  }
  // Review Focus 2: max_tokens reached mid-burst (5 = 4 + 1 of the second burst).
  {
    const auto script = [](MockTok& t) { return words(t, "p0 p1 p2 p3 p4 p5 p6 p7 p8"); };
    const Run spec = serve(3, {4}, "/v1/completions", completion(5, false), script);
    CHECK_EQ(spec.pos, prompt + 5);
    CHECK_EQ(spec.truncations.size(), 1U);
    CHECK_EQ(spec.truncations[0], prompt + 5);
    const json body = json::parse(spec.body);
    CHECK_EQ(body.at("usage").at("completion_tokens"), 5);
    CHECK_EQ(body.at("choices").at(0).at("finish_reason"), std::string("length"));
    same_as_plain("max_tokens mid-burst", "/v1/completions", completion(5, false), script,
                  prompt + 5);
    same_as_plain("max_tokens mid-burst, streamed", "/v1/completions", completion(5, true), script,
                  prompt + 5);
    // A burst that ends exactly at max_tokens: no truncation.
    const Run exact = serve(3, {4}, "/v1/completions", completion(8, false), script);
    CHECK(exact.truncations.empty());
    CHECK_EQ(exact.pos, prompt + 8);
  }
  // Review Focus 5: a streamed tool call (3-byte pieces), thinking on, in bursts; the
  // OutputStream sees the same pieces in the same order, so every frame is the same.
  {
    const std::string text =
        "plan it</think>\n\nI'll read it.\n<tool_call>\n<function=read>\n<parameter=filePath>\n"
        "/w/x.cc\n</parameter>\n</function>\n</tool_call>";
    const auto script = [&text](MockTok& t) {
      std::vector<uint32_t> s = t.raw(text, 3);
      s.push_back(248046);
      for (uint32_t id : t.raw("after", 1)) s.push_back(id);
      return s;
    };
    json chat = {{"model", "mock-model"},
                 {"messages", json::array({{{"role", "user"}, {"content", "hi"}}})},
                 {"stream", true}};
    const uint32_t n = static_cast<uint32_t>((text.size() + 2) / 3) + 1;   // pieces + EOS
    Fixture probe;
    probe.tmpl.think_tag = true;
    const uint32_t p =
        static_cast<uint32_t>(probe.tok.encode(probe.tmpl.render(chat.at("messages"), {}, true)).size());
    same_as_plain("streamed tool call", "/v1/chat/completions", chat, script, p + n, true);
    chat["stream"] = false;
    same_as_plain("tool call", "/v1/chat/completions", chat, script, p + n, true);
    const Run spec = serve(3, {4}, "/v1/chat/completions", chat, script, true);
    const json body = json::parse(spec.body);
    CHECK_EQ(body.at("choices").at(0).at("finish_reason"), std::string("tool_calls"));
    CHECK_EQ(body.at("usage").at("completion_tokens"), n - 1);
  }
  std::printf("mtp_server_test: all cases passed\n");
  return 0;
}
