// Spec 19e (plan 19e Task 2), host only: the server with --spec lookup over a scripted engine
// that verifies external drafts (MockEngine::step_drafts, a greedy target whose output is the
// script). Every response must be the plain server's bytes (modulo the `created` clock),
// with the engine left at the same position; the drafts must be exactly what a fresh
// PromptLookup over the prompt and the ids consumed so far proposes; the policy, the session
// seed and the constructor's refusals are checked.
#include <cstdint>
#include <cstdio>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "check.h"
#include "server/mock.h"
#include "server/prompt_lookup.h"
#include "server/server.h"

namespace {

using json = nlohmann::json;

struct Config {
  bool lookup = false;
  uint32_t k = 3;            // the engine's verify_k() and the policy's max
  uint32_t min_match = 2;
  uint32_t history = 0;
};

server::Options options(const Config& c) {
  server::Options o;
  o.host = "127.0.0.1";
  o.port = 0;
  o.served_model = "mock-model";
  o.eos_ids = {248046};
  o.spec_lookup = c.lookup;
  o.lookup_min_match = c.min_match;
  o.lookup_history = c.history;
  o.lookup_adaptive.max_k = c.k;
  o.lookup_adaptive.cost = server::MtpCost::int8_head().with_free_drafts();
  return o;
}

struct Fixture {
  explicit Fixture(const Config& c) : cfg(c) {
    engine.lookup_k = c.lookup ? c.k : 0;
    server = std::make_unique<server::Server>(server::Deps{tok, tmpl, engine}, options(c));
    CHECK(server->start());
  }
  ~Fixture() { server->stop(); }
  Config cfg;
  MockTok tok;
  MockTemplate tmpl;
  MockEngine engine;
  std::unique_ptr<server::Server> server;
};

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

std::string post(Fixture& f, const std::string& path, const json& body) {
  httplib::Client client("127.0.0.1", f.server->bound_port());
  std::string raw;
  const auto response = client.Post(path, httplib::Headers(), body.dump(), "application/json",
                                    [&raw](const char* d, size_t n) {
                                      raw.append(d, n);
                                      return true;
                                    },
                                    nullptr);
  CHECK(response);
  CHECK_EQ(response->status, 200);
  return normalise(raw, body.value("stream", false));
}

struct Run {
  std::string body;
  uint32_t pos = 0;
  size_t steps = 0;                          // step() calls = ids the engine produced
  size_t iterations = 0;                     // step_drafts() calls
  size_t drafted = 0;
  std::vector<uint32_t> truncations;
};

// Replays the engine's recorded iterations against a fresh matcher over the prompt and the
// ids each iteration emitted: every proposal must be what that matcher proposes (with the
// recorded proposal's own length as K - the policy's choice, checked elsewhere).
void check_proposals(const MockEngine& e, uint32_t min_match) {
  server::PromptLookup ref;
  ref.append(e.last_prompt);
  size_t at = 0;   // script position of the pending id
  for (size_t it = 0; it < e.proposals.size(); ++it) {
    CHECK_EQ(e.pendings[it], e.target(at));
    ref.append(e.target(at));
    const std::vector<uint32_t>& d = e.proposals[it];
    if (!d.empty()) {
      CHECK(ref.propose(uint32_t(d.size()), min_match) == d);
    }
    size_t j = 0;
    while (j < d.size() && d[j] == e.target(at + 1 + j)) ++j;
    for (size_t i = 1; i <= j; ++i) ref.append(e.target(at + i));
    at += j + 1;
  }
}

template <class Script>
Run serve(const Config& c, const std::string& path, const json& body, Script script) {
  Fixture f(c);
  f.engine.script = script(f.tok);
  Run r;
  r.body = post(f, path, body);
  r.pos = f.engine.pos();
  r.steps = f.engine.step_times.size();
  r.iterations = f.engine.proposals.size();
  for (const auto& d : f.engine.proposals) r.drafted += d.size();
  r.truncations = f.engine.truncations;
  if (c.lookup) check_proposals(f.engine, c.min_match);
  return r;
}

json completion(const std::string& prompt, uint32_t max_tokens, bool stream,
                std::vector<std::string> stop = {}) {
  json b = {{"model", "mock-model"}, {"prompt", prompt}, {"max_tokens", max_tokens},
            {"stream", stream}};
  if (!stop.empty()) b["stop"] = stop;
  if (stream) b["stream_options"] = {{"include_usage", true}};
  return b;
}

const char* kPrompt =
    "def f ( a , b ) : return a + b \n def g ( x ) : return f ( x , x ) \n print ( g ( 3 ) )";

// The plain server and --spec lookup at several K and n give the same bytes and end at the
// same position; lookup verified something and kept drafts (the script copies the prompt).
template <class Script>
size_t same_as_plain(const char* what, const std::string& path, const json& body, Script script) {
  const Run plain = serve(Config{}, path, body, script);
  size_t total_iterations = 0, total_steps = 0, total_drafted = 0, truncations = 0;
  for (uint32_t k : {1u, 3u}) {
    for (uint32_t n : {2u, 3u}) {
      Config c;
      c.lookup = true;
      c.k = k;
      c.min_match = n;
      const Run spec = serve(c, path, body, script);
      if (spec.body != plain.body) {
        std::fprintf(stderr, "%s: --spec lookup (K %u, n %u) differs from plain\nplain:  %s\n"
                     "lookup: %s\n", what, k, n, plain.body.c_str(), spec.body.c_str());
        std::exit(1);
      }
      CHECK_EQ(spec.pos, plain.pos);
      CHECK(spec.iterations > 0);
      total_iterations += spec.iterations;
      total_steps += spec.steps;
      total_drafted += spec.drafted;
      truncations += spec.truncations.size();
    }
  }
  CHECK(total_drafted > 0);   // every script copies some of the prompt
  std::printf("%s: identical to plain at K 1/3, n 2/3; %zu iterations, %zu drafts, %zu ids, "
              "%zu truncations, pos %u\n",
              what, total_iterations, total_drafted, total_steps, truncations, plain.pos);
  return truncations;
}

std::vector<uint32_t> words(MockTok& t, const std::string& text) { return t.encode(text); }

void equality_cases() {
  // The output echoes the prompt (a copy-like edit), then diverges, then EOS.
  const auto echo = [](MockTok& t) {
    std::vector<uint32_t> s = words(t, "def g ( x ) : return f ( x , x ) \n print ( new stuff");
    s.push_back(248046);
    return s;
  };
  same_as_plain("echo then EOS", "/v1/completions", completion(kPrompt, 64, false), echo);
  same_as_plain("echo then EOS, streamed", "/v1/completions", completion(kPrompt, 64, true), echo);
  // max_tokens lands inside an accepted run: the engine is truncated to it.
  CHECK(same_as_plain("max_tokens inside a burst", "/v1/completions",
                      completion(kPrompt, 7, false), echo) > 0);
  // A stop string inside an accepted run.
  CHECK(same_as_plain("stop inside a burst", "/v1/completions",
                      completion(kPrompt, 64, true, {"x ,"}), echo) > 0);
  // EOS inside the run: the script reaches EOS where the prompt continued.
  const auto eos_mid = [](MockTok& t) {
    std::vector<uint32_t> s = words(t, "def f ( a");
    s.push_back(248046);
    for (uint32_t id : words(t, ", b ) : return")) s.push_back(id);
    return s;
  };
  same_as_plain("EOS inside a run", "/v1/completions", completion(kPrompt, 64, false), eos_mid);
  // Chat with a tool call that copies the prompt's text into its argument.
  const auto tool = [](MockTok& t) {
    std::vector<uint32_t> s = t.raw("<tool_call>\n<function=edit>\n<parameter=old>\n", 5);
    for (uint32_t id : t.encode("def g ( x ) : return f ( x , x )")) s.push_back(id);
    for (uint32_t id : t.raw("\n</parameter>\n</function>\n</tool_call>", 5)) s.push_back(id);
    s.push_back(248046);
    return s;
  };
  json chat = {{"model", "mock-model"},
               {"messages", {{{"role", "user"}, {"content", kPrompt}}}},
               {"max_tokens", 80},
               {"stream", true},
               {"tools", {{{"type", "function"},
                           {"function", {{"name", "edit"}, {"parameters", {{"type", "object"}}}}}}}}};
  same_as_plain("chat tool call", "/v1/chat/completions", chat, tool);
}

// K follows spec 8 §10's policy with free drafts: every draft right keeps K at max; with no
// match at all every iteration is a plain step (no drafts, nothing verified).
void policy_and_counts() {
  const auto echo_long = [](MockTok& t) {
    std::string text;
    for (int rep = 0; rep < 6; ++rep) text += std::string(kPrompt) + " \n ";
    return t.encode(text);
  };
  Config c;
  c.lookup = true;
  c.k = 3;
  const Run r = serve(c, "/v1/completions", completion(kPrompt, 120, false), echo_long);
  // A full echo: after the first few ids every iteration verifies 3 drafts and keeps them.
  CHECK(r.iterations < 45);
  CHECK(r.drafted >= 3 * (r.iterations - 6));
  const auto fresh = [](MockTok& t) {
    std::string text;
    for (int i = 0; i < 40; ++i) text += "w" + std::to_string(i) + " ";
    return t.encode(text);
  };
  const Run p = serve(c, "/v1/completions", completion("one two three", 40, false), fresh);
  CHECK_EQ(p.drafted, size_t(0));
  CHECK_EQ(p.iterations, size_t(40));
  std::printf("policy: echo %zu iterations / %zu drafts for 120 ids; fresh text 40 plain "
              "iterations\n", r.iterations, r.drafted);
}

// Two requests on one server: the second reuses the matcher (its prompt extends the first's)
// and, with history, also matches the first request's generated ids.
void session() {
  for (uint32_t history : {0u, 1u}) {
    Config c;
    c.lookup = true;
    c.history = history;
    Fixture f(c);
    // Request 1 generates a phrase that is NOT in request 2's prompt.
    f.engine.script = words(f.tok, "alpha beta gamma delta epsilon zeta");
    f.engine.script.push_back(248046);
    (void)post(f, "/v1/completions", completion("one two three", 32, false));
    const size_t first_iters = f.engine.proposals.size();
    // Request 2 repeats that phrase after "alpha beta".
    f.engine.proposals.clear();
    f.engine.pendings.clear();
    f.engine.script = words(f.tok, "alpha beta gamma delta epsilon zeta");
    f.engine.script.push_back(248046);
    (void)post(f, "/v1/completions", completion("one two three four", 32, false));
    size_t drafted = 0;
    for (const auto& d : f.engine.proposals) drafted += d.size();
    std::printf("session, history %u: request 2 drafted %zu ids in %zu iterations (request 1: "
                "%zu iterations)\n", history, drafted, f.engine.proposals.size(), first_iters);
    if (history == 0) {
      check_proposals(f.engine, c.min_match);   // the reused matcher is a fresh one
      CHECK_EQ(drafted, size_t(0));
    } else {
      CHECK(drafted >= 3);   // "alpha beta" -> gamma delta epsilon from request 1's output
    }
  }
}

void refusals() {
  MockTok tok;
  MockTemplate tmpl;
  MockEngine engine;   // verify_k() == 0
  Config c;
  c.lookup = true;
  bool threw = false;
  try {
    server::Server s({tok, tmpl, engine}, options(c));
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);
  engine.lookup_k = 3;
  engine.mtp = 3;   // one proposer per server
  threw = false;
  try {
    server::Server s({tok, tmpl, engine}, options(c));
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);
  engine.mtp = 0;
  c.min_match = 1;
  threw = false;
  try {
    server::Server s({tok, tmpl, engine}, options(c));
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);
  // Off: an engine that could verify is never asked to.
  Config off;
  Fixture f(off);
  f.engine.lookup_k = 3;
  f.engine.script = words(f.tok, "a b c");
  (void)post(f, "/v1/completions", completion("a b c a b", 3, false));
  CHECK(f.engine.proposals.empty());
  std::printf("refusals and the default: ok\n");
}

}  // namespace

int main() {
  equality_cases();
  policy_and_counts();
  session();
  refusals();
  std::printf("lookup_server_test: ok\n");
  return 0;
}
