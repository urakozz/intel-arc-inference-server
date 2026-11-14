// Open-work 27, host only: the server with prompt lookup inside MTP (`--spec auto` / `mtp`'s default) over a scripted
// MTP engine with lookup inside it (MockEngine::step_drafts_or_chosen; a greedy target whose output is the script).
//   - every response is byte-identical to the plain server's and to plain --spec mtp's (modulo `created`) across EOS,
//     a stop string, max_tokens inside an accepted run and a tool call - each inside a lookup run - and the engine
//     never ends past the plain server's position;
//   - the proposals are exactly what a fresh server::LookupInDFlash with no window over a fresh matcher proposes,
//     iteration by iteration (lookup entered, MTP iterations in between);
//   - the MTP iterations keep the per-step draft choice (step_chosen with the server's StepK) for greedy requests, and
//     a sampled request's MTP iterations run step_many (AdaptiveK's K);
//   - no repeat: every iteration is MTP's; the constructor's refusals; off, the engine is never asked.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "check.h"
#include "server/lookup_in_dflash.h"
#include "server/mock.h"
#include "server/prompt_lookup.h"
#include "server/server.h"

namespace {

using json = nlohmann::json;

enum class Mode { Off, Mtp, MtpLookup };
constexpr uint32_t kK = 3;

server::Options options(Mode m) {
  server::Options o;
  o.host = "127.0.0.1";
  o.port = 0;
  o.served_model = "mock-model";
  o.eos_ids = {248046};
  if (m != Mode::Off) {   // b70-serve's --spec mtp / auto: the per-step choice over the int8 head's costs
    o.mtp_auto = true;
    o.mtp_adaptive.max_k = kK;
    o.mtp_adaptive.cost = server::MtpCost::int8_head();
  }
  o.spec_mtp_lookup = m == Mode::MtpLookup;
  return o;
}

struct Fixture {
  explicit Fixture(Mode m) {
    if (m != Mode::Off) {
      engine.mtp = kK;
      engine.chooses = true;
      engine.mtp_lookup_on = true;
      engine.bursts = {3, 1, 4, 2};   // the scripted MTP acceptance
    }
    server = std::make_unique<server::Server>(server::Deps{tok, tmpl, engine}, options(m));
    CHECK(server->start());
  }
  ~Fixture() { server->stop(); }
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
  size_t lookup_iterations = 0, mtp_iterations = 0, chosen_iterations = 0, lookup_drafts = 0;
};

// Replays the recorded iterations against a fresh policy (no window) over a fresh matcher (the prompt, then every
// pending id and kept draft): every proposal must be the policy's.
void check_proposals(const MockEngine& e) {
  server::PromptLookup m;
  m.append(e.last_prompt);
  server::LookupInDFlashOptions o;
  o.window = 0;
  o.max_k = kK;
  server::LookupInDFlash ref(o);
  size_t at = 0;
  for (size_t it = 0; it < e.proposals.size(); ++it) {
    CHECK_EQ(e.pendings[it], e.target(at));
    m.append(e.target(at));
    const std::vector<uint32_t>& d = e.proposals[it];
    CHECK(ref.propose(m) == d);
    const size_t kept = e.returned[it] - 1;
    if (!d.empty()) {
      size_t j = 0;
      while (j < d.size() && d[j] == e.target(at + 1 + j)) ++j;
      CHECK_EQ(kept, j);
      ref.observe(uint32_t(d.size()), uint32_t(kept));
    }
    for (size_t i = 1; i <= kept; ++i) m.append(e.target(at + i));
    at += kept + 1;
  }
}

template <class Script>
Run serve(Mode mode, const std::string& path, const json& body, Script script) {
  Fixture f(mode);
  f.engine.script = script(f.tok);
  Run r;
  r.body = post(f, path, body);
  r.pos = f.engine.pos();
  r.lookup_iterations = f.engine.lookup_iterations;
  r.mtp_iterations = f.engine.dflash_iterations;   // the engine's own iterations: MTP's here
  r.chosen_iterations = f.engine.chosen_iterations;
  for (const auto& d : f.engine.proposals) r.lookup_drafts += d.size();
  if (mode == Mode::MtpLookup) check_proposals(f.engine);
  return r;
}

json completion(const std::string& prompt, uint32_t max_tokens, bool stream, std::vector<std::string> stop = {}) {
  json b = {{"model", "mock-model"}, {"prompt", prompt}, {"max_tokens", max_tokens}, {"stream", stream}};
  if (!stop.empty()) b["stop"] = stop;
  if (stream) b["stream_options"] = {{"include_usage", true}};
  return b;
}

// 40 distinct words of a "file", then 30 words of other text.
std::string file_prompt() {
  std::string p;
  for (int i = 0; i < 40; ++i) p += "f" + std::to_string(i) + " ";
  for (int i = 0; i < 30; ++i) p += "o" + std::to_string(i) + " ";
  return p;
}

// The output re-emits the file's first `n` words (an edit's re-emitted file), then `tail`.
std::vector<uint32_t> reemit(MockTok& t, int n, const std::string& tail) {
  std::string text;
  for (int i = 0; i < n; ++i) text += "f" + std::to_string(i) + " ";
  return t.encode(text + tail);
}

template <class Script>
void same_as_plain(const char* what, const std::string& path, const json& body, Script script) {
  const Run off = serve(Mode::Off, path, body, script);
  const Run mtp = serve(Mode::Mtp, path, body, script);
  const Run lk = serve(Mode::MtpLookup, path, body, script);
  if (lk.body != off.body || mtp.body != off.body) {
    std::fprintf(stderr, "%s: differs\noff:        %s\nmtp:        %s\nmtp-lookup: %s\n", what, off.body.c_str(),
                 mtp.body.c_str(), lk.body.c_str());
    std::exit(1);
  }
  CHECK(lk.pos <= off.pos && mtp.pos <= off.pos);
  CHECK(lk.lookup_iterations > 0);   // the copy entered lookup
  CHECK(lk.mtp_iterations > 0);      // and the head drafted before (and after) it ...
  CHECK_EQ(lk.chosen_iterations, lk.mtp_iterations);   // ... with the per-step choice, every MTP iteration
  CHECK(mtp.chosen_iterations > 0 && mtp.lookup_iterations == 0);   // plain --spec mtp: no lookup
  std::printf("%s: identical to plain and to --spec mtp; %zu lookup iterations (%zu drafts), %zu MTP (all per-step "
              "chosen), pos %u\n",
              what, lk.lookup_iterations, lk.lookup_drafts, lk.mtp_iterations, lk.pos);
}

void equality_cases() {
  const std::string p = file_prompt();
  const auto then_eos = [](MockTok& t) {
    std::vector<uint32_t> s = reemit(t, 30, "new tail words here");
    s.push_back(248046);
    return s;
  };
  same_as_plain("re-emit then EOS", "/v1/completions", completion(p, 80, false), then_eos);
  same_as_plain("re-emit then EOS, streamed", "/v1/completions", completion(p, 80, true), then_eos);
  same_as_plain("max_tokens inside a lookup run", "/v1/completions", completion(p, 17, false), then_eos);
  same_as_plain("stop inside a lookup run", "/v1/completions", completion(p, 80, true, {"f20 f21"}), then_eos);
  const auto eos_mid = [](MockTok& t) {
    std::vector<uint32_t> s = reemit(t, 19, "");
    s.push_back(248046);
    for (uint32_t id : t.encode("f19 f20 f21")) s.push_back(id);
    return s;
  };
  same_as_plain("EOS inside a lookup run", "/v1/completions", completion(p, 80, false), eos_mid);
  const auto tool = [](MockTok& t) {
    std::vector<uint32_t> s = t.raw("<tool_call>\n<function=edit>\n<parameter=new>\n", 5);
    for (uint32_t id : reemit(t, 30, "")) s.push_back(id);
    for (uint32_t id : t.raw("\n</parameter>\n</function>\n</tool_call>", 5)) s.push_back(id);
    s.push_back(248046);
    return s;
  };
  json chat = {{"model", "mock-model"},
               {"messages", {{{"role", "user"}, {"content", p}}}},
               {"max_tokens", 120},
               {"stream", true},
               {"tools", {{{"type", "function"},
                           {"function", {{"name", "edit"}, {"parameters", {{"type", "object"}}}}}}}}};
  same_as_plain("chat tool call", "/v1/chat/completions", chat, tool);
}

// A sampled request: no per-step choice (greedy only) - its MTP iterations are step_many at AdaptiveK's K; lookup
// still enters on the copy (the mock target is the script whatever the sampling).
void sampled_uses_adaptive_k() {
  const auto then_eos = [](MockTok& t) {
    std::vector<uint32_t> s = reemit(t, 30, "new tail words here");
    s.push_back(248046);
    return s;
  };
  json b = completion(file_prompt(), 80, false);
  b["temperature"] = 0.7;
  b["seed"] = 1;
  const Run lk = serve(Mode::MtpLookup, "/v1/completions", b, then_eos);
  const Run off = serve(Mode::Off, "/v1/completions", b, then_eos);
  CHECK(lk.body == off.body);
  CHECK(lk.lookup_iterations > 0 && lk.mtp_iterations > 0);
  CHECK_EQ(lk.chosen_iterations, size_t(0));
  std::printf("sampled: %zu lookup iterations, %zu MTP iterations at AdaptiveK's K, none per-step chosen\n",
              lk.lookup_iterations, lk.mtp_iterations);
}

// No repeat of 8 ids: every iteration is the head's.
void no_repeat_is_mtp() {
  const auto fresh = [](MockTok& t) {
    std::vector<uint32_t> s = t.encode("n1 n2 n3 n4 n5 n6 n7 n8 n9 n10 n11 n12 n13 n14 o28 o29 x y z");
    s.push_back(248046);
    return s;
  };
  const Run lk = serve(Mode::MtpLookup, "/v1/completions", completion(file_prompt(), 40, false), fresh);
  const Run off = serve(Mode::Off, "/v1/completions", completion(file_prompt(), 40, false), fresh);
  CHECK(lk.body == off.body);
  CHECK_EQ(lk.lookup_iterations, size_t(0));
  CHECK_EQ(lk.chosen_iterations, lk.mtp_iterations);
  std::printf("no repeat: %zu MTP iterations, no lookup\n", lk.mtp_iterations);
}

void refusals() {
  MockTok tok;
  MockTemplate tmpl;
  const auto throws = [&](MockEngine& engine, const server::Options& o) {
    try {
      server::Server s({tok, tmpl, engine}, o);
    } catch (const std::invalid_argument&) {
      return true;
    }
    return false;
  };
  {
    MockEngine engine;   // MTP without lookup inside it
    engine.mtp = kK;
    CHECK(throws(engine, options(Mode::MtpLookup)));
  }
  {
    MockEngine engine;
    engine.mtp = kK;
    engine.mtp_lookup_on = true;
    engine.lookup_k = 3;
    server::Options o = options(Mode::MtpLookup);
    o.spec_lookup = true;
    CHECK(throws(engine, o));
  }
  // Off (B70_SPEC_LOOKUP=0): the engine's lookup path is never asked.
  Fixture f(Mode::Mtp);
  f.engine.script = tok.encode("a b c");
  (void)post(f, "/v1/completions", completion("a b c a b", 3, false));
  CHECK(f.engine.proposals.empty());
  std::printf("refusals and plain --spec mtp: ok\n");
}

}  // namespace

int main() {
  equality_cases();
  sampled_uses_adaptive_k();
  no_repeat_is_mtp();
  refusals();
  std::printf("mtp_lookup_server_test: ok\n");
  return 0;
}
