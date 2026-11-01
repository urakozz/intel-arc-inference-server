// Spec 19f (plan 19f Task 3, Review Focus 3-4), host only: the server with --spec dflash-lookup over a scripted DFlash
// engine with lookup inside it (MockEngine::step_drafts_or_many; a greedy target whose output is the script, a drafter
// window of 16 so that a far copy fits a small prompt).
//   - every response is byte-identical to the plain server's and to plain --spec dflash's (modulo `created`) across
//     EOS, a stop string, max_tokens inside an accepted run and a tool call - each inside a lookup run - and the
//     engine ends at the same position;
//   - the proposals are exactly what a fresh server::LookupInDFlash over a fresh matcher proposes, iteration by
//     iteration (lookup entered, DFlash iterations in between);
//   - the drafter stays warm: after the request the mock drafter's ring equals the one plain --spec dflash leaves (a
//     mutant whose lookup iterations skip the projection is caught);
//   - the constructor's refusals; off, the engine is never asked.
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

enum class Mode { Off, DFlash, DFlashLookup };
constexpr uint32_t kWindow = 16, kK = 7;

server::Options options(Mode m) {
  server::Options o;
  o.host = "127.0.0.1";
  o.port = 0;
  o.served_model = "mock-model";
  o.eos_ids = {248046};
  if (m != Mode::Off) {   // b70-serve's --spec dflash: the AdaptiveK loop over the drafter's costs
    o.mtp_auto = true;
    o.mtp_adaptive.max_k = kK;
    o.mtp_adaptive.cost = server::MtpCost::dflash(true, true);
  }
  o.spec_dflash_lookup = m == Mode::DFlashLookup;
  return o;
}

struct Fixture {
  Fixture(Mode m, bool project_lookup = true) {
    if (m != Mode::Off) {
      engine.mtp = kK;
      engine.window = kWindow;
      engine.bursts = {3, 1, 5, 2};   // the scripted DFlash acceptance
    }
    engine.project_lookup = project_lookup;
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
  std::vector<uint32_t> ring;
  std::vector<uint32_t> session;   // the prompt, then the script (the target's output)
  size_t lookup_iterations = 0, dflash_iterations = 0, lookup_drafts = 0;
  // The positions the engine holds: the session's first pos ids.
  std::vector<uint32_t> committed() const {
    return std::vector<uint32_t>(session.begin(), session.begin() + std::min<size_t>(pos, session.size()));
  }
};

// Replays the recorded iterations against a fresh policy over a fresh matcher (the prompt, then every pending id and
// kept draft): every proposal must be the policy's. Each iteration's kept count is what the engine returned.
void check_proposals(const MockEngine& e) {
  server::PromptLookup m;
  m.append(e.last_prompt);
  server::LookupInDFlashOptions o;
  o.window = kWindow;
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
Run serve(Mode mode, const std::string& path, const json& body, Script script, bool project_lookup = true) {
  Fixture f(mode, project_lookup);
  f.engine.script = script(f.tok);
  Run r;
  r.body = post(f, path, body);
  r.pos = f.engine.pos();
  r.session = f.engine.last_prompt;
  r.session.insert(r.session.end(), f.engine.script.begin(), f.engine.script.end());
  CHECK(r.pos <= r.session.size());
  r.ring = f.engine.ring;
  r.lookup_iterations = f.engine.lookup_iterations;
  r.dflash_iterations = f.engine.dflash_iterations;
  for (const auto& d : f.engine.proposals) r.lookup_drafts += d.size();
  if (mode == Mode::DFlashLookup) check_proposals(f.engine);
  return r;
}

json completion(const std::string& prompt, uint32_t max_tokens, bool stream, std::vector<std::string> stop = {}) {
  json b = {{"model", "mock-model"}, {"prompt", prompt}, {"max_tokens", max_tokens}, {"stream", stream}};
  if (!stop.empty()) b["stop"] = stop;
  if (stream) b["stream_options"] = {{"include_usage", true}};
  return b;
}

// 40 distinct words of a "file", then 30 words of other text: the file is far beyond the window of 16 when the
// output starts re-emitting it.
std::string far_prompt() {
  std::string p;
  for (int i = 0; i < 40; ++i) p += "f" + std::to_string(i) + " ";
  for (int i = 0; i < 30; ++i) p += "o" + std::to_string(i) + " ";
  return p;
}

// The output re-emits the file's first `n` words (an edit's re-emitted file), then `tail`.
std::vector<uint32_t> reemit(MockTok& t, int n, const std::string& tail) {
  std::string text;
  for (int i = 0; i < n; ++i) text += "f" + std::to_string(i) + " ";
  std::vector<uint32_t> s = t.encode(text + tail);
  return s;
}

template <class Script>
void same_as_plain(const char* what, const std::string& path, const json& body, Script script) {
  const Run off = serve(Mode::Off, path, body, script);
  const Run df = serve(Mode::DFlash, path, body, script);
  const Run lk = serve(Mode::DFlashLookup, path, body, script);
  if (lk.body != off.body || df.body != off.body) {
    std::fprintf(stderr, "%s: differs\noff:           %s\ndflash:        %s\ndflash-lookup: %s\n", what,
                 off.body.c_str(), df.body.c_str(), lk.body.c_str());
    std::exit(1);
  }
  // Where the engine ends after the last id depends on where the bursts end (the ids handed back with truncate_to past
  // an EOS or a finished tool call) - plain --spec dflash already ends at a different position from the plain server
  // in the tool-call case - so the session is checked by content: never past the plain server's, and its positions
  // exactly the prompt and the ids the target produced.
  CHECK(lk.pos <= off.pos && df.pos <= off.pos);
  CHECK(lk.lookup_iterations > 0);   // the copy entered lookup
  CHECK(lk.dflash_iterations > 0);   // and DFlash drafted before (and after) it
  // The drafter as warm as plain DFlash leaves it: every committed position projected, nothing else.
  CHECK(lk.ring == lk.committed());
  CHECK(df.ring == df.committed());
  std::printf("%s: identical to plain and to --spec dflash; %zu lookup iterations (%zu drafts), %zu DFlash, "
              "pos %u, the ring = every committed position (as plain DFlash's)\n",
              what, lk.lookup_iterations, lk.lookup_drafts, lk.dflash_iterations, lk.pos);
}

void equality_cases() {
  const std::string p = far_prompt();
  const auto then_eos = [](MockTok& t) {
    std::vector<uint32_t> s = reemit(t, 30, "new tail words here");
    s.push_back(248046);
    return s;
  };
  same_as_plain("re-emit then EOS", "/v1/completions", completion(p, 80, false), then_eos);
  same_as_plain("re-emit then EOS, streamed", "/v1/completions", completion(p, 80, true), then_eos);
  // max_tokens inside an accepted lookup run.
  same_as_plain("max_tokens inside a lookup run", "/v1/completions", completion(p, 17, false), then_eos);
  // A stop string inside the copy.
  same_as_plain("stop inside a lookup run", "/v1/completions", completion(p, 80, true, {"f20 f21"}), then_eos);
  // EOS inside the copy: the target stops where the file went on.
  const auto eos_mid = [](MockTok& t) {
    std::vector<uint32_t> s = reemit(t, 19, "");
    s.push_back(248046);
    for (uint32_t id : t.encode("f19 f20 f21")) s.push_back(id);
    return s;
  };
  same_as_plain("EOS inside a lookup run", "/v1/completions", completion(p, 80, false), eos_mid);
  // A chat tool call whose argument re-emits the file.
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

// The warm ring is not vacuous: a mutant whose lookup iterations skip the projection ends with a shorter ring.
void ring_mutant() {
  const std::string p = far_prompt();
  const auto then_eos = [](MockTok& t) {
    std::vector<uint32_t> s = reemit(t, 30, "new tail words here");
    s.push_back(248046);
    return s;
  };
  const Run df = serve(Mode::DFlash, "/v1/completions", completion(p, 80, false), then_eos);
  const Run bad = serve(Mode::DFlashLookup, "/v1/completions", completion(p, 80, false), then_eos, false);
  CHECK(bad.ring != df.ring);
  std::printf("ring mutant (no projection in lookup): caught (%zu vs %zu positions)\n", bad.ring.size(),
              df.ring.size());
}

// No far copy (the output copies the prompt's last words, inside the window): every iteration is DFlash's.
void inside_window_is_dflash() {
  const std::string p = far_prompt();
  const auto near = [](MockTok& t) {
    std::vector<uint32_t> s = t.encode("o20 o21 o22 o23 o24 o25 o26 o27 o28 o29 x y z");
    s.push_back(248046);
    return s;
  };
  const Run lk = serve(Mode::DFlashLookup, "/v1/completions", completion(p, 40, false), near);
  const Run off = serve(Mode::Off, "/v1/completions", completion(p, 40, false), near);
  CHECK(lk.body == off.body);
  CHECK_EQ(lk.lookup_iterations, size_t(0));
  std::printf("a copy inside the window: %zu DFlash iterations, no lookup\n", lk.dflash_iterations);
}

void refusals() {
  MockTok tok;
  MockTemplate tmpl;
  bool threw = false;
  {
    MockEngine engine;   // no drafter
    engine.mtp = kK;
    try {
      server::Server s({tok, tmpl, engine}, options(Mode::DFlashLookup));
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    CHECK(threw);
  }
  {
    MockEngine engine;
    engine.mtp = kK;
    engine.window = kWindow;
    engine.lookup_k = 3;
    server::Options o = options(Mode::DFlashLookup);
    o.spec_lookup = true;
    threw = false;
    try {
      server::Server s({tok, tmpl, engine}, o);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    CHECK(threw);
  }
  // --spec dflash (the mode off): the engine's lookup path is never asked.
  Fixture f(Mode::DFlash);
  f.engine.script = tok.encode("a b c");
  (void)post(f, "/v1/completions", completion("a b c a b", 3, false));
  CHECK(f.engine.proposals.empty());
  std::printf("refusals and plain --spec dflash: ok\n");
}

}  // namespace

int main() {
  equality_cases();
  ring_mutant();
  inside_window_is_dflash();
  refusals();
  std::printf("dflash_lookup_server_test: ok\n");
  return 0;
}
