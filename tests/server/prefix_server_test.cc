// Spec 7 C4 (plan 7c Task 2): the request path with the prefix cache, over HTTP, on a
// mock engine whose output depends on its state AND its KV cache (tests/server/mock.h,
// StateMockEngine). Every sequence runs against two servers, the cache on and the cache
// off (--prefix-cache-gb 0, today's reset() + full prefill); the responses must be equal,
// and the cached run's `cached_tokens` must be what the plan says.
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "check.h"
#include "server/mock.h"
#include "server/prefix_cache.h"
#include "server/server.h"

namespace {

using json = nlohmann::json;

struct MallocAlloc : server::HostAlloc {
  size_t live = 0;
  void* alloc(size_t n) override {
    live += n;
    return std::malloc(n);
  }
  void free(void* p, size_t n) override {
    live -= n;
    std::free(p);
  }
};

struct Fixture {
  explicit Fixture(size_t cache_bytes) {
    for (int i = 0; i < 24; ++i) engine.words.push_back(tok.id_of("w" + std::to_string(i)));
    options.host = "127.0.0.1";
    options.port = 0;
    options.served_model = "mock-model";
    options.eos_ids = {248046};
    options.prefix_cache_bytes = cache_bytes;
    options.prefix_alloc = &alloc;
    server = std::make_unique<server::Server>(server::Deps{tok, tmpl, engine}, options);
    CHECK(server->start());
  }
  ~Fixture() { server->stop(); }

  struct Reply {
    int status = 0;
    std::string content;
    uint32_t cached = 0, prompt_tokens = 0;
    std::vector<uint32_t> prompt_ids, out_ids;
  };
  Reply chat(const json& messages, uint32_t max_tokens, const json& extra = json::object()) {
    json body = {{"model", "mock-model"}, {"messages", messages}, {"max_tokens", max_tokens},
                 {"return_token_ids", true},
                 {"chat_template_kwargs", {{"enable_thinking", false}}}};
    body.update(extra);
    httplib::Client client("127.0.0.1", server->bound_port());
    const auto response = client.Post("/v1/chat/completions", body.dump(), "application/json");
    CHECK(response);
    Reply r;
    r.status = response->status;
    if (r.status != 200) return r;
    const json j = json::parse(response->body);
    const json& msg = j.at("choices").at(0).at("message");
    r.content = msg.at("content").is_null() ? "" : msg.at("content").get<std::string>();
    r.cached = j.at("usage").at("prompt_tokens_details").at("cached_tokens").get<uint32_t>();
    r.prompt_tokens = j.at("usage").at("prompt_tokens").get<uint32_t>();
    r.prompt_ids = j.at("prompt_token_ids").get<std::vector<uint32_t>>();
    r.out_ids = j.at("choices").at(0).at("token_ids").get<std::vector<uint32_t>>();
    return r;
  }

  MockTok tok;
  MockTemplate tmpl;
  StateMockEngine engine;
  MallocAlloc alloc;
  server::Options options;
  std::unique_ptr<server::Server> server;
};

json msg(const std::string& role, const std::string& content) {
  return {{"role", role}, {"content", content}};
}

std::string words(const std::string& stem, int n) {
  std::string s;
  for (int i = 0; i < n; ++i) s += (i ? " " : "") + stem + std::to_string(i % 7);
  return s;
}

// Runs the same request on both servers; the responses must agree.
struct Pair {
  Fixture on{size_t(1) << 20}, off{0};
  Fixture::Reply both(const json& messages, uint32_t max_tokens,
                      const json& extra = json::object()) {
    const Fixture::Reply a = on.chat(messages, max_tokens, extra);
    const Fixture::Reply b = off.chat(messages, max_tokens, extra);
    CHECK_EQ(a.status, b.status);
    if (a.status != 200) return a;
    if (a.content != b.content) {
      std::fprintf(stderr, "cache on:  '%s'\ncache off: '%s'\n", a.content.c_str(),
                   b.content.c_str());
      CHECK(a.content == b.content);
    }
    CHECK(a.out_ids == b.out_ids);
    CHECK_EQ(b.cached, 0u);
    return a;
  }
};

void test_conversation() {
  Pair t;
  const std::string system = words("sys", 37);
  json conv = json::array({msg("system", system), msg("user", "first question here")});
  const auto r1 = t.both(conv, 9);
  CHECK_EQ(r1.cached, 0u);   // cold
  const size_t len1 = r1.prompt_ids.size();

  // A side request (opencode's title generation) diverging at position 0.
  const auto side = t.both(json::array({msg("user", "make a title")}), 5);
  CHECK_EQ(side.cached, 0u);

  // Turn 2 resends turn 1 and its answer: restore from turn 1's request-end snapshot
  // (prompt + the 9 generated ids), the side request having evicted the card.
  conv.push_back(msg("assistant", r1.content));
  conv.push_back(msg("user", "second question"));
  const auto r2 = t.both(conv, 7);
  CHECK_EQ(r2.cached, uint32_t(len1 + 9));
  CHECK(t.on.engine.state_loads == 1 && t.on.engine.kv_loaded == len1 + 9);

  // Turn 3 continues the resident session: nothing copied.
  conv.push_back(msg("assistant", r2.content));
  conv.push_back(msg("user", "third"));
  const size_t loads = t.on.engine.state_loads;
  const auto r3 = t.both(conv, 6);
  CHECK_EQ(r3.cached, uint32_t(r2.prompt_ids.size() + 7));
  CHECK_EQ(t.on.engine.state_loads, loads);

  // Turn 2 again after turn 3 (a client retry): restores below the resident pos.
  conv.erase(conv.end() - 2, conv.end());
  const auto r2b = t.both(conv, 7);
  CHECK(r2b.content == r2.content);
  CHECK(r2b.cached <= r2.prompt_ids.size());
  CHECK(r2b.cached > len1);
  CHECK_EQ(t.off.engine.resets, size_t(5));   // cache off: reset() per request
  CHECK(t.on.alloc.live <= (size_t(1) << 20));
  std::printf("conversation: cold %u, side %u, turn 2 restore %u, turn 3 continue %u,"
              " retry %u; cache on == cache off OK\n",
              r1.cached, side.cached, r2.cached, r3.cached, r2b.cached);
}

void test_failure() {
  // Review Focus 3: a request that fails mid-prefill leaves the resident session unknown;
  // the next request cannot continue from the half-written state. Here the failing
  // request itself continues the resident session (turn 1's prompt + answer) and throws
  // after three ids, so a server that still trusted the resident ids would continue the
  // next turn from a card that has consumed "\n user: x".
  Pair t;
  json conv = json::array({msg("system", words("sys", 21)), msg("user", "hello there")});
  const auto r1 = t.both(conv, 6);
  const uint32_t bad = t.on.tok.id_of("BAD");
  CHECK_EQ(t.off.tok.id_of("BAD"), bad);
  t.on.engine.bad_id = bad;
  t.off.engine.bad_id = bad;
  conv.push_back(msg("assistant", r1.content));
  json failing = conv;
  failing.push_back(msg("user", "x BAD four"));
  const size_t loads = t.on.engine.state_loads;
  const auto f = t.both(failing, 4);
  CHECK_EQ(f.status, 500);
  CHECK_EQ(t.on.engine.state_loads, loads);          // it was a continuation
  json next = conv;
  next.push_back(msg("user", "y z"));
  const auto r2 = t.both(next, 5);                   // equal to the cold server's
  CHECK_EQ(t.on.engine.state_loads, loads + 1);      // a restore, not a continuation
  CHECK(r2.cached >= uint32_t(r1.prompt_ids.size() + 6));
  std::printf("failure: after a failed prefill the next request restores at %u, never"
              " continues (Review Focus 3) OK\n", r2.cached);
}

void test_sampling() {
  // Review Focus 4: the stored state is the prefill's, the request-end snapshot covers
  // the ids actually generated (sampled), not the argmax.
  Pair t;
  json conv = json::array({msg("system", words("sys", 30)), msg("user", "sample please")});
  const json sampled = {{"temperature", 0.8}, {"seed", 3}};
  const auto r1 = t.both(conv, 8, sampled);
  const auto g1 = t.on.chat(conv, 8);   // greedy: the argmax continuation differs
  (void)t.off.chat(conv, 8);
  CHECK(g1.content != r1.content);
  (void)t.both(json::array({msg("user", "side")}), 3);
  conv.push_back(msg("assistant", r1.content));
  conv.push_back(msg("user", "next"));
  const auto r2 = t.both(conv, 6);
  CHECK_EQ(r2.cached, uint32_t(r1.prompt_ids.size() + 8));
  std::printf("sampling: turn 2 restores the sampled turn 1 at %u and matches the cold run"
              " (Review Focus 4) OK\n", r2.cached);
}

void test_same_prompt() {
  // Review Focus 5: the same prompt twice. The second restores at len - 1 from the
  // prompt-end snapshot and feeds exactly one id; cached_tokens == len - 1.
  Pair t;
  const json conv = json::array({msg("system", words("sys", 13)), msg("user", "again")});
  const auto a = t.both(conv, 6);
  const size_t prefilled = t.on.engine.prefilled, ingested = t.on.engine.ingested;
  const auto b = t.both(conv, 6);
  CHECK(a.content == b.content);
  CHECK_EQ(b.cached, uint32_t(b.prompt_ids.size() - 1));
  CHECK_EQ(t.on.engine.prefilled, prefilled);
  CHECK_EQ(t.on.engine.ingested, ingested + 1);
  // After a side request: the same restore, with the KV from the host.
  (void)t.both(json::array({msg("user", "side")}), 2);
  const auto c = t.both(conv, 6);
  CHECK(c.content == a.content);
  CHECK_EQ(c.cached, uint32_t(c.prompt_ids.size() - 1));
  std::printf("same prompt: the second restores at %u of %zu and feeds one id (Review Focus 5)"
              " OK\n", b.cached, b.prompt_ids.size());
}

void test_history_drops_last_id() {
  // The next turn's history does not carry the generated ids (a client that re-renders
  // the answer): it diverges right after the previous prompt. After a side request the
  // prompt-end snapshot at len - 1 is the deepest hit.
  Pair t;
  json conv = json::array({msg("system", words("sys", 25)), msg("user", "go")});
  const auto a = t.both(conv, 5);
  (void)t.both(json::array({msg("user", "side")}), 2);
  conv.push_back(msg("assistant", "something else entirely"));
  conv.push_back(msg("user", "next"));
  const auto b = t.both(conv, 5);
  CHECK_EQ(b.cached, uint32_t(a.prompt_ids.size() - 1));
  std::printf("history diverging at the previous prompt's end restores at %u (len - 1) OK\n",
              b.cached);
}

void test_small_budget() {
  // A budget below one session's entries: requests still succeed and agree.
  Fixture tiny(96);
  Fixture off(0);
  json conv = json::array({msg("system", words("sys", 40)), msg("user", "q")});
  for (int turn = 0; turn < 4; ++turn) {
    const auto a = tiny.chat(conv, 5);
    const auto b = off.chat(conv, 5);
    if (!(a.status == 200 && a.content == b.content))
      std::fprintf(stderr, "turn %d: %d on '%s' off '%s'\n", turn, a.status, a.content.c_str(), b.content.c_str());
    CHECK(a.status == 200 && a.content == b.content);
    CHECK(tiny.alloc.live <= 96);
    conv.push_back(msg("assistant", a.content));
    conv.push_back(msg("user", "more " + std::to_string(turn)));
    if (turn == 1) {   // both servers: the MockTok ids depend on the words seen
      (void)tiny.chat(json::array({msg("user", "side")}), 2);
      (void)off.chat(json::array({msg("user", "side")}), 2);
    }
  }
  std::printf("small budget: 4 turns under a 96-byte budget agree with the cold server OK\n");
}

}  // namespace

int main() {
  test_conversation();
  test_failure();
  test_sampling();
  test_same_prompt();
  test_history_drops_last_id();
  test_small_budget();
  std::printf("prefix_server_test OK\n");
  return 0;
}
