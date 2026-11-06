// Plan 13b Task 4 (Review Focus 1, 3, 6; B2 / B4 on the host): the server's scheduler path over HTTP -
// server::Server with a cli::ReplicaEngine over two mock cards (tests/server/mock.h StateMockEngine, whose ids
// depend on its state AND its KV cache) - against today's serial server on one such card. Every response under
// the replica server must be the serial server's for the same request (content, ids, finish reason), alone and
// with two clients at once, streamed or not, the prefix cache on (with --prefix-split-last) and off; a client that
// disconnects frees its card; a full queue is a 503; --log-requests writes one record per request with t_start the
// arrival. Host only.
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "check.h"
#include "cli/replica_engine.h"
#include "server/mock.h"
#include "server/prefix_cache.h"
#include "server/server.h"

namespace {

using json = nlohmann::json;
using Clock = std::chrono::steady_clock;

struct MallocAlloc : server::HostAlloc {
  void* alloc(size_t n) override { return std::malloc(n); }
  void free(void* p, size_t) override { std::free(p); }
};

// A mock card that can be slow (to hold a slot) - StateMockEngine otherwise.
struct Card : StateMockEngine {
  std::atomic<int> step_us{0};
  std::atomic<uint64_t> steps{0};
  uint32_t step(const server::Sampling& s) override {
    if (step_us != 0) std::this_thread::sleep_for(std::chrono::microseconds(step_us.load()));
    ++steps;
    return StateMockEngine::step(s);
  }
  // The sampler re-arms per request, as EngineAdapter's does (seed_rng after prefill / ingest).
  void prefill(const std::vector<uint32_t>& ids) override {
    rng = 0;
    StateMockEngine::prefill(ids);
  }
  void ingest(const std::vector<uint32_t>& ids) override {
    rng = 0;
    StateMockEngine::ingest(ids);
  }
};

struct Reply {
  int status = 0;
  std::string content, finish;
  std::vector<uint32_t> out_ids;
  uint32_t cached = 0;
  std::string body;
};

// One server: `replica` = two mock cards behind the scheduler; otherwise the serial server on one card.
struct Fixture {
  Fixture(bool replica, size_t cache_bytes, bool split, const std::string& log_dir = "", size_t queue = 4) {
    for (Card* c : {&c0, &c1})
      for (int i = 0; i < 24; ++i) c->words.push_back(tok.id_of("w" + std::to_string(i)));
    options.host = "127.0.0.1";
    options.port = 0;
    options.served_model = "mock-model";
    options.eos_ids = {248046};
    options.prefix_cache_bytes = cache_bytes;
    options.prefix_alloc = &alloc;
    options.prefix_split_last = split;
    options.log_requests_dir = log_dir;
    options.queue_depth = queue;
    if (replica) {
      engine = std::make_unique<cli::ReplicaEngine>(
          std::vector<cli::ReplicaEngine::Card>{{&c0, 8}, {&c1, 8}}, cache_bytes != 0, true);
      server = std::make_unique<server::Server>(server::Deps{tok, tmpl, c0}, options, engine.get());
    } else {
      server = std::make_unique<server::Server>(server::Deps{tok, tmpl, c0}, options);
    }
    CHECK(server->start());
  }
  ~Fixture() { server->stop(); }

  json body(const json& messages, uint32_t max_tokens, const json& extra) {
    json b = {{"model", "mock-model"}, {"messages", messages}, {"max_tokens", max_tokens},
              {"return_token_ids", true}, {"chat_template_kwargs", {{"enable_thinking", false}}}};
    b.update(extra);
    return b;
  }
  Reply chat(const json& messages, uint32_t max_tokens, const json& extra = json::object()) {
    httplib::Client client("127.0.0.1", server->bound_port());
    client.set_read_timeout(60, 0);
    const auto response = client.Post("/v1/chat/completions", body(messages, max_tokens, extra).dump(),
                                      "application/json");
    CHECK(response);
    Reply r;
    r.status = response->status;
    r.body = response->body;
    if (r.status != 200) return r;
    const json j = json::parse(response->body);
    const json& choice = j.at("choices").at(0);
    const json& msg = choice.at("message");
    r.content = msg.at("content").is_null() ? "" : msg.at("content").get<std::string>();
    r.finish = choice.at("finish_reason").get<std::string>();
    r.out_ids = choice.at("token_ids").get<std::vector<uint32_t>>();
    r.cached = j.at("usage").at("prompt_tokens_details").at("cached_tokens").get<uint32_t>();
    return r;
  }
  // Streamed: the content deltas concatenated, the finish reason. `stop_after` > 0: the client hangs up after that
  // many chunks.
  Reply stream(const json& messages, uint32_t max_tokens, size_t stop_after = 0) {
    httplib::Client client("127.0.0.1", server->bound_port());
    client.set_read_timeout(60, 0);
    json b = body(messages, max_tokens, json::object());
    b["stream"] = true;
    std::string raw;
    size_t chunks = 0;
    const auto response = client.Post(
        "/v1/chat/completions", httplib::Headers(), b.dump(), "application/json",
        [&](const char* data, size_t size) {
          raw.append(data, size);
          return stop_after == 0 || ++chunks < stop_after;
        },
        nullptr);
    Reply r;
    if (stop_after != 0) return r;
    CHECK(response);
    r.status = response->status;
    size_t at = 0;
    while (at < raw.size()) {
      const size_t end = raw.find("\n\n", at);
      if (end == std::string::npos) break;
      const std::string frame = raw.substr(at, end - at);
      at = end + 2;
      if (frame == "data: [DONE]") continue;
      const json j = json::parse(frame.substr(6));
      if (j.at("choices").empty()) continue;
      const json& c = j.at("choices").at(0);
      if (c.contains("delta") && c.at("delta").contains("content") && c.at("delta").at("content").is_string())
        r.content += c.at("delta").at("content").get<std::string>();
      if (c.contains("finish_reason") && c.at("finish_reason").is_string()) r.finish = c.at("finish_reason");
    }
    return r;
  }

  MockTok tok;
  MockTemplate tmpl;
  Card c0, c1;
  MallocAlloc alloc;
  server::Options options;
  std::unique_ptr<cli::ReplicaEngine> engine;
  std::unique_ptr<server::Server> server;
};

json msg(const std::string& role, const std::string& content) { return {{"role", role}, {"content", content}}; }

std::string words(const std::string& stem, int n) {
  std::string s;
  for (int i = 0; i < n; ++i) s += (i ? " " : "") + stem + std::to_string(i % 7);
  return s;
}

void same(const char* what, const Reply& got, const Reply& want) {
  if (got.status != want.status || got.content != want.content || got.out_ids != want.out_ids ||
      got.finish != want.finish) {
    std::fprintf(stderr, "%s: replica '%s' (%s, %d), serial '%s' (%s, %d)\n", what, got.content.c_str(),
                 got.finish.c_str(), got.status, want.content.c_str(), want.finish.c_str(), want.status);
    std::exit(1);
  }
}

// A conversation of turns and side requests, sequential: the replica server answers as the serial one, cache on
// (split-last) and off.
void test_sequential(size_t cache, bool split) {
  Fixture rep(true, cache, split), ser(false, cache, split);
  const std::string system = words("sys", 41);
  json conv = json::array({msg("system", system), msg("user", "first question here")});
  const json seeded = {{"temperature", 0.8}, {"seed", 7}};
  const json stop = {{"stop", json::array({"w5"})}};
  for (int turn = 0; turn < 4; ++turn) {
    const Reply a = rep.chat(conv, 9), b = ser.chat(conv, 9);
    same("turn", a, b);
    if (cache != 0 && turn > 0) CHECK(a.cached > 0);
    // A side request (a title) and a seeded / stopped one between the turns.
    same("side", rep.chat(json::array({msg("user", "make a title " + std::to_string(turn))}), 5),
         ser.chat(json::array({msg("user", "make a title " + std::to_string(turn))}), 5));
    same("seeded", rep.chat(conv, 6, seeded), ser.chat(conv, 6, seeded));
    same("stop", rep.chat(conv, 30, stop), ser.chat(conv, 30, stop));
    conv.push_back(msg("assistant", a.content));
    conv.push_back(msg("user", "next " + std::to_string(turn)));
  }
  std::printf("sequential (cache %s%s): every reply equals the serial server's\n", cache ? "on" : "off",
              split ? ", split-last" : "");
}

// B2 on the host: two clients at once; each reply equals the same request on a fresh serial server.
void test_concurrent() {
  Fixture rep(true, size_t(1) << 20, true);
  rep.c0.step_us = rep.c1.step_us = 300;
  const json a = json::array({msg("system", words("sa", 30)), msg("user", "main session")});
  const json b = json::array({msg("user", "a subagent's question " + words("sb", 9))});
  // MockTok numbers a word at its first sight: every fixture sees both prompts in one order first.
  const auto warm = [&](Fixture& f) {
    for (const json* m : {&a, &b}) (void)f.tok.encode(f.tmpl.render(*m, json::array(), false));
  };
  warm(rep);
  Reply ra, rb, sb;
  std::thread ta([&] { ra = rep.chat(a, 40); });
  std::thread tb([&] {
    rb = rep.chat(b, 25, {{"temperature", 0.7}, {"seed", 3}});
    sb = rep.stream(b, 25);
  });
  ta.join();
  tb.join();
  Fixture s1(false, size_t(1) << 20, true), s2(false, size_t(1) << 20, true), s3(false, size_t(1) << 20, true);
  warm(s1);
  warm(s2);
  warm(s3);
  same("concurrent A", ra, s1.chat(a, 40));
  same("concurrent B", rb, s2.chat(b, 25, {{"temperature", 0.7}, {"seed", 3}}));
  const Reply plain_b = s3.chat(b, 25);
  CHECK(sb.content == plain_b.content && sb.finish == plain_b.finish);
  // Both cards did work.
  CHECK(rep.c0.steps > 0 && rep.c1.steps > 0);
  std::printf("concurrent: two clients at once, each reply the serial server's (streamed too)\n");
}

// Review Focus 6: a client that hangs up mid-stream frees its card; a full queue is a 503.
void test_disconnect_and_busy() {
  {
    Fixture rep(true, 0, false);
    rep.c0.step_us = rep.c1.step_us = 2000;
    const json longer = json::array({msg("user", "write forever")});
    (void)rep.stream(longer, 100000, 3);   // hangs up after 3 chunks
    // Both cards serve two new requests at once, promptly: the cancelled one no longer holds a card.
    const auto t0 = Clock::now();
    Reply x, y;
    std::thread tx([&] { x = rep.chat(json::array({msg("user", "x")}), 20); });
    std::thread ty([&] { y = rep.chat(json::array({msg("user", "y")}), 20); });
    tx.join();
    ty.join();
    const double s = std::chrono::duration<double>(Clock::now() - t0).count();
    CHECK_EQ(x.status, 200);
    CHECK_EQ(y.status, 200);
    CHECK(s < 5.0);   // 20 steps at 2 ms each on its own card; the abandoned request would take 200 s
    const uint64_t steps = rep.c0.steps + rep.c1.steps;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    CHECK_EQ(rep.c0.steps + rep.c1.steps, steps);   // nothing runs after the last request
  }
  {
    Fixture rep(true, 0, false, "", /*queue=*/0);
    rep.c0.step_us = rep.c1.step_us = 2000;
    Reply a, b;
    std::thread ta([&] { a = rep.chat(json::array({msg("user", "a")}), 300); });
    std::thread tb([&] { b = rep.chat(json::array({msg("user", "b")}), 300); });
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    const Reply c = rep.chat(json::array({msg("user", "c")}), 3);
    CHECK_EQ(c.status, 503);
    CHECK(c.body.find("server busy") != std::string::npos);
    ta.join();
    tb.join();
    CHECK_EQ(a.status, 200);
    CHECK_EQ(b.status, 200);
  }
  std::printf("disconnect frees the card; both slots busy and no queue: 503\n");
}

// --log-requests from the scheduler path: one record per request, server.cc's fields, t_start <= t_first_token <=
// t_end, the ids the reply carried.
void test_request_log() {
  const std::filesystem::path dir =
      std::filesystem::temp_directory_path() / ("b70-replica-log-" + std::to_string(::getpid()));
  std::filesystem::create_directories(dir);
  {
    Fixture rep(true, size_t(1) << 20, true, dir.string());
    Reply a, b;
    std::thread ta([&] { a = rep.chat(json::array({msg("user", "log me")}), 7); });
    std::thread tb([&] { b = rep.chat(json::array({msg("user", "and me")}), 5); });
    ta.join();
    tb.join();
    std::vector<json> records;
    for (const auto& e : std::filesystem::directory_iterator(dir)) {
      std::ifstream f(e.path());
      std::stringstream ss;
      ss << f.rdbuf();
      records.push_back(json::parse(ss.str()));
    }
    CHECK_EQ(records.size(), size_t(2));
    for (const json& r : records) {
      for (const char* k : {"t_start", "t_first_token", "t_end", "endpoint", "request", "prompt_ids", "out_ids",
                            "response_text"})
        CHECK(r.contains(k));
      CHECK(r.at("t_start").get<double>() <= r.at("t_first_token").get<double>());
      CHECK(r.at("t_first_token").get<double>() <= r.at("t_end").get<double>());
      const auto ids = r.at("out_ids").get<std::vector<uint32_t>>();
      CHECK(ids == a.out_ids || ids == b.out_ids);
    }
  }
  for (const auto& e : std::filesystem::directory_iterator(dir)) std::filesystem::remove(e.path());
  std::filesystem::remove(dir);
  std::printf("request log: one record per request, t_start <= t_first_token <= t_end\n");
}

}  // namespace

int main() {
  test_sequential(0, false);
  test_sequential(size_t(1) << 20, false);
  test_sequential(size_t(1) << 20, true);
  test_concurrent();
  test_disconnect_and_busy();
  test_request_log();
  std::printf("replica_server_test: all cases passed\n");
  return 0;
}
