// Open-work 27 under --parallel replica, host only: prompt lookup inside MTP on each card (server::Scheduler over
// cli::ReplicaEngine over two scripted MTP cards with lookup inside them) against the serial server on one such card.
//   - each request's reply, and every iteration its card ran (the pending ids, the lookup proposals - wide ones and
//     confirmed entries included - the per-step chosen MTP iterations), equal the same request alone on the serial
//     server, sent one at a time and two at once (one a card): the policy state is the request's own;
//   - a sampled (seeded) request's MTP iterations are AdaptiveK's step_many there as on one card;
//   - lookup off: no card is asked to propose; several slots on a card: refused.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "check.h"
#include "cli/replica_engine.h"
#include "server/mock.h"
#include "server/server.h"

namespace {

using json = nlohmann::json;
constexpr uint32_t kK = 3;
constexpr uint32_t kWide = 15;   // runtime::Engine::kWideK

// MockEngine with open-work 28's engine side: lookup runs up to kWide drafts, and with `choose` a candidate the head
// confirms (the mock head drafts the script, so it agrees when the candidate's first two ids are the script's next).
struct Card : MockEngine {
  size_t confirmed = 0, wide_runs = 0;
  uint32_t lookup_wide_k() override { return kWide; }
  std::vector<uint32_t> step_drafts_or_confirmed(const server::Sampling& s, uint32_t k, const Proposer& propose,
                                                 const DraftChoice& choose, const LookupConfirm& confirm) override {
    pendings.push_back(target(at));
    std::vector<uint32_t> d = propose(target(at));
    if (d.size() > kWide) throw std::logic_error("mock: more lookup drafts than the wide list");
    if (d.empty() && choose && confirm.candidate && k > 0) {
      const std::vector<uint32_t> cand = confirm.candidate();
      if (cand.size() >= 2 && cand[0] == target(at + 1) && cand[1] == target(at + 2)) {
        confirm.confirmed();
        ++confirmed;
        d = cand;
      }
    }
    proposals.push_back(d);
    std::vector<uint32_t> out;
    if (d.empty()) {
      ++dflash_iterations;
      out = choose ? step_chosen(s, k, choose) : step_many(s, k);
    } else {
      ++lookup_iterations;
      wide_runs += d.size() > kK;
      size_t j = 0;
      while (j < d.size() && d[j] == target(at + 1 + j)) ++j;
      for (size_t i = 0; i <= j; ++i) out.push_back(step(s));
    }
    returned.push_back(out.size());
    return out;
  }
};

void arm(Card& c) {
  c.mtp = kK;
  c.chooses = true;
  c.mtp_lookup_on = true;
  c.bursts = {3, 1, 4, 2};
}

server::Options options(bool lookup) {
  server::Options o;
  o.host = "127.0.0.1";
  o.port = 0;
  o.served_model = "mock-model";
  o.eos_ids = {248046};
  o.mtp_auto = true;   // b70-serve's --spec auto: the per-step choice over the int8 head's costs
  o.mtp_adaptive.max_k = kK;
  o.mtp_adaptive.cost = server::MtpCost::int8_head();
  o.spec_mtp_lookup = lookup;
  return o;
}

// The output re-emits the file's first 30 words, then other words and EOS.
std::vector<uint32_t> reemit(MockTok& t, const std::string& stem) {
  std::string text;
  for (int i = 0; i < 30; ++i) text += stem + std::to_string(i) + " ";
  std::vector<uint32_t> s = t.encode(text + "new tail words here");
  s.push_back(248046);
  return s;
}

// `replica`: two cards behind the scheduler; otherwise the serial server on c0. MockTok numbers a word at its first
// sight: every fixture sees the prompts in one order, then the script (the `stem` file re-emitted, on both cards).
struct Fixture {
  Fixture(bool replica, const std::vector<std::string>& prompts, const std::string& stem, bool lookup = true) {
    for (const std::string& p : prompts) (void)tok.encode(p);
    arm(c0);
    arm(c1);
    c0.script = c1.script = reemit(tok, stem);
    if (replica) {
      engine = std::make_unique<cli::ReplicaEngine>(
          std::vector<cli::ReplicaEngine::Card>{{&c0, 4096}, {&c1, 4096}}, false, true);
      server = std::make_unique<server::Server>(server::Deps{tok, tmpl, c0}, options(lookup), engine.get());
    } else {
      server = std::make_unique<server::Server>(server::Deps{tok, tmpl, c0}, options(lookup));
    }
    CHECK(server->start());
  }
  ~Fixture() { server->stop(); }
  json post(const json& body) {
    httplib::Client client("127.0.0.1", server->bound_port());
    client.set_read_timeout(60, 0);
    const auto response = client.Post("/v1/completions", body.dump(), "application/json");
    CHECK(response);
    CHECK_EQ(response->status, 200);
    return json::parse(response->body).at("choices");
  }
  MockTok tok;
  MockTemplate tmpl;
  Card c0, c1;
  std::unique_ptr<cli::ReplicaEngine> engine;
  std::unique_ptr<server::Server> server;
};

json completion(const std::string& prompt, uint32_t max_tokens, const json& extra = json::object()) {
  json b = {{"model", "mock-model"}, {"prompt", prompt}, {"max_tokens", max_tokens}, {"return_token_ids", true}};
  b.update(extra);
  return b;
}

// 40 distinct words of a "file" (stem f or g), then 30 words of other text.
std::string file_prompt(const std::string& stem) {
  std::string p;
  for (int i = 0; i < 40; ++i) p += stem + std::to_string(i) + " ";
  for (int i = 0; i < 30; ++i) p += "o" + std::to_string(i) + " ";
  return p;
}

// What a card ran for one request.
struct Trace {
  json choices;
  std::vector<uint32_t> pendings;
  std::vector<std::vector<uint32_t>> proposals;
  std::vector<size_t> returned;
  size_t lookup = 0, mtp = 0, chosen = 0, confirmed = 0, wide = 0;
};
Trace trace(const Card& c, const json& choices) {
  return {choices, c.pendings, c.proposals, c.returned, c.lookup_iterations, c.dflash_iterations,
          c.chosen_iterations, c.confirmed, c.wide_runs};
}

void same(const char* what, const Trace& got, const Trace& want) {
  if (got.choices != want.choices || got.pendings != want.pendings || got.proposals != want.proposals ||
      got.returned != want.returned || got.chosen != want.chosen || got.confirmed != want.confirmed) {
    std::fprintf(stderr, "%s: replica %s (%zu lookup, %zu MTP, %zu chosen, %zu confirmed), serial %s (%zu, %zu, %zu, "
                 "%zu)\n", what, got.choices.dump().c_str(), got.lookup, got.mtp, got.chosen, got.confirmed,
                 want.choices.dump().c_str(), want.lookup, want.mtp, want.chosen, want.confirmed);
    std::exit(1);
  }
}

// The request alone on a fresh serial server.
Trace serial(const std::vector<std::string>& seen, const std::string& stem, const json& body) {
  Fixture s(false, seen, stem);
  const json choices = s.post(body);
  return trace(s.c0, choices);
}

// One request at a time: card 0 serves each (the first free card); greedy and seeded.
void one_at_a_time() {
  for (const bool seeded : {false, true}) {
    const std::string p = file_prompt("f");
    Fixture rep(true, {p}, "f");
    const json extra = seeded ? json{{"temperature", 0.7}, {"seed", 5}} : json::object();
    const json body = completion(p, 80, extra);
    const Trace got = trace(rep.c0, rep.post(body));
    const Trace want = serial({p}, "f", body);
    same(seeded ? "one request, seeded" : "one request, greedy", got, want);
    CHECK(got.lookup > 0 && got.mtp > 0);
    if (seeded) {
      CHECK_EQ(got.chosen, size_t(0));   // AdaptiveK's step_many, as on one card
    } else {
      CHECK_EQ(got.chosen, got.mtp);     // every MTP iteration per-step chosen
      CHECK(got.confirmed > 0 && got.wide > 0);   // open-work 28's entry and wide runs on the card
    }
    CHECK(rep.c1.proposals.empty());
    std::printf("one request (%s): equal to the serial server, %zu lookup iterations (%zu wide, %zu confirmed), "
                "%zu MTP (%zu per-step chosen)\n",
                seeded ? "seeded" : "greedy", got.lookup, got.wide, got.confirmed, got.mtp, got.chosen);
  }
}

// Two at once, one a card: each card's iterations are its own request's alone.
void two_at_once() {
  const std::string pa = file_prompt("f"), pb = file_prompt("g");
  // One script for both cards: the f file re-emitted (b's lookup finds no f copy in its prompt: MTP only there).
  Fixture rep(true, {pa, pb}, "f");
  rep.c0.step_ms = rep.c1.step_ms = 1;
  const json ba = completion(pa, 80), bb = completion(pb, 80, {{"temperature", 0.7}, {"seed", 9}});
  json ca, cb;
  std::thread ta([&] { ca = rep.post(ba); });
  std::thread tb([&] { cb = rep.post(bb); });
  ta.join();
  tb.join();
  const std::vector<uint32_t> ids_a = rep.tok.encode(pa);
  const bool a_on_0 = rep.c0.last_prompt == ids_a;
  CHECK(a_on_0 || rep.c1.last_prompt == ids_a);
  const Trace ga = trace(a_on_0 ? rep.c0 : rep.c1, ca), gb = trace(a_on_0 ? rep.c1 : rep.c0, cb);
  same("two at once: A", ga, serial({pa, pb}, "f", ba));
  same("two at once: B", gb, serial({pa, pb}, "f", bb));
  CHECK(ga.lookup > 0);
  CHECK_EQ(gb.lookup, size_t(0));
  std::printf("two at once: each card's iterations its own request's alone (A %zu lookup, B %zu)\n", ga.lookup,
              gb.lookup);
}

void off_and_refusals() {
  {
    const std::string p = file_prompt("f");
    Fixture rep(true, {p}, "f", /*lookup=*/false);
    (void)rep.post(completion(p, 40));
    CHECK(rep.c0.proposals.empty() && rep.c1.proposals.empty());
  }
  {
    MockTok tok;
    MockTemplate tmpl;
    MockEngine e;
    e.mtp = kK;
    e.mtp_lookup_on = true;
    MockBatchEngine slots;   // several slots on one card: no lookup inside MTP there
    bool threw = false;
    try {
      server::Server s({tok, tmpl, e}, options(true), &slots);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    CHECK(threw);
  }
  std::printf("off: no card proposes; several slots on a card: refused\n");
}

}  // namespace

int main() {
  one_at_a_time();
  two_at_once();
  off_and_refusals();
  std::printf("replica_lookup_server_test: ok\n");
  return 0;
}
