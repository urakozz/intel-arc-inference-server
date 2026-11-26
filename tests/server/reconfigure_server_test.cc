// b70-serve's reconfigure in place (POST /v1/control/reconfigure, cli/serve_reconfigure.h), host only: the route's
// refusals (a non-object body, a server that switches nothing), each key applied or refused with its reason, the
// proposer switched off and back on through the adapter, and every arm's reply the same bytes as before the switch.
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "check.h"
#include "cli/flags.h"
#include "cli/serve_reconfigure.h"
#include "server/mock.h"
#include "server/normalise.h"
#include "server/server.h"

namespace {

using json = nlohmann::json;
using cli::flags::Arg;
using cli::flags::Help;

// The flags the test names (b70-serve's own, by name).
struct Flags {
  [[=Help("S", "")]] std::string spec = "auto";
  [[=Help("K", "")]] Arg<uint32_t> spec_max;
  [[=Help("bf16|int8", "")]] std::string kv_cache;
  [[=Help("auto|L", "")]] std::string max_len = "auto";
  [[=Help("auto|gpu|cpu", "")]] std::string host_experts = "auto";
};

server::Options base() {
  server::Options o;
  o.host = "127.0.0.1";
  o.port = 0;
  o.served_model = "mock-model";
  o.eos_ids = {248046};
  return o;
}

httplib::Result post(server::Server& s, const std::string& path, const std::string& body) {
  httplib::Client client("127.0.0.1", s.bound_port());
  return client.Post(path, body, "application/json");
}

std::string complete(server::Server& s) {
  const json body = {{"model", "mock-model"}, {"prompt", "one two three"}, {"max_tokens", 12}};
  const auto r = post(s, "/v1/completions", body.dump());
  CHECK(r);
  CHECK_EQ(r->status, 200);
  json j = json::parse(normalise(r->body, false));
  j.erase("id");   // the server's request counter
  return j.dump();
}

void loopback() {
  CHECK(server::is_loopback("127.0.0.1"));
  CHECK(server::is_loopback("127.1.2.3"));
  CHECK(server::is_loopback("::1"));
  CHECK(server::is_loopback("::ffff:127.0.0.1"));
  CHECK(!server::is_loopback("10.0.0.1"));
  CHECK(!server::is_loopback("::ffff:10.0.0.1"));
  CHECK(!server::is_loopback("fe80::1"));
  CHECK(!server::is_loopback(""));
  std::printf("loopback: 127.0.0.0/8, ::1 and the IPv4-mapped form only\n");
}

void nothing_in_place() {
  MockTok tok;
  MockTemplate tmpl;
  MockEngine engine;
  server::Server s({tok, tmpl, engine}, base());
  CHECK(s.start());
  const auto bad = post(s, "/v1/control/reconfigure", "[1]");
  CHECK(bad);
  CHECK_EQ(bad->status, 400);
  const auto none = post(s, "/v1/control/reconfigure", R"({"spec": "off"})");
  CHECK(none);
  CHECK_EQ(none->status, 409);
  CHECK(none->body.find("restart it with the new flags") != std::string::npos);
  s.stop();
  std::printf("refusals: a non-object body 400, a server without reconfigure 409\n");
}

void switch_spec() {
  MockTok tok;
  MockTemplate tmpl;
  MockEngine engine;
  engine.script = tok.encode("w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12");
  engine.mtp = 3;
  engine.bursts = {4, 1, 3};
  server::Options o = base();
  o.mtp_auto = true;
  o.mtp_adaptive.max_k = 3;
  cli::InPlace in;
  in.spec = "mtp";
  in.spec_auto = true;
  in.spec_max = 3;
  in.host_experts_why = "it is for another model";
  std::vector<std::pair<uint32_t, bool>> calls;
  in.set_proposer = [&](uint32_t k, bool on) {
    calls.push_back({k, on});
    engine.mtp = on ? k : 0;
  };
  o.reconfigure = cli::reconfigure<Flags>(in, o);
  server::Server s({tok, tmpl, engine}, o);
  CHECK(s.start());
  const std::string mtp = complete(s);
  CHECK(engine.burst_calls > 0);

  const auto r = post(s, "/v1/control/reconfigure",
                      R"({"spec": "off", "spec-max": 2, "kv-cache": "int8", "max-len": 4096, "host-experts": "cpu",
                          "model": "other", "no-such-flag": 1, "spec-max-x": true})");
  CHECK(r);
  CHECK_EQ(r->status, 200);
  const json j = json::parse(r->body);
  CHECK_EQ(j.at("applied").size(), size_t(2));
  CHECK(j.at("applied")[0] == (json{{"setting", "spec"}, {"from", "mtp"}, {"to", "off"}}));
  CHECK(j.at("applied")[1] == (json{{"setting", "spec-max"}, {"from", "3"}, {"to", "2"}}));
  std::vector<std::string> refused;
  for (const json& x : j.at("refused")) refused.push_back(x.at("setting").get<std::string>() + ": " +
                                                          x.at("reason").get<std::string>());
  CHECK_EQ(refused.size(), size_t(6));
  // nlohmann::json's objects iterate in key order
  CHECK(refused[0] == "host-experts: it is for another model");
  CHECK(refused[1] == "kv-cache: needs a restart: it changes the memory plan");
  CHECK(refused[2] == "max-len: needs a restart: it changes the memory plan");
  CHECK(refused[3] == "model: serving another model needs a restart");
  CHECK(refused[4] == "no-such-flag: not a b70-serve flag");
  CHECK(refused[5] == "spec-max-x: not a b70-serve flag");
  CHECK_EQ(engine.mtp, 0U);
  CHECK_EQ(calls.back().second, false);
  engine.burst_calls = 0;
  CHECK(complete(s) == mtp);
  CHECK_EQ(engine.burst_calls, size_t(0));   // off: step() only

  const auto back = post(s, "/v1/control/reconfigure", R"({"spec": "auto"})");
  CHECK(back);
  CHECK_EQ(back->status, 200);
  CHECK(json::parse(back->body).at("applied")[0] == (json{{"setting", "spec"}, {"from", "off"}, {"to", "mtp"}}));
  CHECK_EQ(engine.mtp, 2U);   // the K the last request set
  engine.ks.clear();
  CHECK(complete(s) == mtp);
  CHECK(engine.burst_calls > 0);
  for (uint32_t k : engine.ks) CHECK(k <= 2U);

  const auto other = post(s, "/v1/control/reconfigure", R"({"spec": "dflash", "spec-max": 4})");
  CHECK(other);
  const json oj = json::parse(other->body);
  CHECK(oj.at("applied").empty());
  CHECK(oj.at("refused")[0].at("reason") == "--spec dflash needs a restart: this server switches between mtp and off");
  CHECK(oj.at("refused")[1].at("reason") == "--spec-max 4: this server's mtp runs K 1..3; a larger K needs a restart");
  s.stop();
  std::printf("spec: mtp -> off -> mtp (auto) through the adapter, K 3 -> 2, every reply the same bytes; restart "
              "flags, unknown keys and another proposer refused with their reasons\n");
}

// Plan 41f: Inkling 2 Mini's in-place switch as b70-serve sets it (serve_inkling): its drafter is the started proposer
// under the name auto (on) / off, K 1..7; --host-experts refused with its reason (every expert is on the card).
void switch_spec_inkling() {
  MockTok tok;
  MockTemplate tmpl;
  MockEngine engine;
  engine.script = tok.encode("w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12");
  engine.mtp = 7;
  engine.bursts = {4, 1, 3};
  server::Options o = base();
  o.mtp_auto = true;
  o.mtp_adaptive.max_k = 7;
  o.mtp_adaptive.cost = server::MtpCost::inkling_dspark();
  cli::InPlace in;
  in.spec = "auto";
  in.spec_auto = true;
  in.spec_max = 7;
  in.host_experts_why = "Inkling 2 Mini keeps every expert on the card";
  in.set_proposer = [&](uint32_t k, bool on) { engine.mtp = on ? k : 0; };
  o.reconfigure = cli::reconfigure<Flags>(in, o);
  server::Server s({tok, tmpl, engine}, o);
  CHECK(s.start());
  const std::string drafted = complete(s);
  const auto r = post(s, "/v1/control/reconfigure", R"({"spec": "off", "host-experts": "cpu"})");
  CHECK(r);
  const json j = json::parse(r->body);
  CHECK(j.at("applied")[0] == (json{{"setting", "spec"}, {"from", "auto"}, {"to", "off"}}));
  CHECK(j.at("refused")[0] == (json{{"setting", "host-experts"}, {"reason", "Inkling 2 Mini keeps every expert on the card"}}));
  CHECK_EQ(engine.mtp, 0U);
  CHECK(complete(s) == drafted);
  const auto back = post(s, "/v1/control/reconfigure", R"({"spec": "auto", "spec-max": 5})");
  CHECK(back);
  CHECK_EQ(json::parse(back->body).at("applied").size(), size_t(2));
  CHECK_EQ(engine.mtp, 5U);
  CHECK(complete(s) == drafted);
  const auto mtp = post(s, "/v1/control/reconfigure", R"({"spec": "mtp"})");
  CHECK(mtp);
  CHECK(json::parse(mtp->body).at("refused")[0].at("reason") ==
        "--spec mtp needs a restart: this server switches between auto and off");
  s.stop();
  std::printf("Inkling: the drafter auto -> off -> auto at K 5, every reply the same bytes; --host-experts and another "
              "proposer refused with their reasons\n");
}

}  // namespace

int main() {
  loopback();
  nothing_in_place();
  switch_spec();
  switch_spec_inkling();
  return 0;
}
