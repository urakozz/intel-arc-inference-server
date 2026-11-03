// The server's chat prompt ids against transformers' (Qwen3.8), over HTTP: the real renderer (chat::Template on the
// snapshot's chat_template.jinja), the real tokenizer (tokenizer.json) and a scripted engine. Two halves:
//
//   1. the committed sets: every request of tests/golden/toolcall (A4) and tests/golden/edit-copy (spec 19f), sent as
//      tools/box_validate/serve_client.py sends it, must come back with exactly the set's <name>.ids - the ids
//      transformers' apply_chat_template gives for the same messages and tools (spec 19f's open question: they did not,
//      because the request was held key-sorted and the template prints tools with tojson and arguments with | items);
//   2. one agentic session per case of tests/server/prompt_order_cases.json (tools/tokenizer/dump_prompt_order.py):
//      turn N (a history whose tool calls carry arguments as a JSON string, keys not alphabetical), the engine writes
//      the case's calls, and turn N+1 echoes the server's own response back as an OpenAI client does (arguments the
//      server's string) plus the tool results. Both prompts must be transformers' ids, and turn N+1 must start with
//      turn N's prompt + the generated ids + EOS: what the prefix cache holds after turn N (PrefixSession's resident),
//      so the next turn restores it instead of prefilling the history again. Nested objects, unicode, numbers, empty
//      arguments, several calls in one turn.
//
// usage: prompt_order_test <repo root>   (exit 77 without the Qwen3.8 snapshot: B70_SNAPSHOT_DIR or the HF cache)
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
#include "server/mock.h"
#include "server/server.h"
#include "tokenizer/chat_template.h"
#include "tokenizer/streamer.h"
#include "tokenizer/tokenizer.h"

namespace {

using ojson = nlohmann::ordered_json;

std::string slurp(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) return "";
  std::stringstream s;
  s << file.rdbuf();
  return s.str();
}

std::vector<uint32_t> read_ids(const std::string& path) {
  std::istringstream s(slurp(path));
  std::vector<uint32_t> ids;
  for (uint32_t id; s >> id;) ids.push_back(id);
  return ids;
}

// template_test's resolution: B70_SNAPSHOT_DIR, else the HF cache's Qwen3.8 snapshot.
std::string snapshot() {
  if (const char* env = std::getenv("B70_SNAPSHOT_DIR")) return env;
  std::string root;
  if (const char* hf = std::getenv("HF_HOME")) root = hf;
  else if (const char* home = std::getenv("HOME")) root = std::string(home) + "/.cache/huggingface";
  else root = ".cache/huggingface";
  return root + "/hub/models--urakozz--Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ/snapshots/84575a18f209992ef96d819b31f924b489e3d55d";
}

struct RealTok : server::TokIface {
  explicit RealTok(const std::string& path) : t(path) {}
  std::vector<uint32_t> encode(std::string_view text) override { return t.encode(text, false); }
  std::string decode(const std::vector<uint32_t>& ids) override { return t.decode(ids, false); }
  struct S : server::StreamerIface {
    explicit S(const tok::Tokenizer& tokenizer) : st(tokenizer) {}
    std::string push(uint32_t id) override { return st.push(id); }
    std::string flush() override { return st.flush(); }
    tok::Streamer st;
  };
  std::unique_ptr<server::StreamerIface> streamer() override { return std::make_unique<S>(t); }
  uint32_t vocab_used() override { return t.vocab_size(); }
  tok::Tokenizer t;
};

struct RealTemplate : server::TemplateIface {
  explicit RealTemplate(const std::string& dir) : tmpl(dir) {}
  std::string render(const nlohmann::ordered_json& messages, const nlohmann::ordered_json& tools, bool think) override {
    return tmpl.render(messages, tools, think);
  }
  chat::Template tmpl;
};

struct LongEngine : MockEngine {
  uint32_t max_len() override { return 131072; }   // the edit / copy set reaches 99k ids
};

constexpr uint32_t kImEnd = 248046;

struct Fixture {
  explicit Fixture(const std::string& snap)
      : tok(snap + "/tokenizer.json"), tmpl(snap), server({tok, tmpl, engine}, options()) {
    CHECK(server.start());
  }
  ~Fixture() { server.stop(); }
  static server::Options options() {
    server::Options o;
    o.host = "127.0.0.1";
    o.port = 0;
    o.served_model = "qwen";
    o.eos_ids = {kImEnd, 248044};   // generation_config.json's
    return o;
  }
  // The response, parsed in its own key order.
  ojson post(const ojson& body) {
    engine.at = 0;
    httplib::Client client("127.0.0.1", server.bound_port());
    client.set_read_timeout(300, 0);
    const auto response = client.Post("/v1/chat/completions", body.dump(), "application/json");
    CHECK(response);
    if (response->status != 200) std::fprintf(stderr, "status %d: %s\n", response->status, response->body.c_str());
    CHECK_EQ(response->status, 200);
    return ojson::parse(response->body);
  }
  RealTok tok;
  RealTemplate tmpl;
  LongEngine engine;
  server::Server server;
};

size_t common_prefix(const std::vector<uint32_t>& a, const std::vector<uint32_t>& b) {
  size_t i = 0;
  while (i < a.size() && i < b.size() && a[i] == b[i]) ++i;
  return i;
}

ojson request(const ojson& messages, const ojson& tools, bool thinking) {
  ojson b = {{"model", "qwen"}, {"messages", messages}};
  if (!tools.is_null() && !tools.empty()) b["tools"] = tools;
  b["chat_template_kwargs"] = {{"enable_thinking", thinking}};
  b["max_tokens"] = 1024;   // the scripted output ends in EOS well before
  b["temperature"] = 0;
  b["return_token_ids"] = true;
  return b;
}

// Half 1: every committed request's ids are the set's.
int run_set(Fixture& f, const std::string& dir) {
  const std::string manifest = slurp(dir + "/manifest.json");
  CHECK(!manifest.empty());
  int bad = 0, n = 0;
  for (const ojson& entry : ojson::parse(manifest)) {
    const std::string name = entry.at("name").get<std::string>();
    const ojson sc = ojson::parse(slurp(dir + "/" + name + ".json"));
    const std::vector<uint32_t> want = read_ids(dir + "/" + name + ".ids");
    f.engine.script = {kImEnd};
    const ojson r = f.post(request(sc.at("messages"), sc.contains("tools") ? sc.at("tools") : ojson(),
                                   sc.value("enable_thinking", false)));
    const auto got = r.at("prompt_token_ids").get<std::vector<uint32_t>>();
    ++n;
    if (got != want) {
      ++bad;
      std::fprintf(stderr, "%s/%s: %zu ids, the set's %zu; first difference at id %zu\n", dir.c_str(), name.c_str(),
                   got.size(), want.size(), common_prefix(got, want));
    }
  }
  std::printf("%s: %d/%d requests' prompt ids are the set's\n", dir.c_str(), n - bad, n);
  return bad;
}

// Half 2: turn N, the model's calls, turn N+1 echoing the server's response.
int run_session(Fixture& f, const ojson& tools, const ojson& c) {
  const std::string name = c.at("name").get<std::string>();
  int bad = 0;
  const auto want_n = c.at("want_n").get<std::vector<uint32_t>>();
  const auto want_n1 = c.at("want_n1").get<std::vector<uint32_t>>();

  f.engine.script = f.tok.encode(c.at("output").get<std::string>());
  f.engine.script.push_back(kImEnd);
  const ojson rn = f.post(request(c.at("messages"), tools, false));
  const auto pn = rn.at("prompt_token_ids").get<std::vector<uint32_t>>();
  const auto out = rn.at("choices").at(0).at("token_ids").get<std::vector<uint32_t>>();
  const ojson& msg = rn.at("choices").at(0).at("message");
  if (pn != want_n) {
    ++bad;
    std::fprintf(stderr, "%s: turn N %zu ids, transformers' %zu; first difference at id %zu\n", name.c_str(), pn.size(),
                 want_n.size(), common_prefix(pn, want_n));
  }
  CHECK_EQ(rn.at("choices").at(0).at("finish_reason"), std::string("tool_calls"));

  // The client appends the assistant message as the server sent it (arguments: the server's string) and the results.
  ojson messages = c.at("messages");
  ojson echoed = {{"role", "assistant"}, {"content", msg.at("content")}, {"tool_calls", ojson::array()}};
  for (const ojson& call : msg.at("tool_calls")) {
    echoed["tool_calls"].push_back({{"id", call.at("id")}, {"type", "function"},
                                    {"function", {{"name", call.at("function").at("name")},
                                                  {"arguments", call.at("function").at("arguments")}}}});
  }
  messages.push_back(echoed);
  size_t i = 0;
  for (const ojson& call : msg.at("tool_calls")) {
    messages.push_back({{"role", "tool"}, {"tool_call_id", call.at("id")}, {"content", c.at("results").at(i++)}});
  }
  CHECK_EQ(i, c.at("results").size());

  f.engine.script = {kImEnd};
  const ojson rn1 = f.post(request(messages, tools, false));
  const auto pn1 = rn1.at("prompt_token_ids").get<std::vector<uint32_t>>();
  if (pn1 != want_n1) {
    ++bad;
    std::fprintf(stderr, "%s: turn N+1 %zu ids, transformers' %zu; first difference at id %zu\n", name.c_str(),
                 pn1.size(), want_n1.size(), common_prefix(pn1, want_n1));
  }
  // What the prefix cache holds after turn N: its prompt, the generated ids and the EOS the engine was fed.
  std::vector<uint32_t> resident = pn;
  resident.insert(resident.end(), out.begin(), out.end());
  resident.push_back(kImEnd);
  const size_t shared = common_prefix(resident, pn1);
  std::printf("%s: turn N %zu ids + %zu generated + EOS; turn N+1 %zu ids share %zu of them (%s), %zu to prefill\n",
              name.c_str(), pn.size(), out.size(), pn1.size(), shared,
              shared == resident.size() ? "all" : "DIVERGES", pn1.size() - shared);
  if (shared != resident.size()) ++bad;
  return bad;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string root = argc > 1 ? argv[1] : ".";
  const std::string snap = snapshot();
  if (slurp(snap + "/tokenizer.json").empty() || slurp(snap + "/chat_template.jinja").empty()) {
    std::printf("SKIP: no Qwen3.8 snapshot at %s\n", snap.c_str());
    return 77;
  }
  Fixture f(snap);
  int bad = 0;
  const ojson cases = ojson::parse(slurp(root + "/tests/server/prompt_order_cases.json"));
  for (const ojson& c : cases.at("cases")) bad += run_session(f, cases.at("tools"), c);
  bad += run_set(f, root + "/tests/golden/toolcall");
  bad += run_set(f, root + "/tests/golden/edit-copy");
  CHECK_EQ(bad, 0);
  std::printf("prompt_order_test OK\n");
  return 0;
}
