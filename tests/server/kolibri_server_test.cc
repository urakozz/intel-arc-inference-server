// Spec 20e: the server's Kolibri-1 dispatch over a mock engine (no device, no checkpoint) - the
// chat-format half of the wiring b70-serve uses for Kolibri (the engine half, KolibriEngine behind
// cli::kolibri::KolibriEngineAdapterT, is kolibri_snapshot_test's). With
// Options::chat_format = ChatFormat::for_model_type("kolibri1"):
//   - the request's chat_template_kwargs reach the template (render_with_kwargs), checked first: a bad
//     reasoning_effort is a 400 naming it, and nothing is rendered;
//   - a message's list of text parts reaches the template as one string (joined with "\n");
//   - the output parsed as Kolibri's: reasoning opened by the model's <think> (the prompt does not end
//     in it), hermes JSON calls, finish_reason tool_calls - in the body and in streamed frames (one
//     character per token) alike; German bytes kept;
//   - both EOS ids [127906, 127901] stop generation with finish_reason stop;
//   - Options::sampling_defaults (from generation_config.json via server::generation_sampling: the
//     vendored tests/tokenizer/kolibri copy, T 1.0, top-p 0.97, top-k 128, sampled) is what a request
//     without temperature / top_p / top_k samples with; a request's own fields win; temperature 0 is
//     greedy; and a Qwen-format server keeps Sampling{} (greedy, T 1.0, top-p 0.95, top-k 20).
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "check.h"
#include "server/mock.h"
#include "server/openai.h"
#include "server/server.h"

namespace {

using json = nlohmann::json;

// Kolibri's template in the shape the server sees: ChatML, the generation prompt "<|im_start|>assistant\n"
// (thinking on) or "...<think>\n\n</think>\n\n" (reasoning_effort none / enable_thinking false).
struct KolibriTemplate : server::TemplateIface {
  int plain_calls = 0, kwargs_calls = 0;
  json last_kwargs, last_messages;
  std::string render(const json& messages, const json&, bool) override {
    ++plain_calls;
    return body(messages) + "<|im_start|>assistant\n";
  }
  std::string render_with_kwargs(const json& messages, const json&, bool think, const json& kwargs) override {
    ++kwargs_calls;
    last_kwargs = kwargs;
    last_messages = messages;
    const bool off = kwargs.value("reasoning_effort", json()).is_string() ? kwargs.at("reasoning_effort") == "none"
                                                                          : !think;
    return body(messages) + "<|im_start|>assistant\n" + (off ? "<think>\n\n</think>\n\n" : "");
  }
  static std::string body(const json& messages) {
    std::string text;
    for (const auto& m : messages)
      text += "<|im_start|>" + m.at("role").get<std::string>() + "\n" + m.at("content").get<std::string>() +
              "<|im_end|>\n";
    return text;
  }
};

// MockEngine, recording the sampling of every step.
struct RecordingEngine : MockEngine {
  std::vector<server::Sampling> seen;
  uint32_t step(const server::Sampling& s) override {
    seen.push_back(s);
    return MockEngine::step(s);
  }
};

std::string slurp(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  CHECK(file.good());
  std::stringstream contents;
  contents << file.rdbuf();
  return contents.str();
}

server::Sampling g_defaults;   // from the vendored generation_config.json, read in main()

struct Fixture {
  explicit Fixture(bool kolibri = true) : options(make_options(kolibri)), server({tok, tmpl, engine}, options) {
    CHECK(server.start());
  }
  ~Fixture() { server.stop(); }
  static server::Options make_options(bool kolibri) {
    server::Options o;
    o.host = "127.0.0.1";
    o.port = 0;
    o.served_model = "kolibri";
    o.eos_ids = {127906, 127901};   // generation_config.json's
    if (kolibri) {
      o.chat_format = server::ChatFormat::for_model_type("kolibri1");
      o.sampling_defaults = g_defaults;
    }
    return o;
  }
  httplib::Client client() { return httplib::Client("127.0.0.1", server.bound_port()); }
  // The engine generates `text` one UTF-8 character per token (a real streamer never emits half a
  // character, and a streamed frame must be valid JSON), then `eos`.
  void script(const std::string& text, uint32_t eos) {
    engine.script.clear();
    for (size_t at = 0; at < text.size();) {
      size_t n = 1;
      while (at + n < text.size() && (static_cast<unsigned char>(text[at + n]) & 0xC0) == 0x80) ++n;
      const std::vector<uint32_t> id = tok.raw(text.substr(at, n), n);
      engine.script.insert(engine.script.end(), id.begin(), id.end());
      at += n;
    }
    engine.script.push_back(eos);
  }

  MockTok tok;
  KolibriTemplate tmpl;
  RecordingEngine engine;
  server::Options options;
  server::Server server;
};

const json kTools = json::parse(R"([{"type":"function","function":{"name":"grep","parameters":{"type":"object",
  "properties":{"pattern":{"type":"string"},"path":{"type":"string"}},"required":["pattern"]}}}])");

json body(const json& kwargs = json::object(), bool stream = false) {
  json b = {{"model", "kolibri"},
            {"messages", json::array({{{"role", "user"}, {"content", "Suche „Größe“ im Code."}}})},
            {"tools", kTools},
            {"stream", stream}};
  if (!kwargs.empty()) b["chat_template_kwargs"] = kwargs;
  return b;
}

json post(Fixture& f, const json& b, int want_status = 200) {
  auto client = f.client();
  const auto response = client.Post("/v1/chat/completions", b.dump(), "application/json");
  CHECK(response);
  if (response->status != want_status) std::fprintf(stderr, "status %d: %s\n", response->status, response->body.c_str());
  CHECK_EQ(response->status, want_status);
  return json::parse(response->body);
}

const std::string kCallOutput =
    "<think>\nIch suche nach „Größe“.\n</think>\n\nIch suche.\n<tool_call>\n{\"name\": \"grep\", \"arguments\": "
    "{\"pattern\": \"Größe\", \"path\": \"/w/maß\"}}\n</tool_call>";

void case_call() {
  Fixture f;
  f.script(kCallOutput, 127906);
  const json r = post(f, body());
  CHECK_EQ(f.tmpl.kwargs_calls, 1);
  CHECK_EQ(f.tmpl.plain_calls, 0);
  const json& choice = r.at("choices").at(0);
  CHECK_EQ(choice.at("finish_reason"), std::string("tool_calls"));
  const json& m = choice.at("message");
  CHECK_EQ(m.at("reasoning_content"), std::string("Ich suche nach „Größe“."));
  CHECK_EQ(m.at("content"), std::string("Ich suche."));
  CHECK_EQ(m.at("tool_calls").size(), 1U);
  CHECK_EQ(m.at("tool_calls").at(0).at("function").at("name"), std::string("grep"));
  const std::string args = m.at("tool_calls").at(0).at("function").at("arguments");
  CHECK(args.find("Größe") != std::string::npos);   // UTF-8, not re-escaped
  CHECK_EQ(json::parse(args), json({{"pattern", "Größe"}, {"path", "/w/maß"}}));
  CHECK(r.at("usage").at("completion_tokens").get<size_t>() < kCallOutput.size());   // a character a token
  std::printf("call: reasoning opened by the model, content, a hermes call, finish_reason tool_calls\n");
}

void case_eos() {
  // <|endoftext|> (127901) stops as <|im_end|> (127906) does; thinking off: the output is the answer.
  for (uint32_t eos : {127906u, 127901u}) {
    Fixture f;
    f.script("Paris.", eos);
    const json r = post(f, body({{"reasoning_effort", "none"}}));
    const json& choice = r.at("choices").at(0);
    CHECK_EQ(choice.at("finish_reason"), std::string("stop"));
    CHECK_EQ(choice.at("message").at("content"), std::string("Paris."));
    CHECK(!choice.at("message").contains("reasoning_content") || choice.at("message").at("reasoning_content").is_null() ||
          choice.at("message").at("reasoning_content") == "");
    CHECK_EQ(r.at("usage").at("completion_tokens"), 6);
  }
  std::printf("eos: 127906 and 127901 both stop (finish_reason stop)\n");
}

void case_kwargs() {
  Fixture f;
  f.script("<think>\nkurz\n</think>\n\nJa.", 127906);
  const json kwargs = {{"reasoning_effort", "low"}, {"enable_thinking", true}};
  const json r = post(f, body(kwargs));
  CHECK_EQ(f.tmpl.last_kwargs, kwargs);
  const json& m = r.at("choices").at(0).at("message");
  CHECK_EQ(m.at("reasoning_content"), std::string("kurz"));
  CHECK_EQ(m.at("content"), std::string("Ja."));
  // A bad value is the client's: 400 naming the field, nothing rendered.
  Fixture g;
  const json bad = post(g, body({{"reasoning_effort", "ultra"}}), 400);
  CHECK(bad.at("error").at("message").get<std::string>().find("reasoning_effort") != std::string::npos);
  CHECK_EQ(g.tmpl.kwargs_calls, 0);
  // List content reaches the template as one string.
  Fixture h;
  h.script("ok", 127906);
  json b = body();
  b["messages"] = json::array({{{"role", "user"}, {"content", json::array({{{"type", "text"}, {"text", "Teil eins"}},
                                                                           {{"type", "text"}, {"text", "Teil zwei"}}})}}});
  post(h, b);
  CHECK_EQ(h.tmpl.last_messages.at(0).at("content"), json("Teil eins\nTeil zwei"));
  std::printf("kwargs: reasoning_effort reaches the template, a bad one is a 400; list content joined\n");
}

void case_stream() {
  Fixture f;
  f.script(kCallOutput, 127906);
  auto client = f.client();
  std::string sse;
  const auto response = client.Post("/v1/chat/completions", httplib::Headers(), body({}, true).dump(),
                                    "application/json", [&sse](const char* d, size_t n) {
                                      sse.append(d, n);
                                      return true;
                                    },
                                    nullptr);
  CHECK(response);
  CHECK_EQ(response->status, 200);
  std::string reasoning, content, finish;
  json calls = json::array();
  for (size_t at = 0, end; (end = sse.find("\n\n", at)) != std::string::npos; at = end + 2) {
    const std::string frame = sse.substr(at, end - at);
    if (frame == "data: [DONE]") continue;
    const json b = json::parse(frame.substr(6));
    const json& c = b.at("choices").at(0);
    const json& d = c.at("delta");
    if (d.contains("reasoning_content")) reasoning += d.at("reasoning_content").get<std::string>();
    if (d.contains("content") && d.at("content").is_string()) content += d.at("content").get<std::string>();
    if (d.contains("tool_calls")) calls.push_back(d.at("tool_calls").at(0));
    if (!c.at("finish_reason").is_null()) finish = c.at("finish_reason");
  }
  CHECK_EQ(reasoning, std::string("Ich suche nach „Größe“."));
  CHECK_EQ(content, std::string("Ich suche."));
  CHECK_EQ(calls.size(), 1U);
  CHECK_EQ(calls.at(0).at("index"), 0);
  CHECK_EQ(calls.at(0).at("function").at("name"), std::string("grep"));
  CHECK_EQ(finish, std::string("tool_calls"));
  std::printf("stream: one character per token, the same reasoning, content and call as the body\n");
}

void case_sampling() {
  CHECK(!g_defaults.greedy);
  CHECK_EQ(g_defaults.temperature, 1.0f);
  CHECK_EQ(g_defaults.top_p, 0.97f);
  CHECK_EQ(g_defaults.top_k, 128u);
  {
    Fixture f;
    f.script("a", 127906);
    post(f, body());   // no sampling fields: the model's defaults
    CHECK(!f.engine.seen.empty());
    const server::Sampling& s = f.engine.seen.front();
    CHECK(!s.greedy);
    CHECK_EQ(s.temperature, 1.0f);
    CHECK_EQ(s.top_p, 0.97f);
    CHECK_EQ(s.top_k, 128u);
    CHECK(!s.has_seed);
  }
  {
    Fixture f;
    f.script("a", 127906);
    json b = body();
    b["temperature"] = 0.6;
    b["top_k"] = 20;
    b["seed"] = 7;
    post(f, b);   // the request's own fields win; the rest stay the model's
    const server::Sampling& s = f.engine.seen.front();
    CHECK(!s.greedy);
    CHECK_EQ(s.temperature, 0.6f);
    CHECK_EQ(s.top_k, 20u);
    CHECK_EQ(s.top_p, 0.97f);
    CHECK(s.has_seed && s.seed == 7u);
  }
  {
    Fixture f;
    f.script("a", 127906);
    json b = body();
    b["temperature"] = 0;
    post(f, b);
    CHECK(f.engine.seen.front().greedy);
  }
  {
    Fixture f(/*kolibri=*/false);   // a Qwen-format server: Sampling{} exactly as before
    f.script("a", 127906);
    post(f, body());
    const server::Sampling& s = f.engine.seen.front();
    CHECK(s.greedy);
    CHECK_EQ(s.temperature, 1.0f);
    CHECK_EQ(s.top_p, 0.95f);
    CHECK_EQ(s.top_k, 20u);
    CHECK_EQ(f.tmpl.plain_calls, 1);
  }
  // generation_sampling's own rules.
  CHECK(server::generation_sampling(json::object()).greedy);
  CHECK(server::generation_sampling({{"do_sample", false}, {"temperature", 0.7}}).greedy);
  CHECK(server::generation_sampling({{"do_sample", true}, {"temperature", 0}}).greedy);
  bool threw = false;
  try {
    (void)server::generation_sampling({{"top_p", 1.5}});
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);
  std::printf("sampling: Kolibri's defaults 1.0 / 0.97 / 128 sampled; request fields win; Qwen keeps Sampling{}\n");
}

}  // namespace

int main(int argc, char** argv) {
  const std::string dir = argc > 1 ? argv[1] : "tests/tokenizer/kolibri";
  g_defaults = server::generation_sampling(json::parse(slurp(dir + "/generation_config.json")));
  case_call();
  case_eos();
  case_kwargs();
  case_stream();
  case_sampling();
  std::printf("kolibri_server_test OK: 5 cases\n");
  return 0;
}
