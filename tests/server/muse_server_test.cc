// Spec 23e Task 3: the server's Muse-Glimmer-30B request path over mock engines (no device, no checkpoint) and the
// REAL chat template (chat::Template over the vendored tests/tokenizer/muse/chat_template.jinja) - the chat-format half
// of what b70-serve wires for model_type muse_glimmer:
//   muse_server_test <tests/tokenizer/muse>
//   - chat_template_kwargs reach the template, checked first: reasoning_strength low / medium / high / xhigh (default
//     the template's high), the top-level OpenAI reasoning_effort mapped onto it (spec 23 decision 3; the kwarg wins
//     when both are given), a bad value of either a 400 naming it with nothing rendered; current_date /
//     knowledge_cutoff strings, tool_namespace_descriptions an object of strings;
//   - an image / video content part is a 400 naming the vision encoder (out of scope, spec 23 §10);
//   - Review Focus 5: a tool call's arguments given as a JSON string are parsed into an object before rendering (the
//     template raises on strings); a string that is not a JSON object is a 400 naming the call - never a 500;
//   - Review Focus 3: <|eot|> (200008) and <|end_of_text|> (200001) stop; <|eom|> (200007) does not - a turn of
//     reasoning, calls and an answer is one generation; reasoning / content / calls in the body and in streamed
//     frames (one character a token) alike, finish_reason tool_calls;
//   - Review Focus 6: a generation cut mid-call by max_tokens: finish_reason length, no call, no markup in content;
//   - decision 4: a request that names no sampling field samples with generation_config.json's T 1.0 / top-k 64 /
//     top-p 0.95, the model's logit multiplier and softcap riding along (model properties, never request fields); a
//     request's own fields win;
//   - Review Focus 4: the softcap applied once, on the host, inside server::filter_probs: the filtered probabilities
//     are softmax(20 tanh(z x 0.196.../20) / T) (fp32 cap) restricted to top-k / top-p, a seeded 10^5-draw
//     chi-square against the capped distribution passes (p >= 0.01) and against the uncapped one fails; the
//     acceptance paths (spec_accept: accept_point_mass, accept_sampled) see exactly that capped p; softcap 0 (every
//     other model) leaves filter_probs as it was;
//   - generation_config.json's stop_strings read as a string or a list (HF allows both; refused naming the key
//     otherwise) and merged with a request's stop (Muse's file has none: a modified copy here);
//   - a restored request reports usage.prompt_tokens_details.cached_tokens (spec 7's, asserted for Muse).
#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "check.h"
#include "server/chat_format.h"
#include "server/mock.h"
#include "server/openai.h"
#include "server/server.h"
#include "server/spec_accept.h"
#include "cli/muse_serve.h"
#include "server/toolcall_muse.h"
#include "tokenizer/chat_template.h"

namespace {

using json = nlohmann::json;
using Ids = std::vector<uint32_t>;

constexpr uint32_t kEom = 200007, kEot = 200008, kEndOfText = 200001;

// Muse's special ids decode to their tags (the server's streamer keeps special tokens); scripted output pieces are
// ids from 300000 (never an EOS id); the prompt is encoded word by word from 1000.
struct MuseTok : server::TokIface {
  std::map<uint32_t, std::string> pieces = {{kEom, "<|eom|>"}, {kEot, "<|eot|>"}, {kEndOfText, "<|end_of_text|>"}};
  std::map<std::string, uint32_t> words;
  uint32_t next_word = 1000, next_raw = 300000;

  uint32_t word(const std::string& w) {
    const auto it = words.find(w);
    if (it != words.end()) return it->second;
    pieces[next_word] = w + " ";
    words[w] = next_word;
    return next_word++;
  }
  std::vector<uint32_t> encode(std::string_view text) override {
    Ids out;
    std::string w;
    for (char c : text) {
      if (c == ' ' || c == '\n') {
        if (!w.empty()) out.push_back(word(w));
        w.clear();
      } else {
        w += c;
      }
    }
    if (!w.empty()) out.push_back(word(w));
    return out;
  }
  Ids raw(const std::string& text) {   // one UTF-8 character a token
    Ids out;
    for (size_t at = 0; at < text.size();) {
      size_t n = 1;
      while (at + n < text.size() && (static_cast<unsigned char>(text[at + n]) & 0xC0) == 0x80) ++n;
      pieces[next_raw] = text.substr(at, n);
      out.push_back(next_raw++);
      at += n;
    }
    return out;
  }
  std::string decode(const std::vector<uint32_t>& ids) override {
    std::string s;
    for (uint32_t id : ids) s += pieces.at(id);
    return s;
  }
  struct S : server::StreamerIface {
    explicit S(MuseTok& t) : t(t) {}
    std::string push(uint32_t id) override { return t.pieces.at(id); }
    std::string flush() override { return ""; }
    MuseTok& t;
  };
  std::unique_ptr<server::StreamerIface> streamer() override { return std::make_unique<S>(*this); }
  uint32_t vocab_used() override { return 202048; }
};

// The real template (the original's file), recording what it rendered.
struct RealTemplate : server::TemplateIface {
  explicit RealTemplate(const std::string& dir) : t(dir) {}
  std::string render(const json& messages, const json& tools, bool think) override {
    ++plain_calls;
    return last = t.render(messages, tools, think);
  }
  std::string render_with_kwargs(const json& messages, const json& tools, bool think, const json& kwargs) override {
    ++kwargs_calls;
    last_kwargs = kwargs;
    return last = t.render(messages, tools, think, kwargs);
  }
  chat::Template t;
  std::string last;
  json last_kwargs;
  int plain_calls = 0, kwargs_calls = 0;
};

struct RecordingEngine : MockEngine {
  std::vector<server::Sampling> seen;
  uint32_t step(const server::Sampling& s) override {
    seen.push_back(s);
    return MockEngine::step(s);
  }
};

std::string g_dir;
server::Sampling g_defaults;   // generation_config.json's + Muse's multiplier and softcap (what b70-serve sets)

std::string slurp(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  CHECK(file.good());
  std::stringstream contents;
  contents << file.rdbuf();
  return contents.str();
}

server::Options muse_options() {
  server::Options o;
  o.host = "127.0.0.1";
  o.port = 0;
  o.served_model = "muse";
  o.eos_ids = {kEndOfText, kEot};   // generation_config.json's
  o.chat_format = server::ChatFormat::for_model_type("muse_glimmer");
  o.sampling_defaults = g_defaults;
  return o;
}

struct Fixture {
  explicit Fixture(server::Options o = muse_options()) : tmpl(g_dir), server({tok, tmpl, engine}, o) {
    CHECK(server.start());
  }
  ~Fixture() { server.stop(); }
  void script(const std::string& text, uint32_t eos) {
    engine.script = tok.raw(text);
    if (eos) engine.script.push_back(eos);
  }
  json post(const json& b, int want = 200) {
    httplib::Client c("127.0.0.1", server.bound_port());
    const auto r = c.Post("/v1/chat/completions", b.dump(), "application/json");
    CHECK(r);
    if (r->status != want) std::fprintf(stderr, "status %d: %s\n", r->status, r->body.c_str());
    CHECK_EQ(r->status, want);
    return json::parse(r->body);
  }
  MuseTok tok;
  RealTemplate tmpl;
  RecordingEngine engine;
  server::Server server;
};

const json kTools = json::parse(R"([{"type":"function","function":{"name":"fs.read_file","description":"Read a file.",
  "parameters":{"type":"object","properties":{"path":{"type":"string"},"offset":{"type":"integer"}},"required":["path"]}}}])");

json body(const json& extra = json::object()) {
  json b = {{"model", "muse"},
            {"messages", json::array({{{"role", "user"}, {"content", "Read src/main.cc"}}})},
            {"tools", kTools},
            {"chat_template_kwargs", {{"current_date", "2026-10-15"}}}};
  for (auto it = extra.begin(); it != extra.end(); ++it) b[it.key()] = it.value();
  return b;
}

std::string message_of(const json& error) { return error.at("error").at("message").get<std::string>(); }

const std::string kTurn =
    " to=self<|message|>I should read the file.<|eom|><|start|>assistant to=fs.read_file<|message|>"
    "<atem:function_calls>\n<atem:invoke name=\"fs.read_file\">\n<atem:parameter name=\"path\">src/main.cc"
    "</atem:parameter>\n<atem:parameter name=\"offset\">7</atem:parameter>\n</atem:invoke>\n</atem:function_calls>";

// ---- kwargs, reasoning strength --------------------------------------------------------------------------------
void case_kwargs() {
  CHECK(server::ChatFormat::for_model_type("muse_glimmer").kind == server::ChatFormat::Kind::Muse);
  CHECK(server::ChatFormat::muse().template_kwargs());
  CHECK_EQ(std::string(server::ChatFormat::muse().name()), std::string("muse"));
  const std::unique_ptr<server::OutputParser> op =
      server::make_output_parser(server::ChatFormat::muse(), "<|start|>assistant", json::object(), kTools);
  CHECK(dynamic_cast<server::MuseOutputStream*>(op.get()) != nullptr);
  struct Want {
    json extra;
    const char* line;
  };
  for (const Want& w : {Want{json::object(), "Reasoning strength: high."},
                        Want{{{"chat_template_kwargs", {{"current_date", "2026-10-15"}, {"reasoning_strength", "low"}}}},
                             "Reasoning strength: low."},
                        Want{{{"reasoning_effort", "medium"}}, "Reasoning strength: medium."},
                        Want{{{"reasoning_effort", "low"},
                              {"chat_template_kwargs", {{"current_date", "2026-10-15"}, {"reasoning_strength", "xhigh"}}}},
                             "Reasoning strength: xhigh."}}) {
    Fixture f;
    f.script(" to=user<|message|>ok", kEot);
    f.post(body(w.extra));
    CHECK(f.tmpl.last.find(w.line) != std::string::npos);
    CHECK(f.tmpl.last.find("Current date: 2026-10-15.") != std::string::npos);
    CHECK_EQ(f.tmpl.kwargs_calls, 1);
  }
  struct Bad {
    json extra;
    const char* names;
  };
  for (const Bad& b : {Bad{{{"chat_template_kwargs", {{"reasoning_strength", "ultra"}}}}, "reasoning_strength"},
                       Bad{{{"reasoning_effort", "minimal"}}, "reasoning_effort"},
                       Bad{{{"reasoning_effort", 3}}, "reasoning_effort"},
                       Bad{{{"chat_template_kwargs", {{"current_date", 20261015}}}}, "current_date"},
                       Bad{{{"chat_template_kwargs", {{"knowledge_cutoff", true}}}}, "knowledge_cutoff"},
                       Bad{{{"chat_template_kwargs", {{"tool_namespace_descriptions", "fs"}}}}, "tool_namespace_descriptions"},
                       Bad{{{"chat_template_kwargs", {{"tool_namespace_descriptions", {{"fs", 1}}}}}},
                           "tool_namespace_descriptions"}}) {
    Fixture f;
    const json e = f.post(body(b.extra), 400);
    CHECK(message_of(e).find(b.names) != std::string::npos);
    CHECK_EQ(f.tmpl.kwargs_calls, 0);
  }
  std::printf("kwargs: reasoning_strength and reasoning_effort reach the template (default high, the kwarg wins), "
              "bad values are 400s naming them, nothing rendered\n");
}

// ---- the vision refusal, string arguments --------------------------------------------------------------------------
void case_request_shapes() {
  for (const char* type : {"image_url", "image", "video", "video_url", "input_audio"}) {
    Fixture f;
    json b = body();
    b["messages"] = json::array({{{"role", "user"},
                                  {"content", json::array({{{"type", "text"}, {"text", "What is this?"}},
                                                           {{"type", type}, {type, {{"url", "x"}}}}})}}});
    const json e = f.post(b, 400);
    CHECK(message_of(e).find("vision encoder") != std::string::npos);
    CHECK(message_of(e).find(type) != std::string::npos);
  }
  // Text parts are concatenated as the template concatenates them.
  {
    Fixture f;
    f.script(" to=user<|message|>ok", kEot);
    json b = body();
    b["messages"] = json::array({{{"role", "user"}, {"content", json::array({{{"type", "text"}, {"text", "Part one. "}},
                                                                             {{"text", "Part two."}}})}}});
    f.post(b);
    CHECK(f.tmpl.last.find("<|start|>user<|message|>Part one. Part two.<|eot|>") != std::string::npos);
  }
  // Review Focus 5: string arguments parsed into an object before rendering.
  const auto history = [](const json& arguments) {
    return json::array({{{"role", "user"}, {"content", "Read it."}},
                        {{"role", "assistant"}, {"content", nullptr},
                         {"tool_calls", json::array({{{"id", "call_7"}, {"type", "function"},
                                                      {"function", {{"name", "fs.read_file"}, {"arguments", arguments}}}}})}},
                        {{"role", "tool"}, {"tool_call_id", "call_7"}, {"content", "int main() {}"}}});
  };
  std::string rendered_object;
  {
    Fixture f;
    f.script(" to=user<|message|>ok", kEot);
    json b = body();
    b["messages"] = history(json{{"path", "a.cc"}, {"offset", 2}});
    f.post(b);
    rendered_object = f.tmpl.last;
  }
  {
    Fixture f;
    f.script(" to=user<|message|>ok", kEot);
    json b = body();
    b["messages"] = history("{\"path\": \"a.cc\", \"offset\": 2}");
    f.post(b);
    CHECK_EQ(f.tmpl.last, rendered_object);
    CHECK(f.tmpl.last.find("<atem:parameter name=\"path\">a.cc</atem:parameter>") != std::string::npos);
    CHECK(f.tmpl.last.find("<|start|>tool fs.read_file<|message|>") != std::string::npos);   // by tool_call_id
  }
  for (const char* bad : {"[1, 2]", "not json", "\"a string\""}) {
    Fixture f;
    json b = body();
    b["messages"] = history(bad);
    const json e = f.post(b, 400);
    CHECK(message_of(e).find("fs.read_file") != std::string::npos);
    CHECK(message_of(e).find("call_7") != std::string::npos);
  }
  std::printf("request shapes: image / video parts refused naming the vision encoder; text parts concatenated; string "
              "arguments parsed (the same render as the object), a non-object string a 400 naming the call\n");
}

// ---- EOS, the turn, streaming, a cut call ---------------------------------------------------------------------
void case_turn() {
  for (uint32_t eos : {kEot, kEndOfText}) {
    Fixture f;
    f.script(" to=self<|message|>Think.<|eom|><|start|>assistant to=user<|message|>Paris.", eos);
    const json r = f.post(body());
    const json& c = r.at("choices").at(0);
    CHECK_EQ(c.at("finish_reason"), std::string("stop"));
    CHECK_EQ(c.at("message").at("reasoning_content"), std::string("Think."));
    CHECK_EQ(c.at("message").at("content"), std::string("Paris."));
    // every scripted id is a completion token but the EOS
    CHECK_EQ(r.at("usage").at("completion_tokens").get<size_t>(), f.engine.script.size() - 1);
  }
  // <|eom|> as its own id does not stop: reasoning, <|eom|>, then the call, then <|eot|>.
  {
    Fixture f;
    f.engine.script = f.tok.raw(" to=self<|message|>R");
    f.engine.script.push_back(kEom);
    const Ids rest = f.tok.raw("<|start|>assistant to=user<|message|>A");
    f.engine.script.insert(f.engine.script.end(), rest.begin(), rest.end());
    f.engine.script.push_back(kEot);
    const json r = f.post(body());
    const json& m = r.at("choices").at(0).at("message");
    CHECK_EQ(m.at("reasoning_content"), std::string("R"));
    CHECK_EQ(m.at("content"), std::string("A"));
    CHECK_EQ(r.at("choices").at(0).at("finish_reason"), std::string("stop"));
  }
  // A reasoning block then a call: one generation, finish_reason tool_calls - body and stream alike.
  {
    Fixture f;
    f.script(kTurn, kEot);
    const json r = f.post(body());
    const json& c = r.at("choices").at(0);
    CHECK_EQ(c.at("finish_reason"), std::string("tool_calls"));
    CHECK_EQ(c.at("message").at("reasoning_content"), std::string("I should read the file."));
    CHECK(c.at("message").at("content").is_null());
    CHECK_EQ(c.at("message").at("tool_calls").size(), 1U);
    CHECK_EQ(c.at("message").at("tool_calls").at(0).at("function").at("name"), std::string("fs.read_file"));
    CHECK_EQ(json::parse(c.at("message").at("tool_calls").at(0).at("function").at("arguments").get<std::string>()),
             json({{"path", "src/main.cc"}, {"offset", 7}}));
  }
  {
    Fixture f;
    f.script(kTurn, kEot);
    json b = body({{"stream", true}});
    httplib::Client client("127.0.0.1", f.server.bound_port());
    std::string sse;
    const auto response = client.Post("/v1/chat/completions", httplib::Headers(), b.dump(), "application/json",
                                      [&sse](const char* d, size_t n) {
                                        sse.append(d, n);
                                        return true;
                                      },
                                      nullptr);
    CHECK(response);
    std::string reasoning, content, finish;
    json calls = json::array();
    for (size_t at = 0, end; (end = sse.find("\n\n", at)) != std::string::npos; at = end + 2) {
      const std::string frame = sse.substr(at, end - at);
      if (frame == "data: [DONE]") continue;
      const json fb = json::parse(frame.substr(6));
      const json& c = fb.at("choices").at(0);
      const json& d = c.at("delta");
      if (d.contains("reasoning_content")) reasoning += d.at("reasoning_content").get<std::string>();
      if (d.contains("content") && d.at("content").is_string()) content += d.at("content").get<std::string>();
      if (d.contains("tool_calls")) calls.push_back(d.at("tool_calls").at(0));
      if (!c.at("finish_reason").is_null()) finish = c.at("finish_reason");
    }
    CHECK_EQ(reasoning, std::string("I should read the file."));
    CHECK_EQ(content, std::string());
    CHECK_EQ(calls.size(), 1U);
    CHECK_EQ(calls.at(0).at("function").at("name"), std::string("fs.read_file"));
    CHECK_EQ(finish, std::string("tool_calls"));
  }
  // Review Focus 6: max_tokens cuts mid-invoke - length, no call, no markup.
  {
    Fixture f;
    f.script(kTurn, kEot);
    const size_t cut = kTurn.find("src/main") + 4;
    const json r = f.post(body({{"max_tokens", cut}}));
    const json& c = r.at("choices").at(0);
    CHECK_EQ(c.at("finish_reason"), std::string("length"));
    CHECK(!c.at("message").contains("tool_calls"));
    CHECK_EQ(c.at("message").at("content"), std::string());
    CHECK_EQ(c.at("message").at("reasoning_content"), std::string("I should read the file."));
  }
  std::printf("turn: <|eot|> and <|end_of_text|> stop, <|eom|> does not; reasoning + call = one generation "
              "(tool_calls), streamed alike; a call cut by max_tokens vanishes (length)\n");
}

// ---- sampling: the defaults, the softcap -------------------------------------------------------------------------
float cap(float z) { return 20.0f * std::tanh(z * 0.19611613513818404f / 20.0f); }

void case_sampling() {
  CHECK(!g_defaults.greedy);
  CHECK_EQ(g_defaults.temperature, 1.0f);
  CHECK_EQ(g_defaults.top_k, 64u);
  CHECK_EQ(g_defaults.top_p, 0.95f);
  CHECK_EQ(g_defaults.logit_mult, 0.19611613513818404f);
  CHECK_EQ(g_defaults.softcap, 20.0f);
  CHECK_EQ(server::Sampling{}.softcap, 0.0f);
  CHECK_EQ(server::Sampling{}.logit_mult, 1.0f);
  {
    Fixture f;
    f.script(" to=user<|message|>a", kEot);
    f.post(body());
    const server::Sampling& s = f.engine.seen.front();
    CHECK(!s.greedy && s.temperature == 1.0f && s.top_k == 64u && s.top_p == 0.95f);
    CHECK(s.softcap == 20.0f && s.logit_mult == 0.19611613513818404f);
  }
  {
    Fixture f;
    f.script(" to=user<|message|>a", kEot);
    f.post(body({{"temperature", 0.7}, {"top_k", 10}, {"seed", 5}, {"softcap", 1}, {"logit_mult", 3}}));
    const server::Sampling& s = f.engine.seen.front();
    CHECK(!s.greedy && s.temperature == 0.7f && s.top_k == 10u && s.top_p == 0.95f && s.has_seed);
    CHECK(s.softcap == 20.0f && s.logit_mult == 0.19611613513818404f);   // model properties: never request fields
  }
  {
    Fixture f;
    f.script(" to=user<|message|>a", kEot);
    f.post(body({{"temperature", 0}}));
    CHECK(f.engine.seen.front().greedy);   // greedy: the device argmax over the raw row (the engine's)
  }

  // Review Focus 4: the capped distribution inside filter_probs.
  const uint32_t V = 16;
  std::vector<float> z(V);
  for (uint32_t i = 0; i < V; ++i) z[i] = 60.0f - 9.0f * float(i) + (i % 3 ? 4.5f : 0.0f);   // raw logits up to 64
  server::Sampling s = g_defaults;
  s.top_k = 0;
  s.top_p = 1.0f;
  const server::FilteredProbs f = server::filter_probs(z.data(), V, s);
  CHECK_EQ(f.ids.size(), size_t(V));
  std::vector<double> want(V);
  double sum = 0;
  float top = cap(*std::max_element(z.begin(), z.end()));
  for (uint32_t i = 0; i < V; ++i) sum += want[i] = std::exp(double(cap(z[i]) - top));
  for (double& w : want) w /= sum;
  for (size_t r = 0; r < f.ids.size(); ++r) CHECK(std::fabs(f.probs[r] / f.mass - want[f.ids[r]]) < 1e-12);
  // the uncapped distribution is far sharper (the raw row at T 1)
  std::vector<double> raw(V);
  double rsum = 0;
  for (uint32_t i = 0; i < V; ++i) rsum += raw[i] = std::exp(double(z[i]) - double(z[0]));
  for (double& w : raw) w /= rsum;
  // a seeded 10^5-draw chi-square: against the capped p (df 15, chi2_0.99 = 30.578) and against the uncapped one
  std::mt19937_64 rng(2310);
  std::vector<uint64_t> n(V, 0);
  const int draws = 100000;
  for (int i = 0; i < draws; ++i) ++n[server::draw(f, rng)];
  double chi_capped = 0, chi_raw = 0;
  for (uint32_t i = 0; i < V; ++i) {
    const double e = want[i] * draws, er = std::max(raw[i] * draws, 1e-9);
    chi_capped += (double(n[i]) - e) * (double(n[i]) - e) / e;
    chi_raw += (double(n[i]) - er) * (double(n[i]) - er) / er;
  }
  CHECK(chi_capped < 30.578);
  CHECK(chi_raw > 1e4);
  // top-k / top-p cut by the RAW order (the cap is monotone): the same ids as the capped row's, at most 64
  server::Sampling k4 = g_defaults;
  k4.top_k = 4;
  const server::FilteredProbs f4 = server::filter_probs(z.data(), V, k4);
  CHECK(f4.ids.size() <= 4u);
  for (size_t r = 0; r < f4.ids.size(); ++r) CHECK_EQ(f4.ids[r], f.ids[r]);
  // The acceptance paths see the same capped p: filter_probs on the raw row with the cap == on a pre-capped row with
  // softcap 0 (exactly), so accept_point_mass / accept_sampled (which call it) draw identically.
  std::vector<float> pre(V);
  for (uint32_t i = 0; i < V; ++i) pre[i] = cap(z[i]);
  server::Sampling plain = s;
  plain.softcap = 0.0f;
  plain.logit_mult = 1.0f;
  const server::FilteredProbs fp = server::filter_probs(pre.data(), V, plain);
  CHECK(fp.ids == f.ids && fp.probs == f.probs && fp.mass == f.mass);
  std::vector<float> rows(size_t(V) * 3), pre_rows(size_t(V) * 3);
  for (int r = 0; r < 3; ++r)
    for (uint32_t i = 0; i < V; ++i) {
      rows[r * V + i] = z[(i + r) % V];
      pre_rows[r * V + i] = pre[(i + r) % V];
    }
  const uint32_t drafts[2] = {3, 1};
  for (uint64_t seed = 0; seed < 200; ++seed) {
    std::mt19937_64 a(seed), b(seed);
    const server::AcceptResult x = server::accept_point_mass(rows.data(), drafts, 2, V, V, s, a);
    const server::AcceptResult y = server::accept_point_mass(pre_rows.data(), drafts, 2, V, V, plain, b);
    CHECK(x.accepted == y.accepted && x.next_token == y.next_token);
  }
  // Other models: softcap 0, multiplier 1 - filter_probs exactly as before (the raw row).
  server::Sampling other;
  other.greedy = false;
  other.top_k = 0;
  other.top_p = 1.0f;
  const server::FilteredProbs fo = server::filter_probs(z.data(), V, other);
  for (size_t r = 0; r < fo.ids.size(); ++r) CHECK(std::fabs(fo.probs[r] / fo.mass - raw[fo.ids[r]]) < 1e-12);
  std::printf("sampling: defaults T 1.0 / top-k 64 / top-p 0.95 + the model's cap (never request fields); the capped "
              "softmax in filter_probs (chi-square %.1f < 30.58 vs capped, %.0f vs raw), the acceptance paths on the "
              "same p, softcap 0 unchanged\n", chi_capped, chi_raw);
}

// ---- stop_strings, cached tokens ---------------------------------------------------------------------------------
void case_stop_strings() {
  json g = json::parse(slurp(g_dir + "/generation_config.json"));
  CHECK(!g.contains("stop_strings"));
  CHECK(server::generation_stop_strings(g).empty());
  g["stop_strings"] = "END";
  CHECK(server::generation_stop_strings(g) == std::vector<std::string>{"END"});
  g["stop_strings"] = json::array({"END", "STOP"});
  CHECK((server::generation_stop_strings(g) == std::vector<std::string>{"END", "STOP"}));
  for (const json& bad : {json(3), json::array({"a", 1}), json::object()}) {
    g["stop_strings"] = bad;
    bool threw = false;
    try {
      (void)server::generation_stop_strings(g);
    } catch (const std::invalid_argument& e) {
      threw = std::string(e.what()).find("stop_strings") != std::string::npos;
    }
    CHECK(threw);
  }
  server::Options o = muse_options();
  o.stop_strings = {"END"};
  {
    Fixture f(o);
    f.script(" to=user<|message|>Hello END more", kEot);
    const json r = f.post(body());
    CHECK_EQ(r.at("choices").at(0).at("finish_reason"), std::string("stop"));
    CHECK_EQ(r.at("choices").at(0).at("message").at("content"), std::string("Hello "));   // cut at the stop string
  }
  {
    Fixture f(o);   // the request's own stop beside generation_config's
    f.script(" to=user<|message|>Hi HALT x END", kEot);
    const json r = f.post(body({{"stop", "HALT"}}));
    CHECK_EQ(r.at("choices").at(0).at("message").at("content"), std::string("Hi "));
  }
  std::printf("stop_strings: a string or a list (refused naming the key otherwise), merged with a request's stop\n");
}

void case_cached_tokens() {
  MuseTok tok;
  RealTemplate tmpl(g_dir);
  StateMockEngine engine;
  engine.blk = 64;
  engine.maxlen = 4096;
  engine.kv.assign(4096, 0);
  engine.words = tok.raw(" to=user<|message|>abcdefgh");
  engine.words.erase(engine.words.begin(), engine.words.begin() + 19);   // generate "a".."h" characters
  struct Alloc : server::HostAlloc {
    void* alloc(size_t n) override { return std::malloc(n); }
    void free(void* p, size_t) override { std::free(p); }
  } alloc;
  server::Options o = muse_options();
  o.prefix_cache_bytes = size_t(64) << 20;
  o.prefix_alloc = &alloc;
  server::Server server({tok, tmpl, engine}, o);
  CHECK(server.start());
  httplib::Client c("127.0.0.1", server.bound_port());
  json b = body({{"max_tokens", 8}, {"temperature", 0}});
  const auto r1 = c.Post("/v1/chat/completions", b.dump(), "application/json");
  const auto r2 = c.Post("/v1/chat/completions", b.dump(), "application/json");
  CHECK(r1 && r2 && r1->status == 200 && r2->status == 200);
  const json j1 = json::parse(r1->body), j2 = json::parse(r2->body);
  CHECK_EQ(j1.at("usage").at("prompt_tokens_details").at("cached_tokens"), 0);
  const uint32_t cached = j2.at("usage").at("prompt_tokens_details").at("cached_tokens");
  CHECK(cached > 0);
  CHECK(engine.state_loads > 0);
  server.stop();
  std::printf("cached_tokens: a restored Muse request reports %u cached of %u\n", cached,
              j2.at("usage").at("prompt_tokens").get<uint32_t>());
}

// ---- b70-serve's template rule (spec 23 decision 1), before the device -----------------------------------------
void case_template_choice() {
  const std::string older_dir = g_dir + "/../../model/muse_glimmer";   // the Vishva007 repos' older template
  const cli::muse::TemplateChoice ok = cli::muse::check_serve_template(g_dir, "");
  CHECK(ok.original && ok.file.empty());
  CHECK_EQ(ok.sha256, std::string(cli::muse::kOriginalTemplateSha256));
  bool threw = false;
  try {
    (void)cli::muse::check_serve_template(older_dir, "");
  } catch (const std::runtime_error& e) {
    const std::string m = e.what();
    threw = m.find("pass --chat-template FILE") != std::string::npos && m.find("114f55eb") != std::string::npos &&
            m.find("older template") != std::string::npos;
    if (!threw) std::fprintf(stderr, "template refusal: %s\n", m.c_str());
  }
  CHECK(threw);
  // --chat-template FILE wins: the original's over the older snapshot, or the older one deliberately (not original).
  const cli::muse::TemplateChoice over = cli::muse::check_serve_template(older_dir, g_dir + "/chat_template.jinja");
  CHECK(over.original && over.file == g_dir + "/chat_template.jinja");
  const cli::muse::TemplateChoice deliberate =
      cli::muse::check_serve_template(g_dir, older_dir + "/chat_template.jinja");
  CHECK(!deliberate.original);
  CHECK_EQ(deliberate.sha256, std::string(cli::muse::kOlderTemplateSha256));
  threw = false;
  try {
    (void)cli::muse::check_serve_template(g_dir, g_dir + "/no_such_template.jinja");
  } catch (const std::runtime_error&) {
    threw = true;
  }
  CHECK(threw);
  std::printf("template rule: the snapshot's original accepted, the older one refused naming --chat-template, "
              "--chat-template FILE wins either way\n");
}

}  // namespace

int main(int argc, char** argv) {
  g_dir = argc > 1 ? argv[1] : "tests/tokenizer/muse";
  g_defaults = server::generation_sampling(json::parse(slurp(g_dir + "/generation_config.json")));
  g_defaults.logit_mult = 0.19611613513818404f;   // model::MuseGlimmerDesc's out_mult / softcap (b70-serve's)
  g_defaults.softcap = 20.0f;
  case_kwargs();
  case_request_shapes();
  case_turn();
  case_sampling();
  case_stop_strings();
  case_cached_tokens();
  case_template_choice();
  std::printf("muse_server_test OK: 7 cases\n");
  return 0;
}
